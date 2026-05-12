# Simple KV Offload

`vllm/v1/simple_kv_offload` implements a minimal CPU-backed prefix cache for
vLLM V1. It is not a second scheduler and it does not own GPU allocation.
Instead, it mirrors the normal KV-cache data structures on CPU, uses those
structures to answer "can CPU provide the next prefix blocks?", and then copies
whole block payloads between the worker's GPU KV cache tensors and pinned CPU
tensors.

The connector is registered as `SimpleCPUOffloadConnector` in
`vllm/distributed/kv_transfer/kv_connector/factory.py`. The public connector
class lives in
`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`
and delegates almost all real work to:

- `manager.py`: scheduler-side policy, block bookkeeping, CPU prefix-cache
  lookup, transfer metadata.
- `worker.py`: worker-side CPU tensor allocation, KV-cache storage discovery,
  asynchronous copy launch and completion polling.
- `metadata.py`: flat scheduler-to-worker copy plans and worker-to-scheduler
  store completions.
- `copy_backend.py` and `cuda_mem_ops.py`: a background-thread wrapper around
  `cuMemcpyBatchAsync`.

## High-Level Shape

Simple CPU offload works by creating a second `KVCacheCoordinator` for CPU:

```text
GPU local cache
  KVCacheManager
    KVCacheCoordinator
      BlockPool + hash->GPU block map

CPU offload cache
  SimpleCPUOffloadScheduler
    CPU KVCacheCoordinator
      CPU BlockPool + hash->CPU block map
```

The CPU coordinator uses the same `KVCacheConfig.kv_cache_groups` as the GPU
coordinator, but with `num_blocks` scaled to the configured CPU byte budget.
That is the main design choice: CPU offload reuses the same prefix-cache
correctness rules as the normal GPU cache. Full attention still scans
left-to-right, sliding-window attention still scans according to its window,
Mamba/null padding remains visible as null blocks, and hybrid models still use
the hybrid coordinator logic.

The worker side separately allocates CPU tensors that mirror the physical GPU
KV-cache storage. It deduplicates shared backing storage, converts each unique
storage into a `[num_blocks, block_bytes]` byte-like view, and allocates a CPU
view with the same per-block layout. Transfers are then just block ID pairs:

```text
store: GPU block id -> CPU block id
load:  CPU block id -> GPU block id
```

For each copy, `copy_blocks()` builds all source addresses, destination
addresses, and sizes for every KV tensor and every block, then submits one
batched CUDA copy on a low-priority CUDA stream.

## KVCacheManager vs KVCacheCoordinator

`KVCacheManager` is the scheduler-facing request API. The scheduler asks it to
find prefix hits, allocate slots, cache newly completed blocks, free a request,
remove out-of-window blocks, and expose block IDs. It hides the internal group
layout behind `KVCacheBlocks`, whose shape is:

```text
KVCacheBlocks.blocks[group_index][block_index] -> KVCacheBlock
```

`KVCacheCoordinator` is the group fan-out layer underneath the manager. It owns
the shared `BlockPool` and one `SingleTypeKVCacheManager` per KV-cache group.
It answers questions that require combining groups:

- how many blocks are needed across all groups;
- how to allocate newly hit local blocks plus external blocks;
- how to allocate new blocks for each group;
- how to cache/free/remove skipped blocks across groups;
- how to find the longest prefix hit in unitary or hybrid layouts.

There are three coordinator modes:

- `KVCacheCoordinatorNoPrefixCache`: no prefix-cache lookup.
- `UnitaryKVCacheCoordinator`: one KV-cache group, one attention type.
- `HybridKVCacheCoordinator`: multiple KV-cache groups, often full attention
  plus sliding-window, chunked-local, MLA, or Mamba-like groups.

Simple CPU offload uses both concepts. The normal GPU `KVCacheManager` remains
the authority for GPU blocks. The offload manager creates only a CPU
`KVCacheCoordinator`, because it needs the coordinator's block pool and cache
lookup machinery, not another full request-level manager.

## Life Of A Block

### 1. Birth On GPU

The scheduler calls `KVCacheManager.allocate_slots()` for a request. The manager
delegates to the GPU coordinator, which allocates `KVCacheBlock`s from the GPU
`BlockPool`. On the worker, those block IDs index into the preallocated GPU KV
cache tensors.

After the model computes full blocks, `KVCacheManager.cache_blocks()` calls
`BlockPool.cache_full_blocks()`. This stamps each full non-null block with a
group-aware block hash and inserts it into the GPU hash map:

```text
(block_hash, group_id) -> GPU KVCacheBlock
```

At this point the block is both a live request block and a GPU prefix-cache
entry. Later, when the request releases it, the block can sit in the GPU free
queue with `ref_cnt == 0` while still remaining discoverable by hash until the
allocator evicts it.

### 2. Store Selection

Simple CPU offload can decide to copy a GPU cached block to CPU in two ways.

In eager mode, `_prepare_eager_store_specs()` follows requests through
`SchedulerOutput`. It accumulates allocated block IDs per request and per group,
then scans only blocks whose KV data is confirmed computed:

```text
confirmed_tokens = request.num_computed_tokens - output_placeholders
ready_blocks_g = confirmed_tokens // group_block_size
```

It skips null blocks, skips blocks with no hash, skips blocks already scheduled
or already present in the CPU hash map, and allocates one CPU block for each
remaining GPU block.

In lazy mode, `_prepare_lazy_store_specs()` walks the GPU free queue with a
cursor. It looks for cached, non-null, eviction-near GPU blocks that are not yet
present on CPU. This is more demand-driven: it tries to keep enough
free-or-offloaded GPU blocks available, rather than offloading every newly
computed block immediately.

In both modes, the CPU block is allocated from the CPU `BlockPool` and stamped
with the same group-aware block hash before the copy starts. The GPU block is
`touch()`ed so the allocator cannot reuse it while the asynchronous store reads
from it.

That extra reference is the handoff between request lifetime and transfer
lifetime. Without it, the owning request could finish, `free_blocks()` could put
the GPU block back on the free queue, and a later allocation could evict its
hash and overwrite the same physical slot before the GPU-to-CPU DMA actually
reads it. The CPU block would then be registered under the old hash but contain
bytes from a different request. `touch()` makes the transfer itself a temporary
owner of the source block; store completion releases that ownership.

Other requests can still share the touched block if they hit the same prefix
hash. `touch()` does not remove the block from the prefix-cache hash map or make
it private to the transfer. It only prevents allocator-style reuse: a block with
positive `ref_cnt` is not an eviction candidate and cannot be handed out as a
fresh slot for different KV bytes.

### 3. GPU To CPU Copy

`build_connector_meta()` packs all store pairs for the scheduler step into flat
lists:

```text
store_event = N
store_gpu_blocks = [g0, g1, ...]
store_cpu_blocks = [c0, c1, ...]
```

The worker receives the metadata. `SimpleCPUOffloadWorker.get_finished()`
launches the GPU-to-CPU copy through `DmaCopyBackend.launch_copy()`. The backend
runs on a background thread, submits `cuMemcpyBatchAsync` on the store stream,
records a CUDA event, and lets the worker poll that event in later steps.

Stores are reported to the scheduler with `SimpleCPUOffloadWorkerMetadata`.
For tensor/pipeline parallel setups, the scheduler waits until every worker rank
has reported the same store event before treating it as complete.

### 4. CPU Residency

When a store event completes, `_process_store_completion()` inserts each CPU
block into the CPU hash map:

```text
(block_hash, group_id) -> CPU KVCacheBlock
```

Then it frees the CPU block and the extra GPU touch reference. "Freeing" the CPU
block here does not erase the hash. It makes the block an evictable prefix-cache
entry in the CPU `BlockPool`, exactly like a GPU prefix-cache block in the GPU
free queue. If the CPU pool later needs the slot, `BlockPool.get_new_blocks()`
will evict the hash metadata before reuse.

So a resident CPU block is usually:

```text
ref_cnt == 0
block_hash != None
present in CPU hash map
physically backed by CPU KV tensor storage
```

### 5. CPU Hit Lookup

For a new request, the scheduler first asks the GPU `KVCacheManager` for local
prefix hits. Then it asks the connector for additional external hits:

```text
connector.get_num_new_matched_tokens(request, local_hit_tokens)
```

Simple CPU offload skips the GPU-hit prefix and asks the CPU coordinator to find
the longest CPU hit in the remaining block hashes. A positive result returns
`(hit_length, True)`, where `True` tells the scheduler this will be an async KV
load.

The scheduler then allocates GPU blocks for those external tokens with
`num_new_tokens = 0` and `num_external_computed_tokens > 0`, moves the request to
`WAITING_FOR_REMOTE_KVS`, and calls `update_state_after_alloc()`.

### 6. CPU To GPU Copy

`update_state_after_alloc()` re-runs the CPU coordinator lookup to recover the
actual CPU blocks. It then maps those CPU blocks to the tail of the computed
range in each GPU group:

```text
total_computed_tokens = already_computed + external_tokens
n_computed_g = ceil(total_computed_tokens / group_block_size)
gpu_ext_start = n_computed_g - num_external_blocks_for_group
```

Null CPU blocks are skipped because they do not correspond to real KV bytes.
The real CPU and GPU blocks are touched to pin both sides while the async load is
in flight. This is the same transfer-lifetime rule in the opposite direction:
the CPU source block must not be evicted and reused before the CPU-to-GPU copy
reads it, and the GPU destination block must not be freed or reused before the
copy fills it. The CPU source can still serve other same-hash load hits while
it is touched; it just cannot be evicted as scratch space. `build_connector_meta()`
then emits:

```text
load_event = M
load_cpu_blocks = [c0, c1, ...]
load_gpu_blocks = [g0, g1, ...]
load_event_to_reqs[M] = [request_id, ...]
```

The worker launches the CPU-to-GPU copy in `get_finished()` and reports
completed loads through `finished_recving`. Once the scheduler sees the request
in `finished_recving`, it calls `KVCacheManager.cache_blocks()` for the loaded
tokens and moves the request back to `WAITING` or `PREEMPTED`. If the load was a
full-prompt hit, vLLM still recomputes the last token so it can produce logits.

At this point the same prefix is hot in GPU memory again, and the next request
can hit the normal local GPU prefix cache without going to CPU.

### 7. Finish And Cleanup

`SimpleCPUOffloadScheduler.request_finished()` returns `(False, None)`: the
normal scheduler is still allowed to free request blocks immediately. In-flight
stores and loads are safe because Simple CPU offload has already added temporary
`touch()` references to the exact blocks involved in the DMA. Completion paths
release those references. This keeps the scheduler fast and local: it does not
need to delay request cleanup just because a background transfer is still using
some of the request's blocks.

If a request is preempted, worker-side `handle_preemptions()` synchronizes all
in-flight transfer events before blocks can be reused.

## Eager vs Lazy Offload

Eager mode is request-oriented. It notices newly confirmed full blocks for
scheduled requests and copies them to CPU as soon as there is CPU capacity. The
code notes one sharp edge: if a request finishes in the same step as its last
full block, that block can be missed because eager mode only stores blocks whose
KV data was confirmed before the current step.

Lazy mode is cache-pressure-oriented. It scans the GPU free queue and offloads
cached blocks that are close to eviction. It avoids copying blocks already
present on CPU and keeps a cursor so each step continues from the prior scan.
The target scan budget is estimated from the configured max batched tokens and
the per-group attention behavior: Mamba, sliding-window, and full attention each
contribute a different number of blocks.

Both modes share the same transfer and CPU residency machinery after the block
pairs have been chosen.

## Hybrid Models And HMA

The codebase uses HMA to mean the hybrid KV cache manager/allocator path:
multiple KV-cache groups can exist at once because different layers have
different cache lifecycles. A Gemma-style model may have full-attention groups
and sliding-window groups. A Mamba hybrid may have attention groups plus
state-space groups. Each group can have its own block table, block size, skip
rule, and cache-hit algorithm.

Simple CPU offload explicitly opts into this path by subclassing `SupportsHMA`.
That matters in three places.

First, connector creation rejects non-HMA connectors when the hybrid manager is
enabled. By default, setting `--kv-transfer-config` makes vLLM disable HMA unless
the user explicitly opts back in with `--no-disable-hybrid-kv-cache-manager`.
Simple CPU offload can pass that check because `SimpleCPUOffloadConnector` is a
`SupportsHMA` connector and implements `request_finished_all_groups()`.

Second, the CPU offload scheduler preserves group structure. It derives the CPU
KV config from the GPU config by keeping the same `kv_cache_groups` and scaling
only the number and size of backing CPU tensors. CPU prefix lookup therefore
uses `HybridKVCacheCoordinator` whenever the GPU side does. The hybrid
coordinator groups identical specs, puts full attention first when present,
aligns hits to the LCM of group block sizes, and iteratively reduces the
candidate hit length until every attention type agrees.

Third, transfer planning is group-aware but worker copy is storage-aware. The
scheduler builds load and store pairs across all groups, using each group's own
block size when deciding which blocks correspond to a token range. It skips null
blocks introduced by sliding-window/Mamba padding. The worker does not need to
know the semantic group of each block; it copies the same logical block IDs
across every unique KV backing tensor. This works because HMA's physical layout
still makes a block ID meaningful for every registered backing storage view.

The full-attention group has one special role in `SimpleCPUOffloadScheduler`:
`fa_gidx` is used as the anchor for counting already materialized full prefix
blocks when building CPU load pairs. The scheduler asserts that such a group
exists. That means the current simple offload path is shaped around hybrid
models that include a full-attention KV group; pure attention-free cases need
careful validation against this assumption.

Current HMA-related limits are also explicit:

- CPU offload asserts `dcp_world_size == 1` and `pcp_world_size == 1`.
- Hybrid coordinator itself asserts DCP/PCP are not supported for hybrid
  attention.
- Null blocks are metadata placeholders only; they are never copied.
- KV events plus HMA have restrictions elsewhere in the test/config path, so
  event coverage is not the same for HMA and non-HMA models.

## Design In One Sentence

Simple CPU offload treats CPU memory as another prefix-cache block pool with the
same hashes and group semantics as the GPU pool, while the GPU `KVCacheManager`
remains the only authority for allocating blocks that attention will read.
