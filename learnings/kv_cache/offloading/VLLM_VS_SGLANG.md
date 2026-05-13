# Mooncake KV Cache Interface: vLLM vs SGLang

This note compares the Mooncake store integration in:

- vLLM fork branch: `ivanium/vllm@feat/mooncake-store-int`, specifically
  `MooncakeStoreConnector`.
- Local SGLang checkout: `/Users/daole/git/sglang`, specifically HiCache's
  Mooncake storage backend.

The closest apples-to-apples comparison is vLLM
`MooncakeStoreConnector` vs SGLang `HiCacheStorage` +
`MooncakeStore`. SGLang also has a separate Mooncake TransferEngine path for
PD disaggregation; that is adjacent but not the same interface as its HiCache
Mooncake L3 backend.

## Executive Summary

vLLM integrates Mooncake as a `KVConnectorBase_V1` implementation. The
connector speaks directly to vLLM's scheduler/worker KV-transfer lifecycle:
the scheduler asks the connector for external prefix hits, allocates GPU blocks
for those hits, and the worker reads/writes Mooncake objects directly into the
registered vLLM GPU KV-cache buffers.

SGLang integrates Mooncake as one backend of a broader hierarchical cache.
HiCache owns the cache policy and metadata: GPU memory is L1, host memory is
L2, and Mooncake is L3. The Mooncake backend implements a generic storage
interface over registered host-memory pages; HiCache handles radix-tree
matching, prefetch, write-back, GPU-host movement, policies, and multi-rank
synchronization.

The short version:

- vLLM is more direct and connector-native. It is attractive if the goal is
  fast integration with vLLM scheduling and direct Mooncake I/O into GPU KV
  blocks.
- SGLang is more architected as a cache hierarchy. It is attractive if the goal
  is replaceable L3 storage, runtime policy control, host-tier reuse, and
  support for richer model/cache layouts.

## Source Map

vLLM fork:

- `vllm/distributed/kv_transfer/kv_connector/v1/base.py`
  defines scheduler-side and worker-side connector hooks such as
  `get_num_new_matched_tokens`, `update_state_after_alloc`,
  `build_connector_meta`, `register_kv_caches`, `start_load_kv`,
  `save_kv_layer`, `wait_for_save`, and `get_finished`.
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py`
  is the public connector class.
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py`
  maps vLLM scheduler state to per-request load/save metadata and performs
  prefix-hit lookup through a ZMQ lookup client.
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py`
  registers GPU KV-cache buffers with `MooncakeDistributedStore`, starts
  background store/load threads, and issues Mooncake batch operations.
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py`
  defines key construction and request metadata.
- `scripts/mooncake/README.md` documents benchmark setup and notes that
  cross-DP hits require a fixed `PYTHONHASHSEED`.

SGLang:

- `python/sglang/srt/mem_cache/hicache_storage.py` defines the generic
  `HiCacheStorage` interface and the v2 `PoolTransfer` API for sidecar pools.
- `python/sglang/srt/mem_cache/storage/backend_factory.py` registers
  `mooncake`, `hf3fs`, `nixl`, `aibrix`, `file`, `dynamic`, and other storage
  backends behind the same factory.
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
  implements the Mooncake L3 backend.
- `python/sglang/srt/managers/cache_controller.py` owns CPU-GPU movement,
  storage prefetch/write-back threads, storage attach/detach, and TP
  synchronization.
- `python/sglang/srt/mem_cache/hiradix_cache.py` integrates HiCache with the
  radix prefix cache, hit counting, write policy, and prefetch policy.
- `python/sglang/srt/disaggregation/mooncake/conn.py` is the separate
  Mooncake PD-transfer path.

## Side-By-Side

| Dimension | vLLM `MooncakeStoreConnector` | SGLang HiCache Mooncake |
| --- | --- | --- |
| Primary abstraction | vLLM `KVConnectorBase_V1` | Generic `HiCacheStorage` backend |
| Cache architecture | External KV store connector attached to scheduler/worker lifecycle | Hierarchical cache: GPU L1, host L2, Mooncake L3 |
| Data path | Mooncake reads/writes registered vLLM KV cache buffers directly | Mooncake reads/writes registered host KV pages; separate HiCache path moves host pages to/from GPU |
| Lookup path | Scheduler asks worker rank 0 over ZMQ; worker calls `batch_is_exist` | HiRadix local match first; storage prefetch thread queries L3 with `batch_exists` |
| Granularity | vLLM block hashes, adjusted for PCP/DCP block size | HiCache page hashes, page-aligned |
| Key shape | `model@tp_rank@pcp@dcp@pp@chunk_hash` | page hash plus rank/layout suffixes like `_<rank>_k`, `_<rank>_v`, or MLA `_k` |
| Async model | Python worker threads for store/load, completion via `get_finished` | CPU-GPU streams plus storage prefetch/backup threads, ack queues, prefetch stop policies |
| Replacement story | New connector class per backend | Backend factory; Mooncake is one L3 backend among several |
| Runtime control | Mostly startup `kv_transfer_config` and env config | Startup or runtime attach/detach of storage backend and policies |
| Multi-rank handling | Lookup expands required TP/PP keys and returns first missing prefix | TP groups use all-reduce to agree on storage hit/completion |
| Layout support | Detects vLLM KV tensor storage by stride; optional cross-layer blocks | Requires host layouts such as `page_first`, `page_first_direct`, `page_head`, `page_first_kv_spilt` |
| Extra state | KV only | v2 `PoolTransfer` supports KV plus side pools such as Mamba/indexer state |

## vLLM Design

`MooncakeStoreConnector` is a standard vLLM V1 KV connector. The public class
mostly delegates to a scheduler-side object and a worker-side object.

On the scheduler side:

1. `get_num_new_matched_tokens()` aligns the prompt to the connector block
   size, then asks `LookupKeyClient.lookup()` how many prefix tokens exist in
   Mooncake.
2. `update_state_after_alloc()` records the GPU block IDs allocated for
   externally matched tokens.
3. `build_connector_meta()` creates `MooncakeStoreConnectorMetadata` containing
   `ReqMeta` objects for loads and saves in the current scheduler step.
4. `request_finished()` can delay freeing blocks until async stores complete.

On the worker side:

1. `register_kv_caches()` registers vLLM KV-cache backing storage with
   Mooncake and records each segment's base address and per-block byte length.
2. `LookupKeyServer` runs in worker rank 0 and answers scheduler lookup
   requests over IPC.
3. `KVCacheStoreSendingThread` deduplicates with `batch_is_exist`, then stores
   missing keys with `batch_put_from_multi_buffers`.
4. `KVCacheStoreRecvingThread` loads keys with
   `batch_get_into_multi_buffers` into the allocated GPU KV blocks.
5. `get_finished()` both enqueues new I/O and polls previously completed
   send/receive work.

The connector does not use vLLM's layerwise `start_load_kv`,
`wait_for_layer_load`, `save_kv_layer`, or `wait_for_save` hooks. Those are
no-ops. The actual work is issued from `get_finished()` after compute has been
launched, using a CUDA event to keep stores from reading KV before the model
has produced it.

## SGLang Design

SGLang's Mooncake integration is not a scheduler connector. It is an L3 storage
backend under HiCache.

HiCache organizes:

```text
GPU KV cache      -> L1
host KV pool      -> L2
Mooncake backend  -> L3
```

`HiRadixCache` tracks prefix metadata in a radix tree. Each node can have GPU
indices, host indices, page hashes, hit counts, and storage state. When a
request arrives, HiRadix first matches local GPU/host prefixes. If enough
missing tokens may exist in L3, it allocates host pages and asks
`HiCacheController.prefetch()` to fetch them asynchronously from storage.

`HiCacheController` owns the mechanics:

- CPU-to-GPU and GPU-to-CPU movement through `write()`, `load()`,
  `start_writing()`, and `start_loading()`.
- Storage read/write queues through `prefetch_thread_func()` and
  `backup_thread_func()`.
- Storage backend creation and runtime attach/detach.
- TP synchronization for storage hit counts and completed prefetch length.

`MooncakeStore` implements the storage operations. It registers the host KV
pool buffer with Mooncake, supports batch existence checks, and uses Mooncake
zero-copy calls:

```text
batch_put_from(...)
batch_get_into(...)
batch_is_exist(...)
```

For MHA, SGLang stores K and V as separate objects per page and rank suffix.
For MLA, it stores one `_k` object. With `tp_lcm_size` and `page_head`,
SGLang can split head shards so deployments with different TP sizes can share
Mooncake objects.

SGLang's Mooncake PD disaggregation path is separate. That path implements
`BaseKVManager`, `BaseKVSender`, and `BaseKVReceiver` over Mooncake
TransferEngine and transfers KV by source/destination indices between prefill
and decode workers. HiCache Mooncake can be used together with PD mode, but it
is not the same interface.

## Interface For Integrating With HiCache

HiCache integration has two different boundaries that are easy to blur:

1. A storage backend boundary, implemented by `HiCacheStorage`.
2. A cache-controller boundary, implemented by `HiCacheController` and
   `HiRadixCache`, that decides when pages move between GPU, host, and
   storage.

The storage backend boundary is intentionally page-oriented and host-memory
oriented. A backend is expected to accept stable page keys, know how to read or
write registered host pages, and report the longest contiguous prefix present
in storage. The minimum backend surface is:

- `register_mem_pool_host(mem_pool_host)`: bind the primary host KV pool. For
  Mooncake this registers the pool's backing tensor with
  `MooncakeDistributedStore`.
- `batch_exists(keys, extra_info) -> int`: return how many page keys exist
  consecutively from the start. HiCache relies on prefix semantics here, not a
  sparse hit bitmap.
- `batch_get_v1(keys, host_indices, extra_info) -> List[bool]`: fetch pages
  directly into host-pool slots identified by token/page indices.
- `batch_set_v1(keys, host_indices, extra_info) -> List[bool]`: write pages
  from host-pool slots into storage.
- Legacy `get`, `set`, `batch_get`, and `batch_set`: still exist for generic
  or older backends, but the high-throughput Mooncake/NIXL/HF3FS path uses the
  v1 zero-copy page calls.
- `clear`, `close`, and `get_stats`: optional operational hooks.

The data ownership contract is different from vLLM's connector contract.
HiCache does not ask a backend to allocate GPU blocks or know request objects.
The backend receives only page keys and host indices. The controller owns host
allocation, GPU allocation, device-host copies, prefetch/write-back queues, and
multi-rank synchronization.

The v2 storage interface extends the same idea to auxiliary state pools:

- `PoolTransfer` names a pool such as `kv`, `mamba`, or `indexer`, and carries
  `host_indices`, optional `device_indices`, storage `keys`, and a
  `PoolHitPolicy`.
- `batch_exists_v2(keys, pool_transfers, extra_info) -> PoolTransferResult`
  returns a usable KV prefix after intersecting KV availability with auxiliary
  pool availability.
- `batch_get_v2` and `batch_set_v2` move all listed pools through the same
  page-key interface and return per-pool success lists.

This matters for models whose reusable state is not just attention K/V. For
example, an indexer pool can require `ALL_PAGES`, while a Mamba-like state pool
can use `TRAILING_PAGES` if only tail state is needed to resume computation.

`HiCacheStorageExtraInfo` is the escape hatch for backend-specific context.
The current important field is `prefix_keys`: the already-known prefix page
hashes preceding the batch being queried or written. Backends that need
chained-key context, remote metadata hints, or request-local namespace context
can consume it without changing the main method signatures.

### Runtime Attach Contract

HiCache can attach storage at startup or later through the radix cache control
path. Attaching a backend creates a `HiCacheStorageConfig` with:

- topology: `tp_rank`, `tp_size`, `pp_rank`, `pp_size`, `attn_cp_rank`,
  `attn_cp_size`;
- model/storage shape: `is_mla_model`, `is_page_first_layout`, `model_name`;
- optional heterogeneous-TP sharing knobs: `tp_lcm_size` and
  `should_split_heads`;
- backend options under `extra_config`;
- metrics flag: `enable_storage_metrics`.

`HiCacheController.attach_storage_backend()` then constructs the backend via
`StorageBackendFactory`, registers the host pool, selects either generic
copying or zero-copy page functions, creates storage prefetch/backup threads,
and creates a TP synchronization group for storage decisions. Detach is also a
first-class operation: it stops storage threads, destroys the storage TP group,
closes the backend if supported, and resets page get/set functions.

The practical integration requirement is that attach/detach must happen with
no in-flight requests. Otherwise background storage threads can still hold
host-page references while the backend is being removed.

### Page Key And Layout Contract

HiCache page keys are derived from token pages using the same chained-prefix
hashing model as the radix cache. Storage operates at `page_size` granularity,
so both lookup and prefetch truncate to page-aligned prefixes.

Mooncake then expands each logical page key into one or more physical object
keys:

- MHA normally stores two objects per page: `_<rank>_k` and `_<rank>_v`.
- MLA stores one `_k` object per page.
- PP and attention-CP suffixes are included when enabled.
- With `tp_lcm_size` and `page_head`, MHA keys can be split by target rank so
  deployments with different TP sizes can share objects.
- `extra_backend_tag` can prepend an additional namespace to all object keys.

The host layout is part of the interface. The Mooncake backend currently
requires `page_first`, `page_first_direct`, `page_head`, or
`page_first_kv_spilt`, because it needs contiguous per-page host buffers for
zero-copy Mooncake calls. A vLLM integration that wants to reuse this interface
would need either a compatible host page pool or an adapter that can expose
vLLM KV blocks as page-contiguous host buffers.

### Request Flow Through HiCache

A request goes through HiCache in this order:

1. `HiRadixCache.match_prefix()` finds GPU and host prefix hits locally.
2. If the remaining page-aligned prefix is large enough, `prefetch_from_storage`
   allocates host pages and enqueues a controller `PrefetchOperation`.
3. The storage prefetch thread computes page hashes, calls
   `storage_backend.batch_exists`, and uses a TP all-reduce minimum so all TP
   ranks agree on the usable storage-hit length.
4. If the hit is below `prefetch_threshold` or host prefetch memory is rate
   limited, the operation is revoked and host pages are released.
5. Otherwise the IO thread calls `batch_get_v1` in storage batches, writes into
   host pages, and increments `completed_tokens` only after each page is ready.
6. The radix/cache scheduler can then load host pages to GPU with
   `HiCacheController.load()` and overlap per-layer host-to-device copies using
   the controller load stream and layer events.
7. On cache backup, GPU pages are first copied to host with `write()`, then
   written to storage by `write_storage()` according to the configured
   write-through or write-back policy.

That flow is the main semantic difference from vLLM's Mooncake connector.
vLLM asks the external store for hits before GPU allocation and loads directly
into allocated GPU blocks. HiCache first makes the storage hit materialize as
host pages, then uses its normal L2-to-L1 path to populate GPU memory.

## Interface For Integrating With vLLM

vLLM's integration boundary is `KVConnectorBase_V1`. Unlike HiCache, it is not
just a storage backend interface. It is a scheduler-and-worker contract for
making external KV visible to vLLM's normal request lifecycle.

The public connector is initialized with:

- `vllm_config`: includes `kv_transfer_config`, connector name, role, and
  connector-specific extra config.
- `role`: producer, consumer, or both, depending on the transfer mode.
- `kv_cache_config`: block size, KV cache groups, and layout information.

The scheduler-side surface is request-aware:

- `get_num_new_matched_tokens(request, num_computed_tokens) -> (count, async)`:
  report how many extra prompt tokens can be satisfied by the connector beyond
  local GPU prefix-cache hits. Returning `None` means lookup is not ready and
  the scheduler should try again later. Returning `async=True` lets the request
  enter a waiting state until the worker reports completion.
- `update_state_after_alloc(request, blocks, num_external_tokens)`: receive the
  physical vLLM blocks allocated for the external prefix. This is the key
  coupling point: the connector now knows exactly where external KV must land.
  The connector does not allocate these blocks; `KVCacheManager.allocate_slots`
  does, and the scheduler passes the resulting block mapping to the connector.
- `build_connector_meta(scheduler_output)`: package per-step transfer metadata
  for workers. This call resets the scheduler-side connector staging state.
- `update_connector_output(connector_output)`: consume worker completion
  information reported back through the executor path.
- `request_finished(request, block_ids) -> (delay_free_blocks, params)`: decide
  whether vLLM may free a request's blocks immediately or must keep them alive
  while an async save/send reads from them.

The worker-side surface is buffer-aware:

- `register_kv_caches(kv_caches)` or `register_cross_layers_kv_cache(...)`:
  bind the connector to vLLM's actual paged KV tensors. Mooncake-style
  connectors use this point to register GPU memory with the transfer engine.
- `bind_connector_metadata(meta)` and `clear_connector_metadata()`: set the
  scheduler-built plan for the current model execution.
- `start_load_kv(forward_context)`, `wait_for_layer_load(layer_name)`,
  `save_kv_layer(layer_name, kv_layer, attn_metadata)`, and
  `wait_for_save()`: the layerwise load/save interface for connectors that
  overlap transfer with attention execution.
- `get_finished(finished_req_ids) -> (finished_sending, finished_recving)`:
  report async transfer completions so the scheduler can unblock remote-load
  requests or finally free blocks kept alive after request finish.
- `get_block_ids_with_load_errors()`: expose failed external loads as concrete
  vLLM block IDs, allowing scheduler-side recovery or accounting.
- Stats, events, handshake metadata, and worker metadata hooks provide optional
  observability and out-of-band setup for more complex transfer modes.

So the vLLM connector contract is centered on scheduled requests and physical
KV blocks. The connector does not merely say "key X exists"; it tells the
scheduler how many new tokens can be skipped, receives the destination blocks
for those tokens, and then transfers bytes into those blocks before the request
continues.

### vLLM Request Flow

A connector-backed request generally goes through this path:

1. `KVCacheManager` finds the local GPU prefix hit.
2. The scheduler calls `get_num_new_matched_tokens()` for any additional
   external prefix.
3. The scheduler allocates vLLM KV blocks for the external hit and calls
   `update_state_after_alloc()`.
4. `build_connector_meta()` sends the transfer plan to workers as part of
   `SchedulerOutput`.
5. The worker binds that metadata and either starts layerwise loading through
   `start_load_kv()` or uses connector-specific async machinery.
6. Once the transfer is complete, `get_finished()` reports finished receiving
   request IDs and the scheduler can run the request with those tokens treated
   as computed.
7. When a request finishes, `request_finished()` can keep blocks alive until
   the connector reports that async saving/sending is done.

The Mooncake store branch described above bends this general interface toward
a whole-block async store/load model. It uses scheduler lookup to decide
external prefix length, `update_state_after_alloc()` to remember destination
GPU blocks, worker memory registration for direct Mooncake I/O, and
`get_finished()` to enqueue and poll background operations. It does not rely on
the layerwise `start_load_kv` / `save_kv_layer` path.

### Direct Contrast

| Concern | HiCache interface | vLLM connector interface |
| --- | --- | --- |
| Primary unit | Page key plus host page index | Request plus vLLM block ID |
| Main boundary | Storage backend under cache controller | Scheduler/worker connector lifecycle |
| Who owns policy | HiRadix/HiCacheController | Usually the connector plus scheduler hooks |
| Who owns GPU allocation | HiCacheController device allocator | vLLM scheduler/KVCacheManager |
| Backend visibility | Backend sees keys and host indices | Connector sees requests, blocks, metadata, and KV tensors |
| Hit return | Longest existing page prefix | Additional matched token count beyond local hit |
| Data destination | Host L2 first, then GPU L1 | Usually directly into vLLM paged GPU KV blocks |
| Async completion | Prefetch/write-back operations and controller queues | `get_finished()` request ID sets |
| Failure reporting | Per-page success lists and prefetch termination | Block IDs via `get_block_ids_with_load_errors()` |
| Extensibility | Replaceable storage backends and v2 side pools | Replaceable connectors, each may define its own metadata |

In short: HiCache abstracts storage below a hierarchical cache. vLLM abstracts
external KV movement as part of scheduling. A HiCache backend can be cleanly
swapped because it has no scheduler knowledge; a vLLM connector can be deeply
efficient because it sees the exact request state and destination GPU blocks.

### What A vLLM Adapter Would Need

To integrate vLLM with a HiCache-style interface, vLLM would need an adapter
layer between `KVConnectorBase_V1` and a HiCache-like controller/backend split:

- Scheduler adapter: translate `get_num_new_matched_tokens()` into page-key
  lookup semantics, including page/block alignment, local-prefix offset, and
  TP/PP/CP namespace inputs.
- Allocation adapter: translate `update_state_after_alloc()` from vLLM block
  IDs into host-page slots and eventual GPU block destinations, or explicitly
  choose a direct-GPU fast path that bypasses host L2.
- Metadata adapter: carry per-request page keys, prefix keys, host indices,
  destination GPU block IDs, and whether the request is waiting on async
  prefetch through `build_connector_meta()`.
- Worker adapter: register memory, issue storage `batch_get`/`batch_set`,
  coordinate host-to-GPU copies if using L2, and report completions through
  `get_finished()`.
- Error adapter: convert failed page loads into vLLM-visible load errors, for
  example via `get_block_ids_with_load_errors()` or recompute fallback.
- Policy adapter: expose HiCache-like knobs for `prefetch_threshold`,
  `best_effort`/`wait_complete`/`timeout` behavior, write policy, storage batch
  size, attach/detach lifecycle, and side-pool requirements.

The cleanest design is probably not to port SGLang's controller wholesale into
vLLM. Instead, preserve vLLM's scheduler connector as the public integration
point, but make the Mooncake-specific worker implementation delegate to a
small storage backend interface with HiCache-like semantics: longest-prefix
existence, page/block get, page/block set, key namespace construction, memory
registration, metrics, and explicit failure reporting.

## Pros Of vLLM

- Direct integration with vLLM scheduling. The connector returns external
  matched tokens through the same interface used by other vLLM KV connectors,
  so the scheduler can allocate GPU slots for remote hits in its normal flow.
- Direct Mooncake I/O into vLLM KV-cache buffers. Avoiding an explicit host L2
  cache can reduce copies and memory pressure when the goal is simply
  Mooncake-to-GPU or GPU-to-Mooncake transfer.
- Focused operational surface. Users configure a `kv_transfer_config` and the
  connector handles lookup, metadata, async I/O, delayed frees, metrics, and
  KV events in one place.
- Useful Mooncake-specific handling is already present: dedup-before-put,
  disk-offload staging budget splitting, `NO_AVAILABLE_HANDLE` pressure
  handling, Prometheus operation metrics, and optional cross-layer block
  registration.
- The key contains TP, PP, prefill-CP, and decode-CP dimensions, so it avoids
  accidentally mixing incompatible rank-local KV shards.

## Cons Of vLLM

- It is a bespoke Mooncake connector rather than a reusable storage backend.
  Backend-specific policy, keying, lookup, async I/O, and layout handling all
  live in the connector.
- The vLLM V1 connector API itself is explicitly experimental. This branch also
  documents use of a custom Mooncake fork for GB200/MNNVL, which raises
  portability and maintenance risk.
- Lookup is a scheduler-to-worker ZMQ round trip. Worker rank 0 becomes the
  lookup proxy for Mooncake existence checks, and lookup failure returns zero
  hit tokens rather than a richer degradation signal.
- Load failures are weakly surfaced. The receive thread logs partial failures
  from `batch_get_into_multi_buffers` but still marks the request as finished;
  the connector does not override `get_block_ids_with_load_errors()`.
- There is no layerwise load/save implementation. All Mooncake work is issued
  through connector `get_finished()`, which is simpler but gives less precise
  overlap than SGLang's layer-by-layer host-to-GPU loading path.
- It has fewer cache policy knobs: no L2 host reuse, no write-through vs
  write-back policy, no prefetch timeout policy, no runtime attach/detach.
- Cross-DP reuse depends on block-hash determinism. The branch README notes
  that `PYTHONHASHSEED` must be fixed for reliable cross-DP external prefix
  hits.
- It does not provide a generic side-pool interface for Mamba, NSA/indexer, or
  other nonstandard per-token state.

## Pros Of SGLang

- Clean separation of concerns. HiCache owns cache policy and scheduling;
  `MooncakeStore` only implements storage operations. This makes Mooncake one
  backend among `file`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, and others.
- More complete cache hierarchy. L1/L2/L3 metadata and movement are part of one
  system, so local GPU/host hits are cheap and Mooncake is queried only for the
  L3 portion.
- Rich policies are already in the interface: `best_effort`,
  `wait_complete`, and `timeout` prefetch stop policies; `write_through`,
  `write_through_selective`, and `write_back` write policies.
- Stronger multi-rank coordination. HiCache uses all-reduce to make TP ranks
  agree on storage hit length and prefetch completion before updating radix
  metadata. For example, with TP=2, rank 0 may find 8 pages in Mooncake while
  rank 1 finds only 6 pages because one rank-local K/V shard is missing; the
  storage-hit all-reduce uses the minimum, so both ranks prefetch only 6 pages.
  Similarly, if rank 0 has finished loading 6 pages into host memory but rank 1
  has only finished 4, the completed-token all-reduce makes both ranks insert
  only the first 4 pages into HiRadix. This prevents the radix tree from
  advertising a prefix that not every TP shard can actually serve.
- Better extensibility for model variants. The v2 `PoolTransfer` interface can
  require auxiliary pools to co-exist with KV pages and supports hit policies
  like all-pages or trailing-pages.
- Better layout support for storage efficiency. `page_first`,
  `page_first_direct`, and `page_head` host layouts make page-level Mooncake
  objects contiguous and efficient to transfer.
- Runtime attach/detach is a real operational feature. Storage can be enabled,
  disabled, or policy-updated without restarting the server, assuming no
  in-flight requests.
- The Mooncake backend can reuse an already-initialized Mooncake TransferEngine
  when the PD-disaggregation configuration matches, reducing duplicated RDMA
  setup.

## Cons Of SGLang

- L3 hits take a two-stage path: Mooncake to host L2, then host L2 to GPU. The
  system overlaps and optimizes this, but direct Mooncake-to-GPU loading like
  vLLM's connector can be simpler and potentially lower latency for pure
  remote-hit restore.
- HiCache requires a host KV pool and suitable host layout. Mooncake storage
  asserts layouts such as `page_first`, `page_first_direct`, `page_head`, or
  `page_first_kv_spilt`; a layer-first Mooncake test is currently disabled.
- The interface has more moving pieces: radix metadata, host references,
  storage queues, prefetch buffers, attach/detach lifecycle, write policies,
  and IO policy tuning.
- The storage API is still evolving. `HiCacheStorage` has v1 and v2 methods,
  deprecated TODOs, and some older methods with broad or mixed return types.
- SGLang's default `best_effort` prefetch can intentionally stop before all L3
  data is loaded. That is good for latency SLOs, but it means an L3 hit is not
  always fully consumed.
- Mooncake setup remains operationally heavy: master service, optional metadata
  service, RDMA device mapping, global segment sizing, optional store service,
  and startup health/warmup behavior.

## Interface Tradeoffs

### Direct Connector vs Storage Backend

vLLM's connector is closer to the execution engine. It can talk in terms of
vLLM requests, scheduler-allocated GPU block IDs, `SchedulerOutput`, and
connector completion sets. This makes it efficient to wire into vLLM, but it
also means Mooncake policy and vLLM scheduler details are intertwined.

SGLang's backend is lower-level and more reusable. It knows how to answer
exists/get/set over keys and host pages. The cache system above it owns
admission, prefetch, write-back, and synchronization. This is cleaner for
multiple storage backends, but adds a host-memory layer and more controller
logic.

### GPU Directness vs Host-Tier Reuse

vLLM registers KV-cache storage and passes GPU buffer addresses/sizes to
Mooncake multi-buffer operations. This is compelling when the external cache is
the only offload tier.

SGLang registers host KV buffers and treats Mooncake as L3. That is compelling
when L2 host reuse matters, when write-back policies matter, and when the
serving engine benefits from maintaining a local host-side prefix cache.

### Synchronous Admission vs Asynchronous Prefetch

vLLM uses Mooncake existence checks during `get_num_new_matched_tokens()`.
If Mooncake says the prefix exists, the scheduler can allocate GPU blocks for
external tokens immediately.

SGLang first matches local radix state, then prefetches from L3 into host pages
according to thresholds and stop policies. This makes latency/cache-hit tradeoff
explicit, but it means the request may proceed with only partial L3 benefit.

### Simpler KV Blocks vs Richer Cache Pools

vLLM's MooncakeStoreConnector is centered on KV blocks. That is enough for
standard attention KV reuse and matches vLLM's connector API.

SGLang's `PoolTransfer` model is broader because a "cache hit" can mean more
than "the attention KV page exists." A `PoolTransfer` describes another state
pool, such as `mamba` or `indexer`, that must be restored alongside the KV
pages. During `batch_exists_v2`, HiCache intersects the KV prefix hit with
these side-pool hits and returns only the longest prefix that is safe to reuse.

The `PoolHitPolicy` controls how strict that intersection is. `ALL_PAGES`
means every page in the reusable KV prefix also needs the side-pool state; this
fits state that is required token-by-token across the whole prefix, such as an
indexer/DSA-style pool. `TRAILING_PAGES` means only the last N pages need the
side-pool state; this fits state that summarizes or resumes the tail of a
sequence, such as Mamba/SWA-like state. This is a better fit for hybrid models
that need more than K/V tensors to be restored correctly.

## Practical Takeaways For vLLM Work

Things worth borrowing from SGLang:

1. Separate storage backend semantics from scheduler connector semantics.
   A thin vLLM connector could delegate exists/get/set to a backend interface
   rather than embedding all Mooncake-specific logic.
2. Add explicit prefetch and write-back policies if vLLM wants Mooncake to be a
   general external cache, not only a direct transfer endpoint.
3. Surface load failures to the scheduler. vLLM already has
   `get_block_ids_with_load_errors()` in the base API; MooncakeStoreConnector
   should use it or otherwise trigger recompute/failure recovery.
4. Consider stronger TP/PP/DP synchronization for external-hit admission,
   especially when only one rank performs lookup.
5. Add a generic way to describe sidecar pools if vLLM wants external cache
   support for Mamba/NSA-like state.
6. Document and enforce key namespace inputs: model identity, rank layout,
   block size/page size, dtype/layout, TP/PP/CP, and hash seed determinism.

Things vLLM may keep from its own design:

1. Direct GPU-buffer registration is a strong fast path and avoids making every
   Mooncake hit pass through a host L2 cache.
2. Integrating with `KVConnectorBase_V1` keeps external KV visible to normal
   vLLM scheduling, metrics, and request lifecycle code.
3. The connector-level metrics are useful and should remain first-class for
   diagnosing Mooncake latency, byte volume, failed keys, and pressure events.

## Bottom Line

vLLM's MooncakeStoreConnector is a direct, pragmatic connector for making
Mooncake look like an external vLLM KV cache. Its strongest feature is direct
Mooncake I/O against registered vLLM KV buffers. Its main risk is that too much
cache policy, error handling, and Mooncake-specific behavior live in one
experimental connector.

SGLang's Mooncake integration is a storage backend inside a larger HiCache
architecture. Its strongest feature is the separation of policy from backend:
prefetch, write-back, multi-rank synchronization, model-specific side pools,
and runtime attach/detach are handled above Mooncake. Its main cost is
complexity and the host-memory hop between Mooncake and GPU KV cache.

For a vLLM Mooncake direction, the best hybrid would keep vLLM's direct
GPU-buffer data path but move toward SGLang's clearer storage abstraction,
policy controls, and failure reporting.
