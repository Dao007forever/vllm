# KV Cache Connector — Beginner Guide

The KV Connector is vLLM's plug-in point for **moving KV cache in and out of
vLLM's GPU paged buffer**. The same interface is used for prefix caching to
disk, CPU offload, prefill/decode (P/D) disaggregation across machines, and
remote KV stores like LMCache/Mooncake/NIXL/HF3FS.

You implement two roles of one class, register it in a factory, and the
scheduler + worker call your hooks at well-defined points in the engine step.

This guide covers V1 (`KVConnectorBase_V1`). Files referenced live under
[vllm/distributed/kv_transfer/kv_connector/v1/](../../../vllm/distributed/kv_transfer/kv_connector/v1/).

## Why It Exists

vLLM already has an in-GPU paged KV cache and a prefix cache. The connector
adds the question: *is there KV for this prefix somewhere else I should
load?* — and the dual: *should I save the KV I just produced for next time?*

"Somewhere else" can be:

- a directory on local disk (toy example),
- pinned host memory on the same node (`SimpleCPUOffloadConnector`),
- a remote LMCache/Mooncake/HF3FS store,
- the GPU memory of a *different* node (P/D disaggregation, NIXL/MoRIIO/P2P).

The connector hides where the data lives. The scheduler and worker only see
"how many tokens can I skip?" and "load/save these blocks."

## The Two-Role Split

Every connector class is instantiated **twice** by
[KVConnectorFactory.create_connector](../../../vllm/distributed/kv_transfer/kv_connector/factory.py):
once in the scheduler process with `role=SCHEDULER`, once in each worker
process with `role=WORKER`. Both instances are the same Python class. Inside
your `__init__` you branch on `self.role`.

```text
                  scheduler process              worker process(es)
                ┌──────────────────────┐    ┌──────────────────────────┐
       Request  │ KVConnector          │    │ KVConnector              │
         ────►  │   role=SCHEDULER     │    │   role=WORKER            │
                │                      │    │                          │
                │ get_num_new_matched_ │    │ start_load_kv            │
                │   tokens             │    │ wait_for_layer_load      │
                │ update_state_after_  │    │ save_kv_layer            │
                │   alloc              │    │ wait_for_save            │
                │ build_connector_meta │───►│ bind_connector_metadata  │
                │ request_finished     │    │ get_finished             │
                │ update_connector_    │◄───│ build_connector_worker_  │
                │   output             │    │   meta                   │
                └──────────────────────┘    └──────────────────────────┘
```

The strict split exists because the scheduler is policy ("does this request
have remote KV? how many blocks should I allocate? when is it free to release
blocks?") and the worker is data movement ("copy these bytes into the GPU
tensor, this layer, this slot"). The interface forces you to keep policy
out of the worker and data movement out of the scheduler.

Communication is one-way per step:

- Scheduler → Worker: a `KVConnectorMetadata` blob produced by
  `build_connector_meta` and attached to `scheduler_output.kv_connector_metadata`.
- Worker → Scheduler: a `KVConnectorWorkerMetadata` (optional) and the
  finished-send / finished-recv sets returned by `get_finished()`.

## Lifecycle Of A Request

This is the order you should keep in your head. Each bullet names the method
and the role.

1. **`on_new_request`** (scheduler) — optional hook the first time the
   scheduler sees a request. Use only for bookkeeping.
2. **`get_num_new_matched_tokens(request, num_computed_tokens)`** (scheduler)
   — return `(extra_tokens, async_load)`. `extra_tokens` is "tokens beyond
   what local prefix cache already covered." Returning `0, False` means *no
   external cache for this request*. The scheduler will allocate blocks for
   those extra tokens. May be called multiple times — must be side-effect
   free.
3. **`update_state_after_alloc(request, blocks, num_external_tokens)`**
   (scheduler) — the scheduler has reserved blocks; record what you need
   later (e.g. which request to issue a load for, which block ids to fill).
   If you returned `async_load=True`, this can be called twice: once when
   the async-load blocks are reserved, again when the rest is allocated.
4. **`build_connector_meta(scheduler_output)`** (scheduler) — once per
   engine step. Produce a `KVConnectorMetadata` describing every load and
   save the worker should do this step. **Resets your scheduler state.**
   Don't modify `scheduler_output` here.
5. **`bind_connector_metadata(meta)`** (worker) — model runner sets the
   metadata before forward.
6. **`handle_preemptions(meta)`** (worker, optional) — gives async-save
   connectors a chance to flush preempted/evicted blocks *before* they are
   overwritten by the upcoming forward.
7. **`start_load_kv(forward_context)`** (worker) — issue all loads (may be
   async). Called once before the forward.
8. **For each attention layer's forward:**
   - **`wait_for_layer_load(layer_name)`** (worker) — block until layer `i`'s
     load is complete. Lets you pipeline loads with compute.
   - Layer runs (Q · K, softmax, V).
   - **`save_kv_layer(layer_name, kv_layer, attn_metadata)`** (worker) —
     start saving that layer (may be async).
9. **`wait_for_save()`** (worker) — barrier at end of forward so paged KV
   isn't overwritten before saves drain.
10. **`get_finished(finished_req_ids)`** (worker) — return ids that have
    finished async send/recv this step. Reported back to the scheduler in
    `KVConnectorOutput.finished_sending` / `finished_recving`.
11. **`build_connector_worker_meta()`** (worker, optional) — extra payload
    to ship back to the scheduler-side connector this step.
12. **`update_connector_output(connector_output)`** (scheduler) — receive
    the worker output, including any finished sends/recvs and worker meta.
13. **`request_finished(request, block_ids)`** (scheduler) — exactly once
    when the request finishes. Returns `(defer_free, kv_transfer_params)`.
    Return `True` if **you** take responsibility for freeing the blocks
    (e.g. you still need to upload them); the scheduler then waits until
    the request id comes back through `get_finished()` before reclaiming.

The model-runner side of this orchestration is in
[KVConnectorModelRunnerMixin._get_kv_connector_output](../../../vllm/v1/worker/kv_connector_model_runner_mixin.py)
— it's a small context manager that calls `bind_connector_metadata`,
`start_load_kv`, then on exit calls `wait_for_save`, `get_finished`,
`get_block_ids_with_load_errors`, `build_connector_worker_meta`, and
`clear_connector_metadata`. Read that file once — it's the single best
overview of how the worker side gets called.

## The Methods, Grouped

### Worker-side (data movement)

The four abstract methods you must implement on the worker:

- `start_load_kv(forward_context, **kwargs)` — kick off loads. Async fine.
- `wait_for_layer_load(layer_name)` — barrier per layer.
- `save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)` — kick off
  save for one layer. Async fine.
- `wait_for_save()` — barrier across all saves.

Optional but very common:

- `register_kv_caches(kv_caches)` / `register_cross_layers_kv_cache(...)` —
  called once at startup so you can pin/register the GPU buffers (e.g.
  RDMA handles for NIXL).
- `set_host_xfer_buffer_ops(copy_op)` — model runner injects an xPU-aware
  H↔D copy primitive your connector can call.
- `handle_preemptions(meta)` — see lifecycle step 6.
- `get_finished(finished_req_ids)` — required if you use async send/recv.
- `get_block_ids_with_load_errors()` — report blocks whose load failed so
  the scheduler can recover (recompute them).
- `shutdown()` — drain threads, close sockets.
- `get_kv_connector_stats()` / `get_kv_connector_kv_cache_events()` —
  observability.

### Scheduler-side (policy)

Three abstracts:

- `get_num_new_matched_tokens(request, num_computed_tokens)` — see
  lifecycle step 2.
- `update_state_after_alloc(request, blocks, num_external_tokens)` — step 3.
- `build_connector_meta(scheduler_output)` — step 4.

Common optionals:

- `bind_gpu_block_pool(gpu_block_pool)` — get a reference to the GPU block
  pool to inc/dec refcounts or iterate prefix-cache blocks.
- `on_new_request(request)`, `update_connector_output(...)`,
  `request_finished(...)` — see lifecycle.
- `take_events()` — yield `KVCacheEvent`s for the event bus (used to
  publish prefix-cache hits/misses to external listeners).
- `get_required_kvcache_layout(vllm_config)` — request a specific KV layout
  (e.g. `"NHD"` or `"HND"`).
- `requires_piecewise_for_cudagraph(extra_config)` — return `True` if you
  use async layer-by-layer load/save. CUDA graphs can't capture your Python
  callbacks; PIECEWISE mode interleaves Python between graph pieces.
- `prefer_cross_layer_blocks` (property) — `True` if your transfers are
  cheaper when one block holds all layers' KV contiguously.

## Metadata Classes

There are three small ABCs you typically subclass:

- `KVConnectorMetadata` — what `build_connector_meta` returns. Holds the
  step's load/save plan. Must be serializable (it crosses scheduler → worker
  IPC). Pattern: a list of per-request "do this" records. See
  [ExampleConnectorMetadata](../../../vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py).
- `KVConnectorWorkerMetadata` — what `build_connector_worker_meta` returns.
  Sent back from each worker; vLLM calls `.aggregate(other)` to fold all
  workers' metas into one before handing it to the scheduler connector.
- `KVConnectorHandshakeMetadata` — out-of-band P/D handshake info (NIXL).

## Connector Output Plumbing

The worker connector doesn't talk directly to the scheduler connector. It
fills in fields on a [KVConnectorOutput](../../../vllm/v1/outputs.py) that
flows back inside the `ModelRunnerOutput`. The scheduler reads those fields
in [scheduler.py](../../../vllm/v1/core/sched/scheduler.py) (`_update_from_kv_xfer_finished`)
and calls `update_connector_output` on its scheduler-side connector. Two
fields drive most of the policy:

- `finished_sending` — ids whose async save has completed; scheduler can
  free their blocks now.
- `finished_recving` — ids whose async load has completed; scheduler can
  schedule them as runnable.

## A Tiny End-to-End Example: ExampleConnector

[ExampleConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)
saves and loads KV as `safetensors` files on a shared filesystem. It is
~450 lines and is the canonical "minimal" implementation. Strip it down in
your head as:

```python
class ExampleConnector(KVConnectorBase_V1):
    # Scheduler: is this prompt cached on disk?
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not self._found_match_for_request(request):
            return 0, False
        # Number of cached tokens beyond what the local prefix cache covered.
        return cached_tokens_aligned_to_block - num_computed_tokens, False

    # Scheduler: blocks got reserved; remember to load into them next step.
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    # Scheduler: produce the step's plan.
    def build_connector_meta(self, scheduler_output):
        meta = ExampleConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(..., is_store=False)   # load
            elif not self._found_match_for_prompt(...):
                meta.add_request(..., is_store=True)    # save
        self._requests_need_load.clear()
        return meta

    # Worker: read files and inject KV into paged buffer.
    def start_load_kv(self, forward_context, **kwargs):
        for req in meta.requests:
            if req.is_store: continue
            for layer_name in forward_context.no_compile_layers:
                kv = safetensors.torch.load_file(filename(layer_name, req))
                inject_kv_into_layer(kv_cache_layer, kv, req.slot_mapping, attn_metadata)

    # Worker: dump the layer KV at the requested slots to a file.
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        for req in meta.requests:
            if req.is_store:
                kv = extract_kv_from_layer(kv_layer, req.slot_mapping)
                safetensors.torch.save_file({"kv_cache": kv.cpu()}, filename(...))

    # Synchronous filesystem I/O: barriers are no-ops.
    def wait_for_layer_load(self, layer_name): return
    def wait_for_save(self): return
```

That's the entire surface for a synchronous connector. Add `get_finished`
and you have an async one.

## The `slot_mapping` Concept

Both load and save are addressed by a `slot_mapping: torch.Tensor[int]` —
one entry per token, giving the linear slot in the paged KV cache (i.e.
`block_id * block_size + offset`). Building this from `(block_ids,
block_size)` is what `ReqMeta.make_meta` does in the example. This is the
same `slot_mapping` the attention kernel uses, which is what makes the
"inject at these positions" pattern work.

For non-MLA caches, the paged layer tensor is shaped roughly
`[2, num_pages, page_size, ...]` (K and V interleaved on axis 0); for MLA
it's `[num_pages, page_size, ...]`. The example handles three layouts
(default, MLA, Triton) — most production connectors will only see one or
two of these.

## Block-Granularity Constraint

`num_computed_tokens` is **block-aligned** in V1. So is the value you
return from `get_num_new_matched_tokens`. If your external cache stores
e.g. 130 tokens and `block_size=16`, you can only claim 128 (`130 // 16
* 16`). `align_to_block_size` at the bottom of the example file is the
typical helper — note it uses `(n-1)//bs*bs`, not `n//bs*bs`, because
vLLM always wants at least one uncached token to start generation from.

## Hybrid Memory Allocator (HMA)

If the deployment uses HMA (multiple KV-cache groups: full attn + SWA, or
attn + Mamba), the factory **rejects** any connector that doesn't inherit
from `SupportsHMA` and implement `request_finished_all_groups(request,
block_ids: tuple[list[int], ...])`. The signature change is "block ids per
group" instead of one flat list. If you're writing a new connector and
plan to support hybrid models, inherit from `SupportsHMA` from day one.
See [base.py](../../../vllm/distributed/kv_transfer/kv_connector/v1/base.py).

## Registration

Add one line in [factory.py](../../../vllm/distributed/kv_transfer/kv_connector/factory.py):

```python
KVConnectorFactory.register_connector(
    "YourConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.your_module",
    "YourConnector",
)
```

External (out-of-tree) connectors are also supported: set
`kv_transfer_config.kv_connector_module_path` to your module and
`kv_connector` to the class name. The factory will import it lazily and
verify the 3-arg constructor `(vllm_config, role, kv_cache_config)`.

## In-Tree Connectors As Reading Order

Once you've read the example, the rest are easier to skim because they all
follow the same shape. Pick by what you want to learn:

- [ExampleConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)
  — minimal sync, file-backed.
- [SimpleCPUOffloadConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)
  — pinned-host CPU offload with a real prefix cache. See also
  [SIMPLE_KV_OFFLOAD.md](../offloading/SIMPLE_KV_OFFLOAD.md).
- [OffloadingConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)
  — generic async-save offload with `handle_preemptions`.
- [NixlConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/nixl/)
  — RDMA P/D disaggregation; good study of `register_kv_caches`,
  `prefer_cross_layer_blocks`, async send/recv, handshake metadata.
- [P2pNcclConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/p2p/)
  — NCCL-based P/D.
- [LMCacheConnectorV1](../../../vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py),
  [MooncakeConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/),
  [HF3FSKVConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/),
  [FlexKVConnectorV1](../../../vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py),
  [MoRIIOConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/moriio/)
  — thin shims that delegate to an external library/service.
- [MultiConnector](../../../vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py)
  — composes multiple connectors in a chain (e.g. local CPU + remote
  store). Useful pattern to study.

## Common Pitfalls

- **Side effects in `get_num_new_matched_tokens`.** The scheduler may call
  it more than once for the same request. Move state writes to
  `update_state_after_alloc`.
- **Forgetting to clear scheduler state in `build_connector_meta`.** The
  docstring says "calling this function will reset the state of the
  connector." If you don't clear `_requests_need_load`-like dicts, you
  will re-emit the same load in the next step.
- **Returning `True` from `request_finished` and forgetting to report the
  id back through `get_finished`.** Blocks will leak — the scheduler is
  waiting for your signal to free them.
- **Mutating `scheduler_output` inside `build_connector_meta`.** It's
  shared; don't.
- **Writing async load/save without setting
  `requires_piecewise_for_cudagraph`.** Your callbacks will be skipped
  during CUDA graph replay and you'll get data races.
- **Ignoring `align_to_block_size`.** Returning a non-block-aligned token
  count from `get_num_new_matched_tokens` will trip asserts in the
  scheduler.
- **Using `self.role` to decide what to do at *call* time instead of at
  *init* time.** Worker methods are only called on the worker instance
  and scheduler methods only on the scheduler instance — except for
  `__init__`, which is what `role` is really for.

## Where To Look In The Repo

- Interface: [base.py](../../../vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- Factory + registry: [factory.py](../../../vllm/distributed/kv_transfer/kv_connector/factory.py)
- Worker plumbing: [kv_connector_model_runner_mixin.py](../../../vllm/v1/worker/kv_connector_model_runner_mixin.py)
- Scheduler plumbing: [scheduler.py](../../../vllm/v1/core/sched/scheduler.py)
  (search for `kv_connector`)
- Tests: [tests/v1/kv_connector/](../../../tests/v1/kv_connector/) — the
  unit tests double as small, isolated usage examples.
