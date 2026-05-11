# SimpleCPUOffload External Cache Hit Drop

## Context

Workload:

- Model: `nvidia/Kimi-K2.5-NVFP4`
- Attention: MLA with SWA + full-attention groups
- TP size: 4
- Block size: 64
- SimpleCPUOffload configured with `cpu_bytes_to_use_per_rank=100000000000`
- CPU blocks observed: `44,470` per rank, about `2.85M` logical tokens per rank
- GPU KV cache observed: `584,704` tokens

Runs inspected:

- Lazy mode: `/home/inf-daole/vigil/logs/pd_tp_dep_simple_offload_nixl_400G_random/2026-05-10/20260510_195215`
- Eager mode after initial per-chunk free-order fix: `/home/inf-daole/vigil/logs/pd_tp_dep_simple_offload_nixl_400G_random/2026-05-11/20260511_044154`

## Important Observations

The vmon `external_cache_hit_rate` can go to `0` while vLLM stdout still shows a nonzero or slowly decaying `External prefix cache hit rate`.

These are different windows:

- vmon appears to report scrape/window deltas.
- vLLM stdout uses `CachingMetrics`, a recent-request rolling metric.

So a vmon zero window and a nonzero vLLM log value can both be true.

GPU prefix cache hit being nonzero does not imply CPU external hit should be nonzero. The scheduler first matches GPU prefix cache, then SimpleCPUOffload checks CPU only for the suffix after the GPU hit:

```text
request blocks: A0 A1 A2 A3 A4 A5
GPU hit:        A0 A1 A2
CPU query:               A3 A4 A5
```

If CPU is missing `A3`, the external hit is `0`, even if CPU contains `A4 A5` or even if CPU contains `A0 A1 A2`.

## Capacity Finding

The configured `400GB` is physical memory across four TP ranks. SimpleCPUOffload does not deduplicate MLA across TP ranks, so the effective logical capacity is not `4 * 44,470` blocks. Each rank stores the same logical MLA block independently.

Effective logical CPU cache capacity per prefill node is therefore approximately:

```text
44,470 blocks/rank * 64 tokens/block = ~2.85M logical tokens
```

At the failure point, average prompt length was already around `62k-80k` tokens. With 75 conversations:

```text
75 * 62k tokens = ~4.65M logical tokens
75 * 80k tokens = ~6.0M logical tokens
```

This exceeds the effective logical SimpleCPUOffload capacity, so CPU eviction is expected. MLA-aware deduplication is still needed if the goal is to make the four TP ranks behave like a shared `400GB` logical cache.

## Lazy Mode Finding

Lazy mode does not mirror the whole GPU prefix cache into CPU.

It scans the GPU free queue from a cursor and stops after `_target_free` covered blocks:

```python
covered < self._target_free
```

For this run:

```text
max_num_batched_tokens = 8192
block_size = 64
target ~= 2 * (8192 / 64) = 256 blocks
```

A long request around `80k` tokens is about:

```text
80k / 64 = ~1250 blocks
```

When a request is freed, GPU inserts request blocks tail-first:

```text
A1249 A1248 ... A3 A2 A1 A0
```

Lazy mode may only copy the first ~256 free-queue candidates:

```text
copied: A1249 ... A994
not reached: A3
```

External prefix matching, however, often needs the next block after the GPU hit, for example `A3`. So lazy mode can preserve far-tail blocks while missing the near-prefix continuation block that CPU lookup needs.

The current lazy policy optimizes for GPU eviction safety, not external-prefix contiguity.

## Eager Mode Finding

The initial eager fix reversed CPU free order within each store event. That was insufficient because chunked prefill stores one long request across many store events.

Before the latest fix, chunked eager behavior was:

```text
request: A0 A1 A2 A3 A4 A5 A6 A7

store event 1: A0 A1 A2 A3
free order:    A3 A2 A1 A0

store event 2: A4 A5 A6 A7
free order:    A7 A6 A5 A4

final CPU free queue:
A3 A2 A1 A0 A7 A6 A5 A4
```

This is still wrong for prefix matching. `A3` can be evicted before `A7`, so later a request with GPU hit `A0 A1 A2` misses CPU immediately at `A3`.

The desired whole-request CPU eviction order is:

```text
A7 A6 A5 A4 A3 A2 A1 A0
```

## Implemented Fix

Implemented in:

- `/home/inf-daole/vllm-mooncake/vllm/v1/simple_kv_offload/manager.py`
- `/home/inf-daole/vllm-mooncake/tests/v1/simple_kv_offload/test_scheduler.py`

Behavior after the fix:

1. Eager store completion registers CPU block hashes so blocks are visible to CPU lookup.
2. Eager CPU blocks are not immediately released into the CPU free/eviction queue.
3. Per-request eager store state accumulates CPU block ids in request/chunk order.
4. On request cleanup, accumulated CPU blocks are released in reverse whole-request order.

This produces:

```text
A7 A6 A5 A4 A3 A2 A1 A0
```

instead of:

```text
A3 A2 A1 A0 A7 A6 A5 A4
```

This fixes the cross-chunk eager eviction-order bug.

## Verification

Completed:

- `compileall` passed for `manager.py` and `test_scheduler.py`.
- Direct targeted invocation passed for the new/affected eager eviction-order tests.

Not completed:

- Full `pytest` run was not possible because `/home/inf-daole/vllm-mooncake/.venv` does not have `pytest` installed.

## Remaining Risks

The eager fix improves eviction priority but does not increase effective CPU capacity.

Because eager CPU blocks are now held out of the CPU free queue until request cleanup, active long requests can hold CPU blocks longer. If active in-flight eager stores exceed CPU capacity, later chunks may stop being stored until some requests finish.

This is the intended tradeoff for preserving whole-request suffix-first eviction order, but it is not a substitute for MLA deduplication.

## Next Recommended Fixes

1. Add MLA-aware SimpleCPUOffload ownership:
   - One TP rank stores the logical MLA block.
   - Non-owner ranks restore via TP-group GPU broadcast.

2. Add instrumentation:
   - CPU blocks cached.
   - CPU evictions.
   - first missing external block index.
   - GPU prefix hit blocks vs CPU suffix hit blocks.
   - eager store blocks pinned per request.

3. Re-run eager mode with the latest whole-request release fix.

4. Revisit lazy policy separately:
   - Lazy should spend some budget on likely prefix-continuation blocks, not only front-of-GPU-free-queue eviction candidates.

