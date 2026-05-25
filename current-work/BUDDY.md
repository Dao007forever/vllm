# Buddy Allocator — Design and Current State

Companion to `UNIFORM_BLOCK.md`, `KV-ALLOCATION.md`, `decoupled_hybrid_pages.md`,
and `TEST_PLAN.md`. Documents the buddy-backed `BlockPool` path on the
`exp/malloc-wt` branch.

For the end-to-end story (how buddy is consumed by the decoupled hybrid-page
layout, and the actual memory savings delivered on Zamba2 / Falcon-Mamba),
see `decoupled_hybrid_pages.md`. This file is just the allocator-level state.

## Goal

Let different KV cache groups allocate chunks of different sizes from a shared
pool, with finer-grained reassignment than Jenga's whole-page eviction.

## Where we are

### Plumbing — done

- `vllm/v1/core/buddy_block_pool.py` — pure power-of-2 buddy allocator over a flat id space. Split on alloc, coalesce on free. No coupling to the rest of vLLM.
- `vllm/v1/core/buddy_free_queue.py` — adapter exposing the `FreeKVCacheBlockQueue` interface (`popleft`, `popleft_n`, `append`, `append_n`, `remove`, `get_all_free_blocks`, `num_free_blocks`) plus `alloc_chunk(order)` / `free_chunk(start_id)` for caller-specified orders. `remove(block)` reads `block.chunk_order` and unlinks the block; cached-block eviction is supported via an `on_evict` callback invoked before a split or coalesce that affects a cached chunk. Supports `max_order > 0`.
- `BlockPool` (`vllm/v1/core/block_pool.py`) — gates on `VLLM_USE_BUDDY_BLOCK_POOL`. When set: swaps the free queue and reads `VLLM_BUDDY_MAX_ORDER`. Prefix caching coexists with buddy mode (the queue invokes `BlockPool._maybe_evict_cached_block` on splits/coalesces).
- `BlockPool.get_new_chunks(order, n)` — additive API. Default callers stay on `get_new_blocks` (= order 0).
- `BlockPool.free_chunks(blocks)` — symmetric free for chunks.
- `SingleTypeKVCacheManager` — takes a `chunk_order` arg (passed through `HybridKVCacheCoordinator` from `KVCacheGroupSpec.chunk_order`, computed in `kv_cache_utils.py` as `ceil(log2(group.page / base_page))`). `VLLM_BUDDY_GROUP_ORDERS` env var is still honored as a manual override for legacy testing. Routes its allocation sites through `get_new_chunks(self._buddy_order, n)`.

### Verified

- 25 unit tests in `tests/v1/core/test_buddy_block_pool.py` cover the core allocator
  (single + multi-order, coalescing, OOM, mixed-order stress) and the adapter
  (chunk roundtrip, mixed order-0/order-k coexistence, `remove()`, eviction-callback
  paths for split/coalesce).
- 3 tests in `tests/v1/worker/test_variable_block_slot_mapping.py` cover the
  variable-block slot-mapping kernel path.
- Harness golden checks (`current-work/golden/`, `current-work/golden_v2/`):
  - Non-hybrid baselines (Qwen3-0.6B) unchanged whether buddy is OFF or ON.
  - Hybrid models (Zamba2-1.2B / 7B, Falcon-Mamba-7B) produce bit-identical
    outputs against the baseline under decoupled hybrid pages (see
    `decoupled_hybrid_pages.md` §6 for numbers).

## What this delivers

When buddy is enabled together with the decoupled hybrid-page layout (the
default behavior of `VLLM_USE_BUDDY_BLOCK_POOL=1` on hybrid models with
heterogeneous group page sizes):

- The shared block-id namespace is reinterpreted as a **base-page-indexed**
  address space, and column-packed slabs are sized to `num_blocks × base_page`
  rather than `num_blocks × common_page`.
- A group whose native page is `2^k × base_page` allocates order-`k` buddy
  chunks. Per-group strided views over the slab give each group access to its
  full native page starting at the chunk's base index. Order > 0 chunks now
  actually buy more bytes.
- Result: prefix-cache granularity stays at each group's native block_size
  (no inflation), and the padding waste from the old `page_size_padded` path
  goes away. End-to-end on Zamba2-7B + prefix-caching ON, max concurrency
  rises from 11.5× to 47.9× at 2048-token context (+316%), outputs
  bit-identical. See `decoupled_hybrid_pages.md` §6.

For non-hybrid models (and hybrid models where all groups already share a
page size), `decouple_active` is False and the layout is unchanged from
baseline; buddy still runs as a drop-in for the LRU free queue.

## Known limitations

- **No tail-of-pool support without slop.** `num_base_blocks` must be a
  multiple of `2**max_order`; ids beyond that prefix live in a separate
  `_tail_free` list. Acceptable; documented in code.
- **Mamba allocation rounding.** A group with `ratio = native_page / base_page`
  that isn't already a power of 2 rounds up to the next power of 2
  (`chunk_order = ceil(log2(ratio))`). For Zamba2-7B mamba this is ~47% waste
  per mamba chunk relative to native — but still a large net win versus the
  baseline's full-chunk padding to `common_page`. See
  `decoupled_hybrid_pages.md` §4.
- **Order assignment is per-group, static.** Each group picks one
  `chunk_order` at config time. Within-group heterogeneity (Stage 5 in
  `TEST_PLAN.md`) is not implemented yet.

## Next iterations (in priority order)

1. **Eviction-policy tuning.** Buddy + caching works correctly via
   `on_evict`, but the prioritization (which cached chunk to evict when an
   alloc forces a split) is still naive. Worth measuring on cache-heavy
   workloads.
2. **Performance.** Micro-benchmark buddy vs `FreeKVCacheBlockQueue` for
   typical allocation patterns. Likely close; verify.

**Within-group heterogeneity (Stage 5 in `TEST_PLAN.md`).** Still deferred.
We initially thought MLA hybrids needed it to close a 9× regression
observed at `bs=16` on `TRITON_MLA`. Re-running the same Kimi-Linear-48B
bench at `bs=64` with the default `FLASHINFER_MLA` backend showed the
buddy already wins **+22% throughput** on top of a 50× capacity gain
(see `bench_kimi_linear_result.json` and `decoupled_hybrid_pages.md` §6.1).
So the gap that within-group mixing would have addressed is just the
small-`bs` × non-tile-tuned-backend corner, which is not the recommended
configuration on MLA hybrids. Keep deferred until a workload appears that
both (a) genuinely needs `bs` pinned small and (b) loses throughput that
per-group block_size + backend selection can't already recover.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `VLLM_USE_BUDDY_BLOCK_POOL` | unset | Set to `1` to route `BlockPool`'s free queue through the buddy adapter and, on hybrid models with heterogeneous group page sizes, activate the decoupled hybrid-page layout (see `decoupled_hybrid_pages.md`). Prefix caching stays enabled. |
| `VLLM_BUDDY_MAX_ORDER` | `0` | Max buddy order when buddy is enabled. `0` ⇒ slab-like single size class. Higher ⇒ split/coalesce active. Constraint: `num_gpu_blocks` should be ≥ `2**max_order` (tail ids handled separately). Typically inferred from the largest per-group `chunk_order` in decoupled mode; the env var is a manual cap. |
| `VLLM_BUDDY_GROUP_ORDERS` | unset | Comma-separated list, indexed by `kv_cache_group_id`. Each group's allocations route through `get_new_chunks(order=..., n)`. Overrides the `chunk_order` derived from page-size ratios; useful for testing. Ignored if buddy is off. |
