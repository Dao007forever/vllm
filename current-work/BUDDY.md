# Buddy Allocator — Design and Current State

Companion to `UNIFORM_BLOCK.md` and `TEST_PLAN.md`. Documents the buddy-backed `BlockPool` path added in the `exp/malloc` branch and what it does / doesn't yet deliver.

## Goal

Let different KV cache groups allocate chunks of different sizes from a shared pool, with finer-grained reassignment than Jenga's whole-page eviction.

## Where we are

### Plumbing — done

- `vllm/v1/core/buddy_block_pool.py` — pure power-of-2 buddy allocator over a flat id space. Split on alloc, coalesce on free. No coupling to the rest of vLLM.
- `vllm/v1/core/buddy_free_queue.py` — adapter exposing the `FreeKVCacheBlockQueue` interface (`popleft`, `popleft_n`, `append`, `append_n`, `remove`, `get_all_free_blocks`, `num_free_blocks`) plus new `alloc_chunk(order)` / `free_chunk(start_id)` for caller-specified orders. Supports `max_order > 0`.
- `BlockPool` (`vllm/v1/core/block_pool.py`) — gates on `VLLM_USE_BUDDY_BLOCK_POOL`. When set: swaps the free queue, forces `enable_caching=False`, and reads `VLLM_BUDDY_MAX_ORDER`.
- `BlockPool.get_new_chunks(order, n)` — additive API. Default callers stay on `get_new_blocks` (= order 0).
- `BlockPool.free_chunks(blocks)` — symmetric free for chunks.
- `SingleTypeKVCacheManager` — reads its group's order from `VLLM_BUDDY_GROUP_ORDERS` (comma list, indexed by `kv_cache_group_id`), routes its 3 allocation sites through `get_new_chunks(self._buddy_order, n)`. Default 0.

### Verified

- 17 unit tests cover the core allocator (single + multi-order, coalescing, OOM, mixed-order stress) and the adapter (chunk roundtrip, mixed order-0/order-k coexistence, remove-not-supported guard).
- Harness golden checks (Qwen3-0.6B + Zamba2-1.2B-instruct):
  - buddy OFF — unchanged.
  - buddy ON, max_order = 0 — match.
  - buddy ON, max_order = 4, all groups order 0 — match (exercises split/coalesce in production traffic).
  - buddy ON, max_order = 4, Qwen3 group order = 1 — match.
  - buddy ON, max_order = 4, Zamba2 group orders = 0,0 — match.
  - buddy ON, max_order = 4, Zamba2 group orders = 0,1 — match.

## What this does NOT yet deliver

The buddy operates on the **shared block-id namespace**, but vLLM's actual KV cache memory is **per-group**: each `KVCacheGroupSpec` owns its own tensor of shape `[num_blocks, block_size_g, num_kv_heads_g, head_dim_g]`. A block-id of 100 means different physical memory in group A vs group B.

Consequence: when a group allocates an order-`k` chunk, the buddy reserves `2**k` consecutive block-ids in the shared namespace. The group's manager still only treats the *first* id as a "logical block" and writes / reads through its own per-group tensor at that index. The other `2**k - 1` ids are reserved (so no other group can take them) but the per-group tensor slots at those indices sit unused.

So:
- ✅ Pool tracking via buddy works. Other groups can't double-allocate the reserved ids.
- ✅ Greedy outputs unchanged — slot mapping and attention are correct for the start id.
- ❌ No memory savings. The reserved ids cost memory in each group's own tensor, the same as if `2**k` separate order-0 allocations had been made.

True cross-group memory sharing requires **memory unification**: one shared byte buffer, per-group views over it, with order > 0 chunks actually buying more bytes (proportional to per-token KV size of the group). That is the Jenga-style refactor and is a separate, larger iteration.

## Known limitations

- **Prefix caching disabled.** Buddy is incompatible with `BlockPool._maybe_evict_cached_block` and `touch`, which rely on `remove()`-from-free-list. Implementing `remove()` on a buddy structure is non-trivial: a freed id may have coalesced into a higher-order chunk and not appear in any lower-order set. Punted.
- **No LRU eviction order.** Buddy hands back whatever id it finds; the original `FreeKVCacheBlockQueue` returns the least-recently-used. With caching off, LRU is moot. If caching is re-enabled, ordering will need attention.
- **No tail-of-pool support without slop.** `num_base_blocks` must be a multiple of `2**max_order`; ids beyond that prefix live in a separate `_tail_free` list. Acceptable; documented in code.
- **No memory unification.** See above.

## Next iterations (in priority order)

1. **Memory unification** (the real win): unified KV byte buffer + per-group views over it. Allocation in bytes, not in shared-id-space. Order > 0 actually delivers savings then. Largest surface change.
2. **`remove()` for the buddy adapter** to re-enable prefix caching. Medium surface.
3. **Eviction policy compatibility**: when `enable_caching=True` and the pool runs OOM, current vLLM evicts cached blocks. Buddy needs an LRU overlay or similar.
4. **Performance**: micro-benchmark buddy vs `FreeKVCacheBlockQueue` for typical allocation patterns. Likely close; verify.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `VLLM_USE_BUDDY_BLOCK_POOL` | unset | Set to `1` to route `BlockPool`'s free queue through the buddy adapter. Forces `enable_caching=False`. |
| `VLLM_BUDDY_MAX_ORDER` | `0` | Max buddy order when buddy is enabled. `0` ⇒ slab-like single size class. Higher ⇒ split/coalesce active. Constraint: `num_gpu_blocks` should be ≥ `2**max_order` (tail ids handled separately). |
| `VLLM_BUDDY_GROUP_ORDERS` | unset | Comma-separated list, indexed by `kv_cache_group_id`. Each group's allocations route through `get_new_chunks(order=..., n)`. Ignored if buddy is off. |
