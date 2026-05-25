# vLLM V1 KV cache memory layout — findings

Companion to `UNIFORM_BLOCK.md` and `BUDDY.md`. This document captures the actually-true memory model in V1, which I had wrong in earlier iterations. It is a reference for what's shared vs. partitioned, what's normalized to enable sharing, and where the waste lives.

Source: paths and line numbers refer to the `exp/malloc` branch.

## TL;DR

1. KV cache tensors in V1 are **column-packed across all KV cache groups**. Layers from different groups occupy the same physical tensor at the same `block_id`-indexed row. Memory is *pooled*, not per-group.
2. To make column-packing work when groups have different per-token KV byte sizes, vLLM **normalizes the physical page size**. Two mechanisms exist:
   a. Adjust `block_size` in tokens so that `block_size × per_token_bytes = common_page_bytes`. No waste, but block_size in tokens can become large.
   b. Pad page size up to the nearest larger via `page_size_padded`. Waste = `common_page − native_page` bytes per allocated block.
3. Truly per-group memory only happens for groups whose specs are incompatible at a level the normalization machinery doesn't address (e.g., Mamba's recurrent state vs. paged attention). These are rare.
4. The waste that buddy + variable-block-size can target is the **padded bytes** in (2b) and the **coarse eviction granularity** in (2a). It is *not* a cross-group memory unification problem — sharing is already there.

## 1. Column-packed tensor layout

Layers are arranged in a 2D grid: rows are KV cache groups, columns are layer-position-within-group.

```
            col 0     col 1   ...   col 9
Group 0:    L5        L11     ...   L59      (FULL attention)
Group 1:    L0        L6      ...   L54      (SWA)
Group 2:    L1        L7      ...   L55      (SWA)
            ====      ====          ====
         tensor 0  tensor 1   ...  tensor 9
```

Each **column** of the grid is a single physical tensor of shape `[num_blocks, block_size, num_kv_heads, head_dim, 2]` (the trailing 2 is K/V). With 6 groups × 10 columns = 60 layers, there are **10** physical tensors, not 60.

A row of the column tensor at index `block_id` is shared by one layer from each group at that column. The invariant that makes this safe: **`block_id` lives in at most one group's block table at a time** — when group A holds it, the row is owned by A's request; when A releases, B can claim.

`byte_offset(block_id) = block_id × page_size_bytes` is identical across all column tensors (they have the same shape). The attention kernel does `cache_tensor[block_table[r, c], inner_offset, …]` directly; no extra indirection.

## 2. Cross-group normalization — required for column packing

For all column tensors to be one shape, every group must agree on:

- `block_size` in tokens,
- per-token bytes (`num_kv_heads × head_dim × dtype_size × 2`),
- and therefore the product, `page_size_bytes`.

vLLM's hybrid coordinator (`_get_kv_cache_groups_uniform_page_size`, `vllm/v1/core/kv_cache_utils.py:1057`) lists the assumptions explicitly:

> *Physical memory per block: Must be the same across all KV cache groups. Breaking this assumption is non-trivial due to memory fragmentation concerns when allocating blocks of different sizes.*

If groups don't natively share a page size, one of two normalizations runs:

### 2a. Adjust `block_size` in tokens

`vllm/v1/core/kv_cache_utils.py:1680`

```python
new_bs = max(common_page // per_token, 1)
aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
```

For hidden-state-style layers, `block_size` is scaled so that `block_size × per_token_bytes` lands exactly on `common_page`. The smaller-per-token group gets **more tokens per block**, the larger-per-token group keeps its natural block size. No byte waste.

Cost: the smaller-per-token group's block size in tokens can grow large. With per-token bytes differing by 30×, MLA-style block_size could end up at hundreds of tokens. Side effects:

- Eviction granularity is `block_size` tokens. Larger block = coarser eviction.
- Prefix cache hits / misses align to block boundaries; large blocks mean fewer cache hits when prefixes don't align.
- The block table has fewer entries per request — small effect on metadata size, larger on the smallest-allocation unit's flexibility.

### 2b. Page-size padding

`vllm/v1/core/kv_cache_utils.py:1551-1561`

```python
# Unify page size by padding layers' page_size to the nearest larger page_size.
size_to_candidate: dict[int, int] = {}
for ps in sm_page_sizes:
    size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
    current_size = layer_spec.page_size_bytes
    candidate = size_to_candidate[current_size]
    if current_size < candidate:
        object.__setattr__(layer_spec, "page_size_padded", candidate)
```

For SWA-MLA layers (and via the same field elsewhere), `block_size` in tokens stays the same, but `page_size_padded` rounds the per-block bytes up to the nearest larger group's page size. Each allocated block of the smaller group reserves `common_page` bytes; only the first `native_page` bytes hold real data.

Cost: **`common_page − native_page` bytes wasted per allocated block**. For MLA + MHA where the per-token ratio is ~30×, padding overhead approaches 97%.

### Which mechanism runs when

- (2a) `block_size` adjustment runs for **`HiddenStateCacheSpec`** layers and similar — `kv_cache_utils.py:1675-1682`.
- (2b) `page_size_padded` runs for SWA + MLA mixtures and similar — `_get_kv_cache_groups_uniform_groups` at `kv_cache_utils.py:1551-1561`, plus `_make_uniform_page_size_specs` and `_pad_uniform_page_size` paths.
- A pure `_get_kv_cache_groups_uniform_spec` path handles models where no normalization is needed because all layers natively share the same spec — `kv_cache_utils.py:964`.

## 3. Where memory really is per-group

When the page-size normalization can't run — for example because a group's spec isn't a plain paged-KV layout (`HiddenStateCacheSpec` returns true for `hidden_specs` and follows a different path; Mamba state has its own `MambaSpec` outside the column-packed family) — that group gets its own tensors, not column-packed with the rest. These cases are narrower than I assumed:

- **`HiddenStateCacheSpec` and `MambaSpec`**: cached *state*, not paged tokens. Don't fit the paged tensor layout. Allocated separately, sized per group.
- **Hybrid-disabled mode** (`--disable-hybrid-kv-cache-manager`): forces a fallback path that doesn't column-pack different attention types. Used for compatibility, not the default.

Outside these, **all paged-KV groups share the column-packed pool** via one of the normalization paths in §2.

## 4. Where the waste lives — and what buddy + variable block_size can reclaim

Two real waste sources:

### Padding waste (2b)

If a group's native page is 2 KB and the common page is 64 KB (32× ratio), every block of that group reserves 62 KB of unused bytes in every column tensor row. Across N reserved blocks, that's `N × 62 KB` of unreachable memory.

What buddy + variable block_size can do: instead of padding, have the column tensor's physical page = the *smallest* group's native page (2 KB). The larger-per-token group then uses **order-k buddy chunks** where `2^k × 2 KB = its_native_page (64 KB)`, i.e., k=5. One block-table entry for the large group spans 32 consecutive rows in the column tensor; the small group uses one row per entry. No padding.

### Coarse eviction in (2a)

If a small-per-token group has had its `block_size` adjusted to hundreds of tokens to match a common page, eviction-by-block at that granularity is too coarse: each freed block returns hundreds of tokens to the pool at once. Buddy lets a group still allocate at the common page in bytes but represent its *logical block size in tokens* as the small native value, by letting one buddy chunk hold multiple smaller logical blocks. This is essentially decoupling allocation granularity (bytes) from logical block size (tokens).

### What's required to unlock either

Both require:

1. **Variable per-group block_size**: each group's manager produces block-table entries whose token-stride is its own native block_size, not the unified value.
2. **Variable slot mapping** (the work I removed in Stage 1–3): one block-table entry maps to a token range spanning *multiple consecutive rows* in the column tensor, not a single row.
3. **Variable attention metadata**: the attention kernel reads `2^order_g` consecutive rows per block-table entry, masking by per-token validity.
4. **Buddy free list** (the work in `BUDDY.md`): supplies variable-size chunks from the shared pool.

(4) is in place. (1) is a small change to `SingleTypeKVCacheManager`. (2) is what I deleted and need to restore. (3) is the largest open piece.

## 5. Reasons I had this wrong earlier

Two mistakes worth recording:

1. I assumed "per-group" meant separate tensors. It doesn't — vLLM column-packs across groups by default, and the per-group separation only kicks in for `HiddenStateCacheSpec` / `MambaSpec` / hybrid-disabled fallback.
2. I framed the gating refactor as "Jenga-style memory unification" (one byte buffer + per-group views). That's not actually needed for the common case (paged attention with heterogeneous per-token sizes), because vLLM's existing normalization already shares memory across groups. The remaining issue is the **padding** in (2b) and the **block-size token-coarseness** in (2a), both of which are downstream of choosing one common page — and both of which buddy + variable block_size address without touching the underlying tensor layout.

## 6. Action items

- Restore the Stage 1–3 variable-slot-mapping work from git.
- Then generalize attention metadata to read multiple rows per block-table entry (Stage 5).
- Wire `SingleTypeKVCacheManager` to allocate at each group's `order_g` based on `native_page / common_page` ratio (currently always 0).
- Verify with the harness: when a hybrid group's order > 0, slot mapping and attention should both produce correct outputs.

## Code-reference index

| Concept | File / location |
|---|---|
| Column tensor allocation | `vllm/v1/worker/gpu_model_runner.py` around `_get_kv_cache_specs` and tensor creation |
| Hybrid coordinator | `vllm/v1/core/kv_cache_coordinator.py:392+` (`HybridKVCacheCoordinator`) |
| Page-size normalization assumptions | `vllm/v1/core/kv_cache_utils.py:1057` (`_get_kv_cache_groups_uniform_page_size`) |
| Block-size adjustment for hidden-state layers | `vllm/v1/core/kv_cache_utils.py:1675-1682` |
| Page-size padding for SWA-MLA | `vllm/v1/core/kv_cache_utils.py:1551-1561` |
| `page_size_padded` field | `vllm/v1/kv_cache_interface.py:149-164` (`AttentionSpec`), `:566-579` (`MLAAttentionSpec`) |
| `BlockPool` (shared free list) | `vllm/v1/core/block_pool.py` |
| Slot mapping uniform kernel | `vllm/v1/worker/block_table.py:325+` (`_compute_slot_mapping_kernel`) |
| Buddy adapter (this branch) | `vllm/v1/core/buddy_block_pool.py`, `vllm/v1/core/buddy_free_queue.py` |
