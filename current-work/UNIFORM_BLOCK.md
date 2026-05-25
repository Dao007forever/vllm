# Investigation: Uniform block-size assumptions in vLLM V1

## Current architecture: per-group block_size is already supported

vLLM V1 already allows **different block sizes across KV cache groups** (e.g., full-attention vs sliding-window vs MLA vs Mamba). The remaining uniform assumption is **within a single group**.

- `KVCacheGroupSpec` / `KVCacheSpec.block_size` — one int per group (`vllm/v1/kv_cache_interface.py:95-115`, `:821-861`)
- `MultiGroupBlockTable.block_tables: list[BlockTable]` — one BlockTable per group (`vllm/v1/worker/block_table.py:223-323`)
- `slot_mappings[gid]` and `block_table[gid]` — dict keyed by group id (`vllm/v1/worker/gpu_model_runner.py:3860-3884`, `:2358-2372`)
- Attention builder reads `self.block_size = kv_cache_spec.block_size` per-group (`vllm/v1/attention/backends/flash_attn.py:330`)

## Where the within-group uniform-block-size assumption is hardcoded

### Slot mapping formula

The identity `slot = block_id * block_size + intra_block_offset`, with `block_column = pos // block_size`, is baked in:

- `vllm/v1/worker/block_table.py:326-381` — Triton `_compute_slot_mapping_kernel`
  - L363: `block_indices = pos // virtual_block_size`
  - L378: `slot_ids = block_numbers * block_size + local_block_offsets`
- `vllm/v1/worker/gpu/block_table.py:217-279` — newer multi-group kernel; takes a `block_sizes` tensor (`tl.load(block_sizes + group_id)`, L251) but still assumes uniformity per group
- `vllm/v1/spec_decode/utils.py:242-271` — spec decode recomputation, same formula
- `vllm/v1/attention/backends/mla/compressor_utils.py:8-50` — MLA variant with `compress_ratio`, same shape

### Cache write kernels

- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:17-124` (and `_reshape_cache_per_token_head` ~L138+)
  - L50: `block_idx = slot_idx // block_size`
  - L51: `block_offset = slot_idx % block_size`
  - `block_size` is a `tl.constexpr` kernel parameter — one specialization per block_size

### Cache tensor shape

- `vllm/v1/attention/backends/flash_attn.py:149` — `(2, num_blocks, block_size, num_kv_heads, head_size)`; requires `block_size % 16 == 0`
- Allocation: `vllm/v1/worker/gpu_model_runner.py:6865-6914` — `attn_backend.get_kv_cache_shape(kernel_num_blocks, shape_block_size, ...)` per group

### Block-table column layout

Column-packed: `block_table[req, col]` is a physical block id; the column stride implicitly equals `block_size` tokens. Changing block_size per-column would require a parallel `block_sizes_per_column` or a cumulative `block_starts` array to find the right column for a given token position.

## Existing precedent for heterogeneity

**"Hybrid blocks"** (closest precedent) splits a memory block into N smaller kernel blocks at attention time — `vllm/v1/worker/block_table.py:47-68`, `map_to_kernel_blocks` at L173-201. Memory-allocator block_size ≠ compute-kernel block_size, but both still uniform.

The newer `vllm/v1/worker/gpu/block_table.py` already passes a **per-group `block_sizes` tensor** to the Triton kernel (L86-88, L251) — so the GPU worker path is conceptually one step closer to per-row or per-column variation than the legacy path.

## What would have to change to break the within-group assumption

If the goal is variable block sizes within one group (e.g., per-request or per-column-range):

1. **Slot mapping**: replace `pos // block_size` with a search over a per-row cumulative-token-count array; replace the multiply with a per-column `block_start_slot[col]`.
2. **`reshape_and_cache`** kernels: same — take `block_starts`/`block_sizes` arrays instead of a constexpr.
3. **KV cache shape**: either keep one large flat `[num_slots, num_heads, head_dim]` pool (drop the `block_size` dim) or keep variable-size blocks in a flat pool with a separate `block_offsets` lookup.
4. **Attention kernels** (FlashAttention, FlashInfer, Triton MLA): all currently take a constexpr block_size for the K/V gather; each needs either a runtime arg or per-block metadata.
5. **`BlockTable` storage**: column-packed `int32` table is fine, but needs a sibling `block_size_per_entry` or a `cu_block_lens` per row.
6. **Allocator** (`vllm/v1/core/kv_cache_manager.py`, `block_pool.py`): currently allocates fixed-size blocks from a free pool; would need a buddy/slab allocator or to keep blocks fixed and let only the *logical* boundary vary.

## Recommendation for next step

The cheapest path is probably: keep physical blocks fixed-size in the pool, but allow the **logical** grouping to vary — i.e., one "logical block" can span multiple physical blocks, expressed as `cu_block_lens` per row. That isolates the change to slot-mapping and the index math while leaving allocator, cache tensor shape, and attention kernels mostly alone.
