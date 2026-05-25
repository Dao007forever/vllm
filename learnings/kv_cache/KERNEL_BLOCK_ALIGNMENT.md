# Kernel Block Alignment Constraint

## The assertion

```python
# vllm/v1/attention/backends/mla/cutlass_mla.py:203
_, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape
assert block_num % (128 / PAGE_SIZE) == 0
```

This fires when, for example, `PAGE_SIZE=32` and `block_num=295`:

```
128 / PAGE_SIZE = 128 / 32 = 4   (pages per kernel tile)
295 % 4 = 3  ≠ 0  →  AssertionError
```

---

## What each variable means

| Variable | Source | Example |
|---|---|---|
| `PAGE_SIZE` | vLLM's `block_size` in tokens (from KV cache config) | 32 tokens |
| `128` | CUTLASS MLA kernel's internal tile size in tokens | fixed at 128 |
| `128 / PAGE_SIZE` | Number of vLLM pages per one kernel tile | 4 pages |
| `block_num` | Width of the page table tensor: `max_model_len // block_size` | 295 |

`block_num = 295` arises from e.g. `max_model_len = 9440` tokens and `block_size = 32`:
```
9440 / 32 = 295
```

---

## Why the kernel needs this

The `sm100_cutlass_mla_decode` kernel (CUTLASS MLA for SM100 / Blackwell) processes the
KV cache in **tiles of 128 tokens**. Each tile consists of `128 / PAGE_SIZE` consecutive
pages from the page table.

The kernel's inner loop iterates over these page groups without a boundary check:

```
for tile_idx in range(block_num // pages_per_tile):
    pages = page_table[:, tile_idx * pages_per_tile : (tile_idx + 1) * pages_per_tile]
    # load 128 tokens, compute attention partial sum
```

If `block_num` is not divisible by `pages_per_tile = 128 / PAGE_SIZE`, the last tile is
**partial** — it would read past the end of the page table into garbage memory. The kernel
avoids adding a boundary check in the inner loop (it would cause warp divergence and hurt
throughput), so it requires the page table width to be pre-aligned.

```
block_num = 295  →  295 / 4 = 73.75 tiles  →  last tile is 3/4 full  →  reads 32 garbage slots
block_num = 296  →  296 / 4 = 74.0  tiles  →  every tile is full     →  safe
```

---

## How vLLM resolves it

### Path 1: block_size inflation (current approach)

`resolve_kv_cache_block_sizes` in `vllm/platforms/interface.py:620` gathers
`kernel_block_alignment_size` from the backend:

```python
kernel_block_alignment_size = max(
    min(s.base if isinstance(s, MultipleOf) else s
        for s in backend_cls.get_supported_kernel_block_sizes()),
    cache_config.block_size,
)
```

For CUTLASS MLA, `get_supported_kernel_block_sizes()` returns `[128]`, so
`kernel_block_alignment_size = 128`.

The attention `block_size` is then forced to a multiple of 128:

```python
attn_block_size = kernel_block_alignment_size * cdiv(
    mamba_page_size,
    kernel_block_alignment_size * attn_page_size_1_token,
)
```

If `block_size` becomes 128 and `max_model_len = 9440`:
```
block_num = 9440 / 128 = 73.75 → ceiled to 74 (padded by vLLM)
74 % (128 / 128) = 74 % 1 = 0  ✓
```

The inflation guarantees alignment because `block_size` itself is now a multiple of the
kernel tile size, so `block_num = max_model_len / block_size` is already in kernel-tile
units — the modulus check trivially passes.

### Path 2: page table padding (the alternative)

Instead of inflating `block_size`, pad the page table to the next aligned width at
construction time:

```python
pages_per_tile = 128 // block_size           # e.g. 128 // 32 = 4
block_num_padded = cdiv(block_num, pages_per_tile) * pages_per_tile
# 295 → cdiv(295, 4) * 4 = 74 * 4 = 296
page_table = F.pad(page_table, (0, block_num_padded - block_num), value=NULL_BLOCK_ID)
```

Padding with `NULL_BLOCK_ID` causes the kernel to read the null block (whose K/V is
zeroed), contributing nothing to the attention sum. This is safe because attention
weights at null slots are zeroed out by sequence-length masking.

Path 2 preserves fine `block_size = 32` granularity for prefix caching while satisfying
the kernel constraint. Path 1 is simpler but inflates block_size, degrading prefix cache
resolution (as discussed in `LINEAR_ATTENTION.md`).

---

## Connection to the block_size inflation problem

This constraint is one of the two reasons vLLM's `resolve_kv_cache_block_sizes` inflates
attention block_size in hybrid models (alongside Mamba page size matching). Even in a
pure-attention model with CUTLASS MLA, the block_size must be a multiple of the kernel
tile size (128 tokens for CUTLASS MLA on SM100).

| Backend | `get_supported_kernel_block_sizes()` | Effective alignment |
|---|---|---|
| FlashInfer MLA | `[32, 64]` | 32 or 64 tokens |
| CUTLASS MLA (SM100) | `[128]` | 128 tokens |
| FlashMLA | `[64]` | 64 tokens |
| Triton | `[MultipleOf(1)]` | any |

The result: the minimum `block_size` vLLM can use for a given model depends on which
attention backend is selected, not just the model architecture. Switching from CUTLASS MLA
to FlashInfer MLA allows `block_size = 32` instead of `128`, recovering 4× finer prefix
cache granularity.

---

## Worked example end-to-end

Model: DeepSeek-V3 (MLA), `max_model_len = 9440`, backend = CUTLASS MLA.

```
attn_page_size_1_token = 512 (bytes, latent dim 512 * fp16 = 1KB, plus rope = 576B ≈ 512B rough)
kernel_block_alignment_size = 128   (CUTLASS MLA requirement)

block_size = 128  (smallest multiple of 128 that fits)
block_num  = ceil(9440 / 128) = 74

74 % (128 / 128) = 74 % 1 = 0  ✓  assertion passes
```

Now with FlashInfer MLA instead:

```
kernel_block_alignment_size = 32   (FlashInfer MLA minimum)

block_size = 32
block_num  = ceil(9440 / 32) = 295

295 % (128 / 32) = 295 % 4 = 3  ✗  assertion would fire IF CUTLASS MLA were used
                                     but FlashInfer MLA has no such assertion — it
                                     directly supports PAGE_SIZE = 32
```

The "block_num=295 % (128/32=4) != 0" error only appears when CUTLASS MLA is the active
backend but the configured `block_size` is smaller than 128 (e.g. 32 tokens from a
previous config or an override). The fix is either:
- Use `block_size = 128` (default when CUTLASS MLA is selected), or  
- Switch to a backend that natively supports smaller page sizes (FlashInfer MLA, FlashMLA).
