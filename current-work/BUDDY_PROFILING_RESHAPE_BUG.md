# Buddy reshape bug at `block_size ≥ 128` (profiling-time)

> **Status (2026-05-25):** fixed on this branch — `gpu_model_runner.py`
> narrows the strided-view guard so attn groups with `chunk_order=0`
> fall through to `.view()` (the upstream path baseline-attn uses).
> Verified at `bs=128` on Kimi-Linear-48B: baseline 3,003 blocks /
> 227.8 tok/s; buddy 74,177 blocks / 261.0 tok/s (×24.7 capacity,
> +14.6% throughput). See "Fix applied" section.

A second issue, separate from the FlashInfer alignment check
(`FLASHINFER_BLOCK_NUM_CHECK.md`). At `--block-size 128` on Kimi-Linear-48B
the bench crashes during the **profiling KV-cache allocation step** —
before any real request runs — because the decouple-strided path in
`_reshape_kv_cache_tensors` over-computes the storage requirement when the
kernel's chosen block size is smaller than the spec block size.

## The crash

```
RuntimeError: setStorage: sizes [1024, 64, 576], strides [73728, 576, 1],
storage offset 0, and itemsize 2 requiring a storage size of 150921216
are out of bounds for storage of size 77520896
```

- spec `block_size = 128`
- kernel `block_size = 64` (FlashInfer-MLA's largest supported size, since
  it declined `bs=128`; or one tier down on the priority list)
- `num_blocks = 512` (this run); base slab size = `512 × 128 × 576 × 2 B
  ≈ 75 MiB ≈ 77,520,896 B` ✓ matches the available storage.
- requested view: `(2*num_blocks, 64, 576) = (1024, 64, 576)` with
  stride `(73728, 576, 1)` along outer dim. The required storage is
  `(1024-1)*73728 + (64-1)*576 + 575 + 1 ≈ 75.4M elements ×
  2 B ≈ 150.9 MiB` — **2× the actual slab**.

## Where it fires

```
_init_minimal_kv_cache_for_profiling()                # gpu_model_runner.py:6179
  ├─ get_kv_cache_config_from_groups(...)
  │     uses min_blocks = max_cudagraph_capture_size
  └─ initialize_kv_cache(...)
        └─ _reshape_kv_cache_tensors(...)              # :6891
              └─ AttentionSpec branch
                    if (page_size_padded or base_page_bytes) is not None:
                        torch.as_strided(raw_tensor, kv_cache_shape, strides)   # :6997 — crashes here
```

This is the *profiling-time* minimal-KV allocation used to size CUDA
graph capture buffers before the real per-layer KV cache is built. Once
the strided view fails, profiling can't complete, so no real cache is
ever allocated.

## Arithmetic of the bug

The attn branch builds the kernel-facing shape and stride like this
(annotated; `gpu_model_runner.py:6921-7005`):

```python
block_byte_stride = base_page_bytes if base_page_bytes is not None
                                   else kv_cache_spec.page_size_bytes
num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
kernel_num_blocks = num_blocks * num_blocks_per_kv_block

kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks, shape_block_size, num_kv_heads, head_size, ...)

if page_size_padded is not None or base_page_bytes is not None:
    page_stride = block_byte_stride // dtype_size
    strides = list(torch.empty(kv_cache_shape).stride())
    strides[inv_order[0]] = page_stride                          # ← BUG
    kv_cache = torch.as_strided(raw_tensor, kv_cache_shape, strides)
else:
    kv_cache = raw_tensor.view(kv_cache_shape)
```

The stride along the kernel-block dim is set to `base_page_bytes /
dtype_size`. That is the right stride when one kernel block consumes one
base page — i.e., when `kernel_block_size == spec.block_size` or when
`chunk_order > 0` and we want adjacent kernel indices to overlap by
exactly one base page (mamba-style).

It is **wrong** when `kernel_block_size < spec.block_size`. In that case
one spec block = `num_blocks_per_kv_block` kernel blocks, all contained
inside a single base page, so the kernel-block stride should be
`kernel_block_size × per_token_bytes`, not `base_page_bytes`. Walking the
kernel-block dim with the larger stride asks for `num_blocks_per_kv_block ×
base_slab_size` worth of storage.

For Kimi-Linear-48B at `bs=128`:

- MLA group: `spec.page_size_bytes = 128 × 1152 B = 147,456 B` (this is
  the **smallest** attn-side page, so the layout sets `base_page = 147,456`)
- KDA group: `spec.page_size_bytes ≈ 2 MiB` → `chunk_order = ceil(log2(
  2 MiB / 147,456 B)) = 4`
- MLA's `chunk_order = 0` (it *is* the base)
- FlashInfer-MLA picks `kernel_block_size = 64`
  → `num_blocks_per_kv_block = 128 / 64 = 2`
  → `kernel_num_blocks = 2 × num_blocks`
- Per-kernel-block byte step should be `64 × 1152 = 73,728 B`
- Code's `page_stride = base_page / dtype_size = 147,456 / 2 = 73,728`
  *elements* = 147,456 B — exactly 2× the correct value.

So the strided view requires double the slab.

## Why baseline doesn't hit this (the real reason)

It's tempting to say "baseline's ratio is 1 so the stride is right by
accident." That's wrong. In baseline at user `bs=64`,
`_align_hybrid_block_size` (`vllm/platforms/interface.py:688`) computes:

```
kernel_block_alignment_size = max(min(backend.supported_sizes), cache_config.block_size)
                            = max(32, 64) = 64           # FlashInfer supports [32, 64]
attn_block_size = 64 * cdiv(mamba_page, 64 * attn_per_token)
                ≈ 64 * cdiv(~2 MiB, 64 * 1152)
                ≈ 64 * 29 = 1856                          # ≈ the 1888 we saw earlier
cache_config.block_size = 1856                            # spec.block_size now 1856
```

then pads **mamba's** page up to attn's. So the worker actually sees:

```
attn group:   spec.block_size = 1856,  page_size_padded = None        ← attn is the larger side
mamba group:  spec.block_size = 1856,  page_size_padded = 1856×1152   ← mamba padded up
```

FlashInfer-MLA's `get_supported_kernel_block_sizes` is `[32, 64]`, so at
`spec.block_size = 1856` the kernel picks 64 and `num_blocks_per_kv_block
= 1856/64 = 29`. **Ratio is 29, not 1.** If the stride formula were the
issue, baseline would explode harder than buddy does.

It doesn't explode because **the attn group never enters the strided
path in baseline**. The branch is:

```python
if (kv_cache_spec.page_size_padded is not None
    or base_page_bytes is not None):
    # strided path  ← buggy when ratio > 1 for attn
    ...
else:
    kv_cache = raw_tensor.view(kv_cache_shape)   # ← ratio-agnostic
```

For baseline-attn, **both flags are None**: `page_size_padded` is None
because attn is the *inflated* side (mamba is the one padded), and
`base_page_bytes` is None because there's no buddy. So baseline-attn
takes the simple `.view()` path. `.view()` doesn't care about ratio — the
slab is `num_blocks × 1856 × 1152 × dtype_size` bytes, the kernel view is
`(29×num_blocks, 64, 576)` of identical total bytes, the reinterpretation
is contiguous and correct.

The strided path *does* run in baseline — for the mamba group, whose
`page_size_padded` is set — but mamba goes through the separate
`MambaSpec` branch in `_reshape_kv_cache_tensors`, which computes its own
stride explicitly and is correct.

|  | `page_size_padded` | `base_page_bytes` | path | bug fires? |
|---|---|---|---|---|
| baseline attn | None | None | `.view()` | no |
| baseline mamba | set | None | strided (MambaSpec) | no (own math) |
| buddy attn | None | **set** | strided (AttentionSpec) | **yes if ratio > 1** |
| buddy mamba | None | set | strided (MambaSpec) | no (own math) |

The branch condition `or base_page_bytes is not None` is added by our WIP
commit (`3f12d80ca`), which widened the guard from upstream's
`page_size_padded is not None`. That widening is what pulls buddy-attn
into the strided path. Upstream's strided path never ran for attn groups
whose `page_size_padded` was None, so the wrong stride formula was simply
unreachable.

## Why `bs=64` and `bs=32` are fine under buddy too

Among the configs that *do* enter the strided path (i.e., buddy on, attn
group), the bug only fires when the kernel splits the spec block:

| Buddy config | spec.block_size | kernel_block_size | ratio | stride correct? |
|---|---|---|---|---|
| `bs=32` | 32 | 32 | 1 | ✓ — stride == one kernel page |
| `bs=64` | 64 | 64 | 1 | ✓ — stride == one kernel page |
| **`bs=128`** | **128** | **64** | **2** | **✗ — stride == 2 × one kernel page** |
| `bs=256` (hypothetical) | 256 | 64 | 4 | ✗ — stride == 4× too large |

At ratio = 1, "one spec block" and "one kernel block" are the same thing,
so the buggy formula (stride = base_page = spec.page_size_bytes)
coincidentally equals the right value (stride = kernel_block_size ×
per_token_bytes). The bug is latent until the kernel chooses a smaller
block size than the spec, which on FlashInfer-MLA happens for any
`bs > 64`.

## Why the strided path is taken at all on this branch

The branch guard `page_size_padded is not None or base_page_bytes is not
None` (the `or base_page_bytes …` clause is added by our WIP commit
`3f12d80ca`). In buddy-decoupled mode `base_page_bytes` is always set —
it's how the runner knows it's in the new layout — so the strided path
runs unconditionally for every attn group, even ones whose `chunk_order`
is 0 and whose view would be naturally contiguous.

For Kimi-Linear's MLA group (`chunk_order = 0`), a plain `.view()` would
have worked at every `bs` (this is exactly what baseline-attn does).
The strided trick is only semantically needed when adjacent kernel
indices should overlap (i.e., `chunk_order > 0` — mamba). For attn
groups it's a no-op at `chunk_order=0` *if* the stride is computed
correctly, and an over-allocation otherwise.

## Fix applied (`vllm/v1/worker/gpu_model_runner.py`)

The strided-view guard at `:6979` is narrowed: under buddy
(`base_page_bytes is not None`), only enter the strided path when this
group's view actually has gaps/overlap — i.e., when
`base_page_bytes < kv_cache_spec.page_size_bytes`. Otherwise (chunk_order
= 0), fall through to `.view()`, trimming the slab's max-overhang tail
first so element counts match:

```python
needs_strided_view = (
    kv_cache_spec.page_size_padded is not None
    or (
        base_page_bytes is not None
        and base_page_bytes < kv_cache_spec.page_size_bytes
    )
)
if needs_strided_view:
    # ... existing strided code ...
else:
    view_numel = torch.Size(kv_cache_shape).numel()
    if raw_tensor.numel() != view_numel:
        raw_tensor = raw_tensor.narrow(0, 0, view_numel)
    kv_cache = raw_tensor.view(kv_cache_shape)
```

Rationale: this is exactly the path baseline-attn already takes under
upstream code (when `page_size_padded` is None). Under buddy at
chunk_order=0, the slab layout for an attn group is contiguous — the
only difference from upstream is the max-overhang tail at the end of
the slab, which the `.narrow()` skips. The strided code is preserved
verbatim for the cases that genuinely need it (page_size_padded set, or
chunk_order > 0 attn groups under buddy).

The mamba `MambaSpec` branch is untouched — it computes its own stride
and was always correct.

## Alternative fixes considered (not applied)

Two reasonable directions; both close the bug. Listed in order of
intrusiveness.

1. **Compute the kernel-block stride from kernel byte size, not base
   page.** Inside the buddy-decoupled branch:

   ```python
   per_token_bytes = kv_cache_spec.page_size_bytes // kv_cache_spec.block_size
   kernel_page_stride = (kernel_block_size * per_token_bytes) // dtype_size
   strides[inv_order[0]] = kernel_page_stride
   ```

   Reverts to `page_stride = base_page / dtype_size` only when
   `kernel_block_size == spec.block_size` (i.e., the kernel didn't split
   the spec block). This is the same value in both cases for attn groups
   with `chunk_order = 0`, and matches the chunk_order > 0 case as well
   because mamba groups go through the separate `MambaSpec` branch
   (different stride math).

2. **Fall back to `.view()` when the view is contiguous.** Check
   `chunk_order == 0` (or equivalently `base_page_bytes ==
   kv_cache_spec.page_size_bytes`) and `page_size_padded is None`, and
   in that case use the simple `raw_tensor.view(kv_cache_shape)` path.
   This is exactly what baseline-attn already does — see the
   "Why baseline doesn't hit this" section above. The fix is essentially
   "narrow our widened guard back down so chunk_order=0 attn groups stay
   on the upstream `.view()` path." Easier to reason about than (1)
   because it's the known-correct baseline path.

Neither fix touches the mamba branch — that one already passes the
strided stride explicitly and is correct.

## Workaround until fixed

Pin `--block-size` to a value where `kernel_block_size ==
spec.block_size` for the selected backend:

- FlashInfer-MLA supports `[32, 64]` → use 32 or 64.
- CutlassMLA forces 128 → only safe if every attn group's
  `spec.block_size` is also 128 *and* there is no mamba group forcing
  base_page smaller. On Kimi-Linear there is a mamba group, so this
  doesn't help.

`bs=64` is the simplest configuration that avoids both this bug and the
FlashInfer alignment issue.

## Code-reference index

| What | Where |
|---|---|
| Crash site | `vllm/v1/worker/gpu_model_runner.py:6997` |
| Wrong stride assignment | `vllm/v1/worker/gpu_model_runner.py:6993-7001` |
| Profiling entry point | `vllm/v1/worker/gpu_model_runner.py:6179` (`_init_minimal_kv_cache_for_profiling`) |
| Mamba branch (correct, separate) | `vllm/v1/worker/gpu_model_runner.py:7007-7038` |
| Backend `get_kv_cache_shape` | `vllm/model_executor/layers/attention/mla_attention.py:1176` |
| Buddy layout doc | `current-work/decoupled_hybrid_pages.md` §6.1 (the "known issues" subsection notes this without the full analysis) |
