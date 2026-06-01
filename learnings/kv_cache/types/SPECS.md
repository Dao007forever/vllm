# KV Cache Specs

A `KVCacheSpec` describes the **physical KV cache format of one layer**: how many
bytes one block (page) occupies, and how much memory the layer needs at peak. It
is the bridge between a layer's attention type and the memory the engine reserves
for it.

All specs live in [`vllm/v1/kv_cache_interface.py`](../../../vllm/v1/kv_cache_interface.py).

---

## The base contract

```python
# vllm/v1/kv_cache_interface.py:94
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int                              # tokens per block

    @property
    def page_size_bytes(self) -> int: ...        # bytes for one block
    @property
    def storage_block_size(self) -> int: ...     # = block_size, unless compressed
    def max_memory_usage_bytes(self, vllm_config) -> int: ...  # peak bytes for the layer
    @classmethod
    def merge(cls, specs) -> KVCacheSpec: ...     # collapse identical layers
```

Two methods carry the weight:

- **`page_size_bytes`** — bytes of one block. Used to slice the raw byte buffer
  into blocks and to count how many blocks fit in the memory budget.
- **`max_memory_usage_bytes`** — worst-case bytes this layer can consume, used at
  startup to divide the GPU memory budget into a block count.

`storage_block_size` is normally equal to `block_size`. It diverges only for
**compressed** specs (DeepSeek V4), which store `block_size` tokens in
`block_size // compress_ratio` physical slots.

---

## The spec hierarchy

```
KVCacheSpec                          (base: block_size only)
├── AttentionSpec                    (+ num_kv_heads, head_size, dtype, kv_quant_mode)
│   ├── FullAttentionSpec            standard full attention (the common case)
│   │   ├── TQFullAttentionSpec      TQ-aware page size (tq_slot_size)
│   │   ├── MLAAttentionSpec         Multi-head Latent Attention (DeepSeek)
│   │   │   └── HiddenStateCacheSpec hidden-state cache (extract_hidden_states)
│   │   └── SinkFullAttentionSpec    attention sink (+ sink_len)
│   ├── ChunkedLocalAttentionSpec    chunked local attention (+ attention_chunk_size)
│   ├── SlidingWindowSpec            sliding window (+ sliding_window)
│   │   └── SlidingWindowMLASpec     sliding window with MLA layout
│   ├── EncoderOnlyAttentionSpec     encoder layers — needs no KV cache (0 bytes)
│   └── CrossAttentionSpec           enc-dec cross attention (caches encoder states)
├── MambaSpec                        Mamba/SSM state (not attention; raw shapes/dtypes)
└── UniformTypeKVCacheSpecs          a *container* grouping per-layer specs of one type
```

`KVCacheSpecKind` ([`kv_cache_interface.py:81`](../../../vllm/v1/kv_cache_interface.py#L81))
is the flat enum tag, resolved by `get_kv_cache_spec_kind()` with **subclass checks
ordered before base classes** so e.g. an `MLAAttentionSpec` reports `MLA_ATTENTION`,
not `FULL_ATTENTION`.

---

## Per-spec page-size formulas

| Spec | `page_size_bytes` (uncompressed, non-quantized) | Peak memory |
|---|---|---|
| `FullAttentionSpec` | `block_size · num_kv_heads · (head_size + head_size_v) · dtype_size` | `cdiv(max_model_len, block_size) · page_size` |
| `SlidingWindowSpec` | same as full | `(cdiv(window-1 + batched, block_size) + 1) · page_size` — capped by window |
| `ChunkedLocalAttentionSpec` | from `AttentionSpec` (2·K·V·head·dtype) | `cdiv(chunk + batched, block_size) · page_size` |
| `MLAAttentionSpec` | `storage_block_size · num_kv_heads · head_size · dtype_size` (or custom 584/656 B for fp8_ds_mla) | full-attention formula |
| `CrossAttentionSpec` | from `AttentionSpec` | `cdiv(max_encoder_len, block_size) · page_size` (Whisper: 1500) |
| `EncoderOnlyAttentionSpec` | from `AttentionSpec` | **0** — encoder layers cache nothing |
| `MambaSpec` | `Σ prod(shape) · dtype_size` over state tensors (+ optional padding) | depends on `mamba_cache_mode` (none/align/all) |

Notes that bite:

- **`AttentionSpec` base** multiplies by `2` (separate K and V), so its
  `real_page_size_bytes = 2 · block · heads · head_size · dtype`. **`FullAttentionSpec`
  overrides** this to `block · heads · (head_size + head_size_v)` — the `2` becomes
  `head_size + head_size_v`, which equals `2·head_size` only when K and V dims match.
- **Quantization** inflates `page_size_bytes`: per-token-head modes (int8/fp8) add
  `2 · block · num_kv_heads · 4` bytes for fp32 scales; NVFP4 packs fp4 data + fp8
  block scales via `nvfp4_kv_cache_full_dim`.
- **`page_size_padded`** lets a spec round its page up (alignment, NVFP4) — when set,
  `page_size_bytes` returns the padded value and the reshape step uses a *strided*
  view so the unused tail bytes are skipped.

---

## `merge`: collapsing identical layers

Within one KV cache group, all layers must share the same spec. `merge()` asserts
this and returns one representative. The base implementation asserts strict
equality; `FullAttentionSpec.merge` is laxer — it unifies the *window sizes* across
layers (allowing `None` + a single concrete window) and rejects mixing sliding
window with chunked-local attention. `MLAAttentionSpec.merge` additionally requires
matching `cache_dtype_str`, `compress_ratio`, and `model_version`.

---

## `UniformTypeKVCacheSpecs` — a container, not a layer

```python
# vllm/v1/kv_cache_interface.py:667
@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    kv_cache_specs: dict[str, KVCacheSpec]   # layer_name -> per-layer spec
```

This is **not** a layer's spec — it is a wrapper holding a dict of per-layer specs.
It is created (`from_specs`) when every layer needs the **same number of token slots**
(same attention "type" and block size) but layers may have **different hidden sizes**
— e.g. all full attention, but layer A has more KV heads than layer B. `is_uniform_type`
is what gates this: same `block_size` for all, and all the same spec family (with
matching window/chunk size where applicable).

Its aggregate `page_size_bytes` is the **sum** across inner specs:

```python
# kv_cache_interface.py:677
@property
def page_size_bytes(self) -> int:
    return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())
```

i.e. the bytes of **one block across the whole tuple of layers**. All layers share a
single block count; that count is derived from this combined page size.

---

## How the physical tensor is allocated per inner type

This is the crux of the `UniformTypeKVCacheSpecs` design: **one shared `num_blocks`,
but a separately-sized, dedicated physical tensor per layer.**

### Step 1 — one block count for the whole group

[`get_kv_cache_config_from_groups`](../../../vllm/v1/core/kv_cache_utils.py#L1262):

```python
# kv_cache_utils.py:1262
if len(kv_cache_groups) == 1 and isinstance(
    kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
):
    num_blocks = (
        available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
    )                                       # <- divides by the SUM of inner page sizes
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    kv_cache_tensors = [
        KVCacheTensor(
            size=per_layer_specs[layer_name].page_size_bytes * num_blocks,  # <- per-layer bytes
            shared_by=[layer_name],                                          # <- NOT shared
        )
        for layer_name in kv_cache_groups[0].layer_names
    ]
```

The single division `available_memory // sum(page_sizes)` yields a block count that
is **common to all layers**. Because the denominator is the *sum* of the per-layer
page sizes, allocating `num_blocks` for every layer at its own page size exactly fills
the budget — and crucially keeps the **block index aligned across layers** (block `k`
means the same logical position in every layer, even though it occupies a different
byte count).

### Step 2 — one tensor per layer

Each layer gets **its own** `KVCacheTensor` with `shared_by=[layer_name]` (a single
name). The size is `per_layer_page_size · num_blocks` — so a layer with bigger hidden
size gets a proportionally bigger tensor, but the same number of blocks.

> Contrast with the **general hybrid path** (the `else` branch, [kv_cache_utils.py:1289](../../../vllm/v1/core/kv_cache_utils.py#L1289)),
> where one tensor of uniform `page_size · num_blocks` is **shared by one layer from
> each group** (`shared_by` has multiple names), and the layers carve out different
> regions via their block tables. UniformType does **not** share tensors across
> layers — each layer is independent, only the block *count* is shared.

### Step 3 — raw byte buffer → typed view

[`_allocate_kv_cache`](../../../vllm/v1/worker/gpu/attn_utils.py#L150) materializes
each `KVCacheTensor` as a flat `int8` buffer of `tensor.size` bytes:

```python
# gpu/attn_utils.py:154
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

Then [`_reshape_kv_cache`](../../../vllm/v1/worker/gpu/attn_utils.py#L169) reinterprets
each raw buffer into the backend's KV cache shape. The reshape works **per attention
group, using the per-layer spec** — not the container. The split back to per-layer
specs happens in [`gpu/attn_utils.py:95`](../../../vllm/v1/worker/gpu/attn_utils.py#L95):

```python
# gpu/attn_utils.py:94
layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
```

So in `_reshape_kv_cache`, `num_blocks = raw.numel() // spec.page_size_bytes` uses the
**per-layer** `page_size_bytes`, and recovers the same `num_blocks` for every layer —
exactly the count chosen in Step 1.

### Mental model

```
available_memory
   │  ÷ (pageA + pageB + pageC)          ← sum of inner page sizes
   ▼
num_blocks  (one number, shared by all layers)
   │
   ├── layer A: torch.zeros(pageA · num_blocks)  → reshape via specA
   ├── layer B: torch.zeros(pageB · num_blocks)  → reshape via specB
   └── layer C: torch.zeros(pageC · num_blocks)  → reshape via specC
       (three independent tensors; block index k aligns across all three)
```

The payoff: layers of differing hidden size live in **one KV cache group with one
block table**, because what the block table indexes is the block *number*, and every
layer agrees on the block count. Differing per-layer byte sizes are absorbed entirely
by giving each layer its own physical tensor.

---

## Special multi-group case: DeepSeek V4

DeepSeek V4 produces **multiple** `UniformTypeKVCacheSpecs` groups (full attention +
sliding-window MLA of different sizes). It takes the dedicated allocator
[`_get_kv_cache_config_deepseek_v4`](../../../vllm/v1/core/kv_cache_utils.py#L1179),
which sizes by "layer tuples" (`get_num_layer_tuples`, `get_page_sizes`) rather than
the single-group formula above. The container utility methods
`get_page_sizes` / `get_num_layer_tuples` / `max_memory_usage_pages` exist mainly to
serve this path.

---

## Related

- [KV_CACHE_ALLOCATION.md](../KV_CACHE_ALLOCATION.md) — end-to-end allocation flow
- [KERNEL_BLOCK_ALIGNMENT.md](../KERNEL_BLOCK_ALIGNMENT.md) — kernel tile vs. block size
- [hybrid/](../hybrid/) — multi-group hybrid models (full + sliding window)
