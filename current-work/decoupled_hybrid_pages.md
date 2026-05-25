# Decoupled hybrid page sizes — design summary

Gated by `VLLM_USE_BUDDY_BLOCK_POOL=1`. Activates whenever a hybrid model
has groups with different native page sizes (Zamba2, Falcon-Mamba, etc.).

## 1. The bottleneck (before)

In vLLM today, all KV cache groups must share one **uniform `page_size_bytes`**
so the column-packed cross-group tensor sharing is byte-aligned. Under
prefix caching, mamba's storage needs to be aligned to a chunk boundary
(typically 256 tokens) and the per-token mamba state is large
(≈ 7 KiB on Zamba2-7B). To keep `attn_page == mamba_page` the engine inflates
**attn's** `block_size` until each attn block holds a chunk's worth of tokens.

For Zamba2-7B + prefix-caching ON:

```
attn_block_size  = 256  tokens    (forced up from kernel-natural 16)
mamba_block_size = 256  tokens
attn_page        = 256 × 28 KiB  = 7,340,032 B
mamba_native     =                 ~962 KiB
mamba_padded     = 7,340,032 B    (= attn_page; the padding is empty space)
```

Two costs from this:
- **Coarse prefix-cache granularity.** A 30-token prompt at
  `attn_block_size=256` fills 0 full blocks → no prefix-cache entry.
- **Concurrency cliff.** Each mamba "block" reserves a full chunk-sized
  slot in shared tensors even though only the first ~13% holds real data,
  collapsing `num_blocks` from 3,905 (prefix OFF) to **731** (prefix ON).

## 2. Column-packed layout — unchanged

We keep the existing column-packed tensor layout. For Zamba2-7B
(13 attn layers + 81 mamba layers split into 7 sub-groups), 13 slabs are
created and each slab is shared by one layer from each group at the same
tuple-index:

```
                    slab_0           slab_1     ...  slab_12

                  shared_by:       shared_by:        shared_by:
                  [attn.0,         [attn.1,          [attn.12,
                   mamba_g1.0,      mamba_g1.1,       mamba_g1.12,
                   mamba_g2.0,      mamba_g2.1,       mamba_g2.12,
                   ...              ...               ...
                   mamba_g7.0]      mamba_g7.1]       mamba_g7.12]
```

What changes is **how each slab is sized and indexed**.

## 3. The new layout — `base_page + chunk_order` per group

Each slab is `num_blocks × base_page` bytes (+ a small tail overhang),
where:

```
base_page          = min(group.page_size_bytes)            # = attn_page (small)
group.chunk_order  = ceil(log2(group.page_size / base_page))
```

For Zamba2-7B + prefix-caching ON, decoupled:

```
attn.page_size_bytes  = 458,752 B  (block_size=16, kernel-natural)
mamba.page_size_bytes = 962,048 B  (native, no padding)
base_page             = 458,752 B
attn.chunk_order      = 0          (one base block per attn allocation)
mamba.chunk_order     = 2          (four base blocks per mamba allocation,
                                    since 962,048 / 458,752 ≈ 2.10 → ceil to 4)
```

Layout inside one slab (`base_page` = 1 cell wide):

```
                 base_idx  0   1   2   3   4   5   6   7   8   9  10  11 ...
                          ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
slab_i  (bytes)           │ b │ b │ b │ b │ b │ b │ b │ b │ b │ b │ b │ b │
                          └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
                              ↑               ↑               ↑
                       1× base = attn   4× base = mamba   1× base = attn
                       block (order 0)  block (order 2)   block (order 0)
```

Slab capacity: `num_blocks` base cells. Total memory budget unchanged:
`13 slabs × num_blocks × base_page`. The first slab is sized
`num_blocks × base_page + (mamba_page - base_page)` to give the strided
view safe room at the tail.

## 4. Allocation example

Suppose the buddy address space is `num_blocks = 16` (toy size). One
request comes in and asks for:

- 5 attn blocks  (cdiv(80 tokens / 16) at `attn_block_size = 16`)
- 1 mamba block per mamba sub-group (× 7 groups → 7 mamba blocks)

The buddy hands out chunks at the group's order. **Order is implicit** —
the request's group is known, so the buddy splits/coalesces accordingly.

```
                  Initial state (all free, max_order = log2(16) = 4):

base_idx          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
                ┌───────────────────────────────────────────────────────────┐
free chunks     │           one  free  order-4  chunk  (16 bases)           │
                └───────────────────────────────────────────────────────────┘


                  After 5 attn allocations (order 0 each):

base_idx          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
                ┌───┬───┬───┬───┬───┬───┬───┬───┬───────┬───────────────────┐
                │A.0│A.1│A.2│A.3│A.4│ . │ . │ . │  free │       free        │
                └───┴───┴───┴───┴───┴───┴───┴───┴───────┴───────────────────┘
                  ↑                       ↑       ↑       ↑
                  attn chunks (1 base)    free    order-1 order-3
                  Buddy split sequence: 4 → 3+3 → 2+2+2 → 1+1+1+1+1 (5 used)


                  Plus 7 mamba allocations (order 2, 4 bases each):

base_idx          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
                ┌───┬───┬───┬───┬───┬───┬───┬───┬───────────────┬───────────┐
                │A.0│A.1│A.2│A.3│A.4│ x │ x │ x │   M.0 (×4)    │ OOM here  │
                └───┴───┴───┴───┴───┴───┴───┴───┴───────────────┴───────────┘
                                  ↑               ↑
                          unaligned —     mamba's order-2 chunk
                          attn can fit    starts at order-2 boundary
                          but mamba needs (idx 8, 12)
                          4-aligned space

                  Only one order-2 chunk is free (idx 8-11). Mamba_g1 takes
                  it; mamba_g2..g7 need to evict cached chunks or fail.
```

Two things to notice:

1. **Attn allocations are 1 base block each** — fine prefix-cache granularity.
   A 16-token prompt produces one cacheable block. (In the old uniform-page
   scheme, the same 16-token prompt would produce 0 cacheable blocks because
   `attn_block_size` was inflated to 256.)

2. **Mamba allocations are 4 base blocks each, on 4-aligned starts.** This
   over-allocates by `(4 × base_page − mamba_native) = (4 × 448 KiB − 940 KiB)
   = 852 KiB` per mamba block — about 47% rounding waste. But the saved
   memory from killing the chunk-size attn inflation dwarfs this.

## 5. Strided view inside a slab

The model runner builds **per-group strided views** over each slab:

```
Attn view of slab_i:                Mamba view of slab_i:

view[k] reads base_page bytes       view[k] reads page_size_bytes bytes
starting at offset k × base_page    starting at offset k × base_page
shape: (num_blocks, attn_inner)     shape: (num_blocks, mamba_inner)
stride[0] = base_page               stride[0] = base_page    ← same stride
                                                              despite larger
                                                              inner shape
```

For mamba, **adjacent indices overlap in memory** by
`(mamba_page − base_page) = 503 KiB`:

```
slab byte offset:   0 ----- 448K ----- 896K ----- 1344K ----- 1792K -----...

mamba.view[0]:      ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃                          (0   → 962K)
mamba.view[1]:              ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃                  (448 → 1410K)
mamba.view[2]:                       ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃         (896 → 1858K)
mamba.view[4]:                                  ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━┃  (1792→ 2754K)
                                                ↑
                                non-overlapping with view[0]
                                (one full mamba chunk = 4 bases away)
```

The overlap is safe because **the buddy guarantees that only one of
{view[0], view[1], view[2], view[3]} can be allocated at a time** — they
all live inside the same order-2 chunk. The block_table only ever records
4-aligned starts for mamba.

## 6. End-to-end numbers (validated)

```
                                  baseline (uniform)   decoupled (this work)
Zamba2-1.2B  prefix=ON
  GPU KV cache size               106,656 tokens       305,621 tokens     +186%
  Max concurrency @ 2048 ctx      52.08×               149.23×            +186%
  Outputs                                              bit-identical to baseline

Zamba2-7B    prefix=ON
  GPU KV cache size                23,552 tokens        98,070 tokens     +316%
  Max concurrency @ 2048 ctx      11.50×                47.89×            +316%
  Outputs                                              bit-identical to baseline

Non-hybrid models (Qwen3-0.6B)    no change            no change           — 
                                  (decouple_active is False when all groups
                                   already have the same page size)
```

### MLA-based hybrids (Kimi-Linear-48B): block_size + backend matters

The Zamba2 wins above are capacity-driven (`+N%` *max concurrency*, same
per-request kernel cost). MLA-based hybrids add a knob: MLA decode cost
scales with `cdiv(context_len, block_size)`, so very small `block_size`
combined with a non-tile-tuned backend can erase the buddy win.

Kimi-Linear-48B has 7 MLA layers (page ~18 KiB at `bs=16`) and 20 KDA
layers (page ~2 MiB). Same 16-prompt × 256-token bench at 8K context, 1×
GB200, `mamba_cache_mode='align'`:

```
              backend       bs   num_blocks   gen     throughput
baseline      TRITON_MLA    16        3,033   2.09 s   1,514 tok/s
buddy         TRITON_MLA    16      587,869  17.93 s     172 tok/s    ×0.11
baseline      FLASHINFER    64        2,955   2.08 s   1,520 tok/s
buddy         FLASHINFER    64      147,195   1.84 s   1,859 tok/s    ×1.22
```

Two things are happening:

1. **MLA decode iterations per request.** At `bs=16`, an 8K-context
   request makes the decode kernel walk 512 block_table entries per query
   per layer; at `bs=64`, 128 entries. At baseline's effective inflated
   `bs ≈ 1920`, only 4–5 entries. The buddy lets MLA stay at its
   kernel-natural small bs, so the iteration count is set by `bs`, not
   inflation.
2. **Backend amortization.** `TRITON_MLA` is the portable fallback and
   handles small `bs` poorly. `FLASHINFER_MLA` (default on SM10) is tuned
   for `bs ∈ {32, 64, 128, …}` with proper tile reuse, and at `bs=64`
   amortizes per-block overhead well enough that the buddy's 50× capacity
   gain becomes a net throughput gain too.

The buddy is therefore **block_size- and backend-sensitive** on MLA
hybrids. Recommended config for Kimi-Linear-style models:

- `block_size ≥ 64` (so FlashInfer-MLA's `block_num × block_size % 128 == 0`
  tile constraint is easy to satisfy and decode iterations are kept
  manageable).
- Default attention backend (do not force `TRITON_MLA`).
- `VLLM_USE_BUDDY_BLOCK_POOL=1`, `VLLM_BUDDY_MAX_ORDER` ≥ the largest
  per-group `chunk_order` (4 for Kimi-Linear at bs=64).

At `bs=16` on this model the buddy is a workload-dependent trade-off (good
for high-concurrency, bad for cold single-batch throughput on
`TRITON_MLA`). At `bs=64` on FlashInfer the buddy is a clean win on both
axes.

Higher block sizes now run end-to-end after the reshape fix
(`BUDDY_PROFILING_RESHAPE_BUG.md`). Verified at `bs=128`:

```
              backend       bs   num_blocks    gen     throughput
baseline      FLASHINFER   128        3,003   13.70 s     227.8 tok/s
buddy         FLASHINFER   128       74,177   13.10 s     261.0 tok/s   ×1.15
```

Capacity gain is ×24.7 here, but throughput at `bs=128` is much lower
than at `bs=64` (1,859 tok/s) — large per-block granularity hurts decode
parallelism. `bs=64` remains the recommended setting; `bs=128` is now
*correct* but not preferred.

## 7. Code touchpoints

```
vllm/platforms/interface.py
  └── _align_hybrid_block_size:
        skip attn_block_size inflation & mamba page padding under buddy
        (keep attn at kernel-natural block_size, mamba at native page).

vllm/v1/core/kv_cache_utils.py
  ├── get_kv_cache_groups:
  │     skip unify_kv_cache_spec_page_size() under buddy.
  ├── get_kv_cache_config_from_groups:
  │     compute base_page = min(group_pages), per-group chunk_order,
  │     tensor.size = num_blocks * base_page + max_overhang.
  ├── resolve_kv_cache_block_sizes:
  │     in decoupled mode keep hash_block_size = GCD of group block_sizes
  │     (don't back off to LCM just because mamba diverges).
  ├── _max_memory_usage_bytes_from_groups:
  │     non-uniform-page-aware concurrency math.
  └── get_kv_cache_configs (the shrink-to-min-num-blocks loop):
        preserve tail overhang when shrinking tensor.size.

vllm/v1/kv_cache_interface.py
  ├── KVCacheConfig.base_page_bytes: int | None
  └── KVCacheGroupSpec.chunk_order: int = 0

vllm/v1/core/kv_cache_coordinator.py
  └── HybridKVCacheCoordinator:
        pass kv_cache_group.chunk_order to each SingleTypeKVCacheManager.

vllm/v1/core/single_type_kv_cache_manager.py
  └── __init__:
        new `chunk_order` arg; sets self._buddy_order from it
        (env-var override still honored for legacy testing).

vllm/v1/worker/gpu_model_runner.py
  └── _reshape_kv_cache_tensors:
        when base_page_bytes is set, use it as the strided stride for
        both attn and mamba views; num_blocks comes from kv_cache_config.
```

The buddy queue itself (`buddy_free_queue.py`, `buddy_block_pool.py`)
needed no changes — variable-order allocation was already wired through
`KVCacheBlock.chunk_order` and `alloc_chunk(order)`.
