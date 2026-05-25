# FlashInfer-MLA `block_num × block_size` alignment check

Notes on the tile-alignment assertion we hit while benchmarking
Kimi-Linear-48B on SM10 (GB200) at `block_size=32`. Documents where the
check lives, what it actually constrains, why it fires, and how the bench
mitigation in `bench_kimi_linear.py` interacts with it.

## The assertion

`flashinfer/mla/_core.py` (FlashInfer 0.6.11.post2, vendored in `.venv`)
has the constraint in two places — one per kernel path:

1. **Cutlass-backed MLA path** — `_check_cutlass_shape` (line 68):

   ```python
   if block_num % (128 / block_size) != 0:
       raise ValueError(
           f"Expected block_num % (128 / block_size) == 0, "
           f"got {block_num=} and {block_size=}"
       )
   ```

   where `block_num = page_table.shape[-1]` and
   `block_size = ckv_kpe_cache.shape[1]`.

2. **TRT-LLM gen MLA path** — `_check_trtllm_gen_mla_shape` (line 192):

   ```python
   block_num = page_table.shape[-1]
   block_size = page_size
   if block_num % (128 / block_size) != 0:
       raise ValueError(
           f"Expected block_num % (128 / block_size) == 0, "
           f"got {block_num=} and {block_size=}"
       )
   ```

   `vllm/v1/attention/backends/mla/flashinfer_mla.py` calls
   `trtllm_batch_decode_with_kv_cache_mla` (imported at module top), so this
   is the path that fires for `FLASHINFER_MLA` decode on SM10.

Equivalent rearrangement:

```
block_num × block_size ≡ 0  (mod 128)
```

## Walkthrough — one request mid-decode

Kimi-Linear-48B on GB200, `--block-size 32`, FlashInfer-MLA selected. One
request with a 287-token prompt is decoding tokens 288 → 320. After 290
tokens written, it holds:

```
cdiv(290, 32) = 10 blocks in its page-table row.

block_table[req=0]:  [ B17 | B42 |  B3 |  B8 | B91 | B12 | B55 | B23 | B77 | B61 ]
                       ^                                                          ^
                       block_num = 10 entries
                       10 × 32 = 320 tokens covered

Check: block_num × block_size ≡ 0 (mod 128)?
       10 × 32 = 320,  320 % 128 = 64           ← NOT 0
       fail
```

The kernel raises
`Expected block_num % (128 / block_size) == 0, got block_num=10, block_size=32`
because `10 % 4 = 2`.

To pass, the page table must round **up** to 12 entries (= 384 tokens of
capacity, of which only 290 hold real data — the rest is padding the
runtime appends as dummy block ids masked out by `seq_lens`):

```
block_table[req=0]:  [ B17 | B42 |  B3 |  B8 | B91 | B12 | B55 | B23 | B77 | B61 | pad | pad ]
                                                                                   ^^^^^^^^^^^
                                                                                   added by runtime
                       block_num = 12,  12 × 32 = 384 ≡ 0 (mod 128)  ✓
```

The same request at `block_size = 64` needs only 5 real blocks and 1 pad
(pad-to-6 ⇒ 6 × 64 = 384). At `block_size = 128` no padding is ever
needed — but FlashInfer-MLA refuses that kernel block size, so it isn't a
real option.

| `block_size` | divisor `128 / bs` | max pad blocks per request |
|---|---|---|
| 16 | 8 | up to 7 (FlashInfer can't pick this) |
| 32 | 4 | up to 3 |
| 64 | 2 | up to 1 |
| 128 | 1 | 0 (FlashInfer declines; CutlassMLA only) |

## What our bench actually saw

```
ValueError: Expected block_num % (128 / block_size) == 0,
got block_num=295 and block_size=32
```

`295 × 32 = 9,440` tokens — beyond `max_model_len=8192`, so this can't be
a real decode. It came from the **CUDA-graph capture path**, which builds
a fake worst-case page table with `block_num` close to `num_gpu_blocks`.
`295 % 4 = 3` → fail.

The bench used to retry with `NUM_GPU_BLOCKS_OVERRIDE = (block_num //
align) * align` (rounding the pool **down** to a multiple of `align`).
That only worked because the failing dispatch was the capture-time path
table, whose inner-dim is bounded by the pool size. For real per-request
decode, the inner-dim is `cdiv(context_len, block_size)` and the pool
size doesn't directly bound it — but real decode is handled separately
by vLLM padding the live block tables.

We now run the bench with `enforce_eager=True` (no CUDA graph capture),
which sidesteps both the capture-time alignment failure and the buddy
reshape bug at `bs ≥ 128` (`BUDDY_PROFILING_RESHAPE_BUG.md`).

## What it actually constrains

- `block_num` is `page_table.shape[-1]` — the **max blocks per request** in
  the batch, *not* `num_gpu_blocks`. The page-table is `(num_seqs, max_blocks)`,
  and the constraint is on the per-request dimension.
- `block_size` is the **kernel** block size (`page_size`), which for
  FlashInfer-MLA must come from `get_supported_kernel_block_sizes() →
  [32, 64]` (see `flashinfer_mla.py:49`). So in practice the divisor
  `128 / block_size` is **4** (at `bs=32`) or **2** (at `bs=64`).
- `block_size = 128` would trivialize the divisor to 1, but FlashInfer-MLA
  declines that kernel size, so 128 is not a way out — it instead forces
  fallback to `CUTLASS_MLA` (which itself requires `block_size=128`) or
  further down the priority list.

The constraint exists because the TRT-LLM gen MLA decode kernel processes
KV in 128-token tiles per CTA; the page-table must be a whole number of
tiles long.

## Why baseline (`bs ≈ 1888`) doesn't fire this

The baseline hybrid-page padding inflates `attn_block_size` to match
`mamba_page` — on Kimi-Linear that's ~1888. The kernel block size that
gets selected at that spec is also large enough that
`block_num × block_size` is always a multiple of 128 for any realistic
page-table inner-dim. The constraint is silently satisfied, not avoided.

The buddy path is what exposes it: by design, MLA keeps its small
kernel-natural `block_size` and the inner-dim grows correspondingly.

## Interaction with backend priority on SM10

From `vllm/platforms/cuda.py:111-122`, the MLA priority list on SM10 is:

```
FLASHINFER_MLA → TOKENSPEED_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA
```

`FLASHINFER_MLA` claims `[32, 64]`; `CUTLASS_MLA` claims `[128]`. At a
given `--block-size` value:

- `bs=16`: none of the high-priority MLA backends accept it. The path
  ends up on `TRITON_MLA`, which is the portable fallback and not
  tile-tuned — this is what produced the +9× regression noted in
  `decoupled_hybrid_pages.md` §6.1.
- `bs=32`: FlashInfer accepts. Subject to the `block_num % 4 == 0`
  constraint, which is fragile under the buddy's small kernel blocks.
- `bs=64`: FlashInfer accepts. Constraint is `block_num % 2 == 0` —
  satisfied by almost any allocation pattern.
- `bs=128`: FlashInfer declines. CutlassMLA accepts with forced
  `_block_size = 128` and its own per-CTA shape rules
  (`vllm/v1/attention/backends/mla/cutlass_mla.py:75,203`). But the buddy's
  decouple-strided reshape has a separate bug at `bs ≥ 128` — see
  `BUDDY_PROFILING_RESHAPE_BUG.md`.

## Recommendation

For MLA hybrids under `VLLM_USE_BUDDY_BLOCK_POOL=1`, set `--block-size 64`
unless there is a specific reason to deviate:

- 64 is FlashInfer's largest supported kernel block size, which keeps the
  alignment divisor at 2 (least restrictive).
- 64 avoids the `bs ≥ 128` reshape bug.
- 64 amortizes per-block decode overhead well enough that the buddy's
  capacity gain becomes a net throughput win
  (`bench_kimi_linear_result.json`: baseline 1,520 → buddy 1,859 tok/s).

If a workload requires smaller `bs`, the proper fix is to teach vLLM's
page-table builder (and the CUDA-graph capture-time fake page-table) to
pad the inner dim to a multiple of `128 / block_size` when the selected
MLA backend is FlashInfer. Until then, `enforce_eager=True` is the
simplest way to keep `bs=32` working under FlashInfer-MLA, since the
constraint is only ever hit on the capture path in practice.

## Code-reference index

| What | Where |
|---|---|
| Constraint, cutlass-backed path | `flashinfer/mla/_core.py:68` (in `.venv`) |
| Constraint, trtllm-gen path | `flashinfer/mla/_core.py:192` |
| FlashInfer-MLA wrapper in vLLM | `vllm/v1/attention/backends/mla/flashinfer_mla.py` |
| FlashInfer-MLA supported kernel sizes | `flashinfer_mla.py:49` → `[32, 64]` |
| MLA backend priority on SM10 | `vllm/platforms/cuda.py:111-122` |
| Bench retry logic | `current-work/bench_kimi_linear.py:167-194` |
