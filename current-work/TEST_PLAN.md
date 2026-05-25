# Test Plan — Non-Uniform Block Size in vLLM

Companion to `UNIFORM_BLOCK.md`. This plan defines how we validate that breaking the within-group uniform-block-size assumption preserves correctness and behaves as expected.

## Goals

1. **Correctness regression detection** — every patch must reproduce identical (or token-equivalent) outputs from a set of fixed prompts on representative models.
2. **Per-group plumbing regression detection** — hybrid models (which already use multiple KV cache groups) must continue to work; this is the closest existing analogue to what we're generalizing.
3. **Targeted heterogeneity test** — once within-group variable block size lands, demonstrate it works on a workload designed to exercise it.

## Testbed models

| Role | Model | Architecture | GPU | Notes |
|---|---|---|---|---|
| Uniform / "normal" baseline | `Qwen/Qwen3-0.6B` | GQA, uniform attention (base) | 0 | Replaces gated Llama-3.2-1B; already cached locally. |
| Hybrid / per-group stress | `Zyphra/Zamba2-1.2B-instruct` | Mamba2 + softmax attention | 1 | Two KV cache groups with very different specs. Needs `--chat`. |

Why these choices:
- Both small enough to load in <90s on a single GB200 and run with `gpu_memory_utilization=0.45`, leaving headroom for parallel use.
- Zamba2's two groups exercise the existing per-group machinery — the most likely regression surface for refactors aimed at heterogeneity.
- Qwen3-0.6B has no attention heterogeneity, so any regression on it isolates the within-group code path.

Models considered and rejected, with reason:
- Llama-3.2-1B-Instruct — gated; current HF token lacks access.
- Gemma-2-2B — has full+SW groups, but SW with `block_size=16` is already near-optimal; weak motivator.
- DeepSeek-V2-Lite — uniformly MLA across layers; no within-model heterogeneity.
- Falcon-H1 family — viable alternative to Zamba2 if it later proves more stable; not adopted now to avoid duplicating coverage.

## Correctness harness

Script: `current-work/test_correctness.py`

- 6 prompts spanning: factual short answer, arithmetic, code completion, multi-step reasoning, recall, instruction following.
- Greedy sampling (`temperature=0`, `seed=0`) for reproducibility.
- Soft semantic checks on prompts 1 / 4 / 5 — emit `WARN` (not `FAIL`) when the expected substring is missing. Empty output is `FAIL`.
- `--chat` flag applies the tokenizer chat template — required for `*-instruct` models.

### Baseline (recorded 2026-05-18)

| Model | Load | Gen | Failures | Warns |
|---|---|---|---|---|
| Qwen3-0.6B | 50.3s | 0.19s | 0 | prompts 4, 5 (small base model — expected) |
| Zamba2-1.2B-instruct (`--chat`) | 77.5s | 3.15s | 0 | prompts 4, 5 (Zamba2 says "4" for the apple problem) |

The warnings reflect *model quality at this scale*, not vLLM correctness. They are stable across runs and become the regression signal: a patch that changes them is suspect.

### Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python current-work/test_correctness.py \
    --model Qwen/Qwen3-0.6B &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python current-work/test_correctness.py \
    --model Zyphra/Zamba2-1.2B-instruct --chat &
wait
```

Logs land under `current-work/logs/`.

## Test stages (mapped to the patch sequence)

### Stage 0 — Baseline (DONE)
- Status: ✅ Outputs captured for both models. Pipeline known-good.

### Stage 1 — Slot-mapping generalization
Triggered when changes land in:
- `vllm/v1/worker/block_table.py` (`_compute_slot_mapping_kernel`)
- `vllm/v1/worker/gpu/block_table.py` (`_compute_slot_mappings_kernel`)
- `vllm/v1/spec_decode/utils.py` (`compute_new_slot_mapping`)

Required checks:
1. Run the harness on both Qwen3-0.6B and Zamba2-1.2B-instruct.
2. Outputs must be **byte-identical** to the Stage 0 baseline (no model quality drift permitted from a refactor that should be a no-op when all blocks are the same size).
3. Token-level diff if outputs differ — emit which prompt regressed.

### Stage 2 — `reshape_and_cache` generalization
Triggered when changes land in:
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

Required checks:
- Same as Stage 1.
- Plus: a microbenchmark of the reshape kernel on a single block_size = baseline value, confirming no ≥5% throughput regression.

### Stage 3 — Block-table layout changes
Triggered when changes land in:
- `vllm/v1/worker/block_table.py` (BlockTable storage)
- `vllm/v1/core/block_pool.py`, `kv_cache_manager.py`

Required checks:
- Same as Stage 1.
- Plus: Zamba2 must successfully allocate KV groups end-to-end (this is the per-group plumbing regression check).
- Plus: confirm `block_table[req, col]` index math still works under the new metadata.

### Stage 4 — Attention kernel changes
Triggered when changes land in `vllm/v1/attention/backends/*` (FlashAttention, Triton MLA, etc.).

Required checks:
- Stage 1 checks.
- Add longer prompts (>1024 tokens) to exercise the K/V gather path more heavily — current 6-prompt set stays well under 100 tokens output, which under-tests this surface.

### Stage 4 — Buddy allocator (DONE)
See `BUDDY.md` for the allocator state and `decoupled_hybrid_pages.md` for
the layout that consumes it. Coverage:
- 25 unit tests (`tests/v1/core/test_buddy_block_pool.py`) — covers
  multi-order alloc/free, coalescing, `remove()`, and the eviction-callback
  paths exercised when prefix caching coexists with buddy.
- 3 slot-mapping tests (`tests/v1/worker/test_variable_block_slot_mapping.py`).
- Harness golden: bit-identical outputs across Qwen3-0.6B, Zamba2-1.2B,
  Zamba2-7B, Falcon-Mamba-7B against Stage 0 baseline. Goldens recorded
  under `current-work/golden/` and `current-work/golden_v2/`.
- **Now delivers cross-group memory savings** via the decoupled hybrid-page
  layout (`base_page` + per-group `chunk_order`). Validated +186% / +316%
  max concurrency on Zamba2-1.2B / 7B with prefix caching on. The earlier
  "no real savings" status was for the original
  shared-id-space-only design; the current implementation reinterprets the
  buddy as a base-page address space and adds per-group strided views over
  one column slab.

### Stage 5 — Within-group heterogeneity (deferred; motivator unclear after Kimi-Linear bench)
**Cross-group** heterogeneity is delivered by Stage 4 — each group picks one
static `chunk_order` based on its native page size, and that delivers the
measured +186%/+316% concurrency win.

**Within-group** mixing (allowing one group's allocations to vary in order
per-request) does **not** have a clean motivator after running the
Kimi-Linear bench more carefully. The previous "+9× regression →
within-group mixing fixes it" framing was an artifact of running at
`block_size=16` on `TRITON_MLA`. With `block_size=64` and the default
`FLASHINFER_MLA` backend, buddy already wins **+22% throughput** on
Kimi-Linear-48B at the same 50× capacity gain — see
`bench_kimi_linear_result.json` and `decoupled_hybrid_pages.md` §6.1.

So the remaining gap that within-group mixing would address is just the
`bs=16` regime — a regime that's not the default and not the recommended
configuration on MLA hybrids. Inside a single group all layers still share
one per-token byte cost, so there's no per-layer memory motivator;
secondary motivations (block-table compactness, fragmentation, allocator
throughput) are real but small and unmeasured.

Net: keep this deferred. Revisit only if a concrete workload (e.g., a
hybrid with `bs` pinned small for an external reason) demonstrates a
measurable win that per-group block_size + backend tuning can't already
deliver.

If revisited, the surface change is:
- Per-block-table-entry order metadata (or `cu_block_lens` per row).
- Slot-mapping kernel takes order from metadata, not from a constexpr.
- Attention K/V gather generalized to read variable-stride entries.
- Test: skewed-length batch on a non-hybrid model (e.g., Qwen3-0.6B) with
  a mix of 64-token and 1024-token requests; correctness vs. uniform;
  measure any throughput / fragmentation delta.

## Tooling gaps to address later

- **Golden-output capture**: the harness currently prints outputs but doesn't diff against a saved reference. Add this before Stage 1 so regression detection is automatic, not visual.
- **Per-group memory reporting**: vLLM prints aggregate KV-cache stats; we need a per-group breakdown for Zamba2 to validate Stage 3. Likely a small print in `kv_cache_manager` behind a debug flag.
- **Longer-context prompt**: needed for Stage 4. A single 2K-token prompt that the harness can optionally enable.
- **Numerical equivalence vs token equivalence**: under greedy sampling, byte-identical output is the bar. If we later need to test non-greedy, capture logprobs and compare with tolerance.

## What this plan does NOT cover

- Multi-GPU TP / PP correctness — single-GPU only for now.
- Speculative decoding — defer until base path is stable.
- Mooncake / disaggregated prefill / external KV connectors.
- Quantized KV cache — separate axis of heterogeneity, treated independently.
