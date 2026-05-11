# Why Mamba Is a Different Cache Problem (Even Among Linear-Attention Models)

[LINEAR_ATTENTION.md](LINEAR_ATTENTION.md) covers the conceptual frame:
linear attention compresses history into a fixed-size summary, breaks
positional retrieval, and forces a different cache shape. This note is the
sibling document for the next layer down — what makes **Mamba**
specifically awkward for the cache subsystem, separate from properties
shared with RWKV/RetNet/GLA.

The short version: Mamba is a *selective* state-space model (state-update
weights are themselves a function of the input) with **two** state tensors
per layer, **two** different kernels (parallel scan for prefill, step
recurrence for decode), and a **block-size constraint that doesn't come
from the math** — block boundaries are imposed by vLLM's allocator, not by
anything Mamba needs.

---

## 1. Mamba's "state" is two tensors, not one

`MambaSpec.shapes` is `tuple[tuple[int, ...], ...]`
([kv_cache_interface.py:532](vllm/v1/kv_cache_interface.py#L532)) — plural.
Every Mamba layer keeps two distinct buffers per request:

- **SSM state** (`d_inner × d_state` typically) — the long-range recurrent
  summary. This is the "linear-attention state" from
  [LINEAR_ATTENTION.md](LINEAR_ATTENTION.md).
- **Conv state** (`d_inner × (d_conv − 1)` typically, with `d_conv = 4`) —
  a tiny rolling window of recent activations feeding the depthwise 1D
  convolution that sits between the input projection and the selective
  scan.

Both have to be persisted, restored, and shipped together. The cache
infrastructure treats them as a single "page" — `page_size_bytes` sums
both — but the kernel addresses them separately.

This is why `MambaSpec` has `page_size_padded`
([kv_cache_interface.py:534](vllm/v1/kv_cache_interface.py#L534)). The two
tensors' raw byte sum may not match attention's `page_size`, so vLLM pads
the Mamba page up to the uniform invariant from
[KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md). Padding bytes are unused
but make the column-packed tensor layout work.

**What this changes from "generic linear attention":** a state hand-off
between layers, between scheduling steps, or between machines must move
*both* tensors atomically. Restoring only the SSM state and zeroing the
conv state would inject a 3-step transient where the convolution sees
fresh-start input — the model produces garbage tokens until the conv
buffer fills. The `align` cache mode (below) exists precisely because of
this conv-state subtlety.

## 2. Selectivity: state-update weights depend on the input

Original S4 had fixed `(A, B, C)` matrices. **Mamba's selective SSM** makes
`A_t, B_t, C_t` functions of the input token at step `t`:

```text
A_t, B_t, C_t = projection(x_t)
S_t = A_t · S_{t-1} + B_t · x_t
y_t = C_t · S_t
```

This is the architectural innovation that earned the "Mamba" name over S4.
For the cache, it means **you cannot cheaply replay a missed prefix**.
With a fixed-`A` SSM, you could in principle precompute powers of `A` and
fast-forward. With selectivity, the weights are determined by the actual
tokens, so replay requires running the full input projection on every
missed token. Replay costs the same as the original compute.

This is the load-bearing reason `MambaManager` does *not* implement
"replay from the latest checkpoint" as a recovery path during eviction or
preemption. If the state is gone, the only way back is to re-run the full
prefix through the model — same as a cold prefill. Selectivity made the
math mandate this.

## 3. Two kernels: parallel scan (prefill) vs step recurrence (decode)

Softmax attention uses one kernel for prefill and decode (FlashAttention
varlen handles both). Mamba2 uses two
([mamba2_attn.py:107-167](vllm/v1/attention/backends/mamba2_attn.py#L107)):

- **Prefill** runs a **chunk-wise parallel scan** that processes
  `chunk_size` tokens at once and produces both the per-token outputs and
  the final state. `chunk_size` is configured via
  `model_config.get_mamba_chunk_size()` — a per-model constant, typically
  256.
- **Decode** runs a **step recurrence** — one token at a time, just
  `S_t = A_t · S_{t-1} + B_t · x_t`.

Two kernels means two metadata shapes. `Mamba2AttentionMetadata` carries
chunk-related fields (`seq_idx`, `chunk_offsets`, `cu_seqlens_chunks`) that
are populated **only when `num_prefills > 0`**
([mamba2_attn.py:148](vllm/v1/attention/backends/mamba2_attn.py#L148)).
Decode-only batches skip the chunk metadata path entirely.

**What this changes for the scheduler:** chunked prefill — vLLM's
mechanism for splitting a long prefill across multiple scheduling steps —
must hand off the SSM state from chunk N's tail to chunk N+1's head. For
attention, chunked prefill just appends K/V to the same paged store; the
"hand-off" is implicit in the block table. For Mamba, the state at the
chunk boundary must explicitly be the input to the next forward. This is
what `last_state_block_idx`
([single_type_kv_cache_manager.py:805](vllm/v1/core/single_type_kv_cache_manager.py#L805))
tracks: which block holds the state to feed into the next step.

## 4. `chunk_size` is not `block_size`

A subtle but important distinction. Mamba2 has *three* sizes that get
confused:

| Name           | Source                                    | What it controls |
|----------------|-------------------------------------------|------------------|
| `block_size`   | `cache_config.block_size` (or aligned)    | Logical block size for the BlockPool — checkpoint cadence in `all` mode. |
| `chunk_size`   | `model_config.get_mamba_chunk_size()`     | Parallel-scan kernel tile. Architectural constant. |
| `d_conv`       | Model architecture (typically 4)          | Conv state lookback length. |

The parallel-scan kernel doesn't care about `block_size`. It takes
arbitrary chunk-aligned spans and produces output. `block_size` exists
only because vLLM's BlockPool needs a unit of allocation — Mamba inherits
the abstraction from attention even though the recurrence has no natural
"block."

This is the inverse of attention's situation. For attention,
`block_size` is constrained by the kernel
([KV_CACHE_ASSUMPTIONS.md §2](KV_CACHE_ASSUMPTIONS.md#2-why-block_size-cannot-be-chosen-freely)).
For Mamba, `block_size` is constrained by *no kernel* — it's a pure
allocation knob — but it must satisfy the uniform-page-size invariant
across groups when paired with attention. So the constraint comes from the
*allocator*, not the math.

## 5. Why `mamba_cache_mode="align"` exists (it's the conv state)

[LINEAR_ATTENTION.md §"Three cache modes"](LINEAR_ATTENTION.md) introduced
the three modes. The `align` mode is Mamba-specific, and the reason is the
conv state.

The depthwise convolution of width `d_conv` reads `d_conv − 1` past tokens'
projected activations. At a block boundary, the next block's first token
needs the previous block's last `d_conv − 1` tokens to seed the conv —
otherwise the convolution at position `block_size` of the new block
attends to zeros, producing wrong outputs.

In `align` mode, two blocks are kept live per request
([kv_cache_interface.py:555](vllm/v1/kv_cache_interface.py#L555)):

```python
return self.page_size_bytes * (2 + self.num_speculative_blocks)
```

The "previous block" holds the conv-state lookback for the start of the
"current block." When the current block fills, the *previous* block can be
freed; the current becomes the new previous. `MambaManager` tracks this
with `last_state_block_idx`
([single_type_kv_cache_manager.py:868-885](vllm/v1/core/single_type_kv_cache_manager.py#L868)):

```python
last_state_block_idx = self.last_state_block_idx.get(request_id)
if (
    last_state_block_idx is not None
    and last_state_block_idx < cdiv(num_computed_tokens, self.block_size) - 1
):
    blocks = self.req_to_blocks[request_id]
    if blocks[last_state_block_idx] != self._null_block:
        self.block_pool.free_blocks([blocks[last_state_block_idx]])
        blocks[last_state_block_idx] = self._null_block
```

This is what `align` "aligns": Mamba block boundaries to attention block
boundaries in a hybrid model, *while* keeping the conv lookback live across
each transition. Without `align`, a hybrid model's Mamba layers would
diverge by a few tokens at every attention block boundary because the conv
state at the boundary was zero-padded.

A pure-Mamba (non-hybrid) model could in principle run with `mode="none"`
because there are no attention boundaries to align to — but the conv-state
issue still exists if you ever evict and resume mid-sequence. `none` is
fine for monolithic in-place inference; `align` is required for chunked
prefill, hybrid models, or anything that crosses a block boundary.

## 6. `num_speculative_blocks`: rollback budget for spec decode

Speculative decoding generates `K` draft tokens, runs them through the
target model, and verifies how many are accepted. Rejected drafts must be
rolled back. For attention, rollback is trivial — don't advance the block
table cursor; the unused K/V at the tail is just unread.

For Mamba, rollback is hard. The state at draft step `K` is *already
overwritten* by the draft tokens (the recurrence advanced). To roll back to
"only the first 2 of 5 drafts accepted," you need the state as of step 2,
not the current state.

`num_speculative_blocks`
([kv_cache_interface.py:537](vllm/v1/kv_cache_interface.py#L537)) reserves
extra blocks per request so the pre-draft state is checkpointed and the
draft runs forward into a separate block. After verification, you discard
the unaccepted-draft block and the pre-draft block becomes the new running
state. The relevant guard in `MambaManager.remove_skipped_blocks`
([single_type_kv_cache_manager.py:865](vllm/v1/core/single_type_kv_cache_manager.py#L865)):

```python
# NOTE (tdoublep) with async scheduling, the num_computed_tokens can contain
# draft tokens from the previous step that may or may not be rejected later.
# This can make us think we are further ahead in the sequence than we actually
# are, so let's assume that all tokens are rejected so we don't free blocks
# that we might actually need.
num_computed_tokens = max(0, num_computed_tokens - self.num_speculative_blocks)
```

The conservative subtraction is mandatory: free the "draft" block
optimistically and you've lost the only copy of valid state. Attention can
afford to be optimistic because rejected K/V is harmless dead bytes; Mamba
cannot because state is overwritten in place.

## 7. DCP and PCP are not supported

`MambaManager.find_longest_cache_hit`
([single_type_kv_cache_manager.py:825-826](vllm/v1/core/single_type_kv_cache_manager.py#L825)):

```python
assert dcp_world_size == 1, "DCP not support mamba now."
assert pcp_world_size == 1, "PCP not support mamba now."
```

DCP (decode context parallelism) and PCP (prefill context parallelism)
shard a request's tokens across ranks for attention computation. For
softmax attention this works because each token's K/V is independent —
rank 0 computes K/V for tokens 0..N/2, rank 1 for N/2..N, then they
exchange.

For Mamba, the recurrence is sequential: `S_t` depends on `S_{t-1}`.
Sharding tokens across ranks breaks the chain. There are parallel-scan
algorithms that can reduce sequential dependency (the chunk-wise scan
already uses one within a chunk) but they don't trivially extend to
cross-rank parallelism without communication on every step.

This is a fundamental architectural mismatch, not an unimplemented feature.
Any cluster-scale plan for Mamba inference has to either replicate (TP for
parameters but not for sequence dimension) or pipeline-parallelize (PP
splits layers, sequence stays whole on each stage). Sequence-dimension
sharding is off the table.

## 8. Hybrid layouts (Jamba, Granite-Hybrid): Mamba is the cheap context, attention is the retrieval

Jamba, Granite-Hybrid, and similar architectures interleave Mamba layers
with attention layers in a fixed ratio. The design rationale matters for
the cache:

- **Mamba layers** carry long-context "feel" cheaply. State is O(1), so a
  256k-context Mamba layer holds 256k tokens of context for the same RAM
  cost as a 4k-context one. This is where the long-context budget goes.
- **Attention layers** carry exact retrieval. Some queries genuinely need
  to attend back to a specific token — a cited fact, a code symbol, a
  named entity. The attention layers are kept (at much lower count) to
  preserve this capability.

The cache implications:

- Memory profile is dominated by the attention layers (linear in N) even
  though there are fewer of them. Mamba layers contribute O(layer_count)
  state per request — meaningful but small.
- Prefix caching benefit is dominated by attention: a prefix hit on
  attention saves O(N · n_attn_layers) of compute; the same hit on Mamba
  saves O(N · n_mamba_layers) of compute, but Mamba's per-layer cost is
  often lower.
- The `align` cache mode is not optional for these models. Without it, the
  Mamba layers desynchronize from attention at block boundaries.

The grouping consequence (
[KV_CACHE_ALLOCATION.md §"Why Groups Exist"](KV_CACHE_ALLOCATION.md#why-groups-exist-at-all)
): Jamba-class models always have at least two groups (one Mamba, one
attention), each with its own block table, eviction rule, and
`SingleTypeKVCacheManager`. The hybrid coordinator fans `find_longest_cache_hit`
across them with the `alignment_tokens` discipline so neither group hits a
prefix the other doesn't.

## 9. What "block size" even means for Mamba

For attention, "block of size 16" means: K/V for 16 consecutive token
positions, addressable by `slot = block_id * 16 + offset`. The 16 is a
real, semantic unit — those 16 tokens' K/V is the contents of the block.

For Mamba, "block of size 16" means: a snapshot of state taken once per 16
input tokens. The block contains *one* `(SSM_state, conv_state)` pair, not
16 of them. The "16" is the **interval between snapshots**, not the
content multiplicity. The slot mapping math
(`slot = block_id * block_size + offset`) doesn't even apply to reading —
Mamba kernels read the state at the block's snapshot point, not at
specific offsets within the block.

This is a clean illustration of the BlockPool being agnostic to what a
block holds: same allocation primitive, very different semantics. It also
explains why Mamba's `block_size` is often much larger than attention's
(64, 128, 256, ...) — there's no kernel cost to a larger block, just a
checkpoint-frequency / memory tradeoff.

## 10. Summary table: where Mamba diverges from attention

| Property                          | Softmax attention            | Mamba                                                  |
|-----------------------------------|------------------------------|--------------------------------------------------------|
| State per layer per request       | O(N) K/V                     | Two fixed tensors: SSM state + conv state              |
| Replay cost from missed prefix    | O(N) per layer               | O(N) per layer (selectivity blocks fast-forward)       |
| Prefill kernel                    | FlashAttention varlen        | Chunk-wise parallel scan                               |
| Decode kernel                     | Same as prefill              | Different: step recurrence                             |
| `block_size` constrained by       | Kernel tile size             | Allocator only (no kernel constraint)                  |
| `chunk_size` constrained by       | (no separate chunk_size)     | Model architecture (`model_config.get_mamba_chunk_size()`) |
| Cross-block lookback              | None                         | `d_conv − 1` tokens of conv state                      |
| Spec-decode rollback              | Free, just don't advance     | Reserve `num_speculative_blocks` for state copy        |
| DCP/PCP support                   | Yes                          | No (sequential recurrence)                             |
| Slot-mapping semantics            | `block * size + offset`      | Block holds one state snapshot; offset is meaningless  |
| Allocator-imposed page padding    | None                         | `page_size_padded` to match attention's page size      |

Each row is a place where the cache infrastructure had to grow a Mamba-
specific affordance — a separate kernel, a separate cache mode, a separate
metadata path, an explicit assertion that a feature isn't supported.

## 11. The general lesson

The cache subsystem was originally designed for one shape of compute
(softmax attention). Adding Mamba required:

- A new `SingleTypeKVCacheManager` subclass with its own scan direction and
  hit semantics (LINEAR_ATTENTION.md covers this part).
- A new `KVCacheSpec` subclass (`MambaSpec`) with multiple state shapes and
  a padding field.
- New scheduler bookkeeping (`last_state_block_idx`,
  `num_speculative_blocks`).
- A new metadata path in attention backends (`Mamba2AttentionMetadata`).
- A new alignment discipline (`alignment_tokens` /
  `mamba_cache_mode="align"`).
- Explicit guards excluding it from features that don't compose
  (`assert dcp_world_size == 1`).

Each of these is small and well-localized. Together they make the point
that "the KV cache" is not a single abstraction with one extension point —
it's a stack of conventions about *what a block contains, when it gets
written, when it can be reused, and what the kernel expects to find
there*. Mamba reuses the bottom of the stack (BlockPool, hashing) and
diverges at the top (everything semantic). When future architectures land
— linear-attention variants, hybrid retrieval modules, learned-cache
schemes — the same pattern will repeat: keep the allocator, replace the
semantics, add a manager subclass, document the new constraints.
