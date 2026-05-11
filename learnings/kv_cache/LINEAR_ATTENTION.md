# Why Linear Attention Is a Different Cache Problem

The other notes in this folder describe vLLM's KV cache for *softmax*
attention: a paged store of per-position K/V tensors, indexed by block,
hashed by content, evicted when blocks fall out of relevance. Linear
attention — Mamba/Mamba-2, RWKV, RetNet, GLA — does not fit into that
machinery, even though vLLM uses the same `BlockPool` underneath. This note
explains where the analogy breaks and what `MambaManager` does instead.

For background on the abstractions referenced here, see
[KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md) (groups, block IDs) and
[KV_CACHE.md](KV_CACHE.md) (prefix caching, find_longest_cache_hit).

---

## What "linear attention" means here

Softmax attention computes, at each position `p`:

```text
out[p] = sum_{q <= p} softmax(Q[p] · K[q]) · V[q]
```

Every past token is addressable by index. The cache stores `K[q], V[q]` for
all `q`, and the next forward picks them up by `q`.

Linear attention rewrites the same shape into a *recurrence*:

```text
S[p]   = f(S[p-1], x[p])         # state update — fixed-size matrix/tensor
out[p] = g(S[p], x[p])
```

Mamba's `S` is `(SSM_state, conv_state)`. RWKV's `S` is `(num_state, den_state)`
plus a small history. RetNet's `S` is a low-rank decay matrix. The shapes
differ; the structural property is the same: **the entire history is
compressed into a fixed-size summary, and there is no way to retrieve token
`q` from `S` without replaying the recurrence**.

That single property is what breaks every assumption the softmax KV cache
is built on.

## Property 1: Memory is O(1) per request, not O(N)

Softmax attention's K/V grows with sequence length: a 32k-token request
holds 32k slots in every layer's K/V tensor. Mamba layers hold *one* state
tensor per request, regardless of length, because `S[p]` overwrites `S[p-1]`.

This shows up in `MambaSpec.max_memory_usage_bytes`
([kv_cache_interface.py:550](vllm/v1/kv_cache_interface.py#L550)):

```python
if mamba_cache_mode == "all":
    return cdiv(max_model_len, block_size) * page_size_bytes  # checkpoints
elif mamba_cache_mode == "align":
    return page_size_bytes * (2 + num_speculative_blocks)
else:
    return page_size_bytes * (1 + num_speculative_blocks)
```

The default ("none") is one block per request. Compare to
`FullAttentionSpec`, where the analogous bound is
`cdiv(max_model_len, block_size) * page_size_bytes` — *always* O(N).

The block-pool abstraction still allocates a logical block ID for each
"block" of state, and the column-packed tensor layout from
[KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md) still applies. But the
semantic content of a Mamba "block" is **a snapshot of the recurrent state
at a token boundary**, not K/V for the tokens in that block.

## Property 2: There is no positional retrieval

The softmax cache answers "give me K, V at position 47." The Mamba cache
cannot. `S[47]` is a function of all of `x[0..47]` mixed irreversibly. To
get K/V-equivalent data at position 47, you would have to replay the
recurrence from a checkpoint at or before 47.

This is why `MambaManager` operates on **state checkpoints**, not on
token-indexed K/V slots. A Mamba block at position `p` stores `S[p]` —
sufficient to continue the sequence from `p+1`, but useless if you want to
reconstruct hidden states for any earlier position.

The downstream effect: every operation that softmax KV cache supports per
position — partial-prefix sharing, mid-sequence eviction-and-reload,
position-specific attention masks — has no analog. The Mamba cache supports
exactly two operations: *resume from checkpoint p* and *advance to
checkpoint q > p*.

## Property 3: Prefix caching is all-or-nothing per checkpoint

Softmax attention's prefix cache trades on a strong property: if two
requests share the first `n` tokens, their K/V at every position in that
prefix is *bit-identical*. You can return cached blocks for the matched
prefix and only compute the divergent suffix.

For linear attention, the equivalent property holds for the *state*: if A
and B share the first `n` tokens, then `S_A[n] == S_B[n]`. So a checkpoint
at position `n` from request A's run can be reused by B.

But that's the only point of reuse. A Mamba checkpoint at position `n`
contains no information about positions `< n`. If B hits the cache at the
last shared position `n` and then diverges, B can resume from `S[n]` and
proceed forward — but if B somehow needed to attend back to position 50
(say a downstream reasoning hop), the Mamba layer simply cannot. The
information is gone, fused into `S`.

This is reflected in `MambaManager.find_longest_cache_hit`
([single_type_kv_cache_manager.py:810](vllm/v1/core/single_type_kv_cache_manager.py#L810)),
which is structured very differently from the full-attention version:

```python
# Search from right to left and early stop when a match is found.
for i in range(max_num_blocks - 1, -1, -1):
    if cached_block := block_pool.get_cached_block(...):
        # ... insert null blocks for positions before i
        computed.extend([block_pool.null_block] * i)
        computed.append(cached)
        break  # we just need the last match — early stopping
```

Three differences from `FullAttentionManager`:

- **Just one matching block returned, not a contiguous run.** The latest
  checkpoint subsumes all earlier ones.
- **R→L scan with early-stop on the first hit.** The latest checkpoint
  furthest into the prefix is the most valuable; nothing earlier needs
  checking.
- **Earlier slots filled with `null_block`.** Those positions exist in the
  request's logical block layout (so token-position arithmetic works), but
  there is no actual stored state for them — the kernel reads the
  checkpoint at position `i` and runs forward from there.

The R→L direction agrees with `SlidingWindowManager`, but for a different
reason. SWA scans R→L because only the tail window matters for the *next*
attention step. Mamba scans R→L because only the latest checkpoint matters
*ever* — earlier checkpoints are obsolete the moment a later one is taken.

## Property 4: Receptive field is the entire prefix, with no fall-off

[KV_CACHE.md](KV_CACHE.md) discusses the receptive-field wall for SWA: at
depth `L`, an SWA layer's K/V depends on tokens in `[p − (L−1)(W−1), p]`,
which is bounded but expanding. For linear attention, the receptive field
of `S[p]` is *always* `[0, p]`, full stop. Every prior token has non-zero
weight in the recurrence (decaying with distance for RetNet/RWKV, but never
zero in finite arithmetic).

So the suffix-digest hashing scheme that KV_CACHE.md refutes for SWA is
even more decisively wrong for linear attention. Two requests that share
only a tail and diverge in the prefix produce different `S` at every
checkpoint past the divergence point. There is no "windowed" approximation
to recover. The chained hash from block 0 forward is the only sound key.

What this leaves on the table: there is no purely cache-side trick to
extend prefix sharing for linear attention. The wins must come from
**state checkpointing across requests** — i.e., persisting `S[n]` for
common prefixes — which is already what Mamba's prefix cache does, indexed
by the same chained block hash. The cross-instance work belongs to
connectors (LMCache, Mooncake, NIXL), not to the local cache scheme.

## Property 5: Three cache modes trade memory for prefix-cache hit rate

`mamba_cache_mode`
([kv_cache_interface.py:536](vllm/v1/kv_cache_interface.py#L536)) selects
how many checkpoints to keep:

| Mode      | State blocks per request   | Prefix cache | Notes |
|-----------|----------------------------|--------------|-------|
| `none`    | 1 (just the running state) | No           | Cheapest. State is overwritten in place each step. |
| `align`   | 2 (current + last block)   | No           | Used to align Mamba block size with attention block size in hybrid models — see Property 6. |
| `all`     | `cdiv(max_model_len / block_size)` checkpoints | Yes | Pays full O(N/block_size) memory for the ability to resume any prefix at block boundaries. |

This is a knob softmax attention doesn't have. For softmax, *all* tokens'
K/V are kept by definition (otherwise the model can't attend back). For
Mamba, keeping any past state at all is a deliberate cost — the running
state is enough for forward inference. Prefix caching is opt-in.

The `align` mode is a curious middle: it doesn't enable prefix caching
(the previous-block state still gets overwritten), but it keeps two blocks
live so that the Mamba layer's logical block boundaries match the attention
layers' block boundaries in a hybrid model. See `MambaManager`'s
`last_state_block_idx` bookkeeping at
[single_type_kv_cache_manager.py:868](vllm/v1/core/single_type_kv_cache_manager.py#L868).

## Property 6: Hybrid models force block-size reconciliation

In a Jamba- or Granite-style hybrid (Mamba layers + attention layers in the
same model), the two attention types want different block sizes:

- Attention layers want `block_size=16` (or whatever the kernel prefers).
- Mamba layers want `block_size` aligned to the SSM checkpoint cadence,
  which is typically much larger.

Mamba prefix caching only makes sense at points where both groups can
produce a consistent prefix-hit length. If attention's block_size is 16 and
Mamba's is 1024, and a hit lands at the 17th attention block (272 tokens),
the Mamba group has no checkpoint there — it has checkpoints at 0, 1024,
2048, etc.

`find_longest_cache_hit` handles this with the `alignment_tokens` parameter
([single_type_kv_cache_manager.py:841](vllm/v1/core/single_type_kv_cache_manager.py#L841)):

```python
if (
    block_size != alignment_tokens
    and (i + 1) * block_size % alignment_tokens != 0
):
    continue
```

Skip any checkpoint whose end position isn't a multiple of
`alignment_tokens` — which is the LCM of attention and Mamba block sizes.
A Mamba prefix hit is only returned at boundaries where attention can also
hit, so the two groups stay synchronized.

This is the linear-attention analog of the hash_block_size = GCD discussion
in [KV_CACHE_ASSUMPTIONS.md §2](KV_CACHE_ASSUMPTIONS.md): different
attention types have different natural granularities, and prefix-cache
correctness in a hybrid model requires the lookup to operate at a
granularity that satisfies all of them.

## Property 7: Eviction is meaningless within a request

For softmax attention, eviction policy is delicate (LRU + reverse-free
tiebreak — see [KV_CACHE_ASSUMPTIONS.md §7](KV_CACHE_ASSUMPTIONS.md#7-why-the-free-queue-is-doubly-linked-not-a-deque)).
For Mamba in `none` or `align` mode, there's nothing to evict from a live
request — there is no per-position storage to choose from. The state simply
moves forward.

`remove_skipped_blocks` in `MambaManager`
([single_type_kv_cache_manager.py:857](vllm/v1/core/single_type_kv_cache_manager.py#L857))
is therefore not "evict positions outside the window" (that's SWA). It's
"free the previous-step block now that the current-step block holds the
state." The earlier block was a transient copy buffer, not historical data.

In `all` mode, all checkpoints are retained for the request's lifetime —
the cache is append-only, no in-request eviction. Cross-request eviction
follows the standard `BlockPool` LRU once the request finishes. So Mamba
inherits the eviction infrastructure but uses a tiny fraction of its
behavior.

## Property 8: Connectors transfer state, not blocks of K/V

Disaggregated prefill, P/D handoff, and offloading to LMCache/NIXL all ship
"the cache" between machines. For attention, this is a sequence of K/V
blocks — one block per `block_size` of prefix per layer per group. For
Mamba, it's *one state tensor* per request per layer per group.

The transfer interface is the same (the connector still binds to
`BlockPool`, still receives block IDs to populate). But the payload size
profile is dramatically different — fixed-cost per request rather than
proportional to prefix length. A 32k-token request that hands off
20MB of attention K/V might also be handing off 200KB of Mamba state.

The flip side: there is no incremental transfer for Mamba. With attention,
a producer can stream blocks layer-by-layer as they're computed
(`save_kv_layer` per layer). For Mamba, the state is only meaningful
*after* the recurrence has consumed the full prefix — partial state at an
intermediate position isn't a usable prefix-cache entry, it's just an
intermediate computation. So the producer's overlap pattern with the
forward pass is more constrained.

## What stays the same

Despite all the above, Mamba layers in vLLM use the same `BlockPool`,
`KVCacheBlock`, hashing scheme, ref-counting, free queue, and connector
interface as attention layers. The reason this works:

- A "block" is just a fixed-size physical slot in a column-packed tensor.
  The slot can hold K/V for `block_size` tokens of softmax attention or
  one snapshot of Mamba state — the BlockPool doesn't care.
- The chained hash works for Mamba checkpoints for the same reason it works
  for attention K/V: identical token prefix → identical state.
  `MambaManager` plugs into the same `BlockHashToBlockMap`.
- Group separation handles the type difference: each Mamba sub-system gets
  its own group with its own block table and its own
  `SingleTypeKVCacheManager` subclass. The coordinator fans `find_longest_cache_hit`
  across groups; whether each group returns a contiguous run or a single
  R→L hit is the manager's business (see
  [KV_CACHE_ALLOCATION.md §"Why Groups Exist At All"](KV_CACHE_ALLOCATION.md#why-groups-exist-at-all)).

The `BlockPool` is the right substrate; it just happens to be hosting a
fundamentally different cache *concept* on top of it.

## The fundamental ceiling

Linear attention's promise is O(1) inference state. The cache implications
follow:

| Property                     | Softmax attention      | Linear attention                       |
|------------------------------|------------------------|----------------------------------------|
| Memory per request           | O(N)                   | O(1) running, O(N/block_size) with prefix cache |
| Positional retrieval         | Yes                    | No                                     |
| Partial-prefix reuse         | Yes (per block)        | No (only full checkpoints)             |
| Cross-prefix-divergence reuse | No (correctness wall) | No (stronger version of same wall)     |
| Mid-sequence eviction        | Yes (SWA window)       | No (state is monolithic)               |
| Hit granularity              | `block_size` tokens    | One checkpoint per `block_size`        |

The right way to read this table: linear attention buys runtime memory and
loses retrieval flexibility. The cache subsystem can't recover the
retrieval flexibility no matter what hash scheme it picks — that
flexibility was traded away at the model architecture layer. The wins that
remain are cross-instance state sharing (connectors) and reducing peak
state-checkpoint memory in `all` mode.

If a future architecture wants both — fixed runtime memory *and*
positional retrieval — it would have to expose a separate retrieval mechanism
alongside the recurrence (some long-conv or sparse-attention sidecar). At
that point the cache problem splits in two: a Mamba-style state cache plus
an attention-style position cache. vLLM's group abstraction is already
shaped to support that — each becomes its own group with its own manager.
