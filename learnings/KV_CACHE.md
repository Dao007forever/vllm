# KV Cache Prefix Caching: Today's Design and How to Evolve It for SWA

## Part 1 — How prefix caching works today

There are two prefix caches in vLLM: the **local GPU prefix cache** (owned by `KVCacheManager`) and an optional **external prefix cache** (owned by a `KVConnector`). They're stacked, not separate paths — the connector adds tokens *beyond* what the local cache hit.

### Cast of classes

Local-cache side (in [vllm/v1/core/](vllm/v1/core/)):
- **`KVCacheManager`** ([kv_cache_manager.py](vllm/v1/core/kv_cache_manager.py)) — request-level API the scheduler calls. Wraps the coordinator + block pool.
- **`KVCacheCoordinator`** (Unitary/Hybrid) ([kv_cache_coordinator.py](vllm/v1/core/kv_cache_coordinator.py)) — fans the lookup across one or more KV-cache groups.
- **`SingleTypeKVCacheManager`** ([single_type_kv_cache_manager.py](vllm/v1/core/single_type_kv_cache_manager.py)) — per-attention-type `find_longest_cache_hit` (FullAttention scans L→R, SlidingWindow R→L, etc.).
- **`BlockPool`** + **`BlockHashToBlockMap`** + **`FreeKVCacheBlockQueue`** ([block_pool.py](vllm/v1/core/block_pool.py), [kv_cache_utils.py](vllm/v1/core/kv_cache_utils.py)) — the physical layer: a hash→block index over an LRU free list, ref-counted blocks.
- **`KVCacheBlock`** + `hash_block_tokens` — block hashing (`SHA256(parent_hash, token_ids, extra_keys)`, chained).

Driver:
- **`Scheduler.schedule()`** ([sched/scheduler.py](vllm/v1/core/sched/scheduler.py)) — orchestrates the whole flow per step.

External-cache side:
- **`KVConnectorBase_V1`** ([kv_connector/v1/base.py](vllm/distributed/kv_transfer/kv_connector/v1/base.py)) — two roles: scheduler-side (lookup, metadata) and worker-side (transfer).

### Per-request flow when prefix caching is involved

When a `WAITING` request is being promoted in `Scheduler.schedule()` (Phase 2), starting at [scheduler.py:610](vllm/v1/core/sched/scheduler.py#L610):

**1. Local cache lookup**
```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```
`KVCacheManager.get_computed_blocks` → `KVCacheCoordinator.find_longest_cache_hit` → per-group `SingleTypeKVCacheManager.find_longest_cache_hit` → walks the request's block hashes through `BlockHashToBlockMap`. Returns the longest matched prefix as a list of already-cached `KVCacheBlock`s and a token count.

**2. External cache lookup** (only if a connector is configured)
```python
if self.connector is not None:
    ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
```
The connector reports how many *additional* tokens beyond the local hit it can supply (from peer GPU, CPU offload, NIXL fabric, LMCache, Mooncake, etc.) and whether the load needs to be async.

**3. Async path** ([scheduler.py:656](vllm/v1/core/sched/scheduler.py#L656)) — if `load_kv_async`, this step schedules `num_new_tokens=0` and the request stalls in a `WAITING_FOR_REMOTE_KVS` state until the connector reports the transfer done; only then does it become eligible for real allocation.

**4. Allocation**
```python
self.kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    ...,
)
```
Inside `allocate_slots`:
- `BlockPool.touch()` reactivates the locally-hit blocks (removes them from the free queue, increments `ref_cnt`).
- `BlockPool.get_new_blocks()` pops new blocks for the *uncached remainder* — including the range the connector promised to fill. Evicts cached blocks via `_maybe_evict_cached_block` if needed.
- These new "external" blocks are allocated empty; the connector will fill them on the worker.

**5. Connector hook for staging the transfer**
```python
self.connector.update_state_after_alloc(request, blocks, num_external_computed_tokens)
```
Tells the connector exactly which physical blocks it must populate and what tokens they correspond to. Some connectors (OffloadingConnector, SimpleCPUOffloadConnector) also bind to the pool up front via `bind_gpu_block_pool` ([scheduler.py:239-242](vllm/v1/core/sched/scheduler.py#L239-L242)) so they can read/write blocks the manager owns.

**6. Per-step metadata** — at the end of `schedule()`:
```python
meta = self.connector.build_connector_meta(scheduler_output)
```
Serialized into `SchedulerOutput` and sent to workers.

### Worker-side: the transfer actually happens

On each forward pass:
1. `bind_connector_metadata(meta)` then `start_load_kv(forward_context)` — kicks off async receives for any pending external prefix.
2. Per attention layer: `wait_for_layer_load(layer_name)` blocks until that layer's KV has landed in the paged buffer ([model_executor/layers/attention/kv_transfer_utils.py:50](vllm/model_executor/layers/attention/kv_transfer_utils.py#L50)). Attention then runs against blocks the manager allocated in step 4.
3. On the producer side, `save_kv_layer` writes to the connector after each layer; `wait_for_save()` flushes at end-of-pass.

### Closing the loop: blocks become local cache entries

After the forward pass, the scheduler calls `KVCacheManager.cache_blocks()` for completed prefill chunks. `BlockPool.cache_full_blocks()` hashes each filled block (`hash_block_tokens` chained from the parent) and inserts it into `BlockHashToBlockMap` — so the *next* request gets a local hit on this prefix without needing the connector at all. **External cache becomes local cache as a side effect of normal forward execution.**

### Async cleanup

When a request finishes, `request_finished(request, block_ids)` returns `(delay_free_blocks, kv_transfer_params)`. If the connector is still pushing blocks somewhere (e.g., back to LMCache, or to a decode peer), the scheduler keeps `ref_cnt > 0` on those blocks until `get_finished()` reports the transfer done — preventing use-after-free.

### Layering summary

```
Scheduler.schedule()
  ├── KVCacheManager.get_computed_blocks()       # local hit (N tokens)
  │     └── Coordinator → SingleTypeManager → BlockHashToBlockMap
  ├── connector.get_num_new_matched_tokens()      # external hit (+M tokens)
  ├── KVCacheManager.allocate_slots()             # touch local + new blocks for ext range
  │     └── BlockPool.touch / get_new_blocks
  ├── connector.update_state_after_alloc()        # tell connector which blocks to fill
  └── connector.build_connector_meta()            # ship plan to workers

Worker forward
  ├── start_load_kv → wait_for_layer_load(L)      # attention waits per layer
  └── save_kv_layer(L) → wait_for_save()          # producer side

Post-forward
  ├── KVCacheManager.cache_blocks()               # external→local promotion
  └── (on finish) connector.request_finished()    # delay_free_blocks if async send
```

The `KVCacheManager` always owns the GPU block pool and the local hash table; the connector is purely additive — it lets the scheduler skip computing tokens because someone else already has the KV, and supplies that KV directly into manager-allocated blocks.

---

## Part 2 — What SWA prefix caching can and cannot do

An earlier draft of this section proposed a **suffix-digest hash** for SWA blocks: hash the window-bounded tail and the absolute block position, and let two requests with the same suffix at the same positions share K/V. **That is unsound for any transformer of depth ≥ 2.** This section explains why, what today's design already extracts correctly, and where the legitimate engineering wins are.

### The correctness wall: K/V receptive field

At every layer L, `K_L[p] = W_K · h_{L-1}[p]` and `V_L[p] = W_V · h_{L-1}[p]`. K/V values are a function of the *layer's input hidden state*, not of the input tokens directly. The hidden state's dependency on tokens grows with depth:

| Stack below layer L | `h_{L-1}[p]` depends on tokens at |
|---|---|
| Embedding only (L = 1) | `{p}` |
| 1 SWA layer (window W) | `[p − (W−1), p]` |
| L−1 stacked SWA layers | `[p − (L−1)(W−1), p]` |
| Any full-attention layer at any depth < L | `[0, p]` — the entire prefix |

The attention *mask* at layer L bounds which K/V entries `Q_L[p]` mixes with at this layer. The K/V *values themselves* are computed from `h_{L-1}`, whose receptive field reaches further back than the mask. The earlier draft conflated these two distinct things.

**Consequence**: two requests with identical window-bounded suffix tokens at identical positions but divergent earlier prefixes do **not** produce identical K/V at any SWA layer beyond the first of a pure-SWA stack. In Gemma-3's hybrid stack (5 SWA : 1 full), the moment any SWA layer sits above any full-attention layer, its K/V depends on the entire prefix — exactly the dependency a full-attention K/V has.

#### Worked example: why "same suffix at same positions" is not enough

`block_size=4`, `sliding_window=8`, two requests:

```
A: tokens = [1, 2, 3, 4,  5, 6, 7, 8,  9,10,11,12]
B: tokens = [99,99,99,99, 5, 6, 7, 8,  9,10,11,12]
```

Block 2 spans positions 8..11. The earlier draft claimed A's and B's block 2 K/V would be bit-identical because tokens at positions 4..11 match. They are not:

- At SWA layer 1 in a pure-SWA stack: `h_1[8]` for A depends on tokens at positions [1..8] = `(2,3,4,5,6,7,8,9)`; for B it depends on `(99,99,99,5,6,7,8,9)`. Different.
- Therefore `K_2[8] = W_K · h_1[8]` differs between A and B. Likewise for positions 9, 10, 11.
- In Gemma-3, divergence happens even faster: one full-attention layer below pulls the entire prefix into `h_{L-1}[8..11]`, so every SWA layer above that point has K/V depending on tokens 0..3.

Returning A's cached block 2 K/V on a B request inserts numerically wrong values into the paged buffer. Subsequent attention reads them. There is no kernel-level signal — the model just gets quietly worse on long conversations with shared tails. This is exactly the failure mode prefix caching is supposed to never have.

### Why the symmetric refutation applies

The earlier draft correctly refuted **bounded-chain hashing** by graph reachability: block i's hash transitively reaches block 0 via the W-stride chain, regardless of how many hops away that is.

The same argument refutes **suffix-digest hashing**, applied to the model's residual stream rather than to the hash chain. Suffix-digest's input edges don't reach token 0, but the model's *computation* edges do. A correct fingerprint must cover a function's full dependency set; otherwise hits between distinct dependency-set values are aliasing collisions — exactly what bounded-chain was rejected for.

### What today's design already extracts correctly

The current chained-hash + R→L scan in `SlidingWindowManager.find_longest_cache_hit` ([single_type_kv_cache_manager.py:486](vllm/v1/core/single_type_kv_cache_manager.py#L486)) returns the legitimate hits:

- **Same-prefix continuation** (multi-turn chat, branching from a shared system prompt) — the prefix matches up to divergence, K/V is provably identical, the chained hash hits. The R→L scan terminates as soon as it has `ceil((W−1)/B)` contiguous matched blocks; same-prefix tails always satisfy this.
- **Window eviction** via `remove_skipped_blocks()` — blocks that fall out of the SWA window are freed. This is the SWA memory win and is independent of the hashing scheme.
- **Cross-instance prefix sharing** through connectors — identical prefixes hit across machines because connectors use the same chained hash.

Today's design is not "leaving hits on the table" for SWA. The hits it omits are hits that would be unsound.

### Aside: L→R vs R→L scanning in `find_longest_cache_hit`

Each `SingleTypeKVCacheManager` walks the request's block-hash list in the direction that matches its attention pattern.

**FullAttentionManager → L→R.** A full-attention layer at position `p` attends to all positions `[0, p]`. Once any block diverges from cache, every subsequent block must also be wrong (its parent hash differs). So scan from block 0, stop at first miss.

```
block_size=4, blocks B0..B2 over tokens [A B C D | E F G H | I J K L]
cache: B0=HIT, B1=HIT, B2=MISS
L→R: B0 hit → B1 hit → B2 miss → stop. Matched = 8 tokens.
```

**SlidingWindowManager → R→L.** SWA at position `p` only attends to `[p − W + 1, p]`. The block at the end of the request only needs `ceil((W−1)/B)` contiguous matched blocks ending at *its* position; earlier matches are irrelevant. So scan from the last block, accumulate contiguous hits, stop when you have W of them.

```
block_size=4, sliding_window=8 → W_blocks=2. Blocks B0..B4.
cache: B0=HIT, B1=MISS, B2=HIT, B3=HIT, B4=HIT.
L→R would give: B0 hit → B1 miss → stop at 4 tokens. Useless — we lose B3+B4 which have valid KV.
R→L: i=4 hit (contig=1) → i=3 hit (contig=2 → enough!) stop.
Result blocks = [NULL, NULL, NULL, B3, B4]. Matched = 20 tokens.
```
Earlier slots get nulled out because the SWA layer literally won't read them — and those nulled-out positions' K/V *would* have been wrong anyway under suffix-digest, since they depend on the diverged earlier prefix through the residual stream.

**MambaManager → R→L.** Mamba caches state at block boundaries; you want the latest checkpoint matching the prefix, so scan from the end.

### Position encoding: a smaller wall behind the receptive-field wall

RoPE applied to K bakes absolute position into the cached tensor for almost every modern SWA model (Llama, Mistral, Gemma-3, Qwen). The earlier draft framed this as the headline correctness wall for divergent-prefix sharing. With the receptive-field wall correctly stated, RoPE is just a tightening — even on a pure-ALiBi or learned-relative model where RoPE wouldn't exclude same-suffix-different-position cases, suffix-digest still aliases across divergent prefixes because the K/V values themselves differ. RoPE excludes a strictly smaller set of would-be hits than receptive field already excludes.

### Where the legitimate engineering wins are

#### 1. Sparser block allocation for SWA layers (the largest practical win)

`HybridKVCacheCoordinator` allocates blocks in lock-step across groups. An SWA layer fundamentally only needs `ceil(W/B)` blocks live at any moment per request, not one block per `block_size` of prefix. Today the allocator over-allocates and `remove_skipped_blocks` evicts after the fact — peak allocation is still proportional to context length per SWA layer. A streaming allocator that *never* allocates blocks for out-of-window positions in SWA groups would shrink peak memory without changing correctness or hashing.

For Gemma-3 with 5:1 SWA:full ratio, peak SWA-group memory drops from O(N) per layer to O(W) per layer. On long contexts this is a substantial win — and it's a pure allocation-layer change, no impact on `find_longest_cache_hit`, no risk to correctness.

#### 2. R→L scan micro-optimization

The TODO at [single_type_kv_cache_manager.py:516-520](vllm/v1/core/single_type_kv_cache_manager.py#L516-L520) — skip-ahead by `ceil(W/B)` blocks rather than one block at a time. Orthogonal to hashing.

#### 3. Connector SWA coverage

Cross-instance K/V reuse for SWA group blocks is just as valid as for full-attention blocks (both rely on identical prefixes), but Mooncake / LMCache / NIXL adoption focuses on the larger full-attention payloads. For long-context Gemma-3-style traffic with shared system prompts, ensuring connectors index SWA group blocks symmetrically (using the existing chained hash, with `group_id` plumbed through) recovers compute. No hash change needed — confirm the worker-side transfer path handles per-group keys correctly.

### What this doesn't help: Mamba / RWKV / pure linear attention

Mamba's `(h, conv_state)` at position `p` is a function of *all* prior tokens with exponentially-decaying weights, not a literal window. There's no token-level "fall-off" point. `MambaManager.find_longest_cache_hit` already does what's possible: longest-prefix match with checkpoint replay, R→L scan. Windowed hashing doesn't apply.

The only way to extend prefix sharing for true linear attention is **state checkpointing across requests** — and that's already chained-hash-equivalent (you need the exact prefix to trust the state). The wins there are connector-side: offload state checkpoints to LMCache/Mooncake so different machines can share.

### Test ideas before any future windowed-hash proposal

Receptive-field aliasing has no kernel-level trip-wire — failures are silent. Any new SWA hash scheme must be guarded by:

- **Two-request K/V comparison at every SWA layer**: `A = [X*N, Y*M]`, `B = [X'*N, Y*M]` with `X ≠ X'`. Compute K/V at every SWA layer for the Y range. They must differ; any cache scheme that returns a hit for B after A primed the cache is unsound.
- **Numerical equivalence test**: run B's prefill cold, then again with the cache primed by A's run. Outputs must be bit-identical to cold. If a windowed hash is wired in incorrectly, they won't be — and the difference may be small enough that eval scores barely move.

### Build order

1. **Sparser SWA block allocation** — measure peak memory delta on long-context Gemma-3 / Mistral workloads. No correctness risk, no hashing change. This is the win the earlier draft was reaching for, achieved via a different mechanism.
2. **R→L scan micro-opt** — close the TODO at [single_type_kv_cache_manager.py:516-520](vllm/v1/core/single_type_kv_cache_manager.py#L516-L520).
3. **Connector SWA audit** — confirm Mooncake / LMCache / NIXL handle SWA group blocks correctly for identical-prefix sharing.

There is no step 4 that recovers a suffix-digest-style "win." It does not exist for transformers of depth ≥ 2.

### Risks if anyone proposes a windowed hash again

- Reviewers must demand the K/V-comparison + numerical-equivalence tests above before any new SWA hash scheme lands.
- Anyone proposing per-layer or per-group hash variants must enumerate the dependency set of each cache entry's *contents* and prove the hash covers it. "The mask only reads W tokens" is not the dependency set; the dependency set is the receptive field of the residual stream feeding K/V.
- `extra_keys` (LoRA, MM hash, cache salt) must remain part of every group's key regardless of scheme.
- Hybrid models (Gemma-3) make the constraint stricter, not looser: even *if* a windowed hash were sound for pure-SWA layers, the moment a full-attention layer appears below an SWA layer, that SWA layer's K/V depends on the full prefix and must use the chained hash.
