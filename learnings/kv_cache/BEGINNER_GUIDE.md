# A Beginner's Guide to vLLM's KV Cache

The other notes in this folder are dense — they assume you already know what
problem the KV cache solves and skip straight to the design questions
("why blocks?", "why this hash scheme?", "why these groups?"). This guide is
the on-ramp. It builds the intuitions from scratch, using small examples,
so the other docs read as confirmations rather than revelations.

There are no novel claims here. Everything is a re-explanation of what
`KV_CACHE.md`, `KV_CACHE_ALLOCATION.md`, `KV_CACHE_ASSUMPTIONS.md`,
`LINEAR_ATTENTION.md`, and `MAMBA.md` already cover. Pointers into those
docs are sprinkled throughout for when you want the engineering depth.

---

## 1. The problem: attention is repetitive, on purpose

When a transformer generates text token by token, each new token's
attention layer needs to look back at *every* previous token. At step 100
the model attends to tokens 0..99; at step 101 it attends to 0..100; at
step 102 it attends to 0..101; and so on.

If you computed everything from scratch each step, you'd redo the K and V
projections for tokens 0..99 a hundred times. Those projections don't
change — token 5's K vector at layer 7 is the same on step 100 as it was on
step 6. So we cache them.

The "KV cache" is exactly this: at each layer, for every token we've seen,
we save the K and V tensors that attention computed. Next step, instead of
recomputing them, we read them out of memory.

That's the entire reason the cache exists. Every design decision below is
in service of "store K and V efficiently and find them again quickly."

### A concrete picture

The cache size depends a lot on *how* the model arranges its attention
heads. Three common designs, all sized to a 32-layer model with hidden
size 4096 and head dim 128, fp16:

**Full Multi-Head Attention (MHA)** — every Q head has its own K and V
head. With 32 KV heads:

- One token's K *or* V at one layer = `32 heads × 128 floats × 2 bytes` = 8 KiB.
- One token, both K and V, all 32 layers = `32 × 2 × 8 KiB` = **512 KiB / token**.
- A 4k-context request ≈ `4096 × 512 KiB` ≈ **2 GiB** just for cache.

**Grouped-Query Attention (GQA)** — multiple Q heads share one K/V head.
Llama-3, Mistral, Qwen-2 use this: 32 Q heads but only 8 KV heads (4 Q
heads per KV head). The K/V projections are smaller, so the cache is
smaller:

- One token's K *or* V at one layer = `8 heads × 128 × 2 bytes` = 2 KiB.
- One token, both K and V, all 32 layers = `32 × 2 × 2 KiB` = **128 KiB / token**.
- A 4k-context request ≈ **512 MiB** — 4× smaller than MHA.

GQA is the dominant design today. The intuition: model quality cares
about Q-head diversity (each head asks a different question) but tolerates
shared K/V (the *answers* don't need as much head diversity). You buy 4×
cache savings for a small quality cost.

**Multi-head Latent Attention (MLA)** — DeepSeek's design. Instead of
caching per-head K and V, cache *one compressed latent vector* per token
plus a tiny shared RoPE component. Each layer reconstructs per-head K/V
from the latent at attention time. With latent dim 512 and RoPE dim 64
(typical for DeepSeek-V2/V3):

- One token, all cached state for one layer = `(512 + 64) × 2 bytes` = ~1.1 KiB.
- One token, all 32 layers = `32 × 1.1 KiB` ≈ **36 KiB / token**.
- A 4k-context request ≈ **144 MiB** — ~14× smaller than MHA, ~3.5× smaller than GQA.

MLA trades extra compute (the per-layer reconstruction matmul) for
dramatically smaller cache. The trade is worth it because cache memory
caps the *batch size* you can fit on a GPU, and a bigger batch amortizes
the extra compute.

**Why this matters for the design that follows.** The numbers shrink, but
the *shape* of the problem doesn't: every design decision in this guide
applies to all three schemes. MHA, GQA, and MLA all need block-based
allocation, prefix caching, group separation in hybrid models, and the
same `BlockPool` machinery. The only thing that changes is `page_size` —
the bytes-per-block — and that ripples cleanly through the code because
the uniform-page-size invariant (§7) is computed per-model from whatever
the layer specs say. Even MLA, with its single latent vector replacing
separate K and V, looks the same to the BlockPool: a fixed-size payload
per block, same as everything else.

A practical consequence: on a single 80 GiB GPU, a 70B-class GQA model
might fit a few hundred concurrent 4k-context requests in cache, while
the same hardware running an MHA model of similar size might fit only a
few dozen. The cache layout decisions below are why those numbers are as
high as they are — and why MLA models can push the batch-size envelope
even further.

## 2. Why we allocate in *blocks*, not per token

Naively you'd think: "one slot per token, looked up by token index." The
real design hands out fixed-size **blocks** — typically 16 tokens at a
time. A 33-token request gets 3 blocks, not 33 slots, and the third block
has 15 empty positions just sitting there waiting to be filled by future
decode steps.

This looks wasteful. Why?

**Three things want a fixed-size unit at once**:

1. **The scheduler bookkeeping.** Allocating and freeing blocks needs to
   be O(1). With per-token slots you'd need a bitmap or a per-request
   arena and either fragmentation or scan cost. With fixed blocks you have
   a single free list of identical-shaped slots — pop one off when you
   need it, push one back when you're done.

2. **The attention kernel.** GPU kernels read K and V in tiles, not
   one-token-at-a-time. FlashAttention issues *one* memory load per
   16-token chunk. Per-token allocation would scatter those 16 tokens
   across the GPU, forcing the kernel to do 16 indirect lookups instead of
   one contiguous load — much slower.

3. **The prefix-cache hash.** vLLM caches prefixes by hashing block
   contents (more on this in §3). Hashing a fixed-size unit is cheap and
   chainable. A token-level hash would multiply hash-table entries by 16×
   without any benefit.

Compare this to a hotel: you book a room, not individual square feet of
floor space. The room has 4 beds and you might only fill 1, but the
operational overhead of treating every square foot as bookable would
dwarf the wasted space.

The **internal fragmentation** in the last block — those 15 empty slots in
the third block of a 33-token request — is real but small in practice.
Continuing decode fills them anyway.

→ For the assertion-level details, see
[KV_CACHE_ASSUMPTIONS.md §1](KV_CACHE_ASSUMPTIONS.md#1-why-allocate-in-fixed-size-blocks-at-all-not-per-token).

## 3. The prefix cache: same prefix, same K/V

Here's the second insight that drives everything: **K and V at position
*p* are deterministic functions of the effective inputs at positions
0..p**. Two requests with the same first 100 tokens compute
*bit-identical* K/V for those 100 tokens at every layer, as long as the
effective inputs are the same too: same model weights, same LoRA/adapters,
same multimodal features or prompt embeddings, same cache namespace, and
same KV-cache group. Same effective input in, same result out.

So if request A has already been processed and we still have its blocks
sitting in GPU memory, request B with the same prefix can just **point at
A's blocks** and skip computing them. This is "prefix caching."

### Real-world payoff

The classic case is a chat application with a long system prompt:

```
System: You are a helpful assistant. [...3000 more tokens of instructions...]
User: Hi
```

```
System: You are a helpful assistant. [...same 3000 tokens...]
User: What's the weather?
```

These two requests share 3000+ tokens of identical prefix. Without prefix
caching, every new conversation pays the prefill cost for those 3000
tokens — a meaningful chunk of latency. With prefix caching, the second
request only computes K/V for "What's the weather?".

### How do we know two prefixes are identical?

We can't compare token sequences directly — that would be O(N) per lookup.
Instead vLLM hashes each block: hash of block 0 covers tokens 0..15,
hash of block 1 covers `(hash_of_block_0, tokens 16..31)`, hash of block
2 covers `(hash_of_block_1, tokens 32..47)`, and so on.

This **chained hash** has a beautiful property: if your block-N hash
matches mine, all of blocks 0..N must also match, because the chain
encodes the whole history. One hash comparison vouches for the entire
prefix.

```
block 0 hash = H(NONE_HASH, "The cat sat on the")
block 1 hash = H(block_0_hash, " mat. The mat was")
block 2 hash = H(block_1_hash, " green. It was a")
```

If two requests both produce `block_2_hash = X`, they share at least 48
tokens of identical prefix, assuming the hash function has not collided.

The real hash key also includes **extra keys** when tokens alone are not
enough to identify the K/V contents: LoRA/adapters, multimodal feature
identifiers, prompt embeddings, and request `cache_salt`. Those keys are
folded into the chain, so "same prefix" really means "same prefix under
the same effective cache identity."

`NONE_HASH` is a per-process random seed (§9 in `KV_CACHE_ASSUMPTIONS.md`)
so different vLLM processes get different hash spaces — preventing one
tenant from poisoning another's cache by guessing hashes.

→ For the chained-hash mechanics see
[KV_CACHE.md "Per-request flow"](KV_CACHE.md#per-request-flow-when-prefix-caching-is-involved)
and the receptive-field discussion in
[KV_CACHE.md "What SWA prefix caching can and cannot do"](KV_CACHE.md#part-2--what-swa-prefix-caching-can-and-cannot-do).

## 4. Why we hash *after* the forward, not before

You might think: "If I know the tokens, I can hash them up front and look
them up in the cache before running the model." vLLM does compute the hash
up front for *lookup* — but it only inserts a `(hash → block_id)` entry
into the cache table **after** the worker has actually written the K/V
contents into the block.

The reason is subtle but critical: the hash is a key into K/V *content*,
not into block IDs. If you inserted the hash before computation, two
requests racing through prefill at the same time could both think they
have valid K/V at a given block when really the block hasn't been written
yet. You'd hand out garbage to whoever looked up second.

So vLLM splits it in two: cheap CPU-side hash computation up front
(for lookup), deferred insertion after the GPU finishes writing. Every
piece of K/V you read from the cache is K/V that was actually computed.

→ See [KV_CACHE_ASSUMPTIONS.md §5](KV_CACHE_ASSUMPTIONS.md#5-why-blocks-are-hashed-after-the-forward-pass-not-before).

## 5. What sharing buys you, and where it stops

Prefix caching only works for **identical** prefixes. If A's tokens are
`[1, 2, 3, 4]` and B's tokens are `[1, 2, 3, 5]`, they share blocks 0..2
(if `block_size=4` then they actually share through token 3, but as soon
as one block diverges, every subsequent block diverges too — its hash
includes the parent's hash).

A natural question: "What if I have similar but not identical prefixes,
or just a similar *suffix*? Surely we can share some K/V?"

For sliding-window attention (SWA), where each token only attends to the
last W tokens, the temptation is strong: *"if both requests have the same
last W tokens, why can't they share?"*

The answer is **no, never**, for any transformer with depth ≥ 2. Even
though the attention *mask* at layer L only reads the last W positions,
the K/V values *at* those positions depend on the residual stream from
earlier layers, which depends on tokens further back. A 1-layer SWA
network could share suffixes; a 2-layer SWA network cannot, because layer
2's K/V at position p was computed from layer 1's hidden state at p,
which mixed tokens p−W+1 through p, and so on. The receptive field of
the K/V *contents* expands with depth even though the per-layer mask
doesn't.

This is the **receptive-field wall**. It rules out a whole class of
"clever" cache-sharing schemes that look like they should work and
silently corrupt outputs if implemented. The current `chained hash + R→L
scan` returns exactly the hits that are *provably* identical — no more.

→ The full argument with worked counterexample is in
[KV_CACHE.md "The correctness wall"](KV_CACHE.md#the-correctness-wall-kv-receptive-field).

## 6. Why different attention types live in different "groups"

Modern models often mix attention types in one stack. Gemma-3 alternates
sliding-window layers with full-attention layers. Jamba mixes Mamba layers
with attention layers. These types **disagree about how their cache
behaves**:

| Property             | Full attention | Sliding window | Mamba |
|----------------------|----------------|----------------|-------|
| When can K/V be freed? | When request ends | When position falls out of window | After each step (in default mode) |
| How to find cache hits? | Scan blocks left-to-right | Scan right-to-left, take last W blocks | Scan right-to-left, take latest only |
| What's a "block"?    | 16 tokens of K/V | 16 tokens of K/V | One snapshot of recurrent state |

You can't put these into a single shared block table. Each cell of that
table would mean different things in different rows. So vLLM partitions
layers into **groups** where every layer in a group has the same answers
to those three questions:

- All full-attention layers go in one group (or a few — see below).
- All SWA layers go in another.
- All Mamba layers go in another.

Each group gets its own block table, its own `find_longest_cache_hit`
implementation, its own eviction logic.

### Why not one block table per layer?

That's the safe extreme. It also wastes huge amounts of GPU memory: an
80-layer model at 128k context would burn ~10 GiB on block-table metadata
alone (the block table is on GPU because the kernel reads it directly).
Layers within the same attention type have **identical lifecycles** —
their block tables would be byte-identical — so making them share is free.

For example, suppose request A has 12 tokens and `block_size=4`, so it owns
three logical blocks:

```
request A block table: [47, 89, 153]
```

In a full-attention stack, layer 0, layer 7, and layer 31 all need the same
logical mapping: positions 0..3 read block 47, positions 4..7 read block 89,
and positions 8..11 read block 153. The K/V **contents** differ per layer
because each layer has different weights and activations, but the answer to
"which physical block stores logical block 2 for request A?" is identical for
every full-attention layer. One table can answer it for all of them.

SWA layers have a different lifecycle, but the sharing logic is the same
within the SWA group. If the active window only needs the last two logical
blocks, every SWA layer agrees that the old block is no longer addressable:

```
full-attention group table: [47, 89, 153]
SWA group table:            [NULL, 89, 153]
```

That SWA table cannot be shared with full attention, because the old block
means different things to the two attention types. But it **can** be shared
by all SWA layers, because they all drop the same old positions and keep the
same live window.

The efficiency win is broader than the table itself. Group-level metadata means
vLLM stores and computes one **block table**, one **slot mapping**, one
`find_longest_cache_hit` result, and one eviction decision per group instead of
per layer. The K/V bytes are still per layer, because every layer computes
different K/V, but the bookkeeping that maps token positions to block IDs is
identical across the group. For 10 layers in one group, that turns 10 copies of
the same metadata work into 1.

### Why split SWA layers across multiple groups in Gemma-3?

If Gemma-3 has 5 SWA layers per 1 full-attention layer, and you put all 50
SWA layers in one group with all 10 full-attention layers in another, then
"a block" in the SWA group would represent 5× more memory than a block in
the full-attention group. The scheduler can no longer ask "do I have N
free blocks?" — it would have to track per-group block sizes,
fragmentation, etc.

vLLM's solution: split the 50 SWA layers into 5 groups of 10. Now every
group has 10 layers, every block represents the same amount of physical
memory, and one global free list works for everyone.

That is the reasoning hierarchy:

1. Layers with the same cache lifecycle can share a **group block table**.
   That means one block-table entry for token positions `[p, p+B)` is reused
   by every layer in the group.
2. Because those layers compute different K/V, that one group block must still
   reserve one K/V payload per layer in the group.
3. Groups are split/padded to the same effective layer count so one group
   block always means the same amount of K/V memory.
4. Equal-sized group blocks make a single global `BlockPool` possible.
5. That global pool is what lets column packing share physical block slots
   across groups instead of preallocating one full block-ID range per group.

→ The full reasoning chain is in
[KV_CACHE_ALLOCATION.md "Why Groups Exist At All"](KV_CACHE_ALLOCATION.md#why-groups-exist-at-all)
through "Why Groups Need Uniform Physical Block Size".

## 7. The column-packing trick

If you have 6 groups (1 full + 5 SWA) of 10 layers each, you have 60
layers total and 6 separate block tables. How do you allocate physical
memory for the K/V tensors?

The clever answer: **each "column" of the layer grid shares a single
physical tensor**, where columns are indexed by *layer-position-within-its-group*.
The previous section's equal group sizes are what make this clean: every group
has a slot 0 layer, a slot 1 layer, ..., up to the same `group_size`, so each
column can hold exactly one layer from every group.

Picture it as a grid:

```
            slot 0   slot 1  ...  slot 9
Group 0:    L5       L11     ...  L59      <- FULL
Group 1:    L0       L6      ...  L54      <- SWA
Group 2:    L1       L7      ...  L55      <- SWA
Group 3:    L2       L8      ...  L56      <- SWA
Group 4:    L3       L9      ...  L57      <- SWA
Group 5:    L4       L10     ...  L58      <- SWA
            ====     ====         ====
          Tensor 0 Tensor 1 ... Tensor 9
```

There are **10 physical tensors**, not 60. Tensor 0 holds K/V for layers
L5, L0, L1, L2, L3, L4 — one layer from each group. Tensor 1 holds
L11, L6, L7, L8, L9, L10. And so on.

### A worked example: one request through the layout

To make the block-ID-to-bytes flow concrete, collapse everything to a
tractable size:

```
              slot 0   slot 1
Group 0:      L0       L2        <- FULL
Group 1:      L1       L3        <- SWA
              ====     ====
            Tensor 0 Tensor 1
```

4 layers, 2 attention types, `group_size = 2` → 2 column tensors.
`block_size = 4`. Global pool has 100 free blocks, IDs 0..99.

**Step 1: a request arrives, scheduler allocates block IDs.**

Request A has 8 tokens — it needs 2 blocks per group. The scheduler
pops 4 IDs from the single global free queue and writes them into
per-group block tables:

```
Group 0's table for A: [10, 11]
Group 1's table for A: [20, 21]
```

All 4 IDs are different — they all came from one global pool, which
hands out unique IDs.

**Step 2: compute slot mappings per group.**

The block table answers "which physical blocks does this request own?",
indexed by *block position* in the request. But the attention kernel
processes individual *tokens*, not blocks — it needs to know, for each
token in the batch, the exact physical slot to write that token's K/V
to. Translating "token at position p" to "slot S" through the block
table on every kernel invocation would cost a memory lookup per token
inside the hot path.

So vLLM **precomputes that translation once**, before the forward pass,
into a tensor called `slot_mapping`. It's a flat tensor with one entry
per token in the current batch: `slot_mapping[i]` is the physical slot
ID for token `i`. The kernel never touches the block table — it just
reads `slot_mapping[i]` and writes K/V there directly.

The translation formula:

```
block_index = token_position // block_size
block_id    = block_table[request][block_index]
offset      = token_position %  block_size
slot_id     = block_id × block_size + offset
```

For request A's 8 tokens, this gives:

```
Group 0 slot mapping (uses block table [10, 11]):
  pos 0 → block_index 0 → block_id 10 → slot 10*4 + 0 = 40
  pos 1 → block_index 0 → block_id 10 → slot 10*4 + 1 = 41
  pos 2 → block_index 0 → block_id 10 → slot 10*4 + 2 = 42
  pos 3 → block_index 0 → block_id 10 → slot 10*4 + 3 = 43
  pos 4 → block_index 1 → block_id 11 → slot 11*4 + 0 = 44
  pos 5 → block_index 1 → block_id 11 → slot 11*4 + 1 = 45
  pos 6 → block_index 1 → block_id 11 → slot 11*4 + 2 = 46
  pos 7 → block_index 1 → block_id 11 → slot 11*4 + 3 = 47

Group 1 slot mapping (uses block table [20, 21]):
  pos 0..3 → block 20 → slots 80, 81, 82, 83
  pos 4..7 → block 21 → slots 84, 85, 86, 87
```

So for this one request, `slot_mapping_group_0 = [40,41,42,43,44,45,46,47]`
and `slot_mapping_group_1 = [80,81,82,83,84,85,86,87]` — both are flat
tensors of length 8 (one entry per token), shipped to the GPU once per
scheduling step.

**Why one slot mapping per group, not per layer.** Every layer in
group 0 (i.e., L0 and L2) needs the *same* slot mapping — they all read
the same block table, all process the same tokens, and the slot
arithmetic doesn't depend on the layer. So vLLM computes the slot
mapping once per group and reuses it across all the group's layers in
the forward pass. This is one of the metadata-sharing wins from §6.

Group 0 and group 1 produce *different* slot mappings for the same
token positions, because they hold different block IDs ([10, 11] vs
[20, 21]).

**Step 3: forward pass — each layer writes K/V.**

Now the heavy lifting. For each layer, the attention kernel receives
two key inputs:

- `kv_cache` — the layer's own tensor view (per-layer, looked up by
  layer name).
- `slot_mapping` — the group's slot mapping tensor (group-shared).

For each token `i` in the batch, the kernel writes:

```
kv_cache[slot_mapping[i]] = compute_KV(token_i)
```

Concretely (in pseudocode for layer L0 processing all 8 tokens):

```
for i in 0..7:
    slot = slot_mapping_group_0[i]      # 40, 41, 42, ..., 47
    K, V = compute_KV(token_at_pos_i)   # different per layer
    kv_cache_L0[slot] = K, V            # writes to Tensor 0 at offset slot*page_size
```

The kernel never sees a "block ID" — it only sees slot IDs. Block-table
indexing happened once on the host (or in a tiny prep kernel) when
slot_mapping was built. The expensive forward-pass kernel just does
direct indexed writes.

Tracing all four layers:

```
L0 (group 0, slot 0 in group → Tensor 0):
   slot_mapping = [40,41,...,47]   → writes slots 40..47 of Tensor 0

L1 (group 1, slot 0 in group → Tensor 0):
   slot_mapping = [80,81,...,87]   → writes slots 80..87 of Tensor 0

L2 (group 0, slot 1 in group → Tensor 1):
   slot_mapping = [40,41,...,47]   → writes slots 40..47 of Tensor 1
   (same slot_mapping as L0, because they're in the same group)

L3 (group 1, slot 1 in group → Tensor 1):
   slot_mapping = [80,81,...,87]   → writes slots 80..87 of Tensor 1
```

Two slot_mapping tensors total (one per group), reused across the
group's layers. Four `kv_cache` tensor views (one per layer), each
pointing into one of the two column tensors.

Notice: L0 and L1 share the same physical Tensor 0, but write to
**disjoint** slot ranges (40-47 vs 80-87) because they have different
block IDs. L0 and L2 use the same slot indices (40-47) but write to
**different** Tensors (0 vs 1) because they're in different columns.

**Where does a single token's K/V end up across all 4 layers?**

Pick the token at position 5. In group 0 it lives in block 11 at offset
1 → slot 45. In group 1 it lives in block 21 at offset 1 → slot 85.

| Layer | Tensor | Slot | Why                                  |
|-------|--------|------|--------------------------------------|
| L0    | 0      | 45   | group 0, block 11, offset 1          |
| L1    | 0      | 85   | group 1, block 21, offset 1          |
| L2    | 1      | 45   | group 0, block 11, offset 1          |
| L3    | 1      | 85   | group 1, block 21, offset 1          |

Four different physical byte ranges for one token's K/V across the 4
layers — exactly what we need. Different layers compute different K/V,
so the storage has to be different.

**Why nothing collides — checking all four pairs:**

- **L0 ↔ L2** (same group, identical block IDs from group 0's table):
  same slot indices, but **different tensors** (0 vs 1). No collision.
- **L1 ↔ L3** (same group, identical IDs from group 1's table): same
  slot indices, different tensors. No collision.
- **L0 ↔ L1** (same Tensor 0): **different slots** (45 vs 85), driven
  by group 0 and group 1 holding different block IDs. No collision.
- **L0 ↔ L3** (different tensors, different slots): can't collide
  twice over.

Both axes of the layer grid are collision-free *by construction*:

- The **column-tensor structure** keeps within-group layers in different
  tensors → different physical memory even when slot indices match.
- The **global block ID pool** keeps across-group layers on different
  slot indices → different addresses even when the tensor is shared.

Now zoom in on a single shared tensor (Tensor 0) to see how its bytes
are reused over time.

### What "block 42 in Tensor 0" actually means

Tensor 0 is one physical allocation on the GPU — `page_size × num_blocks`
bytes, contiguous. It's divided into `num_blocks` fixed-size slots. "Slot
42" is the byte range `[42 × page_size, 43 × page_size)` — a fixed,
unmoving region inside that one allocation.

The 6 layers in Tensor 0's `shared_by` list (L5, L0, L1, L2, L3, L4 —
one from each group) all use Tensor 0 as their K/V storage. They each
have their own *tensor view* on top — a PyTorch
`.view(...).permute(...)` chain that names the dimensions
`[num_blocks, 2, block_size, num_kv_heads, head_size]` — but all 6
views are aliases of the same underlying bytes. When L5 reads slot 42
and L0 reads slot 42, they're touching the same physical memory.

That sounds dangerous: 6 layers all able to clobber each other's K/V?
The safety comes from a scheduler invariant: **block ID 42 lives in at
most one group's block table at a time.** Block IDs are popped from a
single global free queue; once a group has block 42 in its table, no
other group can have it until that group frees it back.

So at any single moment, only *one* of those 6 layers has slot 42 in its
currently-active block table. The other 5 layers' tables contain
different block IDs, and they're touching other slots. There's no
concurrent writing — slot 42 is shared *over time*, not in parallel.

### A concrete timeline

```
T0 (startup):  Tensor 0 slot 42 = zeros, unowned.

T1: Request A arrives. Scheduler allocates block 42 to group 0 for A.
    L5 (group 0, slot 0) processes A's tokens, writes K/V to Tensor 0 slot 42.
    → "L5 owns those bytes" — they hold L5's K/V at A's position p.

T2: Request A finishes. Block 42 returns to the free pool.
    → Tensor 0 slot 42 still contains L5's old K/V (stale, unreferenced).

T3: Request B arrives. Scheduler allocates block 42 to group 3 for B.
    L2 (group 3, slot 0) processes B's tokens, overwrites Tensor 0 slot 42.
    → "L2 owns those bytes" now — same physical memory, different content.
```

The bytes don't permanently belong to any one layer. As block 42 gets
recycled across requests and possibly across groups, ownership rotates.
The shape interpretation is unchanged across rotations — every layer
agrees on `page_size` — only the contents and the "current owner" change.

The single-writer-at-a-time property is what makes this safe: the
doubly-linked free queue (§9) plus ref counting (§10) guarantee that
block 42 only re-enters circulation after every reference to it has been
dropped. So a fresh allocation of block 42 to a new group always finds
the slot unreferenced — never racing with a previous owner.

### Why this matters (it's not the allocation count)

You might reasonably ask: if all the physical tensors are allocated *once*
at startup and stay around forever, why care whether it's 10 tensors or
60? The allocation cost is one-time and tiny.

First, what these layouts have in common. **All three layouts have 6
groups with 6 separate block tables** — Gemma-3 needs 6 groups
regardless of how the physical memory is laid out (it's forced by
attention-type lifecycle, §6). The decision is *not* "how many block
tables" — it's "how is the physical memory structured under those 6
tables."

Two stacked distinctions matter, not one:

- **Per-layer → per-group**: collapse 60 block tables down to 6 by
  letting layers of the same attention type share a table. This is the
  metadata-cost win (§6). Both per-group and column-packed do this.

- **Per-group → column-packed**: keep the 6 block tables, but rearrange
  physical memory. This is the layout choice we're examining here.

So when comparing per-group to column-packed, hold the block-table
structure constant (6 tables in both) and focus on the physical-memory
side.

#### Per-group layout

One tensor per group, sized to hold that group's layers' K/V:

```
Group 0 tensor: [num_blocks_per_group, layers_per_group=10, page_size_per_layer]
Group 1 tensor: [num_blocks_per_group, 10, ...]
... (6 group tensors total)
```

Each group's tensor is exclusively for that group's layers. The 6
allocations don't share bytes. To find a block:
`group_tensor[block_id, layer_slot]`.
The equalized group size is what makes `layer_slot` a stable second axis:
block 42 can contain slot 0, slot 1, ..., slot 9 for every group.

You could still make the scheduler hand out globally unique logical IDs
in this layout, but the meaning would be weaker. ID 42 would not be
*one shared physical slot*; it would be "slot 42 in whichever group's
private tensor owns the ID." When group 0 holds ID 42, the live bytes
are at `group_0_tensor[…, 42, …]`; when 42 is freed and reallocated to
group 1, the live bytes move to `group_1_tensor[…, 42, …]` — a
*different* physical region. The slot-42 region in `group_0_tensor`
still exists, allocated at startup; it's just unreferenced. So global
logical uniqueness alone does not pool memory. Column packing is what
makes a global block ID correspond to a reusable physical slot across
groups.

The cost: per-group capacities have to be decided **statically at
startup** — "group 0 gets `num_blocks_per_group` slots, group 1 gets
`num_blocks_per_group` slots, ...". If the global block-ID range is
10,000 blocks and every group tensor must be indexable by any ID in that
range, then every group tensor needs 10,000 block slots. That is painful
for SWA: even if an SWA group only keeps `ceil(W / block_size)` blocks
live per request, its private tensor still reserves the whole 10,000-slot
address space so `group_tensor[block_id, layer_slot]` is valid for any
assigned global ID. If SWA group 1 finishes its windowing pass and frees
blocks while full-attention group 0 is starved, group 0 can't claim group
1's freed slots. They live in physically different tensors. Memory is
partitioned, not pooled.

#### Per-layer layout

60 block tables, 60 tensors. Each layer is fully independent. Block IDs
across layers are completely separate.

The cost: every layer has its own block table on the GPU, and every
layer needs its own `slot_mapping` computed on every forward pass. The
block table is a `(max_num_reqs, max_blocks_per_req)` int32 tensor that
lives on GPU — for an 80-layer model at 128k context with 512
concurrent requests, per-layer tables cost ~1.3 GiB just for
*metadata*. That GiB comes directly out of the KV cache budget.

The arithmetic:

```
max_blocks_per_req = 128k tokens / 16 tokens per block = 8192 blocks
one layer table     = 512 reqs × 8192 blocks × 4 bytes = 16 MiB
80 layer tables     = 80 × 16 MiB = 1280 MiB ≈ 1.25 GiB
```

That's before counting `slot_mapping` tensors or any other scheduler
metadata.

#### Column-packed layout (current)

`group_size` column tensors, each shared by one layer from every group.
6 block tables (same as per-group), and now the global-uniqueness
invariant on block IDs becomes physically load-bearing:

```
Column 0 tensor: shared by L5(g0), L0(g1), L1(g2), L2(g3), L3(g4), L4(g5)
Column 1 tensor: shared by L11(g0), L6(g1), ...
... (10 column tensors total)
```

Each column tensor is block-major: `column_tensor[block_id]`. The
`layer_slot` has already selected which column tensor this layer uses.
So column packing is the same `layer_slot` structure rotated out of the
group tensor and into the tensor list: instead of
`group_tensor[block_id, layer_slot]`, the layer chooses
`column_tensor_for_layer_slot[block_id]`.

The structural difference from per-group is that **slot 42 is one
fixed physical region per column**, period. Whichever group currently
holds block 42 reuses those same bytes. When group 1 frees block 42 and
group 0 reclaims it, the bytes at slot 42 of every column tensor stay
put; only the writing layer changes (e.g., L0 → L5). There is no
`group_X_tensor[…, 42, …]` for each group sitting around unreferenced
— there's just one slot 42 per column, currently in use by whoever
holds the ID.

This is the actual structural win: **physical memory is pooled across
groups, not partitioned.** When SWA groups have lighter demand
(windowing frees blocks mid-request), those freed slots return to a
shared pool that the full-attention group can immediately reuse. In a
per-group layout, that memory would just sit idle until the request
finishes.

A second, smaller win: the per-layer view shape stays
`[num_blocks, 2, block_size, num_kv_heads, head_size]` — kernel reads
`block_table[req][i]` and indexes directly into the layer's own tensor
with no layer-index arithmetic. Per-group layouts would force the
shape to `[num_blocks, layers_per_group, ...]` and require every kernel
to take an extra layer index. Cheap to fix, but invasive across every
attention backend.

So column packing isn't about "fewer allocations" — that's
third-order. It's about, in order of importance:
**(1) cross-group physical-memory pooling** (one slot 42 reused across
groups, not 6 slot-42s sitting in 6 partitioned tensors), and
**(2) kernel shape unchanged**. The fact that it also happens to use 10
tensors instead of 60 is a side effect of the layout, not the reason for
it.

### Why 10 tensors and not 1?

A natural follow-up: if cross-group sharing is the win, why stop at 10
column tensors? Why not collapse everything into a single tensor of
`num_blocks` slots and have all 60 layers index into it?

Because **layers within the same group share a block table.** That's the
load-bearing fact.

When group 0 holds block 42, *all 10 of its layers* (L5, L11, L17, ...,
L59) want to write K/V to slot 42 — they look up the same block ID in
the same table. But each of those 10 layers computes *different* K/V
values (different weights, different activations). If they all wrote to
one shared slot 42, they'd clobber each other.

So the layout needs at least 10 distinct "slot 42" regions — one per
layer-position-within-its-group. That's the floor: **`group_size`
distinct memory regions** (where `group_size` is the layers-per-group
count). The "10" in the column layout is exactly that floor; it's not a
tunable.

The orthogonality is the whole trick:

- **Within a group** (rows of the grid): layers share a block table →
  identical block IDs at the same time → collision if storage is shared
  → **must be separate memory** (different columns).
- **Across groups** (columns of the grid): scheduler invariant
  guarantees no two groups hold the same block ID simultaneously → safe
  to share storage → **can collapse into one tensor per column**.

Column packing is the unique layout that respects both constraints:
separate memory exactly along the axis where collisions would happen,
shared memory along the axis where they wouldn't.

### Could the 10 column tensors be one giant tensor instead?

Conceptually yes — you could allocate one giant tensor of
`10 × num_blocks × page_size` bytes and slice it into 10 sub-regions:

```
giant[0 : N*P]                -> column 0
giant[N*P : 2*N*P]            -> column 1
...
```

(where `N = num_blocks` and `P = page_size`). The kernel wouldn't
notice — each layer's view would still have its own `data_ptr()` and
`stride[0] = page_size`, and "block 42" would still mean offset
`42 × page_size` from the layer's start.

The current design picks 10 separate `torch.zeros()` allocations
because:

- Each per-layer view is a clean `.view(dtype).view(shape).permute(...)`
  chain starting at offset 0 of its own tensor — no `narrow()`
  arithmetic to track per-layer offsets.
- Multiple smaller allocations let PyTorch's caching allocator place
  them independently. One giant allocation requires a single contiguous
  free range, which can fail under fragmentation when smaller pieces
  would still fit.
- Bookkeeping is simpler: each `KVCacheTensor` is one logical unit with
  its own `shared_by` list. With one giant tensor you'd need the same
  metadata anyway, plus per-layer offsets bolted on.

These are real but secondary. The total memory and the kernel-level
access pattern are identical either way. **The 10 isn't really about
"how many allocations"; it's about "how many distinct address spaces
the layout needs," which is fixed at `group_size` by the within-group
collision argument above.**

→ Full layout details with byte-level diagrams in
[KV_CACHE_ALLOCATION.md "The Structural Mechanism: Shared Column Tensors"](KV_CACHE_ALLOCATION.md#the-structural-mechanism-shared-column-tensors).

## 8. Why blocks are stored block-major in memory

Inside one block (e.g., block 42 in Tensor 0), the layout is:

```
[2, block_size, num_kv_heads, head_size]
 ^   ^           ^             ^
 |   |           |             head dim
 |   |           num KV heads
 |   token-within-block (0..15)
 K vs V
```

The 16 tokens of a block are laid out **contiguously** — token 0's K is
right next to token 1's K is right next to token 2's K. This is called
"block-major" and is the opposite of "token-major" (where token 0 of
block 0 would sit next to token 0 of block 1).

Why block-major? Because **the kernel reads a block at a time**. When
FlashAttention processes a tile, it issues one contiguous load for all 16
tokens' K and V. Block-major makes that load a single linear stride.
Token-major would force scattered reads.

Block-major also makes copies cheap. When a request gets preempted and
its blocks are swapped to CPU, that's one `memcpy` per block per layer of
`page_size` bytes — the largest possible contiguous chunk.

→ See [KV_CACHE_ASSUMPTIONS.md §3](KV_CACHE_ASSUMPTIONS.md#3-why-slot-layout-is-block-major-not-token-major).

## 9. The free list is doubly-linked, and that's not arbitrary

When a block becomes unused, it goes onto a **free queue**: candidates
for eviction when memory runs low. The free queue is a hand-rolled
doubly-linked list — not Python's `collections.deque`.

Why? Because you sometimes need to **pull a block out of the middle**:

- Request A finishes, its blocks (which still hold valid K/V!) go to the
  free queue.
- Request B arrives with the same prefix. It hits the cache — it wants
  blocks that are *currently sitting in A's just-freed list* but haven't
  been evicted yet.
- B's block lookup pulls those blocks out of the middle of the free list
  in O(1) (because doubly-linked) and increments their ref count.

A `deque` only supports O(1) operations at the *ends*. Middle removal
would be O(N) — and N here can be tens of thousands of blocks.

There's also a subtle eviction order trick. When request A frees its
blocks, they're added to the queue in **reverse** order (tail block first,
head block last). The reasoning: tail blocks are the least likely to be
hit by future requests (they correspond to A's specific continuation),
while head blocks (system prompt, shared prefix) might serve many future
requests. Putting tail blocks at the front of the eviction queue means
they get evicted first.

→ See [KV_CACHE_ASSUMPTIONS.md §7](KV_CACHE_ASSUMPTIONS.md#7-why-the-free-queue-is-doubly-linked-not-a-deque).

## 10. Reference counting without locks

When two requests share a cached block, vLLM tracks a `ref_cnt` per block
— it's literally `block.ref_cnt += 1` on a Python attribute, no atomics,
no locks.

This is fine because **the scheduler is single-threaded**. Every code
path that touches `ref_cnt` runs inside `Scheduler.schedule()` or
`Scheduler.update_from_output()`, and there's exactly one scheduler thread
per engine. No race possible.

This is load-bearing for performance: at thousands of blocks per step,
even a `threading.Lock` per touch would show up in profiles. The
single-threaded scheduler is also what makes the doubly-linked free list
safe (no concurrent middle-removal), and what makes the hash table
lock-free.

If anyone ever proposes to parallelize the scheduler, they'll touch all
three of these at once.

→ See [KV_CACHE_ASSUMPTIONS.md §8](KV_CACHE_ASSUMPTIONS.md#8-why-ref-counting-has-no-atomics)
and the "Patterns" section at the bottom of that doc.

## 11. When memory runs out: preempt the *youngest*

Eventually a request will be unable to allocate the blocks it needs. The
scheduler then **preempts** another running request — kicks it out of
running state, frees its blocks, sends it back to WAITING.

Counterintuitively, FCFS preempts the *most recently arrived* request,
not the oldest. Reason: the oldest running request has the most
computed K/V (it's been running longest and has the longest context).
Preempting it would throw away tens of thousands of tokens of work.
Preempting the youngest throws away the least.

Preempted requests don't get killed — they retry, often hitting the
cache for their own already-computed prefix on the second try. So
preemption is conservative and reversible; eviction (dropping cache
entries entirely) is more aggressive.

→ See [KV_CACHE_ASSUMPTIONS.md §12](KV_CACHE_ASSUMPTIONS.md#12-why-preempt-the-latest-request-not-the-oldest).

## 12. Two sentinels: null blocks and PAD_SLOT_ID

You'll see two "missing" markers in the code:

- **null block** (`is_null=True`): appears in block tables for SWA or
  Mamba groups when a request has logical positions that the layer won't
  read. For example, in a SWA group, blocks before the active window
  contain null entries — they don't hold valid K/V because the layer
  evicted that range. Null blocks have `ref_cnt=0` permanently and never
  enter the free queue.

- **`PAD_SLOT_ID = -1`**: appears in the per-token `slot_mapping` for
  tokens that exist in the batched tensor but aren't real (CUDA-graph
  padding, decode positions past a request's end). Backends treat -1 as
  an invalid/padded slot and skip the write or route it through a no-op
  path.

Why both? Because the sentinels live at different layers. Null blocks
belong to the *block table* (group-level metadata) and mean "this
position has no storage." `PAD_SLOT_ID` belongs to the *slot mapping*
(per-token) and means "this token isn't real, don't write its K/V
anywhere." Different problems, different sentinels.

Why -1 specifically? Because 0 is a valid block ID and a valid slot ID,
so it can't be a sentinel.

→ See [KV_CACHE_ASSUMPTIONS.md §4](KV_CACHE_ASSUMPTIONS.md#4-why-the-null-block--pad_slot_id--1-exists).

## 13. Linear attention (Mamba, RWKV) is a different problem

Everything so far assumed *softmax attention*: every past token's K/V is
addressable by index, and we cache them. **Linear attention** (Mamba,
RWKV, RetNet) doesn't work that way. It compresses the entire history
into a fixed-size **state tensor** that gets updated each step:

```
S[p]   = f(S[p-1], x[p])           # state evolves
out[p] = g(S[p], x[p])             # output read from state
```

The state has *no positional structure*. You can't ask it "what was K at
position 47?" — the information has been irreversibly mixed into the
running state. You can only resume from a *checkpoint* — a snapshot of
the state at some past position.

This single property changes almost everything about caching:

| Aspect              | Softmax attention             | Linear attention               |
|---------------------|-------------------------------|--------------------------------|
| Memory per request  | O(N) — one slot per token     | O(1) — one state tensor        |
| Position lookup     | Yes                           | No                             |
| Partial sharing     | Per-block (every 16 tokens)   | Per-checkpoint only            |
| Hit granularity     | Continuous run of cached blocks | Latest single checkpoint     |

vLLM still uses the same `BlockPool`, the same hashing, the same group
abstraction for Mamba layers — but the *contents* of a "block" mean
something completely different. Instead of "K/V for 16 tokens," a Mamba
block holds "one snapshot of the recurrent state." The "block size"
becomes the *interval between snapshots*, not the number of cached
tokens per block.

`MambaManager.find_longest_cache_hit` reflects this: it scans
right-to-left and returns just *one* matching checkpoint (the latest
one), filling all earlier slots with null blocks. There's no need for a
contiguous run of hits — the latest checkpoint subsumes all earlier
ones.

### Mamba's extra wrinkles

Even within "linear attention," Mamba has a few quirks:

- **Two state tensors per layer**: a recurrent SSM state plus a small
  rolling conv state for the depthwise convolution. Both must move
  together.
- **Selectivity**: state-update weights depend on the input, so you
  can't "fast-forward" through a missed prefix. Replay costs the same as
  cold prefill.
- **Two kernels**: a chunk-wise parallel scan for prefill, a step
  recurrence for decode. They have different metadata.
- **Speculative decoding rollback is hard**: drafts overwrite state
  in place. To roll back, you have to reserve extra blocks for
  pre-draft snapshots.
- **No DCP/PCP**: you can't shard the recurrence across ranks because
  each step depends sequentially on the previous one.

These are detailed in [MAMBA.md](MAMBA.md).

## 14. The whole picture, in one walk-through

Let's trace what happens when a request arrives at vLLM:

```
Request: [tokens A B C D E F G H I J K L]   (12 tokens, block_size=4)
```

**Step 1: lookup the prefix cache.** Compute block hashes:
```
block_0 hash = H(NONE_HASH, [A B C D])
block_1 hash = H(block_0, [E F G H])
block_2 hash = H(block_1, [I J K L])
```
Walk these through `BlockHashToBlockMap`. Suppose blocks 0 and 1 hit
(found in the table, pointing at physical block IDs 47 and 89).
Block 2 misses.

**Step 2: external cache.** If a connector is configured, ask whether
peers/CPU/storage have block 2. Suppose no — proceed.

**Step 3: allocate.** Touch blocks 47 and 89 (pull them out of the free
queue if they were on it, increment their ref counts). Pop a fresh block
ID from the free queue for the missing block — say, block 153.
Now this request's block table is `[47, 89, 153]`.

**Step 4: schedule.** Write the block table to GPU. Compute slot mapping
for the tokens that need to be processed (only block 2's tokens, since
0 and 1 are cached). Slot mapping = `block_id × block_size + offset`
per token.

**Step 5: forward pass.** The kernel reads K/V for the cached prefix from
blocks 47 and 89 (at the appropriate addresses inside the column tensors),
computes K/V for block 2, and writes the new K/V to block 153's slots.

**Step 6: cache the new block.** After the forward has filled block 153
with valid K/V, vLLM inserts the already-computed block hash
`(block_2_hash → 153)` into `BlockHashToBlockMap`. Now another request
with the same first 12 tokens will hit all three blocks.

**Step 7: continue decoding.** Each new token writes K/V to whichever
block currently has space. When a block fills, hash it and add it to the
table. Repeat.

**Step 8: finish.** When the request finishes, decrement ref counts on
all its blocks. Blocks whose ref count hits zero go to the free queue
(in reverse order — see §9). They're not evicted yet — just candidates
for eviction when someone else needs memory. If a future request hits
their hash, they get pulled out of the queue and reused.

That's the whole loop. Every concept above is a refinement of one of
these eight steps.

## 15. The big takeaways

If you remember nothing else:

1. **The cache exists because attention is repetitive** — same prefix
   means same K/V, so don't compute it twice.

2. **Blocks are the unit of everything**. They make allocation O(1),
   make kernels efficient, and make hashing chainable. The block size
   is a joint constraint between the scheduler, the attention kernel,
   and the hash table.

3. **The chained hash is the key correctness mechanism**. Two requests
   hitting the same hash are *provably* sharing the same prefix. Anything
   weaker (suffix hashes, windowed hashes) silently corrupts outputs.

4. **Groups partition layers by lifecycle.** Different attention types
   (full / SWA / Mamba) need different block tables, different scan
   directions, different eviction rules. A group is the unit "all my
   layers agree on cache behavior."

5. **The uniform-block-size invariant is what lets one global block ID
   space serve all groups.** Column-packed allocation builds on it to
   share physical tensors across groups.

6. **The single-threaded scheduler is load-bearing** for ref counting,
   the doubly-linked free list, and the lock-free hash table. It's why
   the code looks "naively concurrent-unsafe."

7. **Linear attention reuses the allocator but has a totally different
   semantic.** Mamba blocks store *state checkpoints*, not K/V — but the
   same `BlockPool`, same hashes, same groups still apply. The
   abstraction is well-chosen.

8. **Failures here are silent.** There's no kernel-level alarm if a
   wrong block gets returned. The model just gets quietly worse on
   specific prompts. That's why the code is dense with assertions and
   why every clever optimization needs a correctness proof, not just
   benchmarks.

When you're ready for more depth:

- Start with [KV_CACHE.md](KV_CACHE.md) for the prefix-caching flow and
  the receptive-field argument.
- Then [KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md) for groups,
  block IDs, and the column-packed tensor layout.
- Then [KV_CACHE_ASSUMPTIONS.md](KV_CACHE_ASSUMPTIONS.md) for the 15
  hidden assumptions and what would break if you changed them.
- Then [LINEAR_ATTENTION.md](LINEAR_ATTENTION.md) and
  [MAMBA.md](MAMBA.md) for how the same machinery hosts a totally
  different architecture.
