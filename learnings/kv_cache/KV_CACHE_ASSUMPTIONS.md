# KV Cache: Hidden Assumptions and Constraints

The first two notes in this folder cover *what* the KV cache does (prefix
caching, allocation, group layout). This one questions *why it is shaped that
way at all* — the assumptions that look arbitrary in the code until you ask
"what would break if we did the obvious other thing?"

For each item below: the assumption, where it lives, and the failure mode it
prevents.

---

## 1. Why allocate in fixed-size blocks at all (not per-token)

The unit of allocation is a block of `block_size` tokens, not one token.
PagedAttention is sometimes presented as "the OS-paging idea applied to KV
cache," which makes the block sound inevitable. It isn't — the design is
making three trades at once:

- **A free list of fixed-size slots is O(1) to allocate and free.** A
  per-token allocator would need either a free-token bitmap (poor cache
  locality on the host side, and the scheduler runs on CPU once per step) or
  per-request arenas (re-introduces fragmentation). The
  [`FreeKVCacheBlockQueue`](vllm/v1/core/kv_cache_utils.py#L162) is a
  doubly-linked list of `KVCacheBlock` objects exactly because the scheduler
  needs O(1) push/pop and O(1) middle-removal (when a cached block gets
  re-touched and yanked out of the free list).
- **Attention kernels read K/V in tiles.** FlashAttention / FlashInfer issue
  one TMA / vector load per `block_size`-tokens chunk. Allocating
  per-token would force scatter-gather loads — `block_table[req][i]` would
  point at a single token, the kernel would issue `block_size`× more
  indirected loads per query.
- **The hash chain has to anchor somewhere.** Prefix caching keys content by
  block hash; you can only hash a *fixed* unit and chain it. A token-level
  hash chain would have N entries per request instead of N/`block_size`,
  which moves the hash table from a per-block payload to a per-token-with-
  overhead payload.

The price is internal fragmentation in the last block of every request —
acceptable because in production, batches average out and the last partial
block is usually being filled by the next decode step anyway.

**Example:** with `block_size=16`, a 33-token request owns three blocks:

```text
block 0: tokens 0..15
block 1: tokens 16..31
block 2: token 32 plus 15 empty future slots
```

The scheduler accounts for 3 fixed slots, not 33 independently managed token
slots. The last 15 positions are internal fragmentation until later decode
steps fill them.

## 2. Why `block_size` cannot be chosen freely

Three layers each impose a constraint, and they don't agree on what they
want:

- **Manager block size** (`cache_config.block_size`, typically 16) — what
  the scheduler hands out, what `find_longest_cache_hit` walks.
- **Hash block size** (
  [`resolve_kv_cache_block_sizes`](vllm/v1/core/kv_cache_utils.py#L569)) —
  GCD of group block sizes when hybrid; allows finer prefix-cache hits
  across groups whose blocks aren't the same size. Across-group divisibility
  is asserted at
  [`kv_cache_coordinator.py:431`](vllm/v1/core/kv_cache_coordinator.py#L431).
- **Kernel block size** —
  [`select_common_block_size`](vllm/v1/worker/utils.py#L260) picks a value
  the *attention backend* can handle, queried through
  `backend.get_supported_kernel_block_sizes()`. FlashInfer wants specific
  values; some backends only accept a fixed integer; Mamba uses a different
  block_size entirely.

When manager ≠ kernel,
[`BlockTable`](vllm/v1/worker/block_table.py#L47) silently splits each
manager block into `block_size / kernel_block_size` kernel blocks via
`map_to_kernel_blocks`. The scheduler still thinks in manager blocks; the
kernel sees twice as many.

**What breaks if you ignore this**: pick a manager block size the kernel
doesn't accept and the worker raises `ValueError("kernel_block_size N must
divide kv_manager_block_size M")` on init. Subtler: choose a hash block size
that doesn't divide one group's block size and prefix-cache lookup would
silently return the wrong span — guarded by the assertion above, so it
crashes loudly instead.

**Example:** suppose the manager uses `block_size=32` but the selected backend
stores pages in kernel blocks of 16 tokens. One scheduler block becomes two
kernel blocks:

```text
manager block B, offsets 0..31
  -> kernel block 2B,   offsets 0..15
  -> kernel block 2B+1, offsets 0..15
```

If the manager block size were 24, that same backend could not split it into
16-token kernel blocks without a partial kernel block.

## 3. Why slot layout is block-major, not token-major

The slot mapping kernel at
[`block_table.py:371`](vllm/v1/worker/block_table.py#L371) computes:

```python
slot_ids = block_numbers * block_size + local_block_offsets
```

Tokens within a block are contiguous in physical memory. Token-major
(`token_idx * num_blocks + block_idx`) would be the natural choice if you
only thought about "logical position → physical address." But:

- A FlashAttention tile at position `p` reads
  `K[block_table[p // block_size]][0 : block_size, :]` — one contiguous
  load. Token-major layout would force the kernel to gather `block_size`
  scattered tokens for every tile.
- The K/V *tensor view* per layer is
  `[num_blocks, 2, block_size, num_kv_heads, head_size]` (see
  [KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md)),
  which makes block-major the only layout where `block_id` is a direct
  first-axis index.

Block-major also means that copying a block (preemption, swap-out, P/D
transfer) is one contiguous memcpy of `page_size` bytes per layer, not
`block_size` strided memcpys.

**Example:** with `block_size=16` and a group block table `[42, 99]`, logical
position 19 maps to block-table index `1` and offset `3`:

```text
block_numbers = 99
local_block_offsets = 3
slot_id = 99 * 16 + 3 = 1587
```

The 16 slots for physical block 99 are contiguous: `1584..1599`.

## 4. Why the null block / `PAD_SLOT_ID = -1` exists

Two distinct sentinels you might confuse:

- **`is_null=True` on a `KVCacheBlock`**
  ([block_pool.py:174](vllm/v1/core/block_pool.py#L174)) — appears in block
  tables for SWA / Mamba groups when a request has gaps (e.g., out-of-window
  positions that the SWA layer literally won't read). Has `ref_cnt`
  permanently zero, never enters the free queue, never gets cached.
- **`PAD_SLOT_ID = -1`**
  ([attention/backends/utils.py:44](vllm/v1/attention/backends/utils.py#L44))
  — written into `slot_mapping` for tokens that exist in the batched tensor
  but aren't real (CUDA-graph batch padding, decode-position tokens past the
  request's end). Attention writes to slot -1 by convention go into a
  no-op / sink address.

Why distinct? Block 0 is a real allocatable block; you cannot use "block id
== 0" as a sentinel because requests use block 0. You also cannot use "slot
id == 0" because slot 0 belongs to block 0 token 0. Hence -1, which is out
of range for both block IDs (uint) and slot IDs.

**What breaks if `PAD_SLOT_ID` were 0**: every CUDA-graph-padded decode
token would write its phantom K/V over the real K/V at block 0 token 0 of
the first request. Silent corruption.

**Example:** an SWA request may keep logical positions while dropping old
physical storage:

```text
logical blocks: [NULL, NULL, 17, 18]
```

That means "positions covered by the first two blocks are outside this layer's
window." Separately, a padded batch token might get:

```text
slot_mapping: [..., -1, -1]
```

Those `-1` entries mean "do not write K/V for these padded tokens."

## 5. Why blocks are hashed *after* the forward pass, not before

`BlockPool.cache_full_blocks()` runs in
[`KVCacheManager.cache_blocks`](vllm/v1/core/kv_cache_manager.py#L414) only
*after* the scheduler step that scheduled the prefill chunk has executed.
The naive design would hash up-front — you know the tokens before you run
the model.

The reason is **the hash isn't a key into the block, it's a key into KV
content**. Two requests with identical token prefixes have identical KV
*only if computation actually happened* on those tokens with the same
weights / dtype / config. Hashing before the forward would let a request
share KV that another request hasn't actually computed yet — a write-write
race in the GPU buffer.

Concretely: `hash_block_tokens` is called over the request's *future*
tokens to prepare lookup keys (in `update_block_hashes`), but
`BlockPool.cache_full_blocks` only inserts a `(block_hash → block_id)`
mapping after the worker has written real K/V into that block ID. The
two-phase split — *lookup hash* computed eagerly on CPU, *insertion*
deferred until after the GPU writes — is what keeps lookup cheap and
insertion safe.

**Example:** request A and request B have the same first 16 tokens and are
scheduled in the same prefill wave. B may compute the same hash as A on CPU,
but B cannot hit A's block yet because A's worker has not written the K/V
contents. The `(hash -> block_id)` entry becomes valid only after A's forward
finishes and `cache_blocks` inserts it.

## 6. Why only *full* blocks get cached

Partial blocks are never inserted into `BlockHashToBlockMap`. The hash
input is `(parent_hash, block_tokens, extra_keys)` — if you hashed a
partial block, the next decode token would land in that block, the hash
would change, and the prior insertion would be a stale orphan in the map.

The fix would be deletion-on-extend, but that needs a back-reference from
block to map entry, costing one pointer per block plus a write on every
decode step. Easier to just wait for the block to fill.

**Side effect**: prefix cache hit granularity is `block_size` tokens — a
3-token shared suffix in a 16-token block is invisible to the cache. This
is the same trade as #1: you lose fine-grained sharing to keep the data
structure simple.

**Example:** with `block_size=16`:

```text
request length 15 -> 0 cacheable blocks
request length 16 -> 1 cacheable block
request length 31 -> 1 cacheable block
request length 32 -> 2 cacheable blocks
```

The partially-filled tail can still be used by the running request, but it is
not inserted into the prefix-cache hash table.

## 7. Why the free queue is doubly-linked, not a deque

[`FreeKVCacheBlockQueue`](vllm/v1/core/kv_cache_utils.py#L162) is implemented
as a hand-rolled doubly-linked list with sentinel head/tail. The docstring
says it's for O(1) middle-removal — *why* is that needed?

When a request hits the prefix cache, the cached blocks already exist in
the free queue (they're freed but not yet evicted).
[`BlockPool.touch`](vllm/v1/core/block_pool.py#L391) has to pull them *out
of the middle* of the free list — they shouldn't be candidates for eviction
anymore. A `collections.deque` only supports O(1) ends, so this would be
O(N) over potentially tens of thousands of blocks per cache hit.

The list also encodes a non-obvious eviction tiebreak
([kv_cache_utils.py:171-178](vllm/v1/core/kv_cache_utils.py#L171)): within
blocks freed at the same time, blocks with *more hash tokens* (i.e., the
tail of a chain) are at the front. The block_pool achieves this by
**reversing the block order** when freeing a request's blocks
([block_pool.py:408](vllm/v1/core/block_pool.py#L408)). Tail blocks evict
first because they're the least likely to be reused — they only help one
specific request continuation. Head-of-chain blocks (system prompts, shared
prefixes) survive longer because more requests will hash-hit them.

Replacing this with a deque loses both invariants.

**Example:** a completed request frees blocks `[10, 11, 12]`. They are returned
to the free queue in reverse priority:

```text
front of eviction queue -> 12, 11, 10 -> back
```

If a new request later hits block 11 as the second block of a shared system
prompt, `touch(11)` removes it from the middle of the queue in O(1), while
block 12 remains easier to evict.

## 8. Why ref counting has no atomics

[`block.ref_cnt += 1`](vllm/v1/core/block_pool.py#L391) and the matching
decrement in `free_blocks` are plain Python attribute writes. There's no
lock, no atomic, no compare-and-swap. This works because **the scheduler is
single-threaded and is the only writer**. Every code path that mutates
`ref_cnt` runs inside `Scheduler.schedule()` or
`Scheduler.update_from_output()`.

This is load-bearing for performance: the scheduler runs once per step on
the critical path, and acquiring even a `threading.Lock` per block touch
would be a measurable hit at thousands of blocks per step.

**What breaks if you parallelize the scheduler**: ref_cnt double-decrement
sends a still-referenced block to the free queue → next allocation hands
out a slot another request is still reading → silent corruption. Any future
multi-threaded scheduler design needs to either shard by request id or
atomicize the block pool.

**Example:** request A and request B share cached block 20. The scheduler
touches it for B:

```text
ref_cnt: 1 -> 2
```

When A finishes:

```text
ref_cnt: 2 -> 1
```

Block 20 must not enter the free queue until B finishes. A lost decrement or
double decrement would put a live block back into circulation.

## 9. Why `NONE_HASH` is randomized per process by default

[`init_none_hash`](vllm/v1/core/kv_cache_utils.py#L95) seeds the hash chain
with `os.urandom(32)` unless `PYTHONHASHSEED` is set. Why not just use a
fixed sentinel like `b"\\0" * 32`?

The same reason CPython randomizes `hash()`: prevent **adversarial cache
poisoning**. With a fixed seed, a request prefix that produces a known
collision with a sensitive cached prefix would let one tenant read
another's cached KV. Per-process randomization makes the chain values
unpredictable across runs.

The cost: cache cannot be reused across process restarts (e.g., a connector
that persisted block hashes to disk would see all-misses after a restart).
Set `PYTHONHASHSEED` if you want cross-process reuse — explicit opt-in for
the security tradeoff.

**Example:** the first block hash is effectively:

```text
hash(random_NONE_HASH, first_16_token_ids, extra_keys)
```

After an engine restart, the same token IDs produce a different chain unless
`PYTHONHASHSEED` fixes the seed. That prevents another process from predicting
the cache key space by default.

## 10. Why all groups use the *same* `num_blocks`, even when SWA doesn't need them

`num_blocks` from `get_num_blocks` in
[kv_cache_utils.py](vllm/v1/core/kv_cache_utils.py) is global. An SWA group
provably never holds more than `ceil(W / block_size)` live blocks per
request, so allocating `num_blocks` slots for it looks wasteful.

It isn't waste, it's the column-packing invariant
([KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md)). A block ID `B` means
"slot `B` in *every* column tensor." If groups had different block counts,
the scheduler couldn't hand any block ID to any group without per-group
offset arithmetic. The "wasted" SWA slots are not actually wasted — they're
shared by full-attention layers in the same column tensor (`shared_by`
covers one layer per group). The total memory is still `available_memory`;
the SWA allocation is paid for by the full layers sharing the column.

The genuine waste is peak per-request blocks held: today's
`HybridKVCacheCoordinator` allocates `ceil(num_tokens / block_size)` blocks
in lock-step across all groups, even though SWA groups only need W's worth.
That's the win flagged in
[KV_CACHE.md](KV_CACHE.md#1-sparser-block-allocation-for-swa-layers-the-largest-practical-win).

**Example:** a hybrid model has one full group and five SWA groups, each with
`num_blocks=10000`. That does not mean vLLM allocated six independent
10000-block pools. In the column-packed layout, one physical tensor column can
be shared by one layer from each group, and block ID 123 is a valid slot in
that column regardless of which group table points to it.

This is the part a per-group layout would lose. If each group had its own
physical tensor but still accepted block IDs from one 10000-ID global range,
then each group tensor would need 10000 addressable slots. The SWA groups
would reserve the same address range as the full-attention group even though
their live window may use far fewer blocks. Column packing keeps the global
address range, but stores it once per column instead of once per group.

## 11. Why workers get clamped to the *minimum* `num_blocks` across ranks

[`unify_kv_cache_configs`](vllm/v1/core/kv_cache_utils.py#L2038):

```python
min_num_blocks = min(c.num_blocks for c in kv_cache_configs)
for c in kv_cache_configs:
    c.num_blocks = min_num_blocks
```

Workers profile their own GPU memory; ranks may differ (different
fragmentation, slightly different free pages). The scheduler, however,
hands out **one global block ID space** that every worker must honor. If
worker A has 10000 blocks and worker B has 9800, the scheduler giving block
9900 to a request would address valid memory on A and out-of-bounds on B.

Pessimistic clamping is the only safe choice without per-worker block ID
namespaces. The cost is throwing away 200 blocks of memory on A — small
compared to the alternative (per-worker block tables, all the indexing
complexity, plus ID translation on every cross-worker transfer).

**Example:**

```text
rank 0 profiles 10000 blocks
rank 1 profiles  9800 blocks
scheduler capacity = 9800 blocks
valid block IDs = 0..9799
```

If the scheduler handed out block 9900, rank 0 would have memory for it and
rank 1 would not.

## 12. Why preempt the *latest* request, not the oldest

When `allocate_slots` returns `None` (out of blocks), the scheduler preempts
([scheduler.py:480-504](vllm/v1/core/sched/scheduler.py#L480)):

```python
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
else:
    preempted_req = self.running.pop()  # last appended
```

FCFS preempts the *most recently arrived* running request. Counterintuitive
— shouldn't a queue evict the oldest? The reasoning is that the oldest
running request has the most computed KV (longest context, most blocks
held). Preempting it means re-running prefill on potentially tens of
thousands of tokens. Preempting the youngest is cheaper to recover from —
it has the smallest in-flight investment.

This is also why preempted requests go back to `WAITING` rather than being
killed: their cache entries may be hit by the very requests that bumped
them, or by themselves on retry. Preemption is conservative; eviction is
aggressive; they're separate mechanisms acting on the same blocks.

**Example:** two running requests need memory:

```text
old request:   12000 computed tokens, hundreds of blocks held
young request:   128 computed tokens, a few blocks held
```

Preempting the old request frees more blocks immediately, but if it is resumed
without cache hits, vLLM may have to replay a long prefill. Preempting the young
request usually costs much less replay work.

## 13. Why `extra_keys` exists and what gets in it

[`hash_block_tokens`](vllm/v1/core/kv_cache_utils.py#L564) takes
`extra_keys` alongside the parent hash and tokens. Without `extra_keys`, two
requests with identical token sequences but different conditioning context
would alias:

| Source of aliasing | Where it's added |
|---|---|
| **LoRA** — different adapters produce different K/V from the same tokens | LoRA name appended to first block's extra_keys |
| **Multi-modal** — image embeddings vary by input even when tokens match | MM hash + per-block-offset |
| **`cache_salt`** — opaque request-supplied string for opting out of cross-request sharing | First block only |

`cache_salt` is the interesting one. It exists not for correctness but for
**experimental isolation**: when benchmarking two prompt variants on the
same prefix, you don't want one to warm the cache for the other. The salt
scopes the hash to a specific "cache namespace." Only the first block
carries the salt because the chained hash propagates it forward.

**What breaks if `extra_keys` were dropped**: a request switching LoRA
adapters mid-prefix would hit cached blocks computed with the previous
adapter's weights. Quietly wrong outputs.

**Example:** two requests tokenize to the same first block:

```text
tokens: [128000, 791, 3931, ...]
request A: LoRA adapter = sql
request B: LoRA adapter = python
```

Without the LoRA key, request B could reuse K/V produced with request A's
adapter weights. With `extra_keys`, the first block hashes differ, and the
parent chain keeps every later block in a separate cache namespace too.

## 14. Why the strided group split (`layers[i::num_groups]`) instead of contiguous slicing

[`_get_kv_cache_groups_uniform_page_size`](vllm/v1/core/kv_cache_utils.py)
splits a same-type layer list as `layers[i::num_groups]` (every Nth layer)
rather than `layers[i*g:(i+1)*g]` (contiguous slice). Both yield groups of
equal size; only one survives pipeline parallelism.

Pipeline parallelism partitions layers into contiguous stage ranges. If a
group held a contiguous slice of layers, that group could land entirely on
one PP stage — and the *other* stages would have an empty group, padded to
keep the per-stage layer count equal. Padding wastes memory and forces
empty-group special cases throughout.

Strided split guarantees every group spans every PP stage. Each PP stage
ends up with `len(group) / pp_size` layers from each group, balanced.

**Example:** a model has 12 SWA layers and 3 SWA groups. Contiguous slicing
would produce:

```text
group 0: layers 0, 1, 2, 3
group 1: layers 4, 5, 6, 7
group 2: layers 8, 9, 10, 11
```

With pipeline stages `[0..5]` and `[6..11]`, group 0 is absent from stage 1 and
group 2 is absent from stage 0. Strided splitting gives:

```text
group 0: layers 0, 3, 6, 9
group 1: layers 1, 4, 7, 10
group 2: layers 2, 5, 8, 11
```

Both pipeline stages now see every group.

## 15. Why `delay_free_blocks` exists for connector finishes

When a request completes, its blocks normally go straight back to the free
queue. But if a connector is still pushing those blocks somewhere (LMCache
write, P/D send to a decode peer), the underlying KV memory must not be
reused yet. `request_finished` returns `(delay_free_blocks,
kv_transfer_params)`; if true, the scheduler holds the blocks until the
connector reports `get_finished` is done.

This is a use-after-free guard, not a correctness optimization. Without it,
the connector would race against `BlockPool.free_blocks → get_new_blocks →
worker writes new K/V` for the same physical slot, and the receiver would
see a torn block — half old request, half new.

The flag is per-block, decided by the connector, because not every connector
cares — synchronous connectors finish before `request_finished` returns, so
they answer false.

**Example:** a producer request finishes with blocks `[200, 201]`, and a
connector is still sending those blocks to a decode worker. If vLLM freed them
immediately, the next request could allocate block 200 and overwrite it while
the connector DMA is still reading. `delay_free_blocks=True` keeps block 200
owned until the connector reports completion.

---

## Patterns that show up across these

A few of these aren't independent — they're applications of the same
underlying constraint.

- **Single-threaded scheduler** is load-bearing for ref-count integrity
  (#8), the doubly-linked free queue (#7), and the lack of locking around
  `BlockHashToBlockMap`. Any plan to parallelize the scheduler hits all
  three at once.
- **Global block ID space** is load-bearing for column-packed tensors
  ([KV_CACHE_ALLOCATION.md](KV_CACHE_ALLOCATION.md)), the `min_num_blocks`
  clamp (#11), and the choice not to split block IDs by group. It also
  makes preemption (#12) and connector transfers (#15) a single global
  problem instead of a per-worker one.
- **The chained hash** is load-bearing for prefix cache correctness across
  divergent prefixes ([KV_CACHE.md](KV_CACHE.md)), `extra_keys` semantics
  (#13), `cache_salt` (#13), and `NONE_HASH` randomization (#9). All four
  are flavors of "scope the cache key correctly."
- **Block-as-unit-of-everything** ties hash granularity (#6), allocation
  granularity (#1), kernel tile size (#2), slot layout (#3), and eviction
  order (#7). The block size is not a tunable; it's the joint in the design
  where four kernels meet.

The recurring failure mode if any of these were violated is **silent KV
corruption** — wrong numbers in the right tensor slots. There is no kernel
trip-wire; the model produces slightly worse outputs, possibly only on
specific prompts, possibly indistinguishable from normal sampling noise.
That is why the surrounding code is dense with assertions and the comments
read like contracts.
