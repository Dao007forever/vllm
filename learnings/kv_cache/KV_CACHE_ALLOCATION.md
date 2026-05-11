# KV Cache Allocation and Group Block IDs

This note explains how vLLM groups KV-cache layers, why groups are shaped to
have uniform physical block sizes, and how logical block IDs map to physical
KV memory.

## Mental Model

vLLM separates two ideas:

- **Logical block ID**: an integer managed by the scheduler/block pool, such as
  `10`.
- **Physical KV memory**: the actual K/V tensor storage used by a specific
  layer or KV-cache group.

KV-cache groups have separate block tables, but their block IDs come from one
shared scheduler-side `BlockPool`. In the current column-packed layout, that
global ID is also a physical slot in the preallocated KV-cache pool: block ID
`10` means slot `10` in each column tensor, not a group-local slot.

That last sentence is layout-dependent. In a hypothetical per-group physical
layout, the scheduler could still hand out globally unique logical IDs, but ID
`10` would map to `group_0_tensor[..., 10, ...]` or
`group_1_tensor[..., 10, ...]` depending on the owning group. Those are
different byte ranges. Column packing is what turns global uniqueness into
cross-group physical-memory pooling.

For a hybrid model, one request may hold different block IDs in different
groups:

```text
group 0 table: [10, 11, 12]
group 1 table: [13, 14, 15]
group 2 table: [16, 17, 18]
```

The group table chooses which global physical slots that group's layers use in
the column-packed layout.

Inside a group, all layers share the same block table, but each layer has its
own KV-cache tensor view:

```text
group 1, layer 7,  block 13 -> K/V memory for layer 7
group 1, layer 13, block 13 -> K/V memory for layer 13
```

The actual key is effectively:

```text
(layer_name, block_id)
```

The group still matters because it selects the block table and attention
metadata for the layer.

## Why Groups Exist At All

Before "why must group blocks be uniform-sized" comes the prior question:
**why have groups at all?** A homogeneous model (every layer is full
attention) could in principle work with a single global block table. The
existence of the group abstraction has to be justified before the layout
choices on top of it.

The answer is that **a group is the equivalence class of layers that share
a KV-cache lifecycle**. Different attention types disagree about three
things, and any one of them is enough to forbid sharing a single block
table:

- **Eviction rule.** A full-attention layer needs K/V at position `p`
  available forever — every later token attends to it. A sliding-window
  layer with window `W` can free K/V at `p` once the request is past
  `p + W`. A Mamba layer doesn't have positional K/V at all; it has state
  checkpoints at block boundaries. One block table can only encode one
  eviction rule. If you tried to share, either SWA layers waste memory by
  pretending to be full, or full-attention layers lose data by pretending
  to be SWA.
- **Cache-hit scan direction.** `FullAttentionManager.find_longest_cache_hit`
  scans block hashes left-to-right; `SlidingWindowManager` scans
  right-to-left and stops once `ceil((W-1)/B)` contiguous blocks match;
  `MambaManager` walks R→L through state checkpoints. These aren't tunings
  — they fall out of the underlying attention pattern's correctness
  requirements (see [KV_CACHE.md §"L→R vs R→L scanning"](KV_CACHE.md#aside-lr-vs-rl-scanning-in-find_longest_cache_hit)).
  A unified block table would need a single scan direction and would lose
  hits in whichever attention type it disagreed with.
- **Block size.** Attention layers typically use `block_size=16`. Mamba
  uses a different block size driven by the SSM kernel's state checkpoint
  granularity. They cannot index into the same block table because position
  `p` maps to different `block_index` values under different `block_size`s.

The code shape mirrors this: each attention type gets its own
[`SingleTypeKVCacheManager`](vllm/v1/core/single_type_kv_cache_manager.py)
subclass (`FullAttentionManager`, `SlidingWindowManager`,
`ChunkedLocalAttentionManager`, `MambaManager`, `CrossAttentionManager`)
with its own `find_longest_cache_hit` and `remove_skipped_blocks`. The
group is what binds all layers of one attention type to one such manager.

### What about per-layer block tables instead?

If groups exist to separate attention types, why not skip the abstraction
and give every layer its own block table? It's the safe extreme — every
layer trivially has its own lifecycle.

It works but it's wasteful, because **layers of the same attention type
have *identical* lifecycles**. Two full-attention layers in the same model
evict at the same time, hit the cache at the same positions, and use the
same `block_size`. Their block tables would be byte-identical. Sharing
costs nothing semantically and saves:

- One block-table tensor per layer on the worker. The block table is a
  `(max_num_reqs, max_num_blocks_per_req)` int32 tensor that lives **on
  GPU** because Triton kernels (slot mapping, attention) read it directly
  via `block_table_ptr`.

  **Why int32, not int64?** Block-table entries are *logical block IDs*,
  not memory addresses. A block ID is just an index in `[0, num_blocks)`.
  PyTorch translates the index into a memory address through strided
  indexing — the kernel never manipulates raw 64-bit pointers. Sizing
  check: even a 60 GiB KV-cache budget with very small (1 KiB) pages
  produces ~60M blocks, which fits comfortably under 2³¹ ≈ 2.1B. Realistic
  configs with KiB-to-MiB pages give 10⁵–10⁶ blocks — nowhere near the
  int32 ceiling.

  The `slot_mapping` tensor *is* int64
  ([block_table.py:327](vllm/v1/worker/block_table.py#L327)) because slot
  IDs are computed as `block_id * block_size + offset` and the
  multiplication is explicitly upcast to int64 before storing
  ([block_table.py:357](vllm/v1/worker/block_table.py#L357):
  `tl.load(block_table_ptr + ...).to(tl.int64)`). Slot IDs index into a
  tensor of `num_blocks * block_size` entries; while that product also
  fits in int32 today, slot mapping is a per-token tensor (much smaller
  than the block table) so the int64 cost is negligible and PyTorch's
  index-tensor API expects int64 anyway. The asymmetry is deliberate:
  int32 where the tensor is large (block table, megabytes), int64 where
  arithmetic happens (slot mapping, kilobytes).

  Concrete numbers for a typical production config with
  `max_num_reqs=512`, `max_model_len=131072`, `block_size=16`:

  ```text
  max_num_blocks_per_req = 131072 / 16 = 8192
  one table = 512 * 8192 * 4 bytes = 16 MiB on GPU
  ```

  An 80-layer model with per-layer tables would burn `80 * 16 MiB = 1.28
  GiB` of GPU memory on metadata alone — memory that comes directly out of
  the KV cache budget. With groups, that drops to one table per group
  (e.g., 6 tables for a Gemma-3-style hybrid → 96 MiB), a 13× reduction.

  At long contexts this gets worse fast: at 1M context the same config
  gives a per-layer table of 128 MiB, so 80 layers would be 10 GiB before
  any K/V is stored. The savings ratio is exactly `n_layers / n_groups`,
  which is why the abstraction is most load-bearing on deep models with
  few group types.
- One `find_longest_cache_hit` walk per layer at scheduling time. With
  shared tables, the manager walks once for the whole group — a constant-
  factor scheduler win that scales with layer count.
- One slot-mapping computation per layer per forward pass. Slot mapping
  is a Triton kernel launch
  ([block_table.py:319](vllm/v1/worker/block_table.py#L319)) that reads
  the block table and produces per-token slot IDs. Per-layer tables would
  force per-layer launches; per-group tables let one launch cover all
  layers in the group.

So groups are the middle path: coarse enough to share metadata where
sharing is free, fine enough to separate where lifecycles diverge.

### How does layer-specific data work if metadata is shared?

The three savings above all involve *one artifact per group*: one block
table, one cache-hit walk result, one slot-mapping tensor. But each layer
in the group still has its own K/V bytes — different layer, different
weights, different K/V. How does that reconcile?

The answer is that the **metadata is logical, the K/V tensor is physical**,
and the dispatch happens at the `forward()` call boundary, not inside the
kernel.

Concretely, when the model runner steps through layers:

- Layer L's `attn.forward()` is called with **layer L's own `kv_cache`
  tensor** (looked up via `kv_caches[layer_name]` — the per-layer dict
  populated at init time from `KVCacheTensor.shared_by`).
- The `attn_metadata` argument carries the **group-shared** `block_table`
  and `slot_mapping`.

Inside the attention backend (e.g.,
[flash_attn.py:752](vllm/v1/attention/backends/flash_attn.py#L752)):

```python
key_cache, value_cache = kv_cache.unbind(0)   # layer-specific
...
flash_attn_varlen_func(
    k=key_cache, v=value_cache,                # layer-specific
    block_table=block_table,                   # group-shared
    ...
)
```

The kernel computes addresses as `kv_cache.data_ptr() + slot_id * stride`.
The slot_id is shared; the data_ptr is layer-specific. Same logical slot,
different physical memory:

```text
slot_mapping (group):    [42,  43,  44, ...]   <- shared across layers in group
                          |
                          v
layer 7 forward(kv_cache=K_cache_7) → kernel writes K_cache_7[42], K_cache_7[43], ...
layer 8 forward(kv_cache=K_cache_8) → kernel writes K_cache_8[42], K_cache_8[43], ...
layer 9 forward(kv_cache=K_cache_9) → kernel writes K_cache_9[42], K_cache_9[43], ...
                          ^
                       different physical tensors, same slot indices
```

This is what makes the column-packed allocation
([§"Shared Column Tensors"](#the-structural-mechanism-shared-column-tensors))
work without per-layer offset arithmetic in the slot mapping. Each layer
already brings its own K/V tensor reference; "slot 42" inside that
reference is unambiguously *that layer's* slot 42. The slot mapping
doesn't need to know which layer is consuming it.

The same pattern applies to all three group-shared artifacts:

| Group-level artifact            | Per-layer counterpart at forward time |
|---------------------------------|---------------------------------------|
| Block table                     | The layer's own `kv_cache` tensor (passed via `kv_caches[layer_name]`) |
| `find_longest_cache_hit` result | (No per-layer step — answer is identical for all layers in group) |
| `slot_mapping` tensor           | The layer's own `kv_cache` tensor (same as block table) |

This is also why `BlockHashWithGroupId` is keyed by `group_id` rather than
`layer_name`: layers within a group cannot disagree about whether a prefix
is cached, so the hash table only needs to distinguish at group
granularity. Layer specificity enters only at forward time, when each
layer's kernel call is given a different physical tensor.

### What about one giant group per attention type?

Within a single attention type, why subdivide further (Gemma-3's 50 SWA
layers split into 5 groups of 10) instead of keeping one giant SWA group?
That's where the *next* section comes in — the uniform-page-size
constraint forces splitting an attention-type's layers into multiple
groups when there are more layers than the smallest other type. Without
that constraint, one group per attention type would suffice.

The hierarchy of reasons is therefore:

1. Different attention types → must be different groups (lifecycle).
2. Same attention type, similar count → one group per type (sharing is
   free).
3. One group block-table entry is shared by every layer in that group, but each
   layer still needs its own K/V payload for that token block.
4. Same attention type, more layers than the smallest type → split into
   equal-sized groups so block-pool slots stay uniform-sized (next
   section).
5. Uniform group block size → one shared `BlockPool` can hand block IDs to
   any group.
6. One shared `BlockPool` plus per-group block tables → column-packed storage
   can share physical block slots across groups while keeping separate block
   tables for different cache lifecycles.

## Why Groups Need Uniform Physical Block Size

vLLM's current allocator wants a simple invariant: one block allocated from the
shared `BlockPool` corresponds to the same amount of physical KV memory,
regardless of which KV-cache group uses it.

This invariant is why groups are not merely semantic. A group block is the
allocation unit, so its physical size is proportional to the number of layers in
that group. Equalizing the effective group size makes `block_id = B` a fixed
size reservation no matter which group owns it.

The proportionality comes from the group-shared block table. If group 0 owns
block ID `B`, every layer in group 0 uses that same block-table entry for the
same token positions. But layer L5's K/V and layer L11's K/V are different
tensors, so the physical group block needs one page of K/V storage per layer in
the group:

```text
group block bytes = layers_per_group * page_size_per_layer
```

Suppose a model has:

```text
10 full-attention layers
50 sliding-window-attention layers
```

If vLLM made only two KV-cache groups:

```text
Group 0: 10 full layers
Group 1: 50 SWA layers
```

then a block used by group 1 would need to reserve 5x more physical KV memory
than a block used by group 0:

```text
Group 0 block = KV memory for 10 layers
Group 1 block = KV memory for 50 layers
```

That would make the shared `BlockPool` variable-sized. The scheduler could no
longer ask a simple question like:

```text
request needs 3 more blocks; do we have at least 3 free blocks?
```

Instead, it would need to reason about different-sized chunks, fragmentation,
and potentially different free capacity per group.

To avoid that, vLLM splits the 50 SWA layers into five groups of 10 layers:

```text
Group 0: 10 full layers
Group 1: 10 SWA layers
Group 2: 10 SWA layers
Group 3: 10 SWA layers
Group 4: 10 SWA layers
Group 5: 10 SWA layers
```

Now each group block represents roughly the same physical amount of memory:

```text
1 block ~= KV memory for 10 layers
```

The design assumption is documented in
[`vllm/v1/core/kv_cache_utils.py`](vllm/v1/core/kv_cache_utils.py):

```python
# Physical memory per block: Must be the same across all KV cache groups.
# Breaking this assumption is non-trivial due to memory fragmentation concerns
# when allocating blocks of different sizes.
```

This assumption is tied to vLLM's centralized scheduling model. The scheduler
does not dynamically query each GPU's free memory for every request. Instead,
vLLM profiles memory once during startup, preallocates KV-cache tensors on the
workers, and gives the scheduler a fixed logical block capacity.

At runtime, the scheduler manages a logical block pool:

```text
free logical blocks: [10, 11, 12, ...]
```

It wants allocation and eviction to be simple operations:

```text
allocate block 10
free block 10
evict block 10
```

For this to work, every block ID must mean one fixed-size slot in the
preallocated KV-cache pool:

```text
block 10 -> one fixed-size slot
block 11 -> one fixed-size slot
block 12 -> one fixed-size slot
...
```

If group blocks had different physical sizes, a single global free list would be
too crude. The scheduler would need per-group capacities, per-group free lists,
and allocation decisions like:

```text
request needs:
  3 blocks in full-attention group
  1 block in SWA group
  0 blocks in another group
```

Eviction would also need to know which group is under memory pressure. vLLM
avoids that complexity by making a logical block correspond to a uniform slot
across groups and by limiting the global block count to what every worker can
support.

### The Structural Mechanism: Shared Column Tensors

The "fragmentation" framing above is the consequence. The mechanism that makes
the uniform-page-size invariant load-bearing sits in the allocator itself at
[`_get_kv_cache_config_uniform_page_size`](vllm/v1/core/kv_cache_utils.py):

```python
# General case:
# We will have group_size memory pools, each is shared by one layer from
# each group. As layers of different groups have different block table,
# they will use different parts of the shared Tensor.
group_size = max(len(group.layer_names) for group in kv_cache_groups)
page_size = get_uniform_page_size(...)
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
for i in range(group_size):
    shared_by = []
    for j in range(len(kv_cache_groups)):
        if i < len(kv_cache_groups[j].layer_names):
            shared_by.append(kv_cache_groups[j].layer_names[i])
    kv_cache_tensors.append(
        KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
    )
```

`group_size` here is the number of **layers per group** (not the number of
groups). The outer loop iterates layer-slots within a group; the inner loop
walks across groups. So each `KVCacheTensor` is shared by one layer from each
group.

For 30 layers in 6 groups of 5 layers each, the layer grid is:

```text
         slot 0   slot 1   slot 2   slot 3   slot 4
Group 0: L5       L11      L17      L23      L29       <- FULL
Group 1: L0       L6       L12      L18      L24       <- SWA
Group 2: L1       L7       L13      L19      L25       <- SWA
Group 3: L2       L8       L14      L20      L26       <- SWA
Group 4: L3       L9       L15      L21      L27       <- SWA
Group 5: L4       L10      L16      L22      L28       <- SWA
         |---|    |---|    |---|    |---|    |---|
       Tensor 0 Tensor 1 Tensor 2 Tensor 3 Tensor 4
```

Packing is column-major across the layer grid, but each physical tensor is
block-major: conceptually `column_tensor[block_id]`. The `layer_slot` has
already selected which column tensor this layer uses. Every tensor has size
`page_size * num_blocks` and exactly `num_blocks` slots. A block ID `B` means
slot `B` in every tensor, so no group needs an offset of its own to translate
block ID -> memory address.

The invariant `page_size` must be the same across groups falls directly out of
this layout: 6 layers reshape the same raw tensor as `[num_blocks, ...]`. They
can only do that consistently if every layer agrees on how many bytes per block
slot.

### Inside one block: byte layout and per-layer reinterpretation

A "block" in the column tensor is `page_size` bytes living at offset
`block_id * page_size` in the raw allocation. The layout *within* those
bytes is the layer's KV cache shape — for FlashAttention-style backends:

```text
one block = page_size bytes
          = [2, block_size, num_kv_heads, head_size] * dtype_bytes
            ^   ^           ^             ^
            |   |           |             head dim
            |   |           number of KV heads (after TP/MQA/GQA)
            |   token position within the block (0..block_size-1)
            K vs V (unbind(0) splits these in flash_attn.py)
```

**Sharing is across groups, not within.** A column tensor (e.g., Tensor 0
in the layer grid above) is shared by exactly *one layer per group*: L5
(group 0), L0 (group 1), L1 (group 2), L2, L3, L4. Layers within the same
group are in different column slots → different column tensors → different
physical memory.

For one specific block (say block 42) in Tensor 0:

```text
column tensor 0 (raw allocation, page_size * num_blocks bytes):
  offset 0           page_size       42*page_size      43*page_size
  |<- block 0     ->|<- block 1   ->|...|<- block 42 ->|<- block 43 ->|...

Block 42 (page_size bytes) — interpreted by each layer's view as:

  L5's view  (group 0, FULL)  : [2, block_size, n_kv_heads, head_size]
  L0's view  (group 1, SWA)   : [2, block_size, n_kv_heads, head_size]
  L1's view  (group 2, SWA)   : [2, block_size, n_kv_heads, head_size]
  ... (one view per layer in shared_by)
```

Each `kv_caches[layer_name]` is a `.view(dtype).view(kv_cache_shape).permute(...)`
of the same raw bytes. PyTorch handles the address arithmetic via strides;
no copy. When the kernel does `kv_cache.data_ptr() + 42 * stride[0]`, it
lands at byte offset `42 * page_size` in the shared column tensor — and
that location contains *whichever layer last wrote to block 42*.

**No collision because of the scheduler invariant.** At any single moment,
the global block ID `42` appears in *at most one* group's block table.
Block IDs are popped from one global `BlockPool` queue, and a popped ID is
ref-counted as held until it goes back to the queue (see
[KV_CACHE_ASSUMPTIONS.md §8](KV_CACHE_ASSUMPTIONS.md#8-why-ref-counting-has-no-atomics)).
So in the layer grid above, even though L5, L0, L1, L2, L3, L4 *could* all
in principle reference block 42, only one of them does at a time. The other
five layers' block tables contain different IDs in their respective slots.

When the request holding block 42 finishes and the block returns to the
pool, the bytes are stale (still hold the old K/V) but unreferenced. A
later request from a different group can claim block 42 → that group's
layer overwrites the bytes → its meaning changes from "L5's K/V at some
position" to "L0's K/V at some position." The shape interpretation is
unchanged (page_size invariant guarantees this); only the contents and
which-layer-owns-this rotates.

**Within a group**, by contrast, every layer is in a *different* column
slot. For group 0 with 5 layers (L5, L11, L17, L23, L29), they occupy
slots 0..4 → Tensors 0..4 → five distinct allocations. Block 42 in
Tensor 0 (L5's storage) and block 42 in Tensor 1 (L11's storage) are
separate `page_size`-byte regions in different physical allocations. The
shared block table just means *all five tensors get accessed at the same
slot index* on every forward pass; the bytes themselves are not shared.

### Why Column Packing Instead of Per-Group or Per-Layer

The same total memory could be cut three ways:

| Layout            | # tensors | Per-tensor size                  |
| ----------------- | --------- | -------------------------------- |
| Per-layer         | 30        | `page_size * num_blocks_per_layer` |
| Per-group         | 6         | `page_size * 5 * num_blocks_per_group` |
| Column (current)  | 5         | `page_size * num_blocks`         |

The columns layout is chosen because:

1. **The global `BlockPool` is physically pooled, not just globally named.**
   Each column tensor has `num_blocks` slots, and a block ID `B` means slot
   `B` in every column. The scheduler can pop block IDs from one queue and
   hand any of them to any group because all groups agree on the same physical
   slot space. In a per-group layout, a globally unique ID would only be a
   logical reservation label: `B` in group 0 and `B` in group 1 would still be
   different byte ranges. Worse, if the global ID range has 10,000 possible
   blocks, every private group tensor must reserve 10,000 slots so any assigned
   ID can be indexed, even if an SWA group's live working set is much smaller.
   To recover similar pooling, the implementation would need per-group offsets
   into a common arena or another indirection layer in the block-table -> memory
   translation; otherwise it would accept partitioned per-group pools.

2. **Layer view shape stays kernel-friendly.** With column packing, each
   layer's view is `[num_blocks, 2, block_size, num_kv_heads, head_size]`,
   indexed directly by block ID. This is exactly what FlashAttention and other
   paged-attention kernels expect for `block_table`. Per-group layout would
   produce `[num_blocks, layers_per_group, ...]` and force every backend to
   take an extra layer-index argument.

   This is the same equalized `layer_slot` axis used two different ways:
   per-group storage would index it inside the tensor as
   `group_tensor[block_id, layer_slot]`; column packing uses it to choose a
   tensor first, then indexes that tensor by block ID.

3. **Layers that share a block table want separate buffers.** All 5 layers in
   group 0 share group 0's block table, so when group 0 holds block ID `B`, all
   5 of them write to slot `B`. They cannot write to slot `B` of the *same*
   tensor without colliding. Layers from different groups, on the other hand,
   always hold different block IDs at the same time (the scheduler enforces
   that), so they *can* safely share a tensor. The column axis is the
   orthogonal one to the block table -- exactly the axis that can collapse
   into shared storage.

4. **Padding for uneven groups is cheap.** The
   `if i < len(kv_cache_groups[j].layer_names)` guard lets the last column be
   shared by fewer layers when groups have unequal sizes (e.g., 13 FULL + 12
   SWA padded to `group_size=13`). The tensor is still allocated at full
   `num_blocks` but with shorter `shared_by` -- no per-group resizing needed.

5. **Fewer allocations.** 5 large allocations instead of 30 small ones reduces
   per-allocation overhead in PyTorch's caching allocator. Secondary, but
   consistent with the others.

For the single-group case where layers have different per-layer page sizes
(e.g., `UniformTypeKVCacheSpecs` covering layers with different hidden sizes),
vLLM falls back to per-layer allocation at
[`_get_kv_cache_config_uniform_type`](vllm/v1/core/kv_cache_utils.py). Column
packing is specifically the multi-group hybrid solution.

## Gemma-3-Style Hybrid Grouping

Gemma-3-style models commonly use a repeated pattern like:

```text
SWA, SWA, SWA, SWA, SWA, FULL
```

For `N` repetitions, that gives:

```text
5N SWA layers
N full-attention layers
```

vLLM does not create groups by contiguous layer ranges. It creates groups that
preserve a uniform layer count per group:

```text
SWA group 0: layer 0,  layer 6,  layer 12, ...
SWA group 1: layer 1,  layer 7,  layer 13, ...
SWA group 2: layer 2,  layer 8,  layer 14, ...
SWA group 3: layer 3,  layer 9,  layer 15, ...
SWA group 4: layer 4,  layer 10, layer 16, ...
FULL group:  layer 5,  layer 11, layer 17, ...
```

Each group has `N` layers. The grouping is for memory/block-table layout, not
execution order. Forward execution still runs layers in normal model order:

```text
0, 1, 2, 3, 4, 5, 6, ...
```

## Code Path

### Startup Memory Sizing

The worker profiles available memory for KV cache:

```python
def determine_available_memory(self) -> int:
    ...
    self.model_runner.profile_run()
    ...
    self.available_kv_cache_memory_bytes = (
        self.requested_memory
        - profile_result.non_kv_cache_memory
        - cudagraph_memory_estimate_applied
    )
    return int(self.available_kv_cache_memory_bytes)
```

The engine core collects the worker memory numbers:

```python
available_gpu_memory = self.model_executor.determine_available_memory()
```

Then it builds per-worker KV-cache configs:

```python
kv_cache_configs = get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_gpu_memory
)
```

If workers differ, vLLM clamps every worker to the smallest block count:

```python
min_num_blocks = min(
    kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
)
for kv_cache_config in kv_cache_configs:
    kv_cache_config.num_blocks = min_num_blocks
```

The scheduler receives a single scheduler-side config:

```python
scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
```

After this point, the scheduler does not allocate based on live GPU memory. It
allocates and evicts logical block IDs inside this fixed capacity. Workers have
already allocated physical KV tensors according to their KV-cache config.

### Group Construction

Hybrid grouping is built in
[`_get_kv_cache_groups_uniform_page_size`](vllm/v1/core/kv_cache_utils.py).

The comments describe the repeated-pattern idea:

```python
# A model with 10 full attention layers and 20 sliding window attention layers
# can be regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
# ...
# there are 3 kv_cache_groups, each of which represents 10 layers.
```

The code first groups layers by KV-cache spec:

```python
same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
for layer_name, layer_spec in kv_cache_spec.items():
    same_type_layers[layer_spec].append(layer_name)
```

Then it picks the group size from the smallest attention-type layer count:

```python
min_num_layers = min([len(layers) for layers in same_type_layers.values()])
group_size = min_num_layers
```

For each attention type, it splits the layer list into equal-sized groups:

```python
num_groups = cdiv(len(layers), group_size)
for i in range(num_groups):
    grouped_layers.append(layers[i::num_groups])
```

The strided split is intentional. In pipeline-parallel cases, it avoids creating
groups that are empty on some pipeline stages and therefore require extra
padding.

### Memory Tensor Allocation

The column-packed allocation is described in detail in
[The Structural Mechanism: Shared Column Tensors](#the-structural-mechanism-shared-column-tensors)
and
[Why Column Packing Instead of Per-Group or Per-Layer](#why-column-packing-instead-of-per-group-or-per-layer).
The relevant call site is:

```python
group_size = max(len(group.layer_names) for group in kv_cache_groups)
page_size = get_uniform_page_size(
    [group.kv_cache_spec for group in kv_cache_groups]
)
num_blocks = get_num_blocks(
    vllm_config, group_size, available_memory, page_size
)
for i in range(group_size):
    shared_by = []
    for j in range(len(kv_cache_groups)):
        if i < len(kv_cache_groups[j].layer_names):
            shared_by.append(kv_cache_groups[j].layer_names[i])
    kv_cache_tensors.append(
        KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
    )
```

`num_blocks` is sized as `available_memory / (page_size * group_size)`, which
makes total memory across all `group_size` column tensors equal to
`available_memory`.

The worker allocates these raw tensors and maps each layer name to its storage:

```python
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

Later, each layer's raw tensor is reshaped into the backend-specific KV-cache
shape:

```python
kv_caches[layer_name] = (
    kv_cache_raw_tensors[layer_name]
    .view(dtype)
    .view(kv_cache_shape)
    .permute(*inv_order)
)
```

### Block Tables Per Group

The worker keeps one block table per KV-cache group:

```python
self.num_kv_cache_groups = len(self.block_sizes)
self.block_tables: list[StagedWriteTensor] = []
for i in range(self.num_kv_cache_groups):
    block_table = StagedWriteTensor(
        (self.max_num_reqs, max_num_blocks),
        dtype=torch.int32,
        device=device,
    )
    self.block_tables.append(block_table)
```

When scheduler-allocated block IDs arrive, they are written per group:

```python
for i in range(self.num_kv_cache_groups):
    block_ids = new_block_ids[i]
    self.block_tables[i].stage_write(req_index, start, block_ids)
```

This is why each group has its own table, while the block IDs written into those
tables are still drawn from one global pool:

```text
full group table: [10, 11, 12]
SWA group table:  [13, 14, 15]
```

Those entries point to different fixed-size physical slots.

### Slot Mapping Per Group

For each token position, vLLM computes a slot ID separately for each group:

```python
group_id = tl.program_id(0)
block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
block_size = tl.load(block_sizes + group_id)
...
block_numbers = tl.load(
    block_table_ptr + req_state_idx * block_table_stride + block_indices
)
slot_ids = block_numbers * block_size + block_offsets
```

So `block_id = 10` is resolved through the current group's block table and
current group's block size.

### Attention Metadata Per Group

During input preparation, vLLM starts with metadata for group 0 and then swaps in
the group-specific block table and slot mapping for later groups:

```python
for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
    cm = copy(cm_base)
    if kv_cache_gid > 0:
        cm.block_table_tensor = _get_block_table(kv_cache_gid)
        cm.slot_mapping = slot_mappings[kv_cache_gid]
```

The resulting attention metadata is assigned to all layers in the attention
group — every layer in the group gets a *reference to the same metadata
object*, not a copy:

```python
for layer_name in attn_group.layer_names:
    attn_metadata_dict[layer_name] = attn_metadata_i
```

This is the line that makes group-level metadata sharing concrete. The
per-layer `attn_metadata_dict` lookup returns identical metadata for every
layer in the group, so there's only one `block_table` tensor and one
`slot_mapping` tensor in memory per group.

Finally, each attention backend receives:

- the layer's own KV-cache tensor (per-layer, looked up by `layer_name`),
  and
- the metadata containing that layer's *group's* block table and slot
  mapping (group-shared).

For FlashAttention, for example
([flash_attn.py:752](vllm/v1/attention/backends/flash_attn.py#L752)):

```python
key_cache, value_cache = kv_cache.unbind(0)   # layer-specific
flash_attn_varlen_func(
    k=key_cache, v=value_cache,                # layer-specific
    block_table=block_table,                   # group-shared
    ...
)
```

The kernel does `kv_cache.data_ptr() + slot_id * stride` to compute write
addresses. Different layers in the same group pass different `kv_cache`
arguments → different `data_ptr()` → different physical addresses, even
though `slot_id` is identical.

That is the final lookup shape:

```text
layer_name -> layer KV tensor                              (per-layer)
layer_name -> attention metadata (shared within group)     (per-group via aliasing)
metadata   -> group block table, group slot_mapping        (per-group)
block id   -> physical slot inside that layer/group KV tensor
```

The "per-layer" rows are just dict lookups by `layer_name`. The "per-group"
rows are the same Python object referenced from every layer in the group.
Layer specificity is purely the choice of `kv_cache` tensor passed to the
kernel — the kernel itself is layer-agnostic.

## Summary

- vLLM splits hybrid models into KV-cache groups so each group block has uniform
  physical size.
- KV-cache groups have separate block tables, but their block IDs are allocated
  from one shared `BlockPool`.
- Physical lookup is layer-specific after the group block table selects the
  block ID:

```text
(layer_name, block_id) -> physical K/V memory
```
