# KV Cache Allocation and Group Block IDs

This note explains how vLLM groups KV-cache layers, why groups are shaped to
have uniform physical block sizes, and how the same logical block ID maps to
different physical memory for different KV-cache groups and layers.

## Mental Model

vLLM separates two ideas:

- **Logical block ID**: an integer managed by the scheduler/block pool, such as
  `10`.
- **Physical KV memory**: the actual K/V tensor storage used by a specific
  layer or KV-cache group.

The same logical block ID can appear in multiple KV-cache groups:

```text
(group 0, block 10) -> physical memory for group 0's layers
(group 1, block 10) -> physical memory for group 1's layers
(group 2, block 10) -> physical memory for group 2's layers
```

So `block_id = 10` is not one shared K/V tensor. It is a group-local slot index.

Inside a group, each layer still has its own K/V slice:

```text
group 1, layer 7,  block 10 -> K/V memory for layer 7
group 1, layer 13, block 10 -> K/V memory for layer 13
```

The actual key is effectively:

```text
(layer_name, kv_cache_group_id, block_id)
```

## Why Groups Need Uniform Physical Block Size

vLLM's current allocator wants a simple invariant: one logical block allocation
corresponds to the same amount of KV memory in every KV-cache group.

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

then one block in group 1 would be 5x larger than one block in group 0:

```text
Group 0 block = KV memory for 10 layers
Group 1 block = KV memory for 50 layers
```

That would make allocation and eviction variable-sized. The scheduler could no
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

For this to work, `block 10` must be a valid slot everywhere the request needs
KV cache:

```text
worker 0, group 0, block 10
worker 0, group 1, block 10
worker 1, group 0, block 10
worker 1, group 1, block 10
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

KV-cache tensors are sized using the maximum number of layers in any group:

```python
group_size = max(len(group.layer_names) for group in kv_cache_groups)
page_size = get_uniform_page_size(
    [group.kv_cache_spec for group in kv_cache_groups]
)
num_blocks = get_num_blocks(
    vllm_config, group_size, available_memory, page_size
)
```

Then vLLM creates raw tensors and records which layer names share each tensor:

```python
for i in range(group_size):
    shared_by = []
    for j in range(len(kv_cache_groups)):
        if i < len(kv_cache_groups[j].layer_names):
            shared_by.append(kv_cache_groups[j].layer_names[i])
    kv_cache_tensors.append(
        KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
    )
```

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

This is why the same integer can exist in several group tables:

```text
full group table: [10, 11, 12]
SWA group table:  [10, 11, 12]
```

Those entries point into different group/layer KV storage.

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
group:

```python
for layer_name in attn_group.layer_names:
    attn_metadata_dict[layer_name] = attn_metadata_i
```

Finally, each attention backend receives:

- the layer's own KV-cache tensor, and
- the metadata containing that layer group's block table.

For FlashAttention, for example:

```python
flash_attn_varlen_func(
    k=key_cache,
    v=value_cache,
    ...
    block_table=block_table,
)
```

That is the final lookup shape:

```text
layer_name -> layer KV tensor
layer_name -> attention metadata
metadata   -> group block table
block id   -> physical slot inside that layer/group KV tensor
```

## Summary

- vLLM splits hybrid models into KV-cache groups so each group block has uniform
  physical size.
- A logical block ID like `10` can appear in multiple group block tables.
- The same block ID does not imply shared physical K/V memory.
- Physical lookup is group- and layer-specific:

```text
(layer_name, kv_cache_group_id, block_id) -> physical K/V memory
```
