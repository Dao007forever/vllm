# Triton Copy Blocks Kernel

This note explains the Triton kernel in:

https://github.com/ivanium/vllm/blob/48c3777798fa3d4eb0599865cae21c0002075e22/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/copy_ops.py#L41

The kernel copies KV-cache blocks between two sets of cache tensors. It does
not compute attention. It is a batched memory-copy kernel that knows how to copy
many `(source block id, destination block id)` pairs across all layers in one
launch.

## Big Picture

The Python caller has:

- `src_caches`: one source KV-cache tensor per layer.
- `dst_caches`: one destination KV-cache tensor per layer.
- `block_mapping`: an `[N, 2]` tensor of block-id pairs.

Each row of `block_mapping` means:

```text
copy source block X into destination block Y
```

The Triton kernel repeats that copy for every layer. If there are `N` block
pairs and `L` layers, the kernel has:

```text
total_jobs = N * L
```

Each job is one logical copy:

```text
(one layer, one source block, one destination block)
```

## CUDA Mental Model

Triton programs are similar to CUDA thread blocks. This line:

```python
pid = tl.program_id(0)
```

is roughly like:

```cuda
int pid = blockIdx.x;
```

And this line:

```python
num_ctas = tl.num_programs(0)
```

is roughly like:

```cuda
int num_ctas = gridDim.x;
```

The kernel then uses a grid-stride loop:

```python
job_id = pid
while job_id < total_jobs:
    ...
    job_id += num_ctas
```

That is the same idea as CUDA code where each block processes work item
`blockIdx.x`, then `blockIdx.x + gridDim.x`, then
`blockIdx.x + 2 * gridDim.x`, and so on.

## Why The Grid Is Small

The caller launches:

```python
grid_size = min(launch_params.num_sms, total_jobs)
```

`DEFAULT_COPY_NUM_SMS` is 16, so this copy kernel intentionally uses only a
small number of Triton programs. The goal is to avoid occupying the whole GPU
with background copy work. Other compute kernels can then run concurrently on
the remaining SMs.

This is useful for KV-cache offloading because copying KV blocks is important,
but it should not completely crowd out model execution.

## Mapping Job IDs To Work

Inside the loop, the kernel maps a flat `job_id` to:

```python
pair_id = job_id % num_pairs
layer_id = job_id // num_pairs
```

For example, if `num_pairs = 3` and `num_layers = 2`, the jobs are:

```text
job 0 -> layer 0, pair 0
job 1 -> layer 0, pair 1
job 2 -> layer 0, pair 2
job 3 -> layer 1, pair 0
job 4 -> layer 1, pair 1
job 5 -> layer 1, pair 2
```

So the kernel copies every requested block pair for layer 0, then every
requested block pair for layer 1, and so on.

## Reading The Block Mapping

`block_mapping` is flattened before being passed to the kernel:

```text
[
  src0, dst0,
  src1, dst1,
  src2, dst2,
  ...
]
```

The kernel reads one pair with:

```python
src_block = tl.load(mapping_ptr + pair_id * 2).to(tl.int64)
dst_block = tl.load(mapping_ptr + pair_id * 2 + 1).to(tl.int64)
```

If `block_mapping[pair_id]` is `[7, 42]`, the current job copies source block 7
to destination block 42 for the current layer.

## Pointer Tables

Instead of passing one tensor pointer per layer as separate kernel arguments,
the caller builds pointer tables:

```python
src_ptr_table = torch.tensor(
    [t.data_ptr() for t in src_tensors], device="cuda", dtype=torch.uint64
)
dst_ptr_table = torch.tensor(
    [t.data_ptr() for t in dst_tensors], device="cuda", dtype=torch.uint64
)
```

So `src_ptrs[layer_id]` is the base address for the source cache tensor for
that layer, and `dst_ptrs[layer_id]` is the base address for the destination
cache tensor for that layer.

The kernel loads those addresses:

```python
src_base = tl.load(src_ptrs + layer_id).to(tl.pointer_type(tl.int64))
dst_base = tl.load(dst_ptrs + layer_id).to(tl.pointer_type(tl.int64))
```

The cast to `tl.pointer_type(tl.int64)` means the kernel copies the cache as
8-byte words. It is treating the tensor memory as raw 64-bit chunks, not as the
original tensor dtype.

## Words Per Block

The caller also builds:

```python
wpb_table = torch.tensor(wpb_list, device="cuda", dtype=torch.int64)
```

`wpb` means "words per block", where one word is an `int64`, or 8 bytes.

The caller computes each layer's value as:

```python
src_wpb = src_t.stride(0) * src_t.element_size() // 8
```

`stride(0)` is how many elements you move to get from one block to the next in
the tensor's first dimension. Multiplying by `element_size()` converts that to
bytes. Dividing by 8 converts bytes to 64-bit words.

The kernel loads the current layer's value:

```python
wpb = tl.load(wpb_ptr + layer_id)
```

Then it computes the starting word offset of each block:

```python
src_off = src_block * wpb
dst_off = dst_block * wpb
```

So:

```text
src_base + src_off
```

points to the beginning of the source KV block, and:

```text
dst_base + dst_off
```

points to the beginning of the destination KV block.

## Vectorized Copy

This is the core copy:

```python
offsets = tl.arange(0, BLOCK_SIZE)
for start in range(0, max_words_per_block, BLOCK_SIZE):
    idx = start + offsets
    mask = idx < wpb
    data = tl.load(src_base + src_off + idx, mask=mask, other=0)
    tl.store(dst_base + dst_off + idx, data, mask=mask)
```

`tl.arange(0, BLOCK_SIZE)` creates a vector:

```text
[0, 1, 2, ..., BLOCK_SIZE - 1]
```

So `idx` is a vector of word offsets. A single `tl.load` loads many contiguous
64-bit words, and a single `tl.store` stores them.

This is one of the biggest syntax differences from CUDA. In CUDA, you often
write code where each thread handles one element. In Triton, you usually write
vector operations directly. Triton then lowers those vector operations to GPU
instructions.

## Why The Mask Exists

Different layers may have different block sizes. The loop upper bound is
`max_words_per_block`, but the current layer may have a smaller `wpb`.

The mask:

```python
mask = idx < wpb
```

prevents out-of-bounds loads and stores for smaller layers.

For example, if:

```text
BLOCK_SIZE = 1024
wpb = 700
```

then offsets `0..699` are valid and offsets `700..1023` are ignored.

The load uses:

```python
other=0
```

for masked-out lanes, but those lanes are not stored because the store uses the
same mask.

## Compile-Time Constants

These kernel arguments are marked as `tl.constexpr`:

```python
max_words_per_block: tl.constexpr
BLOCK_SIZE: tl.constexpr
```

That means Triton knows their values at compile time and can specialize the
generated kernel.

The caller picks the copy vector size with:

```python
block_size = min(triton.next_power_of_2(words_per_block), 1024)
num_warps = min(max(block_size // 32, 1), 32)
```

So the vector width is a power of two up to 1024 words. Since each word is
8 bytes, the largest vector copy chunk is:

```text
1024 * 8 = 8192 bytes
```

## CUDA-Style Pseudocode

This is not exactly what Triton emits, but it captures the logic:

```cuda
__global__ void copy_blocks_kernel(...) {
    int pid = blockIdx.x;
    int num_ctas = gridDim.x;

    for (int job_id = pid; job_id < total_jobs; job_id += num_ctas) {
        int pair_id = job_id % num_pairs;
        int layer_id = job_id / num_pairs;

        int64_t src_block = mapping[pair_id * 2];
        int64_t dst_block = mapping[pair_id * 2 + 1];

        int64_t* src_base = (int64_t*)src_ptrs[layer_id];
        int64_t* dst_base = (int64_t*)dst_ptrs[layer_id];

        int64_t wpb = words_per_block[layer_id];

        int64_t src_off = src_block * wpb;
        int64_t dst_off = dst_block * wpb;

        for (int i = 0; i < wpb; i++) {
            dst_base[dst_off + i] = src_base[src_off + i];
        }
    }
}
```

The real Triton version performs the inner copy as vectorized masked
load/store operations instead of a scalar `for (int i = 0; i < wpb; i++)`
loop.

## Summary

The kernel is a batched KV-cache block copy:

- One logical job copies one block pair for one layer.
- A grid-stride loop lets a small number of Triton programs process many jobs.
- Pointer tables let the kernel handle all layers in one launch.
- `wpb` tells the kernel how many 64-bit words belong to each layer's block.
- `tl.arange`, `tl.load`, and `tl.store` express vectorized memory movement.
- Masks keep loads and stores valid when layers have different block sizes.

If you can read CUDA, the main Triton shift is that a program often describes a
whole vector of element work at once, instead of describing one CUDA thread's
single element.
