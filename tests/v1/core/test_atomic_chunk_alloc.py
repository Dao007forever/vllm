# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomic multi-chunk allocation and allocator satisfiability (AC-4).

* ``BlockPool.get_new_blocks`` must be all-or-nothing: a batch that cannot be
  fully satisfied leaves the free count and every ref count exactly as before.
* ``BlockAllocator.can_allocate`` reports whether a ``{span: count}`` demand can
  be carved from the current free structure, so admission can reject
  fragmentation-infeasible requests up front.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.buddy_free_queue import BuddyAllocator
from vllm.v1.core.eviction_policy import LRUEvictionPolicy
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def _buddy(n, max_order):
    return BuddyAllocator(
        [KVCacheBlock(i) for i in range(n)],
        max_order=max_order,
        evictor=LRUEvictionPolicy(),
    )


def test_can_allocate_counts_splittable_capacity():
    q = _buddy(8, max_order=2)
    # Fresh pool: two order-2 chunks (starts 0 and 4).
    assert q.can_allocate({4: 2})
    assert not q.can_allocate({4: 3})
    # Each order-2 chunk splits into two order-1 chunks -> 4 total.
    assert q.can_allocate({2: 4})
    assert not q.can_allocate({2: 5})
    # ... and into eight order-0 base blocks.
    assert q.can_allocate({1: 8})
    assert not q.can_allocate({1: 9})
    # Out-of-range span is never satisfiable.
    assert not q.can_allocate({8: 1})


def test_can_allocate_joint_and_threshold():
    q = _buddy(8, max_order=2)
    # Fresh: two order-2 chunks = 8 base blocks.
    assert q.can_allocate({4: 2})
    assert not q.can_allocate({4: 3})
    # Mixed spans compete for the same base blocks: one span-4 (4 base) plus
    # 4 span-1 == 8 fits; one more base block does not.
    assert q.can_allocate({4: 1, 1: 4})
    assert not q.can_allocate({4: 1, 1: 5})
    assert q.can_allocate({2: 1, 1: 6})
    assert q.can_allocate({})


def test_can_allocate_rejects_fragmentation():
    q = _buddy(8, max_order=2)
    held = [q.allocate(1) for _ in range(8)]  # allocate all base blocks
    # Free only even-id blocks; their odd buddies remain allocated so nothing
    # coalesces -> four scattered order-0 free blocks, no order-1 chunk exists.
    for b in held:
        if b.block_id % 2 == 0:
            q.free(b)
    assert q.num_free_blocks == 4
    # Enough raw base blocks for a span-1 demand...
    assert q.can_allocate({1: 4})
    assert not q.can_allocate({1: 5})
    # ...but a single span-2 chunk cannot be formed despite 4 free base blocks.
    assert not q.can_allocate({2: 1})


def test_get_new_blocks_is_atomic_on_failure(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    pool = BlockPool(
        num_gpu_blocks=8, enable_caching=True, hash_block_size=1, max_allocation_span=4
    )

    free_before = pool.get_num_free_blocks()
    refs_before = [b.ref_cnt for b in pool.blocks]

    # Request far more span-4 blocks than the pool can ever provide; the
    # batch must fail and roll back completely.
    with pytest.raises((ValueError, RuntimeError)):
        pool.get_new_blocks(num_blocks=100, base_span=4)

    assert pool.get_num_free_blocks() == free_before
    assert [b.ref_cnt for b in pool.blocks] == refs_before


def test_get_new_blocks_success_then_free_roundtrips(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    pool = BlockPool(
        num_gpu_blocks=8, enable_caching=True, hash_block_size=1, max_allocation_span=4
    )

    free_before = pool.get_num_free_blocks()
    chunks = pool.get_new_blocks(1, base_span=2)
    assert len(chunks) == 1
    assert chunks[0].ref_cnt == 1
    # One span-2 block consumes 2 base blocks.
    assert pool.get_num_free_blocks() == free_before - 2
    pool.free_blocks(chunks)
    assert pool.get_num_free_blocks() == free_before
    assert chunks[0].ref_cnt == 0


def test_free_blocks_prepend_roundtrips_in_buddy_mode(monkeypatch):
    """Regression: BlockPool.free_blocks(..., prepend=True) — used by
    remove_skipped_blocks for mamba/SWA scratch recycling — must work under the
    buddy allocator (which has no global reuse ordering and treats prefer_reuse
    as a hint)."""
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    pool = BlockPool(
        num_gpu_blocks=8, enable_caching=True, hash_block_size=1, max_allocation_span=4
    )
    free_before = pool.get_num_free_blocks()
    chunks = pool.get_new_blocks(1, base_span=2)
    assert pool.get_num_free_blocks() == free_before - 2
    pool.free_blocks(chunks, prepend=True)
    assert pool.get_num_free_blocks() == free_before
    assert chunks[0].ref_cnt == 0
