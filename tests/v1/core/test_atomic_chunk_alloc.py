# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomic multi-chunk allocation and allocator satisfiability (AC-4).

* ``BlockPool.get_new_chunks`` must be all-or-nothing: a batch that cannot be
  fully satisfied leaves the free count and every ref count exactly as before.
* ``can_allocate_chunks`` reports whether a given ``(order, count)`` demand can
  be carved from the current free structure, so admission can reject
  fragmentation-infeasible requests up front.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def test_can_allocate_chunks_counts_splittable_capacity():
    blocks = [KVCacheBlock(i) for i in range(8)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=2, on_evict=None)
    # Fresh pool: two order-2 chunks (starts 0 and 4).
    assert q.can_allocate_chunks(2, 2)
    assert not q.can_allocate_chunks(2, 3)
    # Each order-2 chunk splits into two order-1 chunks -> 4 total.
    assert q.can_allocate_chunks(1, 4)
    assert not q.can_allocate_chunks(1, 5)
    # ... and into eight order-0 base blocks.
    assert q.can_allocate_chunks(0, 8)
    assert not q.can_allocate_chunks(0, 9)
    # Out-of-range order is never satisfiable.
    assert not q.can_allocate_chunks(3, 1)


def test_can_allocate_demands_joint_and_threshold():
    blocks = [KVCacheBlock(i) for i in range(8)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=2, on_evict=None)
    # Fresh: two order-2 chunks = 8 base blocks.
    assert q.can_allocate_demands({2: 2})
    assert not q.can_allocate_demands({2: 3})
    # Mixed orders compete for the same base blocks: one order-2 (4 base) plus
    # 4 order-0 == 8 fits; one more base block does not.
    assert q.can_allocate_demands({2: 1, 0: 4})
    assert not q.can_allocate_demands({2: 1, 0: 5})
    assert q.can_allocate_demands({1: 1, 0: 6})
    assert q.can_allocate_demands({})


def test_can_allocate_demands_rejects_fragmentation():
    blocks = [KVCacheBlock(i) for i in range(8)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=2, on_evict=None)
    held = [q.alloc_chunk(order=0) for _ in range(8)]  # allocate all base blocks
    # Free only even-id blocks; their odd buddies remain allocated so nothing
    # coalesces -> four scattered order-0 free blocks, no order-1 chunk exists.
    for b in held:
        if b.block_id % 2 == 0:
            q.append(b)
    assert q.num_free_blocks == 4
    # Enough raw base blocks for an order-0 demand...
    assert q.can_allocate_demands({0: 4})
    assert not q.can_allocate_demands({0: 5})
    # ...but a single order-1 chunk cannot be formed despite 4 free base
    # blocks: fragmentation is rejected up front.
    assert not q.can_allocate_demands({1: 1})


def test_larger_span_first_avoids_heuristic_allocation_failure():
    """A feasible mixed-span demand can fail if the smaller span is allocated
    first (the uncached-first heuristic splits the only large uncached chunk),
    but succeeds when the larger span is allocated first. This is why the
    coordinator allocates larger-span groups first."""
    from vllm.v1.core.kv_cache_utils import (
        BlockHash,
        make_block_hash_with_group_id,
    )

    def build():
        # Drain the pool to exactly {cached order-0 chunk id0, uncached
        # order-2 chunk 4..7}, with no uncached small chunks.
        blocks = [KVCacheBlock(i) for i in range(8)]
        q = BuddyFreeKVCacheBlockQueue(blocks, max_order=2, on_evict=lambda b: True)
        a = q.alloc_chunk(order=2)  # holds chunk 0..3
        b = q.alloc_chunk(order=2)  # holds chunk 4..7
        q.append(a)  # free 0..3 back (uncached order 2)
        z = q.alloc_chunk(order=0)  # split 0..3 -> id0, leaving id1, id2..3
        z.block_hash = make_block_hash_with_group_id(BlockHash(b"\x09" * 32), 0)
        q.append(z)  # cached order-0 id0 (preserved, not coalesced)
        q.alloc_chunk(order=0)  # hold id1
        q.alloc_chunk(order=1)  # hold id2..3
        q.append(b)  # free 4..7 back (uncached order 2)
        return q

    # Smaller-span first: order-0 splits the only order-2 chunk, order-2 fails.
    q1 = build()
    q1.alloc_chunk(order=0)
    with pytest.raises(ValueError):
        q1.alloc_chunk(order=2)

    # Larger-span first: both succeed (order-0 falls back to evicting cached).
    q2 = build()
    big = q2.alloc_chunk(order=2)
    small = q2.alloc_chunk(order=0)
    assert big.block_id == 4
    assert small.block_id == 0


def test_get_new_chunks_is_atomic_on_failure(monkeypatch):
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


def test_get_new_chunks_success_then_free_roundtrips(monkeypatch):
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
    buddy allocator. The buddy queue previously lacked prepend_n, so this path
    raised AttributeError and killed the engine whenever it was hit (e.g. with
    prefix caching disabled)."""
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    pool = BlockPool(
        num_gpu_blocks=8, enable_caching=True, hash_block_size=1, max_allocation_span=4
    )
    free_before = pool.get_num_free_blocks()
    chunks = pool.get_new_blocks(1, base_span=2)
    assert pool.get_num_free_blocks() == free_before - 2
    # The prepend=True path that used to crash the buddy allocator.
    pool.free_blocks(chunks, prepend=True)
    assert pool.get_num_free_blocks() == free_before
    assert chunks[0].ref_cnt == 0
