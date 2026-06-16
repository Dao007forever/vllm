# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BuddyAllocator, the variable-span buddy allocator.

Reclaim model: the allocator's free structure only holds truly-free memory.
Cached-but-reusable blocks live in an owned EvictionPolicy; ``allocate``
reclaims LRU candidates (dropping their hashes via ``on_evict``) when short,
``free`` routes a cached block to the policy and an uncached one to the pool
(coalescing), and ``reuse`` pulls a candidate back out on a prefix-cache hit.
"""

import contextlib

import pytest

from vllm.v1.core.buddy_free_queue import BuddyAllocator
from vllm.v1.core.eviction_policy import LRUEvictionPolicy
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def _evict_resetting(evicted: list):
    """on_evict that records the block and resets its hash, mirroring
    BlockPool._maybe_evict_cached_block."""

    def _on_evict(block):
        evicted.append(block.block_id)
        block.reset_hash()
        return True

    return _on_evict


def _make(n=16, max_order=3, evicted=None):
    evictor = LRUEvictionPolicy(
        on_evict=_evict_resetting(evicted) if evicted is not None else None
    )
    blocks = [KVCacheBlock(i) for i in range(n)]
    return BuddyAllocator(blocks, max_order=max_order, evictor=evictor), evictor


def _cache(block):
    """Mark a block as carrying a prefix-cache hash."""
    block.block_hash = ("FAKE", block.block_id)
    return block


def _drain(q):
    """Allocate every base block as span-1 chunks; return them."""
    held = []
    while True:
        try:
            held.append(q.allocate(1))
        except ValueError:
            break
    return held


def test_order_0_only_slab():
    q, _ = _make(8, max_order=0)
    assert q.num_free_blocks == 8
    got = [q.allocate(1) for _ in range(3)]
    assert q.num_free_blocks == 5
    for b in got:
        q.free(b)
    assert q.num_free_blocks == 8


def test_multi_order_alloc_free_roundtrip():
    q, _ = _make(16, max_order=4)
    assert q.max_order == 4
    b1 = q.allocate(4)
    b2 = q.allocate(1)
    b3 = q.allocate(8)
    assert q.num_free_blocks == 16 - (4 + 1 + 8)
    assert b1.block_id % 4 == 0 and b1.base_span == 4
    assert b2.base_span == 1
    assert b3.block_id % 8 == 0 and b3.base_span == 8
    q.free(b2)
    q.free(b1)
    q.free(b3)
    assert q.num_free_blocks == 16


def test_base_span_and_alignment():
    q, _ = _make(16, max_order=4)
    for span in (1, 2, 4, 8):
        h = q.allocate(span)
        assert h.base_span == span
        assert h.block_id % span == 0
        q.free(h)


def test_null_block_id_zero_preserved_across_cycles():
    q, _ = _make(8, max_order=3)
    for _ in range(3):
        first = q.allocate(1)
        assert first.block_id == 0, "NULL_BLOCK_ID=0 invariant broken"
        q.free(first)  # coalesces back for the next cycle


def test_allocated_chunks_dont_overlap():
    q, _ = _make(16, max_order=4)
    allocs: list[tuple[int, int]] = []
    for span in [1, 1, 2, 4, 1, 8]:
        try:
            blk = q.allocate(span)
        except ValueError:
            break
        assert blk.block_id % span == 0
        allocs.append((blk.block_id, span))
    intervals = sorted(allocs)
    for i in range(len(intervals) - 1):
        end_i = intervals[i][0] + intervals[i][1]
        assert end_i <= intervals[i + 1][0], (
            f"overlap between {intervals[i]} and {intervals[i + 1]}"
        )


def test_free_coalesces_uncached_buddies():
    q, _ = _make(16, max_order=4)
    head = q.allocate(16)
    assert q.num_free_blocks == 0
    q.free(head)  # uncached -> pool, coalesces fully
    free = q.get_all_free_blocks()
    assert len(free) == 1 and free[0].base_span == 16


def test_buddy_xor_coalesce_merges_to_min_id():
    q, _ = _make(8, max_order=3)
    a = q.allocate(4)
    b = q.allocate(4)
    assert a.block_id == 0 and b.block_id == 4  # buddies at order 2
    q.free(a)
    q.free(b)
    frees = q.get_all_free_blocks()
    assert len(frees) == 1
    assert frees[0].block_id == 0 and frees[0].base_span == 8
    assert q.num_free_blocks == 8


def test_free_routes_cached_to_evictor_uncached_to_pool():
    q, evictor = _make(8, max_order=2, evicted=[])
    a = q.allocate(1)
    b = q.allocate(1)
    _cache(a)
    q.free(a)  # cached -> evictor candidate
    q.free(b)  # uncached -> pool
    assert evictor.num_reclaimable == 1
    assert a in evictor.reclaimable_chunks()
    # num_free_blocks counts pool free + reclaimable candidates.
    assert q.num_free_blocks == 8


def test_reclaim_prefers_pool_then_evicts_lru_candidate():
    evicted: list[int] = []
    q, evictor = _make(8, max_order=2, evicted=evicted)
    held = _drain(q)
    assert len(held) == 8 and q.num_free_blocks == 0
    # Free four as cached candidates, four back to the pool.
    for b in held[:4]:
        _cache(b)
        q.free(b)
    for b in held[4:]:
        q.free(b)
    assert evictor.num_reclaimable == 4
    assert q.num_free_blocks == 8
    # The four pool blocks satisfy allocations with no eviction.
    [q.allocate(1) for _ in range(4)]
    assert evicted == []
    # The next allocation must reclaim an LRU candidate.
    nxt = q.allocate(1)
    assert len(evicted) == 1
    assert nxt.block_hash is None, "reclaimed memory handed out clean"


def test_reuse_pulls_candidate_back_out():
    q, evictor = _make(8, max_order=2, evicted=[])
    b = q.allocate(1)
    _cache(b)
    q.free(b)
    assert evictor.num_reclaimable == 1
    q.reuse(b)  # prefix-cache hit
    assert evictor.num_reclaimable == 0
    assert b.prev_free_block is None and b.next_free_block is None


def test_oom_raises_when_no_free_and_no_candidates():
    q, _ = _make(8, max_order=3, evicted=[])
    q.allocate(8)  # whole pool, uncached -> caller holds it
    with pytest.raises(ValueError):
        q.allocate(1)


def test_can_allocate_majorization_and_fragmentation():
    q, _ = _make(8, max_order=2, evicted=[])
    # Fresh: two order-2 chunks = 8 base blocks.
    assert q.can_allocate({4: 2})
    assert not q.can_allocate({4: 3})
    assert q.can_allocate({4: 1, 1: 4})
    assert not q.can_allocate({4: 1, 1: 5})
    assert q.can_allocate({2: 1, 1: 6})
    assert q.can_allocate({})

    # Fragment: hold odd ids, free even ids -> four scattered order-0 blocks.
    held = _drain(q)
    for b in held:
        if b.block_id % 2 == 0:
            q.free(b)
    assert q.num_free_blocks == 4
    assert q.can_allocate({1: 4})
    assert not q.can_allocate({1: 5})
    # Four free base blocks but no order-1 chunk can be carved.
    assert not q.can_allocate({2: 1})


def test_can_allocate_counts_reclaimable_candidates():
    # Pool seeded as two order-1 chunks (n=4, max_order=1).
    q, evictor = _make(4, max_order=1, evicted=[])
    held = [q.allocate(2), q.allocate(2)]
    assert q.num_free_blocks == 0
    _cache(held[0])
    q.free(held[0])  # span-2 cached candidate -> evictor
    q.free(held[1])  # span-2 uncached -> pool
    assert evictor.num_reclaimable == 2
    # Pool alone has one span-2 chunk; the reclaimable candidate provides the
    # second, so a two-chunk span-2 demand is feasible (and three is not).
    assert q.can_allocate({2: 2})
    assert not q.can_allocate({2: 3})


def test_tail_blocks_isolated_and_drain_last():
    # n=12, max_order=3 -> aligned prefix one order-3 chunk (0..7); 8..11 tail.
    q, _ = _make(12, max_order=3)
    aligned = [q.allocate(1).block_id for _ in range(8)]
    assert all(i < 8 for i in aligned), aligned
    tail = [q.allocate(1).block_id for _ in range(4)]
    assert sorted(tail) == [8, 9, 10, 11]


def test_order0_falls_back_to_tail_when_aligned_exhausted():
    q, _ = _make(12, max_order=3)
    big = q.allocate(8)  # consumes the whole aligned region
    assert big.block_id == 0
    b = q.allocate(1)  # must come from the tail
    assert b.block_id in (8, 9, 10, 11)


def test_num_free_blocks_matches_chunk_sum_through_ops():
    q, _ = _make(16, max_order=3)

    def pool_base(alloc):
        return sum(b.base_span for b in alloc.get_all_free_blocks())

    assert q.num_free_blocks == pool_base(q) == 16
    held = []
    for span in [1, 4, 2, 1, 8]:
        with contextlib.suppress(ValueError):
            held.append(q.allocate(span))
        assert q.num_free_blocks == pool_base(q)
    for b in held:
        q.free(b)
        assert q.num_free_blocks == pool_base(q)
    assert q.num_free_blocks == 16
