# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural invariants of BuddyAllocator (AC-7).

Free-count accounting, buddy-XOR coalescing, NULL_BLOCK_ID=0 preservation
across alloc/coalesce cycles, tail-block isolation/fallback, and the reclaim
boundary (no eviction while uncached free memory remains).
"""

import contextlib

import pytest

from vllm.v1.core.buddy_free_queue import BuddyAllocator
from vllm.v1.core.eviction_policy import LRUEvictionPolicy
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def _make(n=16, max_order=3, on_evict=None):
    evictor = LRUEvictionPolicy(on_evict=on_evict)
    return BuddyAllocator(
        [KVCacheBlock(i) for i in range(n)], max_order=max_order, evictor=evictor
    )


def _pool_base(q):
    """Total free base blocks recomputed from the pool chunk heads."""
    return sum(b.base_span for b in q.get_all_free_blocks())


def test_num_free_blocks_matches_chunk_sum_through_ops():
    q = _make(16, 3)
    assert q.num_free_blocks == _pool_base(q) == 16
    held = []
    for span in [1, 4, 2, 1, 8]:
        with contextlib.suppress(ValueError):
            held.append(q.allocate(span))
        assert q.num_free_blocks == _pool_base(q)
    for b in held:
        q.free(b)
        assert q.num_free_blocks == _pool_base(q)
    assert q.num_free_blocks == 16


def test_buddy_xor_coalesce_merges_to_min_id():
    q = _make(8, 3)  # one order-3 chunk covering ids 0..7
    a = q.allocate(4)
    b = q.allocate(4)
    assert a.block_id == 0 and b.block_id == 4  # buddies at order 2
    q.free(a)
    q.free(b)
    frees = q.get_all_free_blocks()
    assert len(frees) == 1
    assert frees[0].block_id == 0
    assert frees[0].base_span == 8
    assert q.num_free_blocks == 8


def test_null_block_id_zero_preserved_across_cycles():
    q = _make(8, 3)
    for _ in range(3):
        first = q.allocate(1)
        assert first.block_id == 0, "NULL_BLOCK_ID=0 invariant broken"
        q.free(first)  # coalesces back to the full chunk for the next cycle


def test_tail_blocks_isolated_and_drain_last():
    # n=12, max_order=3 -> aligned prefix is one order-3 chunk (0..7); 8..11 tail.
    q = _make(12, 3)
    aligned = [q.allocate(1).block_id for _ in range(8)]
    assert all(i < 8 for i in aligned), aligned
    tail = [q.allocate(1).block_id for _ in range(4)]
    assert sorted(tail) == [8, 9, 10, 11]


def test_order0_falls_back_to_tail_when_aligned_exhausted():
    q = _make(12, 3)
    big = q.allocate(8)  # consumes the whole aligned region
    assert big.block_id == 0
    b = q.allocate(1)  # must come from the tail
    assert b.block_id in (8, 9, 10, 11)


def test_no_eviction_while_uncached_free_remains():
    evicted: list[int] = []

    def on_evict(block):
        evicted.append(block.block_id)
        block.reset_hash()
        return True

    q = _make(8, 3, on_evict=on_evict)
    # A plain uncached alloc/free cycle must never reclaim a candidate.
    blk = q.allocate(1)
    q.free(blk)
    assert evicted == []
    # Even allocating the whole pool (uncached) triggers no eviction.
    held = []
    with contextlib.suppress(ValueError):
        while True:
            held.append(q.allocate(1))
    assert evicted == []
