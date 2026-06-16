# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural invariants of BuddyFreeKVCacheBlockQueue (AC-7).

Complements test_buddy_free_queue.py (alignment, split eviction, coalesce-skip,
the two uncached/cached alloc passes) with: free-count accounting, buddy-XOR
coalescing, NULL_BLOCK_ID=0 preservation across alloc/coalesce cycles,
tail-block isolation and fallback, and on_evict timing.
"""

import contextlib

import pytest

from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def _make(n=16, max_order=3, on_evict=None):
    return BuddyFreeKVCacheBlockQueue(
        [KVCacheBlock(i) for i in range(n)], max_order=max_order, on_evict=on_evict
    )


def _free_base_from_chunks(q):
    """Total free base blocks recomputed from the free chunk heads, each of
    size base_span (tail-fallback ids have base_span 1)."""
    return sum(b.base_span for b in q.get_all_free_blocks())


def test_num_free_blocks_matches_chunk_sum_through_ops():
    q = _make(16, 3)
    assert q.num_free_blocks == _free_base_from_chunks(q) == 16
    held = []
    for order in [0, 2, 1, 0, 3]:
        with contextlib.suppress(ValueError):
            held.append(q.alloc_chunk(order=order))
        # Accounting stays consistent after every allocation.
        assert q.num_free_blocks == _free_base_from_chunks(q)
    for b in held:
        q.append(b)
        assert q.num_free_blocks == _free_base_from_chunks(q)
    assert q.num_free_blocks == 16


def test_buddy_xor_coalesce_merges_to_min_id():
    q = _make(8, 3)  # one order-3 chunk covering ids 0..7
    a = q.alloc_chunk(order=2)
    b = q.alloc_chunk(order=2)
    assert a.block_id == 0 and b.block_id == 4  # buddies at order 2
    # Freeing both buddies coalesces back to a single order-3 chunk at min id.
    q.append(a)
    q.append(b)
    frees = q.get_all_free_blocks()
    assert len(frees) == 1
    assert frees[0].block_id == 0
    assert frees[0].base_span == 8
    assert q.num_free_blocks == 8


def test_null_block_id_zero_preserved_across_cycles():
    q = _make(8, 3)
    for _ in range(3):
        first = q.popleft()
        assert first.block_id == 0, "NULL_BLOCK_ID=0 invariant broken"
        q.append(first)  # coalesces back to the full chunk for the next cycle


def test_tail_blocks_isolated_and_drain_last():
    # n=12, max_order=3 -> aligned prefix is one order-3 chunk (0..7); 8..11 tail.
    q = _make(12, 3)
    assert q._tail_free == [8, 9, 10, 11]
    # Order-0 allocations drain the aligned region (ids < 8) first.
    aligned = [q.alloc_chunk(order=0).block_id for _ in range(8)]
    assert all(i < 8 for i in aligned), aligned
    # Only then do allocations come from the isolated tail list.
    tail = [q.alloc_chunk(order=0).block_id for _ in range(4)]
    assert sorted(tail) == [8, 9, 10, 11]


def test_order0_falls_back_to_tail_when_aligned_exhausted():
    q = _make(12, 3)
    big = q.alloc_chunk(order=3)  # consumes the whole aligned region
    assert big.block_id == 0
    b = q.alloc_chunk(order=0)  # must come from the tail
    assert b.block_id in (8, 9, 10, 11)


def test_on_evict_not_called_on_plain_append():
    evicted: list[int] = []

    def on_evict(block):
        evicted.append(block.block_id)
        return True

    q = _make(8, 3, on_evict=on_evict)
    blk = q.alloc_chunk(order=0)
    # A plain append with no cached buddy to coalesce and no split must not
    # trigger any eviction.
    q.append(blk)
    assert evicted == []
