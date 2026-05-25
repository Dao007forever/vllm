# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the standalone BuddyBlockPool."""

import pytest

from vllm.v1.core.buddy_block_pool import (
    BuddyAllocationError,
    BuddyBlockPool,
)


def test_init_validates_size() -> None:
    BuddyBlockPool(num_base_blocks=16, max_order=4)  # ok
    with pytest.raises(ValueError):
        BuddyBlockPool(num_base_blocks=0, max_order=2)
    with pytest.raises(ValueError):
        BuddyBlockPool(num_base_blocks=16, max_order=-1)
    with pytest.raises(ValueError):
        # 17 is not a multiple of 2**3 = 8.
        BuddyBlockPool(num_base_blocks=17, max_order=3)


def test_initial_pool_is_all_max_order_chunks() -> None:
    pool = BuddyBlockPool(num_base_blocks=32, max_order=3)  # 4 chunks of 8
    assert pool.free_chunks_per_order() == [0, 0, 0, 4]
    assert pool.free_base_blocks() == 32
    assert pool.num_allocated_chunks() == 0


def test_allocate_at_max_order() -> None:
    pool = BuddyBlockPool(num_base_blocks=32, max_order=3)
    b = pool.allocate(3)
    # No splits — direct pop from max order.
    assert pool.free_chunks_per_order() == [0, 0, 0, 3]
    assert pool.num_allocated_chunks() == 1
    assert b % 8 == 0


def test_allocate_smaller_splits_chain() -> None:
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)  # one chunk of 8
    b0 = pool.allocate(0)  # split: 8 -> 4 + (2 + (1 + (1)))
    # Asking for order 0 from a max-order pool requires log2(8) = 3 splits.
    # After: free_chunks_per_order = [0, 1, 1, 0] (buddies at orders 0,1,2).
    # Wait — we keep one half at each lower order and split the other half
    # all the way down. So the splits produce buddies at orders 2, 1, 0.
    # Allocated order = 0. Free at orders 2, 1, 0.
    assert pool.free_chunks_per_order() == [1, 1, 1, 0]
    assert pool.num_allocated_chunks() == 1
    assert b0 == 0  # smallest id wins through the chain


def test_free_coalesces_buddies() -> None:
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)
    a = pool.allocate(0)  # 0
    b = pool.allocate(0)  # 1, buddy of a at order 0
    # State: allocated {0,1}; free chunks at orders [0]=0, [1]=1 (chunk at 2),
    # [2]=1 (chunk at 4). After we free a and b they should coalesce back up
    # to a single chunk of order 3.
    pool.free(a)
    pool.free(b)
    assert pool.free_chunks_per_order() == [0, 0, 0, 1]
    assert pool.free_base_blocks() == 8


def test_free_without_coalesce() -> None:
    # Buddy is allocated → freeing doesn't coalesce.
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)
    a = pool.allocate(0)
    b = pool.allocate(0)
    pool.free(a)
    # b is still allocated, so a's order-0 buddy is not free → no coalesce.
    assert 0 in [a, b]
    assert pool.num_allocated_chunks() == 1
    assert pool.free_chunks_per_order()[0] == 1


def test_allocate_out_of_range() -> None:
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)
    with pytest.raises(ValueError):
        pool.allocate(-1)
    with pytest.raises(ValueError):
        pool.allocate(4)


def test_oom_when_pool_exhausted() -> None:
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)
    pool.allocate(3)  # one chunk of 8 — pool empty
    with pytest.raises(BuddyAllocationError):
        pool.allocate(0)


def test_free_invalid() -> None:
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)
    with pytest.raises(ValueError):
        pool.free(0)  # nothing allocated
    a = pool.allocate(2)
    pool.free(a)
    with pytest.raises(ValueError):
        pool.free(a)  # double-free


def test_reuse_after_free() -> None:
    pool = BuddyBlockPool(num_base_blocks=8, max_order=3)
    a = pool.allocate(1)
    pool.free(a)
    b = pool.allocate(1)
    # The freed chunk should be reused (after coalescing back up to order 3,
    # then re-splitting to order 1).
    assert pool.free_base_blocks() == 8 - 2
    pool.free(b)
    assert pool.free_base_blocks() == 8


def test_mixed_orders_stress() -> None:
    pool = BuddyBlockPool(num_base_blocks=64, max_order=6)  # one chunk of 64
    chunks = []
    # Allocate a mix.
    for order in [0, 0, 1, 1, 2, 2, 3, 3, 4]:  # 1+1+2+2+4+4+8+8+16 = 46
        chunks.append((pool.allocate(order), order))
    assert pool.free_base_blocks() == 64 - 46
    assert pool.num_allocated_chunks() == 9

    # Free everything in a non-trivial order.
    import random
    rng = random.Random(0)
    rng.shuffle(chunks)
    for cid, _ in chunks:
        pool.free(cid)
    assert pool.free_base_blocks() == 64
    assert pool.num_allocated_chunks() == 0
    # Coalescing should restore the pool to a single max-order chunk.
    assert pool.free_chunks_per_order() == [0, 0, 0, 0, 0, 0, 1]


def test_adapter_order_0_only() -> None:
    """The BlockPool-facing adapter degenerates to slab when max_order=0."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    n = 8
    blocks = [KVCacheBlock(i) for i in range(n)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=0)
    assert q.num_free_blocks == n
    popped = q.popleft_n(3)
    assert q.num_free_blocks == n - 3
    q.append_n(popped)
    assert q.num_free_blocks == n


def test_adapter_multi_order_chunks() -> None:
    """alloc_chunk / free_chunk roundtrip at higher orders."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    n = 16
    blocks = [KVCacheBlock(i) for i in range(n)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    assert q.max_order == 4

    b1 = q.alloc_chunk(order=2)  # 4 base blocks
    b2 = q.alloc_chunk(order=0)  # 1 base block
    b3 = q.alloc_chunk(order=3)  # 8 base blocks
    assert q.num_free_blocks == n - (4 + 1 + 8)
    # All starts must be order-aligned and chunk_order recorded on head.
    assert b1.block_id % 4 == 0 and b1.chunk_order == 2
    assert b2.chunk_order == 0
    assert b3.block_id % 8 == 0 and b3.chunk_order == 3

    q.append(b2)
    q.append(b1)
    q.append(b3)
    assert q.num_free_blocks == n


def test_adapter_mixed_order_and_order_0() -> None:
    """popleft (legacy order-0 path) coexists with alloc_chunk."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    n = 16
    blocks = [KVCacheBlock(i) for i in range(n)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    # Allocate a few singles, a large chunk, more singles. Free in mixed order.
    singles = q.popleft_n(3)
    large = q.alloc_chunk(order=3)  # 8 blocks
    more_singles = q.popleft_n(2)
    assert q.num_free_blocks == n - 3 - 8 - 2

    q.append_n(singles)
    q.append(large)
    q.append_n(more_singles)
    assert q.num_free_blocks == n


def test_adapter_oom_falls_back_via_value_error() -> None:
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    n = 8
    blocks = [KVCacheBlock(i) for i in range(n)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=3)
    q.alloc_chunk(order=3)  # uses the whole pool
    with pytest.raises(ValueError):
        q.alloc_chunk(order=0)


def test_chunk_order_set_on_alloc() -> None:
    """alloc_chunk records chunk_order on the returned head block."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    for order in (0, 1, 2, 3):
        h = q.alloc_chunk(order=order)
        assert h.chunk_order == order
        assert h.block_id % (1 << order) == 0
        q.append(h)


def test_remove_uses_block_chunk_order() -> None:
    """remove(block) reads block.chunk_order — no order parameter."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)

    # Allocate then free a chunk so it sits in its order's LRU.
    head = q.alloc_chunk(order=2)
    q.append(head)
    # After the free, head is back in a free list at some order >= 2
    # (eager coalesce may have promoted it). Read the order off the block.
    assert head.chunk_order >= 2
    free_before = q.num_free_blocks
    q.remove(head)
    assert q.num_free_blocks == free_before - (1 << head.chunk_order)
    # Re-appending should restore the count.
    q.append(head)
    assert q.num_free_blocks == free_before


def test_remove_of_mid_lru_is_o1() -> None:
    """remove unlinks a block in the middle of an order's LRU correctly."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    # max_order=0 so each freed block lives independently in order-0 LRU.
    blocks = [KVCacheBlock(i) for i in range(8)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=0)
    # Pop several singles, then free in a known order to seed the LRU.
    a, b, c, d = q.popleft_n(4)
    for blk in (a, b, c, d):
        q.append(blk)
    # Free list head→tail order is now: remaining initial blocks, then a,b,c,d
    # in append order. Remove the middle (c).
    free_before = q.num_free_blocks
    q.remove(c)
    assert q.num_free_blocks == free_before - 1
    # c is no longer in any list.
    assert c.prev_free_block is None and c.next_free_block is None
    # The list is still well-formed (no dangling).
    visible = q.get_all_free_blocks()
    assert c not in visible
    # popping all remaining yields exactly free_before - 1 blocks.
    assert len(q.popleft_n(q.num_free_blocks)) == free_before - 1


def test_popleft_auto_splits_from_higher_order() -> None:
    """popleft with no order-0 chunks must split the next-larger LRU head."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    # max_order=2 means the pool is seeded with 4 order-2 chunks (16/4).
    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=2)
    # Initially order-0 LRU is empty. popleft must split an order-2 chunk
    # into two order-1 buddies, then one of those into two order-0 buddies.
    b = q.popleft()
    assert b.chunk_order == 0
    # Sibling at order 0 and buddy at order 1 should now be in their LRUs.
    free_per_order = [0, 0, 0]
    for blk in q.get_all_free_blocks():
        free_per_order[blk.chunk_order] += 1
    assert free_per_order[0] >= 1, "split should leave a sibling at order 0"
    assert free_per_order[1] >= 1, "split should leave a buddy at order 1"


def test_append_coalesces_with_free_buddy() -> None:
    """append eagerly merges with a free buddy at the same order."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    # Alloc two order-0 buddies; freeing both should coalesce all the way
    # back to a single order-4 chunk.
    head4 = q.alloc_chunk(order=4)
    assert q.num_free_blocks == 0
    q.append(head4)
    free = q.get_all_free_blocks()
    assert len(free) == 1 and free[0].chunk_order == 4


def test_on_evict_fires_when_splitting_cached_chunk() -> None:
    """alloc that walks up and splits a cached parent invokes on_evict."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    evicted: list[KVCacheBlock] = []
    q = BuddyFreeKVCacheBlockQueue(
        blocks, max_order=4, on_evict=lambda b: (evicted.append(b), True)[1]
    )
    # Simulate a cached order-2 chunk sitting in the LRU: alloc + append.
    head = q.alloc_chunk(order=2)
    # Pretend the BlockPool registered a hash for it.
    head._block_hash = "FAKE_HASH_FOR_HEAD"  # noqa: SLF001
    q.append(head)
    # head re-entered the LRU at some order >= 2 with its hash intact.
    assert head.block_hash is not None
    assert head in q.get_all_free_blocks()
    evicted.clear()

    # Now ask for order 0. The only free chunk is at head's order; the
    # queue must walk up, pop head, evict its hash, then split down.
    leaf = q.alloc_chunk(order=0)
    assert leaf.chunk_order == 0
    assert evicted == [head], (
        f"expected on_evict to fire once for head, got {evicted}"
    )


def test_on_evict_fires_for_both_contributors_on_coalesce() -> None:
    """append that coalesces two cached siblings evicts both hashes."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    evicted: list[KVCacheBlock] = []
    q = BuddyFreeKVCacheBlockQueue(
        blocks, max_order=4, on_evict=lambda b: (evicted.append(b), True)[1]
    )
    # Two order-2 buddies: chunks at start ids 0 and 4 (0 ^ 4 = 4 → buddies).
    a = q.alloc_chunk(order=2)
    b = q.alloc_chunk(order=2)
    # They should be buddies at order 2 (covering [0..3] and [4..7]).
    assert {a.block_id, b.block_id} == {0, 4}
    a._block_hash = "HASH_A"  # noqa: SLF001
    b._block_hash = "HASH_B"  # noqa: SLF001
    # Free a first — sits in order-2 LRU with its hash. No coalesce yet
    # because b is still allocated.
    q.append(a)
    assert evicted == [], "no coalesce → no eviction"
    assert a.block_hash == "HASH_A"
    # Free b — buddy of a at order 2. Coalesce fires; both hashes evicted.
    q.append(b)
    evicted_ids = {blk.block_id for blk in evicted}
    assert evicted_ids == {a.block_id, b.block_id}, (
        f"expected both contributors evicted on coalesce, got {evicted}"
    )


def test_on_evict_skipped_when_block_has_no_hash() -> None:
    """on_evict is not called when the affected chunk has no hash."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    calls = 0

    def on_evict(_b):
        nonlocal calls
        calls += 1
        return True

    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4, on_evict=on_evict)
    # Pure alloc/free of an order-2 chunk that was never cached.
    h = q.alloc_chunk(order=2)
    q.append(h)
    assert calls == 0, "alloc+free without caching shouldn't evict anything"
    # Force a split of an uncached chunk — also no eviction.
    _ = q.alloc_chunk(order=0)
    assert calls == 0


def test_remove_then_reappend_preserves_chunk_order() -> None:
    """Round-trip: remove → caller holds chunk → append returns it intact."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    head = q.alloc_chunk(order=2)
    q.append(head)  # back in some order's LRU
    saved_order = head.chunk_order
    q.remove(head)
    # chunk_order remains set on the block while caller holds it.
    assert head.chunk_order == saved_order
    q.append(head)
    # And the queue still accounts for it correctly.
    assert q.num_free_blocks == 16


def test_allocated_chunks_dont_overlap() -> None:
    pool = BuddyBlockPool(num_base_blocks=16, max_order=4)
    allocs: list[tuple[int, int]] = []
    for order in [0, 0, 1, 2, 0, 3]:  # 1+1+2+4+1+8 = 17 > 16 — last fails
        try:
            start = pool.allocate(order)
            allocs.append((start, 1 << order))
        except BuddyAllocationError:
            break
    # Verify no two allocated chunks overlap.
    intervals = sorted(allocs)
    for i in range(len(intervals) - 1):
        end_i = intervals[i][0] + intervals[i][1]
        assert end_i <= intervals[i + 1][0], (
            f"overlap between {intervals[i]} and {intervals[i + 1]}"
        )
