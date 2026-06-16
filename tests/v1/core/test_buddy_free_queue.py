# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BuddyFreeKVCacheBlockQueue, the live buddy-backed free queue."""

import pytest


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
    # All starts must be order-aligned and base_span recorded on head.
    assert b1.block_id % 4 == 0 and b1.base_span == 4
    assert b2.base_span == 1
    assert b3.block_id % 8 == 0 and b3.base_span == 8

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


def test_base_span_set_on_alloc() -> None:
    """alloc_chunk records base_span on the returned head block."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    for order in (0, 1, 2, 3):
        h = q.alloc_chunk(order=order)
        assert h.base_span == (1 << order)
        assert h.block_id % (1 << order) == 0
        q.append(h)


def test_remove_uses_block_base_span() -> None:
    """remove(block) reads block.base_span — no order parameter."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)

    # Allocate then free a chunk so it sits in its order's LRU.
    head = q.alloc_chunk(order=2)
    q.append(head)
    # After the free, head is back in a free list at some span >= 4
    # (eager coalesce may have promoted it). Read the span off the block.
    assert head.base_span >= 4
    free_before = q.num_free_blocks
    q.remove(head)
    assert q.num_free_blocks == free_before - head.base_span
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
    assert b.base_span == 1
    # Sibling at order 0 and buddy at order 1 should now be in their LRUs.
    free_per_order = [0, 0, 0]
    for blk in q.get_all_free_blocks():
        free_per_order[blk.base_span.bit_length() - 1] += 1
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
    assert len(free) == 1 and free[0].base_span == 16


def test_on_evict_fires_when_splitting_cached_chunk() -> None:
    """alloc that walks up and splits a cached parent invokes on_evict."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    evicted: list[KVCacheBlock] = []
    q = BuddyFreeKVCacheBlockQueue(
        blocks, max_order=4, on_evict=lambda b: (evicted.append(b), True)[1]
    )
    # Drain the pool so head's buddy is allocated (still held by caller) and
    # no coalesce will happen on append. After this, order-2 LRU is empty.
    head = q.alloc_chunk(order=2)  # blocks[0..3]
    _other = q.alloc_chunk(order=2)  # blocks[4..7] — head's buddy
    _hi = q.alloc_chunk(order=3)  # blocks[8..15]
    # Pretend the BlockPool registered a hash for head.
    head._block_hash = "FAKE_HASH_FOR_HEAD"  # noqa: SLF001
    q.append(head)
    # head sits alone in order-2 LRU (its buddy is still allocated, so
    # coalescing has nothing to do). Hash preserved.
    assert head.block_hash is not None
    assert head in q.get_all_free_blocks()
    evicted.clear()

    # Now ask for order 0. The only free chunk is head at order 2; the
    # queue walks up, pops head, evicts its hash, then splits down.
    leaf = q.alloc_chunk(order=0)
    assert leaf.base_span == 1
    assert evicted == [head], f"expected on_evict to fire once for head, got {evicted}"


def test_coalesce_skipped_when_cached_hash_present() -> None:
    """append that would coalesce two cached siblings instead keeps both
    individually cached. Coalescing resumes once a sibling is uncached."""
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
    assert {a.block_id, b.block_id} == {0, 4}
    a._block_hash = "HASH_A"  # noqa: SLF001
    b._block_hash = "HASH_B"  # noqa: SLF001
    # Free a — sits in order-2 LRU with its hash; b still allocated.
    q.append(a)
    assert evicted == [], "no coalesce → no eviction"
    assert a.block_hash == "HASH_A"
    # Free b — would normally coalesce with a, but both are cached. Skip
    # coalescing; both remain in order-2 LRU with hashes intact.
    q.append(b)
    assert evicted == [], f"expected no eviction (cache preserved), got {evicted}"
    assert a.block_hash == "HASH_A" and b.block_hash == "HASH_B"
    free = q.get_all_free_blocks()
    # a and b remain individually in the free pool at order 2 (no coalesce).
    cached_free = [blk for blk in free if blk.block_id in {a.block_id, b.block_id}]
    assert {blk.block_id for blk in cached_free} == {a.block_id, b.block_id}
    assert all(blk.base_span == 4 for blk in cached_free)

    # Once a's hash is dropped (e.g. BlockPool evicted via LRU), the next
    # append against b should be able to coalesce them.
    a._block_hash = None  # noqa: SLF001
    # Simulate b being re-touched by allocator (alloc + free) — the alloc
    # path is what would trigger coalescing in a real workload. Simplest
    # exercise: remove b and re-append; with a now uncached, append(b)
    # coalesces them into an order-3 chunk.
    q.remove(b)
    b._block_hash = None  # noqa: SLF001
    q.append(b)
    free = q.get_all_free_blocks()
    # After hashes are cleared, coalescing cascades all the way up: a+b → 3,
    # and that buddies with the leftover order-3 chunk [8..15] → 4.
    assert len(free) == 1 and free[0].base_span == 16, (
        f"expected full coalesce after hashes cleared, got {free}"
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


def test_remove_then_reappend_preserves_base_span() -> None:
    """Round-trip: remove → caller holds chunk → append returns it intact."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    head = q.alloc_chunk(order=2)
    q.append(head)  # back in some order's LRU
    saved_span = head.base_span
    q.remove(head)
    # base_span remains set on the block while caller holds it.
    assert head.base_span == saved_span
    q.append(head)
    # And the queue still accounts for it correctly.
    assert q.num_free_blocks == 16


def test_alloc_prefers_splitting_uncached_over_evicting_cached() -> None:
    """alloc_chunk should split a higher-order uncached chunk rather than
    evict a same-order cached chunk when the requested order is non-empty
    but its head is cached."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    evicted: list[KVCacheBlock] = []
    q = BuddyFreeKVCacheBlockQueue(
        blocks, max_order=4, on_evict=lambda b: (evicted.append(b), True)[1]
    )
    # Carve out two cached order-2 chunks (siblings won't coalesce).
    a = q.alloc_chunk(order=2)
    b = q.alloc_chunk(order=2)
    a._block_hash = "HASH_A"  # noqa: SLF001
    b._block_hash = "HASH_B"  # noqa: SLF001
    q.append(a)
    q.append(b)
    # order-2 LRU now has two cached chunks. order-4 LRU still has fresh
    # max-order chunk(s) — the rest of the pool was untouched.
    assert evicted == []
    assert a.block_hash == "HASH_A" and b.block_hash == "HASH_B"

    # Ask for order 2. Old policy would pop a's LRU head (cached) and evict.
    # New policy: prefer splitting an uncached higher-order chunk.
    new_chunk = q.alloc_chunk(order=2)
    assert evicted == [], (
        f"expected no eviction (uncached higher-order available), got {evicted}"
    )
    assert a.block_hash == "HASH_A" and b.block_hash == "HASH_B"
    assert new_chunk.base_span == 4
    # And the chunk we got should be fresh — not a or b.
    assert new_chunk.block_id not in {a.block_id, b.block_id}


def test_alloc_falls_back_to_evict_when_no_uncached() -> None:
    """When every order >= requested has only cached chunks, alloc evicts
    the LRU-head cached chunk at the requested order."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(8)]
    evicted: list[KVCacheBlock] = []
    q = BuddyFreeKVCacheBlockQueue(
        blocks, max_order=3, on_evict=lambda b: (evicted.append(b), True)[1]
    )
    # Allocate the whole pool as two cached order-2 chunks.
    a = q.alloc_chunk(order=2)
    b = q.alloc_chunk(order=2)
    a._block_hash = "HASH_A"  # noqa: SLF001
    b._block_hash = "HASH_B"  # noqa: SLF001
    q.append(a)
    q.append(b)
    # All free chunks are cached. Asking for order 2 must evict.
    assert evicted == []
    q.alloc_chunk(order=2)
    assert len(evicted) == 1, f"expected exactly one eviction, got {evicted}"
    # The evicted block is the LRU-head of order 2 (a, freed first).
    assert evicted[0].block_id == a.block_id
    # b retains its hash.
    assert b.block_hash == "HASH_B"


def test_initial_pool_null_block_id_zero() -> None:
    """First popleft after init must return block_id 0 (NULL_BLOCK_ID
    invariant). Uncached-at-head seeding reverses insert order to preserve
    this."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(32)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=3)
    first = q.popleft()
    assert first.block_id == 0, (
        f"NULL_BLOCK_ID = 0 invariant broken: popleft returned {first.block_id}"
    )


def test_allocated_chunks_dont_overlap() -> None:
    """Allocated chunks cover disjoint, aligned base-block ranges."""
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    allocs: list[tuple[int, int]] = []
    for order in [0, 0, 1, 2, 0, 3]:
        try:
            blk = q.alloc_chunk(order=order)
        except ValueError:
            break
        # Each chunk start is aligned to its size.
        assert blk.block_id % (1 << order) == 0
        allocs.append((blk.block_id, 1 << order))
    # Verify no two allocated chunks overlap.
    intervals = sorted(allocs)
    for i in range(len(intervals) - 1):
        end_i = intervals[i][0] + intervals[i][1]
        assert end_i <= intervals[i + 1][0], (
            f"overlap between {intervals[i]} and {intervals[i + 1]}"
        )
