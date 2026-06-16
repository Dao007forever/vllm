# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power-of-two buddy allocator for variable-span KV-cache blocks.

A ``BlockAllocator`` (see ``kv_cache_allocator.py``) that hands out aligned
chunks of ``2**order`` base blocks. Used when groups need different page sizes
(decoupled hybrid paging); the default span-1 path uses the LRU
``FreeKVCacheBlockQueue`` instead.

Reclaim model: this allocator's free structure only ever holds *truly-free*
memory. Cached-but-reusable blocks live in an ``EvictionPolicy`` the allocator
*owns* (``self._evictor``). When ``allocate`` can't satisfy a request from free
memory it reclaims the policy's least-recently-used candidate (dropping its
hash via the policy's ``on_evict`` hook), frees that memory back to itself —
which eagerly coalesces — and retries. Because the free structure never holds
cached chunks, there is no preserve-cached special case and coalescing is
always unconditional.

Free chunks are organised as one doubly-linked list per order, threaded through
the ``prev_free_block`` / ``next_free_block`` slots of ``KVCacheBlock`` (the same
slots the LRU queue and the eviction policy use; a block lives in exactly one of
them at a time). A per-order set of free chunk start-ids backs O(1) buddy-
membership tests during coalesce — pointer presence alone cannot distinguish a
pool-free block from one held by the eviction policy, since both use the same
link slots.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.v1.core.eviction_policy import LRUEvictionPolicy
from vllm.v1.core.kv_cache_allocator import EvictionPolicy
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def _make_sentinel(block_id: int = -1) -> KVCacheBlock:
    """Fake head/tail node — never returned to callers."""
    return KVCacheBlock(block_id=block_id)


class BuddyAllocator:
    def __init__(
        self,
        blocks: list[KVCacheBlock],
        max_order: int = 0,
        evictor: EvictionPolicy | None = None,
    ) -> None:
        """Args:
        blocks: One ``KVCacheBlock`` per base-block id, ordered by id.
        max_order: Largest buddy order. ``2**max_order`` need not divide
            ``len(blocks)``; any tail ids beyond the largest aligned prefix are
            served only as order-0 chunks (they cannot coalesce up).
        evictor: The eviction policy holding cached reclaim candidates. When
            ``None`` (prefix caching disabled), a no-op policy is used —
            nothing is ever tracked, so ``allocate`` never reclaims.
        """
        if max_order < 0:
            raise ValueError(f"max_order must be non-negative, got {max_order}")
        n = len(blocks)
        if n == 0:
            raise ValueError("blocks must be non-empty")
        self._max_order = max_order
        self._evictor: EvictionPolicy = (
            evictor if evictor is not None else LRUEvictionPolicy(on_evict=None)
        )
        self._blocks_by_id: dict[int, KVCacheBlock] = {b.block_id: b for b in blocks}

        # Per-order free list (head -> ... -> tail) plus a per-order set of free
        # chunk start-ids for O(1) buddy-membership checks.
        self._heads: list[KVCacheBlock] = [
            _make_sentinel() for _ in range(max_order + 1)
        ]
        self._tails: list[KVCacheBlock] = [
            _make_sentinel() for _ in range(max_order + 1)
        ]
        for k in range(max_order + 1):
            self._heads[k].next_free_block = self._tails[k]
            self._tails[k].prev_free_block = self._heads[k]
        self._pool_starts: list[set[int]] = [set() for _ in range(max_order + 1)]
        self._pool_free: int = 0

        # Seed the largest aligned prefix as max-order chunks. Tail ids (not a
        # multiple of 2**max_order from 0) can't form valid buddy chunks, so
        # they live in a side fallback list served only for order-0 requests.
        chunk = 1 << max_order
        aligned_n = (n // chunk) * chunk
        if aligned_n == 0:
            raise ValueError(
                f"len(blocks)={n} too small for max_order={max_order} "
                f"(need at least {chunk})"
            )
        # ``_pool_add`` links at the HEAD, so each insert pushes earlier inserts
        # back. Seed in REVERSE id order so block 0 ends up at the head: the
        # very first allocate(1) (which BlockPool consumes as the null block)
        # walks up to max_order, pops block 0's chunk, and splits down to
        # return id 0 — preserving the hardcoded NULL_BLOCK_ID = 0 invariant.
        for start in range(aligned_n - chunk, -1, -chunk):
            head = blocks[start]
            head.base_span = 1 << max_order
            self._pool_add(head, max_order)
        self._tail_free: list[int] = list(range(aligned_n, n))
        self._tail_allocated: set[int] = set()

    # --------------------------- pool helpers -------------------------------
    def _pool_add(self, block: KVCacheBlock, order: int) -> None:
        """Link ``block`` at the head of order ``order`` and mark it free."""
        head = self._heads[order]
        nxt = head.next_free_block
        assert nxt is not None
        head.next_free_block = block
        block.prev_free_block = head
        block.next_free_block = nxt
        nxt.prev_free_block = block
        self._pool_starts[order].add(block.block_id)
        self._pool_free += 1 << order

    def _unlink(self, block: KVCacheBlock) -> None:
        prev = block.prev_free_block
        nxt = block.next_free_block
        assert prev is not None and nxt is not None
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        block.prev_free_block = None
        block.next_free_block = None

    def _pool_remove(self, block: KVCacheBlock, order: int) -> None:
        """Take ``block`` out of order ``order``'s free list."""
        self._unlink(block)
        self._pool_starts[order].discard(block.block_id)
        self._pool_free -= 1 << order

    def _pool_pop_head(self, order: int) -> KVCacheBlock | None:
        head = self._heads[order]
        first = head.next_free_block
        assert first is not None
        if first is self._tails[order]:
            return None
        self._pool_remove(first, order)
        return first

    def _smallest_available_order(self, order: int) -> int | None:
        for k in range(order, self._max_order + 1):
            if self._pool_starts[k]:
                return k
        return None

    @staticmethod
    def _span_to_order(span: int) -> int:
        order = span.bit_length() - 1
        if span < 1 or (1 << order) != span:
            raise ValueError(f"span must be a power of two, got {span}")
        return order

    # --------------------------- BlockAllocator -----------------------------
    @property
    def max_order(self) -> int:
        return self._max_order

    @property
    def num_free_blocks(self) -> int:
        """Base blocks available: free pool plus reclaimable candidates."""
        return self._pool_free + self._evictor.num_reclaimable

    def allocate(self, span: int = 1) -> KVCacheBlock:
        """Allocate one chunk of ``span`` (a power-of-two count of) base blocks.

        Returns the head ``KVCacheBlock`` of the chunk with ``base_span`` set.
        Reclaims LRU eviction candidates when free memory is short.
        """
        order = self._span_to_order(span)
        if order > self._max_order:
            raise ValueError(f"span {span} exceeds max order {self._max_order}")
        while True:
            src = self._smallest_available_order(order)
            if src is not None:
                break
            # Unaligned tail ids can serve order-0 requests only.
            if order == 0 and self._tail_free:
                bid = self._tail_free.pop()
                self._tail_allocated.add(bid)
                self._pool_free -= 1
                blk = self._blocks_by_id[bid]
                blk.base_span = 1
                return blk
            # Reclaim a cached candidate and retry; the freed memory coalesces.
            victim = self._evictor.pop_victim()
            if victim is None:
                raise ValueError(f"No free chunk of order >= {order} available")
            self._free_to_pool(victim)
        block = self._pool_pop_head(src)
        assert block is not None
        # Split src -> order: high-half buddies go back to their order's list.
        while src > order:
            src -= 1
            buddy = self._blocks_by_id[block.block_id + (1 << src)]
            buddy.base_span = 1 << src
            self._pool_add(buddy, src)
        block.base_span = 1 << order
        return block

    def free(self, block: KVCacheBlock, *, prefer_reuse: bool = False) -> None:
        """Return ``block`` to the allocator. A cached block becomes a reclaim
        candidate in the eviction policy; an uncached block returns to the free
        pool (and eagerly coalesces)."""
        if block.block_hash is not None:
            self._evictor.track(block, prefer_reuse=prefer_reuse)
        else:
            self._free_to_pool(block)

    def reuse(self, block: KVCacheBlock) -> None:
        """Prefix-cache hit: pull a cached candidate back out of the policy."""
        self._evictor.untrack(block)

    def _free_to_pool(self, block: KVCacheBlock) -> None:
        """Return a chunk to the free pool, coalescing with free buddies up to
        ``max_order``."""
        bid = block.block_id
        if bid in self._tail_allocated:
            self._tail_allocated.discard(bid)
            self._tail_free.append(bid)
            self._pool_free += 1
            return
        order = block.base_span.bit_length() - 1
        if order < 0 or order > self._max_order:
            raise ValueError(
                f"free: base_span {block.base_span} (order {order}) out of "
                f"range [0, {self._max_order}] for block {block.block_id}"
            )
        while order < self._max_order:
            buddy_id = bid ^ (1 << order)
            if buddy_id not in self._pool_starts[order]:
                break
            self._pool_remove(self._blocks_by_id[buddy_id], order)
            bid = min(bid, buddy_id)
            order += 1
        merged = self._blocks_by_id[bid]
        merged.base_span = 1 << order
        self._pool_add(merged, order)

    def can_allocate(self, demand_by_span: dict[int, int]) -> bool:
        """Whether a joint ``{span: num_blocks}`` demand is satisfiable from the
        free pool plus reclaimable candidates.

        A free chunk of order ``j >= k`` contributes its base blocks to every
        threshold ``k``; a power-of-two demand multiset is satisfiable iff for
        every threshold ``k`` the base blocks demanded by orders ``>= k`` do not
        exceed those available in chunks of order ``>= k`` (a majorization
        condition). Candidates are counted at their stored order without
        simulating the coalescing reclaim would trigger — this matches the
        legacy buddy's behavior (which never coalesced cached chunks) and is
        sound, since reclaim+coalesce can only do better than the check
        promises. Unaligned tail-fallback ids only help order 0.
        """
        demand_by_order: dict[int, int] = {}
        for span, count in demand_by_span.items():
            if count <= 0:
                continue
            order = span.bit_length() - 1
            if span < 1 or (1 << order) != span:
                return False
            if order > self._max_order:
                return False
            demand_by_order[order] = demand_by_order.get(order, 0) + count
        if not demand_by_order:
            return True

        free_counts = [len(self._pool_starts[k]) for k in range(self._max_order + 1)]
        for blk in self._evictor.reclaimable_chunks():
            o = blk.base_span.bit_length() - 1
            if 0 <= o <= self._max_order:
                free_counts[o] += 1
        tail_free = len(self._tail_free)

        for k in range(self._max_order + 1):
            demand_base = sum(c << o for o, c in demand_by_order.items() if o >= k)
            free_base = sum(free_counts[j] << j for j in range(k, self._max_order + 1))
            if k == 0:
                free_base += tail_free
            if demand_base > free_base:
                return False
        return True

    @staticmethod
    def normalize_span(natural_span: int) -> int:
        """Round a natural span up to the next power of two — the only spans
        this buddy allocator can carve and align. ``normalize_span(3) == 4``,
        ``normalize_span(4) == 4``, ``normalize_span(1) == 1``."""
        if natural_span <= 1:
            return 1
        return 1 << (natural_span - 1).bit_length()

    @classmethod
    def for_max_span(
        cls,
        blocks: list[KVCacheBlock],
        max_allocation_span: int,
        evictor: EvictionPolicy | None = None,
    ) -> BuddyAllocator:
        """Construct from a neutral max *span* (base blocks per logical block),
        translating span->order internally."""
        max_order = (max(1, max_allocation_span) - 1).bit_length()
        return cls(blocks, max_order=max_order, evictor=evictor)

    # ---------------------------- introspection -----------------------------
    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """All truly-free chunk heads (pool, low order to high, then tail-
        fallback ids). Excludes reclaim candidates. For tests/introspection."""
        out: list[KVCacheBlock] = []
        for k in range(self._max_order + 1):
            cur = self._heads[k].next_free_block
            tail = self._tails[k]
            while cur is not None and cur is not tail:
                out.append(cur)
                cur = cur.next_free_block
        out.extend(self._blocks_by_id[bid] for bid in self._tail_free)
        return out

    def reclaimable_chunks(self) -> Sequence[KVCacheBlock]:
        """Cached reclaim candidates held by the eviction policy."""
        return self._evictor.reclaimable_chunks()
