# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-order LRU buddy-backed free queue.

Drop-in replacement for ``FreeKVCacheBlockQueue`` when variable-size
allocations are needed. Differences vs. the LRU queue:

- Free blocks are organised as one doubly-linked LRU list **per buddy order**.
  Each list threads through the ``prev_free_block`` / ``next_free_block``
  attributes of ``KVCacheBlock`` (same pointer slots the LRU queue uses, so a
  block lives in at most one queue at a time).
- ``alloc_chunk(order)`` pops the LRU head of that order's list; if empty it
  walks up, pops a larger chunk, and splits — pushing the buddy half(ves)
  back into lower-order LRU tails. Returns the head block with
  ``chunk_order = order`` recorded on it.
- ``append(block)`` reads ``block.chunk_order``, attempts eager coalesce with
  its buddy if the buddy is also free at the same order, otherwise links the
  block into the appropriate order's LRU tail.
- ``remove(block)`` reads ``block.chunk_order`` and unlinks the block from
  its LRU in O(1) — no order parameter, same signature as the LRU queue's
  ``remove``. The chunk leaves the free pool entirely; the caller is
  responsible for tracking it.
- ``popleft()`` and ``popleft_n()`` delegate to ``alloc_chunk(order=0)``.

The structure preserves the buddy invariant that a free chunk at order ``k``
starts at an id that is a multiple of ``2**k``, but storage is now per-order
LRU rather than per-order sets.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.v1.core.kv_cache_utils import KVCacheBlock


def _make_sentinel(block_id: int = -1) -> KVCacheBlock:
    """Fake head/tail node — never returned to callers."""
    return KVCacheBlock(block_id=block_id)


class BuddyFreeKVCacheBlockQueue:
    def __init__(
        self,
        blocks: list[KVCacheBlock],
        max_order: int = 0,
        on_evict: Callable[[KVCacheBlock], bool] | None = None,
    ) -> None:
        """Args:
            blocks: One ``KVCacheBlock`` per base-block id, ordered by id.
            max_order: Largest buddy order. ``2**max_order`` need not divide
                ``len(blocks)``; any tail ids beyond the largest aligned
                prefix are added as order-0 chunks (cannot coalesce up).
            on_evict: Optional callback fired before a cached chunk loses
                its identity due to a split (in ``alloc_chunk``) or a
                coalesce (in ``append``). The callback should drop the
                block's hash from the prefix-cache map and call
                ``block.reset_hash()``. Signature mirrors
                ``BlockPool._maybe_evict_cached_block``: returns True if
                anything was evicted; the return value is ignored here.
                When ``None`` (default), no eviction is performed — only
                safe when prefix caching is disabled at the pool level.
        """
        if max_order < 0:
            raise ValueError(f"max_order must be non-negative, got {max_order}")
        n = len(blocks)
        if n == 0:
            raise ValueError("blocks must be non-empty")
        self._max_order = max_order
        self._on_evict = on_evict
        self._blocks_by_id: dict[int, KVCacheBlock] = {
            b.block_id: b for b in blocks
        }

        # Per-order LRU: heads[k] -> ... -> tails[k]. Lowest id at head end
        # initially; subsequent appends go to the tail (= MRU end).
        self._heads: list[KVCacheBlock] = [
            _make_sentinel() for _ in range(max_order + 1)
        ]
        self._tails: list[KVCacheBlock] = [
            _make_sentinel() for _ in range(max_order + 1)
        ]
        for k in range(max_order + 1):
            self._heads[k].next_free_block = self._tails[k]
            self._tails[k].prev_free_block = self._heads[k]

        # Seed the largest aligned prefix as max-order chunks. Any tail ids
        # (not a multiple of 2**max_order from 0) live in a side fallback
        # list, NOT in the order-0 LRU. Two reasons:
        # 1. ``NULL_BLOCK_ID = 0`` is hardcoded in the attention/mamba
        #    kernels — they assume the first popleft() (which becomes
        #    BlockPool.null_block) returns id 0. Keeping order-0 LRU empty
        #    at init means popleft must walk up to max_order, where the
        #    LRU head is block 0, and splits down through the low halves
        #    to return block 0.
        # 2. Tail blocks can't form valid buddy chunks (their buddies
        #    would be out-of-range), so eager coalesce would waste cycles
        #    checking buddies that don't exist. Keeping them in a side
        #    list lets popleft drain the aligned pool first and only fall
        #    back to tail blocks under pressure.
        chunk = 1 << max_order
        aligned_n = (n // chunk) * chunk
        if aligned_n == 0:
            raise ValueError(
                f"len(blocks)={n} too small for max_order={max_order} "
                f"(need at least {chunk})"
            )
        # ``_link_tail`` puts uncached blocks at the LRU HEAD (so they pop
        # first), which means each insert pushes earlier inserts back. Seed
        # in REVERSE id order so that block 0 ends up at the head — this
        # preserves the NULL_BLOCK_ID = 0 invariant: the very first popleft
        # (which BlockPool consumes as the null block) returns id 0.
        for start in range(aligned_n - chunk, -1, -chunk):
            head = blocks[start]
            head.chunk_order = max_order
            self._link_tail(head, max_order)
        # Tail blocks: pop from the end (LIFO) so iterating popleft sees
        # higher ids first, matching the LRU-queue convention that initial
        # block order is by id.
        self._tail_free: list[int] = list(range(aligned_n, n))
        self._tail_allocated: set[int] = set()

        self.num_free_blocks: int = n

    # ----------------------- LRU linked-list helpers ------------------------
    def _link_tail(self, block: KVCacheBlock, order: int) -> None:
        """Insert ``block`` at the MRU end of order ``order``.

        Cached blocks (``block_hash is not None``) go to the TAIL — they age
        out via LRU and are evicted last. Uncached blocks go to the HEAD so
        they are popped first, preserving cached chunks. Combined with the
        coalesce-skip rule in ``append``, this means the order-K LRU layout
        is: ``[uncached newest → ... → uncached oldest][cached oldest → ...
        → cached newest]``. The head's hash state tells us in O(1) whether
        any uncached chunk exists at this order — used by ``alloc_chunk``
        to prefer splitting a higher-order uncached chunk over evicting a
        same-order cached one.
        """
        if block.block_hash is None:
            # Uncached: link at HEAD so it pops first.
            head = self._heads[order]
            nxt = head.next_free_block
            assert nxt is not None
            head.next_free_block = block
            block.prev_free_block = head
            block.next_free_block = nxt
            nxt.prev_free_block = block
        else:
            # Cached: link at TAIL (MRU end) for LRU eviction.
            tail = self._tails[order]
            prev = tail.prev_free_block
            assert prev is not None
            prev.next_free_block = block
            block.prev_free_block = prev
            block.next_free_block = tail
            tail.prev_free_block = block

    def _unlink(self, block: KVCacheBlock) -> None:
        """Unlink ``block`` from whatever per-order LRU it's currently in."""
        prev = block.prev_free_block
        nxt = block.next_free_block
        if prev is None or nxt is None:
            raise RuntimeError(
                f"_unlink called on block not in any LRU: {block}"
            )
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        block.prev_free_block = None
        block.next_free_block = None

    def _pop_head(self, order: int) -> KVCacheBlock | None:
        """Pop the LRU head of order ``order``, or None if empty."""
        head = self._heads[order]
        first = head.next_free_block
        assert first is not None
        if first is self._tails[order]:
            return None
        self._unlink(first)
        return first

    def _is_in_queue(self, block: KVCacheBlock) -> bool:
        return (
            block.prev_free_block is not None
            and block.next_free_block is not None
        )

    # --------------------------- public surface -----------------------------
    @property
    def max_order(self) -> int:
        return self._max_order

    def popleft(self) -> KVCacheBlock:
        """Single base-block alloc (= alloc_chunk(0))."""
        return self.alloc_chunk(order=0)

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        if n > self.num_free_blocks:
            raise ValueError(
                f"Cannot pop {n} blocks (only {self.num_free_blocks} free)"
            )
        return [self.alloc_chunk(order=0) for _ in range(n)]

    def alloc_chunk(self, order: int) -> KVCacheBlock:
        """Allocate a chunk of ``2**order`` base blocks.

        Returns the head ``KVCacheBlock`` of the chunk (start id), with
        ``chunk_order = order`` set on it.
        """
        if order < 0 or order > self._max_order:
            raise ValueError(
                f"order {order} out of range [0, {self._max_order}]"
            )
        # Pass 1: prefer an UNCACHED chunk to avoid evicting a cached one.
        # Because _link_tail puts uncached blocks at the HEAD of each order's
        # LRU, checking the head's hash is an O(1) test for "any uncached
        # chunk available at this order". We walk up looking for the smallest
        # order whose head is uncached.
        src = order
        while src <= self._max_order:
            head = self._heads[src].next_free_block
            if (
                head is not self._tails[src]
                and head.block_hash is None
            ):
                break
            src += 1
        if src > self._max_order:
            # Pass 2: no uncached chunk anywhere. Fall back to evicting the
            # LRU-head cached chunk at the smallest non-empty order.
            src = order
            while src <= self._max_order:
                if (
                    self._heads[src].next_free_block
                    is not self._tails[src]
                ):
                    break
                src += 1
        if src > self._max_order:
            # Fall back to tail blocks for order-0 requests only — tail
            # ids aren't aligned for higher orders.
            if order == 0 and self._tail_free:
                bid = self._tail_free.pop()
                self._tail_allocated.add(bid)
                self.num_free_blocks -= 1
                blk = self._blocks_by_id[bid]
                blk.chunk_order = 0
                return blk
            raise ValueError(
                f"No free chunk of order >= {order} available"
            )
        block = self._pop_head(src)
        assert block is not None
        # If the popped chunk is cached, drop its hash before reusing the
        # memory. This covers both same-order eviction (src == order, no
        # uncached found) and the older split-on-cached case (src > order
        # — should be rare with the uncached-first pass but possible if a
        # higher-order chunk got cached somehow).
        if self._on_evict is not None and block.block_hash is not None:
            self._on_evict(block)
        # Split src -> order: at each level, the high-half buddy goes into
        # its order's LRU tail; the low half continues down.
        while src > order:
            src -= 1
            buddy_id = block.block_id + (1 << src)
            buddy = self._blocks_by_id[buddy_id]
            buddy.chunk_order = src
            self._link_tail(buddy, src)
        block.chunk_order = order
        self.num_free_blocks -= 1 << order
        return block

    def free_chunk(self, start_id: int) -> None:
        """Back-compat: locate the block by id and call ``append`` on it.

        Callers that hold a ``KVCacheBlock`` should prefer ``append(block)``.
        """
        self.append(self._blocks_by_id[start_id])

    def remove(self, block: KVCacheBlock) -> None:
        """Remove ``block`` (a free-chunk head) from its order's LRU.

        Decrements ``num_free_blocks`` by ``2**block.chunk_order``. The
        caller takes ownership of the chunk; ``append(block)`` returns it.
        """
        if not self._is_in_queue(block):
            raise RuntimeError(
                f"remove called on block not in any LRU: {block}"
            )
        order = block.chunk_order
        self._unlink(block)
        self.num_free_blocks -= 1 << order

    def append(self, block: KVCacheBlock) -> None:
        """Return ``block`` (head of a chunk at ``block.chunk_order``) to the
        free pool. Eagerly coalesces with its buddy if also free at the same
        order, recursively up to ``max_order``."""
        bid = block.block_id
        # Tail-block fast path: not part of the buddy address space, just
        # return to the side fallback list.
        if bid in self._tail_allocated:
            self._tail_allocated.discard(bid)
            self._tail_free.append(bid)
            self.num_free_blocks += 1
            return
        order = block.chunk_order
        if order < 0 or order > self._max_order:
            raise ValueError(
                f"append: chunk_order {order} out of range "
                f"[0, {self._max_order}] for block {block.block_id}"
            )
        # Coalesce as far up as buddies are free.
        size_returned = 1 << order
        while order < self._max_order:
            buddy_id = bid ^ (1 << order)
            # Tail-region order-0 ids may have no buddy in the pool.
            buddy = self._blocks_by_id.get(buddy_id)
            if buddy is None:
                break
            if (
                not self._is_in_queue(buddy)
                or buddy.chunk_order != order
            ):
                break
            # Preserve cached identity: if either contributor holds a cache
            # hash, coalescing would discard a valid prefix-cache entry. Stop
            # and leave both in their order's LRU. They can still be reused
            # individually; coalescing resumes once both sides are uncached.
            cur = self._blocks_by_id[bid]
            if self._on_evict is not None and (
                cur.block_hash is not None or buddy.block_hash is not None
            ):
                break
            self._unlink(buddy)
            bid = min(bid, buddy_id)
            order += 1
        merged = self._blocks_by_id[bid]
        merged.chunk_order = order
        self._link_tail(merged, order)
        self.num_free_blocks += size_returned

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for b in blocks:
            self.append(b)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return all free chunk heads in iteration order (head → tail per
        order, low order to high order, then tail-fallback ids). Intended
        for tests/introspection."""
        out: list[KVCacheBlock] = []
        for k in range(self._max_order + 1):
            cur = self._heads[k].next_free_block
            tail = self._tails[k]
            while cur is not None and cur is not tail:
                out.append(cur)
                cur = cur.next_free_block
        out.extend(self._blocks_by_id[bid] for bid in self._tail_free)
        return out
