# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU eviction policy for cached KV-cache blocks.

An allocator owns one of these (the "reclaim model"): released-but-cached blocks
are *tracked* here as reclaim candidates rather than returned to the allocator's
free structure, so they retain their prefix-cache hash and can still serve a hit.
When the allocator runs short it calls ``pop_victim`` to reclaim the least-
recently-used candidate; ``on_evict`` drops that block's hash before its memory
is reused.

The candidate list is a doubly-linked LRU threaded through the
``prev_free_block`` / ``next_free_block`` slots of ``KVCacheBlock`` — the same
slots the free queues use. A block lives in the allocator's free structure *or*
in this policy's candidate list, never both, so the pointers never conflict.

Ordering: head = least-recently-used (reclaimed first), tail = most-recently-
used (reclaimed last). A freshly released candidate is the most recent, so it
links at the tail; ``prefer_reuse`` (e.g. mamba scratch recycling) links at the
head so the block is reclaimed/reused first.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.v1.core.kv_cache_utils import KVCacheBlock


def _make_sentinel() -> KVCacheBlock:
    """Fake head/tail node — never returned to callers."""
    return KVCacheBlock(block_id=-1)


class LRUEvictionPolicy:
    def __init__(
        self,
        on_evict: Callable[[KVCacheBlock], bool] | None = None,
    ) -> None:
        """Args:
        on_evict: Fired on the block ``pop_victim`` is about to surrender,
            before its memory is reused. Should drop the block's hash from the
            prefix-cache map and reset it (mirrors
            ``BlockPool._maybe_evict_cached_block``); the return value is
            ignored. ``None`` only when prefix caching is disabled (nothing is
            ever tracked, so it never fires).
        """
        self._on_evict = on_evict
        self._head = _make_sentinel()
        self._tail = _make_sentinel()
        self._head.next_free_block = self._tail
        self._tail.prev_free_block = self._head
        self.num_reclaimable: int = 0

    def _unlink(self, block: KVCacheBlock) -> None:
        prev = block.prev_free_block
        nxt = block.next_free_block
        assert prev is not None and nxt is not None
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        block.prev_free_block = None
        block.next_free_block = None

    def track(self, block: KVCacheBlock, *, prefer_reuse: bool = False) -> None:
        if prefer_reuse:
            # Reclaim/reuse first: link at the LRU (head) end.
            anchor = self._head
            nxt = anchor.next_free_block
            assert nxt is not None
            anchor.next_free_block = block
            block.prev_free_block = anchor
            block.next_free_block = nxt
            nxt.prev_free_block = block
        else:
            # Most-recently-used: link at the tail end (reclaimed last).
            anchor = self._tail
            prev = anchor.prev_free_block
            assert prev is not None
            prev.next_free_block = block
            block.prev_free_block = prev
            block.next_free_block = anchor
            anchor.prev_free_block = block
        self.num_reclaimable += block.base_span

    def untrack(self, block: KVCacheBlock) -> None:
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"untrack called on a block not tracked here: {block}")
        self._unlink(block)
        self.num_reclaimable -= block.base_span

    def pop_victim(self) -> KVCacheBlock | None:
        victim = self._head.next_free_block
        assert victim is not None
        if victim is self._tail:
            return None
        self._unlink(victim)
        self.num_reclaimable -= victim.base_span
        if self._on_evict is not None:
            self._on_evict(victim)
        return victim

    def reclaimable_chunks(self) -> list[KVCacheBlock]:
        out: list[KVCacheBlock] = []
        cur = self._head.next_free_block
        while cur is not None and cur is not self._tail:
            out.append(cur)
            cur = cur.next_free_block
        return out
