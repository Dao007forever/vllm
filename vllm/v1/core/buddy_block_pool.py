# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power-of-2 buddy allocator over a flat base-block id space.

Sits alongside ``BlockPool`` as an experimental free-list structure that
allows variable-size allocations. A "chunk" of order ``k`` covers
``2**k`` consecutive base blocks; its starting id is always a multiple
of ``2**k``. Buddy of a chunk at order ``k`` is the chunk obtained by
flipping bit ``k`` of the start id (``start ^ (1 << k)``).

This module is not yet wired into the KV cache manager — it is a building
block for a later iteration. Keep it self-contained.
"""

from __future__ import annotations


class BuddyAllocationError(RuntimeError):
    """Raised when no free chunk satisfies an allocation request."""


class BuddyBlockPool:
    def __init__(self, num_base_blocks: int, max_order: int) -> None:
        """Args:
            num_base_blocks: Total number of base-sized blocks owned by the
                pool. Must be a multiple of ``2**max_order``.
            max_order: Largest allocation order. Pool is seeded with chunks
                of this order.
        """
        if num_base_blocks <= 0:
            raise ValueError("num_base_blocks must be positive")
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        chunk = 1 << max_order
        if num_base_blocks % chunk != 0:
            raise ValueError(
                f"num_base_blocks ({num_base_blocks}) must be a multiple of "
                f"2**max_order ({chunk})"
            )

        self.num_base_blocks = num_base_blocks
        self.max_order = max_order
        # free_lists[order] = set of starting base-block ids for free chunks
        # of size 2**order. Sets give O(1) membership for coalescing.
        self._free: list[set[int]] = [set() for _ in range(max_order + 1)]
        for start in range(0, num_base_blocks, chunk):
            self._free[max_order].add(start)
        # block_id -> order. Tracks live allocations for free() validation.
        self._alloc_order: dict[int, int] = {}

    def allocate(self, order: int) -> int:
        """Return the starting base-block id of a free chunk of size ``2**order``.

        Splits a larger chunk if no chunk of the requested order is free.
        Raises ``BuddyAllocationError`` if no large-enough chunk exists.
        """
        if order < 0 or order > self.max_order:
            raise ValueError(
                f"order {order} out of range [0, {self.max_order}]"
            )
        cur = order
        while cur <= self.max_order and not self._free[cur]:
            cur += 1
        if cur > self.max_order:
            raise BuddyAllocationError(
                f"no free chunk of order >= {order} available"
            )
        block = self._free[cur].pop()
        # Split down to the requested order: at each step, the high half
        # becomes a free buddy at order-1, the low half continues splitting.
        while cur > order:
            cur -= 1
            buddy = block + (1 << cur)
            self._free[cur].add(buddy)
        self._alloc_order[block] = order
        return block

    def free(self, block_id: int) -> None:
        """Return a previously-allocated chunk to the pool, coalescing buddies."""
        if block_id not in self._alloc_order:
            raise ValueError(f"block_id {block_id} is not allocated")
        order = self._alloc_order.pop(block_id)
        while order < self.max_order:
            buddy = block_id ^ (1 << order)
            if buddy in self._free[order]:
                self._free[order].remove(buddy)
                block_id = min(block_id, buddy)
                order += 1
            else:
                break
        self._free[order].add(block_id)

    def free_base_blocks(self) -> int:
        """Total free base blocks across all orders."""
        return sum(len(s) * (1 << o) for o, s in enumerate(self._free))

    def free_chunks_per_order(self) -> list[int]:
        return [len(s) for s in self._free]

    def num_allocated_chunks(self) -> int:
        return len(self._alloc_order)
