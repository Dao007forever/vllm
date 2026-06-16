# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator-neutral interface for the KV-cache free-block pool.

``BlockPool`` owns a free-block allocator behind this interface so the
allocation strategy is swappable. Two implementations satisfy it today: the
default LRU ``FreeKVCacheBlockQueue`` and the variable-span
``BuddyFreeKVCacheBlockQueue``. The vocabulary here is deliberately
allocator-agnostic — callers speak in blocks and counts, never in
implementation concepts such as buddy orders or bit arithmetic. Any
strategy-specific capability (e.g. variable-span chunk allocation) lives behind
the pool, not in the generic KV-cache managers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


@runtime_checkable
class KVCacheBlockAllocator(Protocol):
    """The free-block management surface a ``BlockPool`` depends on.

    Implementations maintain the set of free blocks (including eviction
    candidates when prefix caching is enabled) and thread each block through a
    single intrusive free list at a time. A block is in at most one allocator.
    Callers must not rely on a particular ordering of free blocks beyond the
    documented LRU/eviction semantics of the concrete implementation.
    """

    # Number of free base blocks currently available.
    num_free_blocks: int

    def popleft(self) -> KVCacheBlock:
        """Remove and return the next free block to hand out."""
        ...

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Remove and return the next ``n`` free blocks."""
        ...

    def append(self, block: KVCacheBlock) -> None:
        """Return a block to the free pool."""
        ...

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Return several blocks to the free pool."""
        ...

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Return several blocks to the free pool, prioritized for immediate
        reuse (the next allocations should prefer them). An allocator with no
        global reuse ordering may treat this the same as ``append_n``."""
        ...

    def remove(self, block: KVCacheBlock) -> None:
        """Take a specific block out of the free pool (caller owns it)."""
        ...

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return all currently-free blocks (for introspection/tests)."""
        ...

    def allocate_spanned_block(self, base_span: int) -> KVCacheBlock:
        """Allocate one logical block spanning ``base_span`` consecutive base
        blocks and return its starting block. ``base_span == 1`` is the
        ordinary single-base-block allocation. Implementations that only
        support single-base-block allocation must raise for ``base_span > 1``.
        """
        ...

    def free_spanned_block(self, block: KVCacheBlock) -> None:
        """Return a block previously obtained from ``allocate_spanned_block``
        to the free pool (its span is recovered from allocator-internal
        bookkeeping)."""
        ...

    def can_allocate_spans(self, demand_by_span: dict[int, int]) -> bool:
        """Whether a joint demand of ``{base_span: num_blocks}`` can be
        satisfied from the current free structure, accounting for any
        alignment/fragmentation the implementation imposes."""
        ...

    @staticmethod
    def normalize_span(natural_span: int) -> int:
        """Round a requested *natural* span (the number of base blocks a
        logical block needs to cover its page) up to the granularity this
        allocator can actually hand out and align starts to.

        The layout sizes tensors and computes logical<->base block-id
        translation (``block_id // base_span``) from the returned value, so
        every consumer must agree on one effective span. The contract is that
        the allocator aligns each spanned chunk's start id to a multiple of the
        returned span. Allocators that support arbitrary spans return
        ``natural_span`` unchanged; the buddy allocator rounds up to the next
        power of two. A span-1-only allocator returns 1 and rejects anything
        larger. This is a static capability because the layout is resolved
        before any allocator instance exists.
        """
        ...
