# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator + eviction-policy interfaces for the KV-cache free-block pool.

``BlockPool`` depends only on ``BlockAllocator``: a real-allocator surface of
``allocate`` / ``free`` (plus ``reuse`` for prefix-cache hits and a feasibility
query). The allocation strategy is swappable behind it. Two implementations
satisfy it today: the default LRU ``FreeKVCacheBlockQueue`` and the
variable-span ``BuddyAllocator``.

Prefix-cache eviction is a separate concern, modelled by ``EvictionPolicy``. An
allocator *owns* an eviction policy and reclaims from it internally when free
memory runs short — ``BlockPool`` never orchestrates reclaim. Cached-but-
reusable blocks live in the policy (the "reclaim model"), so the allocator's
free structure only ever holds truly-free memory. The policy carries an
``on_evict`` hook that drops the reclaimed block's hash from the prefix-cache
map (owned by ``BlockPool``); ``BlockPool`` keeps the map for lookup/insert and
signals a hit via ``BlockAllocator.reuse``.

The vocabulary here is deliberately allocator-agnostic — callers speak in blocks
and spans (base-block counts), never in implementation concepts such as buddy
orders or bit arithmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.v1.core.kv_cache_utils import KVCacheBlock


@runtime_checkable
class BlockAllocator(Protocol):
    """The allocation surface ``BlockPool`` depends on.

    An implementation manages the set of truly-free blocks and reclaims from its
    own eviction policy when short. A block is in at most one allocator. Callers
    must not rely on a particular ordering of free blocks beyond the documented
    semantics of the concrete implementation.
    """

    # Base blocks available to satisfy allocations: truly-free memory plus the
    # reclaimable (cached eviction-candidate) blocks the policy holds.
    num_free_blocks: int

    def allocate(self, span: int = 1) -> KVCacheBlock:
        """Allocate one logical block spanning ``span`` consecutive base blocks
        and return its starting block. ``span == 1`` (the default) is the
        ordinary single-base-block allocation.

        When truly-free memory cannot satisfy the request, the allocator
        reclaims LRU eviction candidates from its policy (dropping their cached
        hashes) and retries, so the returned memory never carries a stale hash.
        Raises if the request cannot be satisfied even after reclaiming.
        Implementations that only support single-base-block allocation must
        raise for ``span > 1``.
        """
        ...

    def free(self, block: KVCacheBlock, *, prefer_reuse: bool = False) -> None:
        """Return a block previously obtained from ``allocate``.

        A block that still carries a prefix-cache hash becomes a reclaim
        candidate (retained by the eviction policy for a possible future hit);
        an uncached block returns to truly-free memory. ``prefer_reuse``
        prioritises the block for the next allocation (e.g. mamba scratch-block
        recycling); allocators with no global reuse ordering may ignore it. The
        block's span is recovered from allocator-internal bookkeeping.
        """
        ...

    def reuse(self, block: KVCacheBlock) -> None:
        """Take a specific cached eviction-candidate back out of the free
        structure because a prefix-cache hit is reusing it (caller owns it
        again). Mirrors a touch on a ref-count-zero cached block."""
        ...

    def can_allocate(self, demand_by_span: dict[int, int]) -> bool:
        """Whether a joint demand of ``{span: num_blocks}`` can be satisfied
        from free memory plus reclaimable candidates, accounting for any
        alignment/fragmentation the implementation imposes."""
        ...

    @staticmethod
    def normalize_span(natural_span: int) -> int:
        """Round a requested *natural* span (the number of base blocks a
        logical block needs to cover its page) up to the granularity this
        allocator can actually hand out and align starts to.

        The layout sizes tensors and computes logical<->base block-id
        translation (``block_id // span``) from the returned value, so every
        consumer must agree on one effective span. The contract is that the
        allocator aligns each spanned chunk's start id to a multiple of the
        returned span. Allocators that support arbitrary spans return
        ``natural_span`` unchanged; the buddy allocator rounds up to the next
        power of two. A span-1-only allocator returns 1 and rejects anything
        larger. This is a static capability because the layout is resolved
        before any allocator instance exists.
        """
        ...


@runtime_checkable
class EvictionPolicy(Protocol):
    """Holds released-but-cached blocks as reclaim candidates, in reuse-priority
    order, and reclaims them on demand.

    A ``BlockAllocator`` owns one of these. It is constructed with an
    ``on_evict`` hook that drops a reclaimed block's hash from the prefix-cache
    map (and resets the block's hash) the moment the block is about to be reused
    for different memory.
    """

    # Total base blocks held as reclaim candidates (sum of candidate spans).
    num_reclaimable: int

    def track(self, block: KVCacheBlock, *, prefer_reuse: bool = False) -> None:
        """Record ``block`` (released by a request but still cached) as a
        reclaim candidate. ``prefer_reuse`` puts it at the reuse-soon end."""
        ...

    def untrack(self, block: KVCacheBlock) -> None:
        """Remove a specific candidate from the structure (a prefix-cache hit is
        reusing it)."""
        ...

    def pop_victim(self) -> KVCacheBlock | None:
        """Remove and return the least-recently-used candidate, after firing
        ``on_evict`` to drop its cached hash. ``None`` if there are no
        candidates to reclaim."""
        ...

    def reclaimable_chunks(self) -> Sequence[KVCacheBlock]:
        """All current candidate chunk heads (for the allocator's feasibility
        fold-in / introspection)."""
        ...
