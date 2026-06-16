# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The allocator-neutral KVCacheBlockAllocator interface (AC-5).

Both built-in allocators satisfy the runtime-checkable interface, a fresh
third-party allocator with no buddy concepts can satisfy it, and the generic
KV-cache surface (group spec, coordinator, manager) names no allocator-specific
concepts.
"""

import inspect

import pytest

from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
from vllm.v1.core.kv_cache_allocator import KVCacheBlockAllocator
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

pytestmark = pytest.mark.cpu_test


def test_both_builtin_allocators_satisfy_protocol():
    lru = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(8)])
    buddy = BuddyFreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(8)], max_order=2)
    assert isinstance(lru, KVCacheBlockAllocator)
    assert isinstance(buddy, KVCacheBlockAllocator)


def test_incomplete_allocator_is_not_an_instance():
    class Partial:
        num_free_blocks = 0

        def popleft(self):  # missing the rest of the surface
            ...

    assert not isinstance(Partial(), KVCacheBlockAllocator)


def test_trivial_third_party_allocator_conforms():
    """A minimal allocator with no buddy concepts satisfies the interface and
    can therefore back a BlockPool."""

    class TrivialAllocator:
        def __init__(self, blocks):
            self._free = list(blocks)
            self.num_free_blocks = len(blocks)

        def popleft(self):
            self.num_free_blocks -= 1
            return self._free.pop()

        def popleft_n(self, n):
            return [self.popleft() for _ in range(n)]

        def append(self, block):
            self._free.append(block)
            self.num_free_blocks += 1

        def append_n(self, blocks):
            for block in blocks:
                self.append(block)

        def prepend_n(self, blocks):
            for block in blocks:
                self.append(block)

        def remove(self, block):
            self._free.remove(block)
            self.num_free_blocks -= 1

        def get_all_free_blocks(self):
            return list(self._free)

        def allocate_spanned_block(self, base_span):
            if base_span != 1:
                raise ValueError("only span 1")
            return self.popleft()

        def free_spanned_block(self, block):
            self.append(block)

        def can_allocate_spans(self, demand_by_span):
            return all(s == 1 for s, c in demand_by_span.items() if c > 0) and (
                sum(demand_by_span.values()) <= self.num_free_blocks
            )

        @staticmethod
        def normalize_span(natural_span):
            if natural_span <= 1:
                return 1
            raise ValueError("only span 1")

    alloc = TrivialAllocator([KVCacheBlock(i) for i in range(4)])
    assert isinstance(alloc, KVCacheBlockAllocator)


def test_generic_surface_has_no_allocator_specific_names():
    """The group spec, coordinator, and single-type manager must not name
    allocator-implementation concepts (buddy order, chunk_order, alloc_chunk,
    bit-shift translation)."""
    import vllm.v1.core.kv_cache_coordinator as coordinator_mod
    import vllm.v1.core.single_type_kv_cache_manager as manager_mod
    import vllm.v1.kv_cache_interface as interface_mod

    forbidden = ("chunk_order", "alloc_chunk", "buddy_order", "_buddy_order")
    for mod in (interface_mod, coordinator_mod, manager_mod):
        src = inspect.getsource(mod)
        for token in forbidden:
            assert token not in src, f"{token!r} found in {mod.__name__}"
