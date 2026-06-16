# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator-owned span normalization.

The decoupled layout requests a *natural* span (base blocks a group's page
needs); the allocator rounds it to a granularity it can carve and align. This
keeps the power-of-two rounding inside the buddy allocator instead of inlined in
the config, so a span-capable allocator could realize a natural span with no
waste. These tests pin the rounding policy of each built-in allocator and the
resolver the config uses to pick one.
"""

import pytest

from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    _active_span_normalizer,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "natural,expected",
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16)],
)
def test_buddy_normalize_span_rounds_to_power_of_two(natural, expected):
    assert BuddyFreeKVCacheBlockQueue.normalize_span(natural) == expected


def test_lru_normalize_span_only_supports_span_one():
    assert FreeKVCacheBlockQueue.normalize_span(1) == 1
    assert FreeKVCacheBlockQueue.normalize_span(0) == 1
    for natural in (2, 3, 5):
        with pytest.raises(ValueError, match="span"):
            FreeKVCacheBlockQueue.normalize_span(natural)


def test_active_span_normalizer_follows_buddy_flag(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    assert _active_span_normalizer() is BuddyFreeKVCacheBlockQueue.normalize_span
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "0")
    assert _active_span_normalizer() is FreeKVCacheBlockQueue.normalize_span


def test_buddy_alloc_records_normalized_span_on_block():
    # A chunk handed out at order k carries base_span == 2**k, the neutral
    # base-block count the rename exposes (no buddy "order" on the block).
    blocks = [KVCacheBlock(i) for i in range(16)]
    q = BuddyFreeKVCacheBlockQueue(blocks, max_order=4)
    head = q.alloc_chunk(order=2)
    assert head.base_span == 4
    assert head.base_span == BuddyFreeKVCacheBlockQueue.normalize_span(3)
