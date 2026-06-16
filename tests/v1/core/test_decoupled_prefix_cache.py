# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache correctness for span>1 (decoupled hybrid) groups (AC-8).

Fragmentation policy: a cached multi-base-block chunk is *preserved* — the
allocator never coalesces a chunk whose buddy holds a live cache hash, and only
drops a cached chunk's hash (via the eviction callback) when an allocation must
reclaim its memory. This keeps the prefix-cache hash map consistent: no stale
entry survives, and a chunk reused by a new request never carries a previous
owner's hash (so a cache "hit" can never return reused memory).
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

pytestmark = pytest.mark.cpu_test


def _buddy_pool(monkeypatch, num_gpu_blocks=8, max_order=2):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    return BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=1,
        max_allocation_span=1 << max_order,
    )


def _cache_block(pool, block, raw_hash, group_id=0):
    key = make_block_hash_with_group_id(raw_hash, group_id)
    block.block_hash = key
    pool.cached_block_hash_to_block.insert(key, block)


def test_span_block_cache_hit_then_evicted_on_reuse(monkeypatch):
    pool = _buddy_pool(monkeypatch)
    raw_hash = BlockHash(b"\x01" * 32)

    # Allocate a span-4 chunk, cache it, and free it back to the pool.
    (blk,) = pool.get_new_blocks(1, base_span=4)
    _cache_block(pool, blk, raw_hash)
    pool.free_blocks([blk])

    # While cached and free, it is a prefix-cache hit and is preserved.
    assert pool.get_cached_block(raw_hash, [0]) == [blk]

    # Under pressure, the next span-4 allocation must reclaim the cached chunk;
    # doing so drops its hash (no stale entry) and resets the block.
    (reused,) = pool.get_new_blocks(1, base_span=4)
    assert reused.block_id == blk.block_id
    assert reused.block_hash is None
    assert pool.get_cached_block(raw_hash, [0]) is None


def test_span_block_cache_survives_unrelated_allocation(monkeypatch):
    # A larger pool so an unrelated span-1 allocation does not need the cached
    # chunk's memory: the cached chunk must be preserved (no premature evict).
    pool = _buddy_pool(monkeypatch, num_gpu_blocks=16, max_order=2)
    raw_hash = BlockHash(b"\x02" * 32)

    (blk,) = pool.get_new_blocks(1, base_span=4)
    _cache_block(pool, blk, raw_hash)
    pool.free_blocks([blk])
    assert pool.get_cached_block(raw_hash, [0]) == [blk]

    # Allocate a few single base blocks elsewhere; the cached span-4 chunk is
    # untouched and still a hit.
    others = pool.get_new_blocks(2, base_span=1)
    assert all(b.block_id != blk.block_id for b in others)
    assert pool.get_cached_block(raw_hash, [0]) == [blk]
    assert blk.block_hash is not None
