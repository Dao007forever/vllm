# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRUEvictionPolicy: the reclaim-candidate store an allocator composes.

Holds released-but-cached blocks in reuse-priority order; ``pop_victim``
surrenders the least-recently-used candidate after firing ``on_evict`` to drop
its hash; ``untrack`` removes a specific candidate on a prefix-cache hit.
"""

import pytest

from vllm.v1.core.eviction_policy import LRUEvictionPolicy
from vllm.v1.core.kv_cache_utils import KVCacheBlock

pytestmark = pytest.mark.cpu_test


def _blk(i, span=1):
    b = KVCacheBlock(i)
    b.base_span = span
    return b


def test_track_untrack_accounting():
    ev = LRUEvictionPolicy()
    a, b = _blk(1), _blk(2, span=4)
    ev.track(a)
    ev.track(b)
    assert ev.num_reclaimable == 1 + 4
    assert set(x.block_id for x in ev.reclaimable_chunks()) == {1, 2}
    ev.untrack(a)
    assert ev.num_reclaimable == 4
    assert a.prev_free_block is None and a.next_free_block is None


def test_pop_victim_is_lru_order():
    ev = LRUEvictionPolicy()
    first, second, third = _blk(10), _blk(11), _blk(12)
    ev.track(first)
    ev.track(second)
    ev.track(third)
    # Oldest (first tracked) is reclaimed first.
    assert ev.pop_victim().block_id == 10
    assert ev.pop_victim().block_id == 11
    assert ev.pop_victim().block_id == 12
    assert ev.pop_victim() is None


def test_prefer_reuse_reclaimed_first():
    ev = LRUEvictionPolicy()
    ev.track(_blk(10))
    ev.track(_blk(11))
    ev.track(_blk(12), prefer_reuse=True)  # jump to the reclaim-soon end
    assert ev.pop_victim().block_id == 12
    assert ev.pop_victim().block_id == 10


def test_pop_victim_fires_on_evict_and_updates_count():
    evicted = []

    def on_evict(block):
        evicted.append(block.block_id)
        block.reset_hash()
        return True

    ev = LRUEvictionPolicy(on_evict=on_evict)
    b = _blk(7, span=2)
    b.block_hash = ("h", 7)
    ev.track(b)
    assert ev.num_reclaimable == 2
    victim = ev.pop_victim()
    assert victim is b
    assert evicted == [7]
    assert b.block_hash is None  # on_evict reset it
    assert ev.num_reclaimable == 0


def test_untrack_unknown_block_raises():
    ev = LRUEvictionPolicy()
    with pytest.raises(RuntimeError):
        ev.untrack(_blk(99))
