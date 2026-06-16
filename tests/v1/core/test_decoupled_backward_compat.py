# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward compatibility of the decoupled allocator changes (AC-9).

* With the buddy path disabled, the block pool uses the unchanged LRU queue and
  the baseline allocation behaviour.
* With the buddy path enabled but every group at span 1 (``max_order == 0``),
  fresh allocation, the null block, free accounting, and prefix-cache hits
  match the uniform path.

Note: the buddy queue deliberately links *uncached* free blocks at the head (so
allocation prefers uncached chunks and preserves cached ones). That makes the
post-eviction reuse *order* differ from the plain LRU queue; this is a
heuristic difference, not a correctness one, so it is not asserted here.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

pytestmark = pytest.mark.cpu_test


def test_buddy_off_uses_lru_baseline(monkeypatch):
    monkeypatch.delenv("VLLM_USE_BUDDY_BLOCK_POOL", raising=False)
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=1)
    assert type(pool.free_block_queue).__name__ == "FreeKVCacheBlockQueue"
    assert pool.null_block.block_id == 0
    blocks = pool.get_new_blocks(5)
    assert [b.block_id for b in blocks] == [1, 2, 3, 4, 5]


def test_buddy_span1_matches_uniform_fresh_alloc(monkeypatch):
    monkeypatch.delenv("VLLM_USE_BUDDY_BLOCK_POOL", raising=False)
    base = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=1)

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    # No spanned groups -> derived buddy order is 0 (the default); a span-1
    # buddy pool must behave identically to the uniform LRU pool.
    buddy = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=1)

    assert base.null_block.block_id == buddy.null_block.block_id == 0
    assert base.get_num_free_blocks() == buddy.get_num_free_blocks()

    a = base.get_new_blocks(5)
    b = buddy.get_new_blocks(5)
    assert [x.block_id for x in a] == [x.block_id for x in b]
    assert base.get_num_free_blocks() == buddy.get_num_free_blocks()


def test_buddy_span1_cache_hit_cycle(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=1)

    (blk,) = pool.get_new_blocks(1)
    raw = BlockHash(b"\x07" * 32)
    key = make_block_hash_with_group_id(raw, 0)
    blk.block_hash = key
    pool.cached_block_hash_to_block.insert(key, blk)
    pool.free_blocks([blk])
    # A span-1 cached block is a normal prefix-cache hit under the buddy queue.
    assert pool.get_cached_block(raw, [0]) == [blk]
