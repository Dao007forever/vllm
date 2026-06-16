# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission accounting for decoupled hybrid paging.

Covers two correctness properties of the decoupled (variable-span) path:

* Base-weighted admission: a group whose logical block spans ``2**order`` base
  blocks must report admission demand in base-block units, because the block
  pool tracks free capacity in base blocks. Counting a span>1 logical block as
  a single base block under-reserves capacity and can OOM after admission.
* Single source of truth: the per-group span comes solely from the group's
  spec/config (the ``base_span`` constructor argument). No environment
  variable may influence it.
"""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

pytestmark = pytest.mark.cpu_test


def _make_manager(base_span: int) -> SlidingWindowManager:
    block_size = 2
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # placeholder, not relevant to the count
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    return SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        max_admission_blocks_per_request=10**9,
        base_span=base_span,
    )


def _base_demand(manager, *args):
    """Mirror the coordinator's per-group base-block demand: the logical-block
    demand scaled by the group's allocation span (``logical * base_span``)."""
    logical = manager.get_num_blocks_to_allocate(*args)
    return logical, logical * manager.base_span


def test_uniform_span_base_demand_equals_logical():
    """span 1: base demand equals logical demand (baseline path)."""
    manager = _make_manager(base_span=1)
    block_size = manager.block_size
    cached = [KVCacheBlock(i + 1) for i in range(10)]

    assert manager.base_span == 1
    logical, base = _base_demand(
        manager, "1", 20 * block_size, cached, 0, 20 * block_size
    )
    assert logical == 20
    assert base == logical


def test_span_gt_one_scales_base_demand():
    """span 4: base demand is the logical demand times the span, covering both
    new blocks and evictable touched cached blocks."""
    span = 4
    block_size = 2
    # New-only request: 20 logical blocks -> 80 base blocks.
    manager_new = _make_manager(base_span=span)
    assert manager_new.base_span == 4
    cached_new = [KVCacheBlock(i + 1) for i in range(10)]
    logical_new, base_new = _base_demand(
        manager_new, "1", 20 * block_size, cached_new, 0, 20 * block_size
    )
    assert logical_new == 20
    assert base_new == logical_new * span == 80

    # Evictable-cached request: the evictable cached block is also counted in
    # base units (the free-capacity check must reserve its span).
    manager_evict = _make_manager(base_span=span)
    block_pool = manager_evict.block_pool
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate
    logical_evict, base_evict = _base_demand(
        manager_evict,
        "req",
        2 * block_size,
        [evictable_block],
        block_size,
        2 * block_size,
    )
    assert logical_evict == 2
    assert base_evict == logical_evict * span == 8


def test_span_source_is_spec_not_env(monkeypatch):
    """The span comes only from the spec/config; environment variables that
    used to override per-group order have no effect anymore (AC-2)."""
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    monkeypatch.setenv("VLLM_BUDDY_GROUP_ORDERS", "7")
    manager = _make_manager(base_span=4)
    # The constructor argument wins; the env var is ignored entirely.
    assert manager.base_span == 4
    assert manager._base_span == 4


def test_blockpool_uses_caller_derived_buddy_order(monkeypatch):
    """In buddy mode the queue supports exactly the caller-derived order (from
    the layout's spans). Otherwise a span>1 demand is rejected by
    ``can_allocate_*`` and the spanned allocation raises."""
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    pool = BlockPool(
        num_gpu_blocks=64,
        enable_caching=False,
        hash_block_size=1,
        max_allocation_span=4,  # a span-4 group; buddy sizes to order 2
    )
    assert pool.free_block_queue.max_order == 2
    # Span 4 (and 2) are schedulable; at order 0 these would be rejected,
    # making span>1 groups unschedulable.
    assert pool.can_allocate_demands({4: 1})
    assert pool.can_allocate_demands({2: 2})


def test_coordinator_sizes_allocator_from_group_spans(monkeypatch):
    """Regression: enabling buddy on a config whose layout assigned a span>1
    group must yield a buddy queue able to carve that span, with no manual
    tuning. The coordinator derives the order from the largest
    ``allocation_base_span``."""
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    block_size = 2
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                allocation_base_span=4,
            )
        ],
    )
    coord = get_kv_cache_coordinator(
        config,
        max_model_len=64,
        max_num_batched_tokens=64,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    # span 4 -> order 2, derived purely from the config.
    assert coord.block_pool.free_block_queue.max_order >= 2
    assert coord.block_pool.can_allocate_demands({4: 1})
