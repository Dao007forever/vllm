# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single config path for the buddy/decoupled allocator (AC-6).

The decoupled mode and its parameters are registered in ``vllm.envs`` and read
through that single registry; no module performs ad-hoc ``os.environ`` parsing
for these flags.
"""

import pytest

import vllm.envs as envs
from vllm.v1.core.block_pool import BlockPool

pytestmark = pytest.mark.cpu_test


def test_envs_expose_buddy_flags(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    assert envs.VLLM_USE_BUDDY_BLOCK_POOL is True
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "0")
    assert envs.VLLM_USE_BUDDY_BLOCK_POOL is False


def test_blockpool_selects_allocator_from_config(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    # The max allocation span comes from the layout and is passed in by the
    # coordinator; there is no env knob for it. The pool forwards it to the
    # allocator, which sizes itself (span 4 -> buddy order 2).
    pool = BlockPool(
        num_gpu_blocks=8, enable_caching=False, hash_block_size=1, max_allocation_span=4
    )
    assert pool._buddy_mode is True
    assert pool.free_block_queue.max_order == 2

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "0")
    pool_off = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=1)
    assert pool_off._buddy_mode is False
