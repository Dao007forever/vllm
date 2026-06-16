# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout resolution for decoupled hybrid paging.

Verifies the attention-span guard (AC-3): in decoupled mode the base page is
anchored to the smallest attention page, attention groups must end up at span 1
so standard-attention strided views stay valid, and a
non-attention (mamba) group whose native page is larger gets a span > 1. An
attention group that cannot be represented at the base page is rejected with a
clear error rather than silently given an unsafe span > 1.
"""

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test


def _attn(head_size=64, block_size=16):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=head_size,
        dtype=torch.float32,
    )


def _mamba(block_size=16, shapes=((2, 8192), (3, 128, 128))):
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=(torch.float32, torch.float32),
        num_speculative_blocks=2,
        mamba_cache_mode="none",
    )


def test_decoupled_attention_span_one_mamba_span_gt_one(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    attn = _attn()
    mamba = _mamba()
    # Precondition for a span > 1 mamba group: its native page exceeds the
    # attention page that becomes the base page.
    assert mamba.page_size_bytes > attn.page_size_bytes

    groups = [
        KVCacheGroupSpec(["attn0"], attn),
        KVCacheGroupSpec(["mamba0"], mamba),
    ]
    cfg = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=attn.page_size_bytes * 4096
    )

    assert cfg.base_page_bytes == attn.page_size_bytes
    # Attention group is pinned to span 1.
    assert groups[0].allocation_base_span == 1
    # Mamba group spans more than one base block, enough to cover its page.
    assert groups[1].allocation_base_span >= 2
    assert groups[1].allocation_base_span * cfg.base_page_bytes >= mamba.page_size_bytes


def test_mamba_span_is_allocator_normalized_natural_span(monkeypatch):
    """The layout requests the natural span ceil(page / base_page) and the
    buddy allocator rounds it; the stored span must equal what the allocator's
    normalize_span produces (the power-of-two rounding now lives there, not in
    the config)."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    attn = _attn()
    mamba = _mamba()
    groups = [
        KVCacheGroupSpec(["attn0"], attn),
        KVCacheGroupSpec(["mamba0"], mamba),
    ]
    cfg = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=attn.page_size_bytes * 4096
    )
    natural = cdiv(mamba.page_size_bytes, cfg.base_page_bytes)
    assert groups[1].allocation_base_span == BuddyFreeKVCacheBlockQueue.normalize_span(
        natural
    )


def test_memory_estimate_uses_same_effective_span_as_layout(monkeypatch):
    """The decoupled memory estimate must count base blocks at the same
    allocator-normalized span the layout assigns, or admission drifts from real
    allocation."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.buddy_free_queue import BuddyFreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    attn = _attn()
    mamba = _mamba()
    groups = [
        KVCacheGroupSpec(["attn0"], attn),
        KVCacheGroupSpec(["mamba0"], mamba),
    ]
    estimate = _max_memory_usage_bytes_from_groups(vllm_config, groups)

    base_page = min(g.kv_cache_spec.page_size_bytes for g in groups)
    group_size = max(len(g.layer_names) for g in groups)
    expected_base_blocks = 0
    for g in groups:
        natural = cdiv(g.kv_cache_spec.page_size_bytes, base_page)
        effective = BuddyFreeKVCacheBlockQueue.normalize_span(natural)
        logical = cdiv(
            g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            g.kv_cache_spec.page_size_bytes,
        )
        expected_base_blocks += logical * effective
    assert estimate == group_size * base_page * expected_base_blocks


def test_memory_estimate_anchors_base_page_to_attention_not_smallest(monkeypatch):
    """Regression: when a non-attention page is SMALLER than the attention
    page, the base page must still anchor to the smallest *attention* page (as
    the layout does), not to the smallest page overall. Anchoring to the
    smaller mamba page would give attention a span > 1 in the estimate while
    the layout pins it to span 1 — capacity planning would diverge from the
    realized tensor layout."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    attn = _attn(head_size=512)  # large attention page
    mamba = _mamba(shapes=((2, 64),))  # tiny mamba page
    # Precondition: the divergent case the old min(all) rule mishandled.
    assert mamba.page_size_bytes < attn.page_size_bytes

    groups = [
        KVCacheGroupSpec(["attn0"], attn),
        KVCacheGroupSpec(["mamba0"], mamba),
    ]
    cfg = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=attn.page_size_bytes * 4096
    )
    # Layout anchored to the attention page, NOT the smaller mamba page.
    assert cfg.base_page_bytes == attn.page_size_bytes
    assert groups[0].allocation_base_span == 1  # attention pinned to span 1

    # The estimate must agree with the layout: same base page, same per-group
    # spans (read off the groups the layout just populated).
    estimate = _max_memory_usage_bytes_from_groups(vllm_config, groups)
    group_size = max(len(g.layer_names) for g in groups)
    expected_base_blocks = 0
    for g in groups:
        logical = cdiv(
            g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            g.kv_cache_spec.page_size_bytes,
        )
        expected_base_blocks += logical * g.allocation_base_span
    assert estimate == group_size * cfg.base_page_bytes * expected_base_blocks


def test_decoupled_max_concurrency_uses_base_page_not_group0(monkeypatch):
    """Reported capacity (max concurrency / "GPU KV cache size: tokens") must
    size per-request blocks by the layout's actual ``base_page_bytes``, not
    ``groups[0]``'s native page. Here group[0] is a mamba group whose native
    page is far smaller than the base (attention) page — the Kimi-Linear
    ordering — so reading groups[0]'s page would inflate per-request blocks and
    under-report capacity by the base/native ratio."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.kv_cache_utils import (
        get_max_concurrency_for_kv_cache_config,
        max_memory_usage_bytes,
    )

    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    mamba = _mamba(shapes=((2, 64),))  # tiny native page
    attn = _attn(head_size=512)  # large page -> becomes base_page
    assert mamba.page_size_bytes < attn.page_size_bytes
    groups = [
        KVCacheGroupSpec(["mamba0"], mamba),  # group[0]: native << base_page
        KVCacheGroupSpec(["attn0"], attn),
    ]
    cfg = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=attn.page_size_bytes * 4096
    )
    assert cfg.base_page_bytes == attn.page_size_bytes
    # group[0]'s native page differs from base_page, so the buggy formula (read
    # groups[0]) and the correct one (read base_page) diverge — making the
    # assertion below meaningful rather than vacuous.
    assert cfg.kv_cache_groups[0].kv_cache_spec.page_size_bytes != cfg.base_page_bytes

    nlpg = max(len(g.layer_names) for g in cfg.kv_cache_groups)
    max_mem = nlpg * max_memory_usage_bytes(
        vllm_config, (g.kv_cache_spec for g in cfg.kv_cache_groups)
    )
    expected = cfg.num_blocks / cdiv(max_mem, cfg.base_page_bytes * nlpg)
    assert get_max_concurrency_for_kv_cache_config(vllm_config, cfg) == expected


def test_decoupled_rejects_attention_span_gt_one(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BUDDY_BLOCK_POOL", "1")
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    small_attn = _attn(head_size=64)
    big_attn = _attn(head_size=256)  # larger page than the base (small) attn page
    assert big_attn.page_size_bytes > small_attn.page_size_bytes

    groups = [
        KVCacheGroupSpec(["a0"], small_attn),
        KVCacheGroupSpec(["a1"], big_attn),
    ]
    with pytest.raises(ValueError, match="attention"):
        get_kv_cache_config_from_groups(
            vllm_config, groups, available_memory=big_attn.page_size_bytes * 4096
        )
