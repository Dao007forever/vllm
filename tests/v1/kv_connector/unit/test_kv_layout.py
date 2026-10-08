# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden tests for the KV layout algebra and the transfer planner.

Every expected value below was printed by CuTe (``include/cute/layout.hpp``)
for the same inputs, or follows from the CuTe docs examples. The planner cases
are the worked examples of the PD transfer proposal: 8 KV heads, 512 B per
head-token, 16 tokens per block.
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_layout import (
    Direct,
    KVPage,
    Layout,
    LayoutError,
    SingleRunStrategy,
    Staged,
    Unsupported,
    block_ratio_plan,
    complement,
    composition,
    head_range,
    is_contiguous,
    logical_divide,
    max_common_layout,
    plan_pair,
    right_inverse,
    staged_plan,
)

pytestmark = pytest.mark.cpu_test

C = 512


def L(shape, stride):
    return Layout(tuple(shape), tuple(stride))


# ---------------------------------------------------------------- algebra


def test_layout_is_a_function_of_colex_coordinates():
    a = L((4, 8), (8, 1))
    assert a(3, 5) == 29
    assert a(23) == a(3, 5)
    assert a.coord(23) == (3, 5)
    assert a.size == 32 and a.cosize == 32
    assert str(a) == "(4,8):(8,1)"


def test_ordered_layout_assigns_strides_by_rank():
    assert Layout.ordered((512, 16, 4), (0, 1, 2)) == L((512, 16, 4), (1, 512, 8192))
    assert Layout.ordered((512, 16, 4), (0, 2, 1)) == L((512, 16, 4), (1, 2048, 512))


def test_coalesce_merges_chained_modes_and_drops_size_one():
    assert L((2, 1, 6), (1, 6, 2)).coalesce() == L((12,), (1,))
    assert L((512, 16, 2), (1, 512, 8192)).coalesce() == L((16384,), (1,))
    assert L((512, 16, 2), (1, 2048, 512)).coalesce() == L((512, 16, 2), (1, 2048, 512))


def test_composition_matches_cute_docs_example():
    a = L((6, 2), (8, 2))
    b = L((4, 3), (3, 1))
    assert composition(a, b) == L((2, 2, 3), (24, 2, 8))


def test_composition_picks_the_first_heads():
    page = L((512, 16, 4), (1, 512, 8192))
    picker = L((512, 16, 2), (1, 512, 8192))
    assert composition(page, picker) == L((512, 16, 2), (1, 512, 8192))
    nhd = L((512, 16, 4), (1, 2048, 512))
    assert composition(nhd, picker) == L((512, 16, 2), (1, 2048, 512))


def test_composition_rejects_non_divisible_shapes():
    try:
        composition(L((3, 4), (1, 3)), L((2,), (2,)))
    except LayoutError:
        pass
    else:
        raise AssertionError("expected a divisibility error")


def test_complement_table():
    assert complement(L((4,), (1,)), 16) == L((4,), (4,))
    assert complement(L((4,), (2,)), 16) == L((2, 2), (1, 8))
    assert complement(L((2, 2), (1, 4)), 16) == L((2, 2), (2, 8))
    assert complement(L((2,), (4,)), 16) == L((4, 2), (1, 8))
    assert complement(L((16,), (1,)), 16) == L((1,), (0,))
    assert complement(L((4,), (2,)), 24) == L((2, 3), (1, 8))
    assert complement(L((512, 2), (1, 8192)), 16384) == L((16,), (512,))
    assert complement(L((2, 2), (2, 8)), 16) == L((2, 2), (1, 4))


def test_complement_refuses_overlap():
    try:
        complement(L((2, 2), (1, 1)), 8)
    except LayoutError:
        pass
    else:
        raise AssertionError("expected a non-injective error")


def test_right_inverse_walks_the_gap_free_prefix():
    assert right_inverse(L((2, 2, 2), (1, 8, 2))) == L((2, 2), (1, 4))
    assert right_inverse(L((2, 2, 2), (1, 4, 2))) == L((2, 2, 2), (1, 4, 2))
    assert right_inverse(L((4, 2), (2, 1))) == L((2, 4), (4, 1))
    assert right_inverse(L((4, 8), (8, 1))) == L((8, 4), (4, 1))
    assert right_inverse(L((2, 2), (1, 3))) == L((2,), (1,))
    assert right_inverse(L((4,), (2,))) == L((1,), (0,))


def test_max_common_layout_hnd_and_nhd():
    src = L((512, 16, 2), (1, 1024, 512))
    dst = L((512, 16, 2), (1, 2048, 512))
    assert max_common_layout(src, dst) == L((512, 2), (1, 8192))
    hnd = L((512, 16, 2), (1, 512, 8192))
    assert max_common_layout(hnd, hnd) == L((16384,), (1,))
    # HND -> NHD over single-byte head-tokens: no run at all.
    assert max_common_layout(L((4, 2), (1, 4)), L((4, 2), (2, 1))) == L((1,), (0,))
    # 2 B per head-token: runs of one head-token.
    assert max_common_layout(L((2, 4, 2), (1, 2, 8)), L((2, 4, 2), (1, 4, 2))) == L(
        (2,), (1,)
    )


def test_logical_divide_matches_cute_docs_example():
    tile, rest = logical_divide(L((4, 2, 3), (2, 1, 8)), L((4,), (2,)))
    assert tile == L((2, 2), (4, 1))
    assert rest == L((2, 3), (2, 8))


def test_logical_divide_lists_the_copies():
    src = L((2, 4, 2), (1, 4, 2))
    dst = L((2, 4, 2), (1, 8, 2))
    vec = max_common_layout(src, dst)
    assert vec == L((2, 2), (1, 8))
    tile_s, rest_s = logical_divide(src, vec)
    tile_d, rest_d = logical_divide(dst, vec)
    assert tile_s == L((4,), (1,)) and rest_s == L((4,), (4,))
    assert tile_d == L((4,), (1,)) and rest_d == L((4,), (8,))
    assert is_contiguous(tile_s) and is_contiguous(tile_d)


# ---------------------------------------------------------------- planner


def test_head_range_with_and_without_replication():
    assert [head_range(r, 4, 8) for r in range(4)] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert [head_range(r, 8, 4) for r in range(8)] == [
        (0, 1),
        (0, 1),
        (1, 2),
        (1, 2),
        (2, 3),
        (2, 3),
        (3, 4),
        (3, 4),
    ]


def page(tp, rank, order, total_heads=8, tokens=16):
    lo, _ = head_range(rank, tp, total_heads)
    return KVPage.attention(tokens, max(1, total_heads // tp), lo, order, unit=C)


def test_example_a_same_tp_hnd():
    for d in range(4):
        plan = plan_pair(page(4, d, "HND"), page(4, d, "HND"))
        assert plan is not None
        assert plan.runs == ((0, 0, 16384),)
    assert plan_pair(page(4, 0, "HND"), page(4, 1, "HND")) is None


def test_example_b_fan_in_hnd():
    d0 = page(2, 0, "HND")
    assert plan_pair(page(4, 0, "HND"), d0).runs == ((0, 0, 16384),)
    assert plan_pair(page(4, 1, "HND"), d0).runs == ((0, 16384, 16384),)
    d0 = page(1, 0, "HND")
    assert [plan_pair(page(4, p, "HND"), d0).runs[0][1] for p in range(4)] == [
        0,
        16384,
        32768,
        49152,
    ]


def test_example_c_fan_out_hnd():
    p0 = page(2, 0, "HND")
    assert plan_pair(p0, page(4, 0, "HND")).runs == ((0, 0, 16384),)
    assert plan_pair(p0, page(4, 1, "HND")).runs == ((16384, 0, 16384),)
    assert plan_pair(p0, page(4, 2, "HND")) is None


def test_examples_in_nhd():
    assert plan_pair(page(4, 1, "NHD"), page(4, 1, "NHD")).runs == ((0, 0, 16384),)
    fan_in = plan_pair(page(4, 1, "NHD"), page(2, 0, "NHD"))
    assert fan_in.vec == L((512, 2), (1, 8192))
    assert fan_in.num_runs == 16 and fan_in.run_len == 1024
    assert fan_in.runs[:3] == ((0, 1024, 1024), (1024, 3072, 1024), (2048, 5120, 1024))
    assert fan_in.src_contiguous and not fan_in.dst_contiguous
    fan_out = plan_pair(page(2, 0, "NHD"), page(4, 1, "NHD"))
    assert fan_out.runs[:2] == ((1024, 0, 1024), (3072, 1024, 1024))
    assert not fan_out.src_contiguous


def test_mixed_orders_same_tp():
    plan = plan_pair(page(4, 1, "HND"), page(4, 1, "NHD"))
    assert plan.num_runs == 32 and plan.run_len == 512


def test_units_scale_bytes():
    plan = plan_pair(page(4, 1, "NHD", tokens=16), page(2, 0, "NHD"))
    units = plan_pair(
        KVPage.attention(16, 2, 2, "NHD"), KVPage.attention(16, 4, 0, "NHD")
    )
    assert units.num_runs == plan.num_runs
    assert [(s * C, d * C, n * C) for s, d, n in units.runs] == list(plan.runs)


def test_staged_fan_in_hnd_sources_into_nhd_is_a_transpose():
    d0 = page(2, 0, "NHD")
    pairs = [plan_pair(page(4, p, "HND"), d0) for p in (1, 0)]
    plan = staged_plan(pairs)
    assert plan is not None
    assert plan.runs == ((0, 0, 16384), (0, 16384, 16384))
    assert plan.fixup.kind == "transpose_hn"
    assert plan.fixup.staging == L((512, 16, 2, 2), (1, 512, 8192, 16384))
    assert plan.fixup.staging.coalesce() == L((32768,), (1,))
    assert plan.fixup.actual.coalesce() == L((512, 16, 4), (1, 2048, 512))
    assert plan.fixup.view_dims == (2, 2, 16)
    assert plan.fixup.perm == (2, 0, 1)


def test_staged_fan_in_nhd_sources_is_a_general_permute():
    d0 = page(2, 0, "NHD")
    plan = staged_plan([plan_pair(page(4, p, "NHD"), d0) for p in (0, 1)])
    assert plan is not None
    assert plan.fixup.staging == L((512, 16, 2, 2), (1, 1024, 512, 16384))
    assert plan.fixup.actual == L((512, 16, 2, 2), (1, 2048, 512, 1024))
    assert plan.fixup.kind == "general"
    assert plan.fixup.view_dims == (2, 16, 2)
    assert plan.fixup.perm == (1, 0, 2)


def test_staged_is_identity_when_orders_agree():
    plan = staged_plan(
        [plan_pair(page(4, p, "HND"), page(2, 0, "HND")) for p in (0, 1)]
    )
    assert plan.fixup.kind == "identity"


def test_staged_fan_out_nhd_has_the_scatter_on_the_source():
    assert staged_plan([plan_pair(page(2, 0, "NHD"), page(4, 1, "NHD"))]) is None


def test_single_run_strategy():
    default = SingleRunStrategy()
    with_transpose = SingleRunStrategy(frozenset({"identity", "transpose_hn"}))
    same = [plan_pair(page(4, 1, "HND"), page(4, 1, "HND"))]
    assert isinstance(default.choose(same), Direct)
    fan_in_hnd = [plan_pair(page(4, p, "HND"), page(2, 0, "HND")) for p in (0, 1)]
    assert isinstance(default.choose(fan_in_hnd), Direct)
    hnd_to_nhd = [plan_pair(page(2, 0, "HND"), page(4, 1, "NHD"))]
    assert isinstance(default.choose(hnd_to_nhd), Unsupported)
    assert isinstance(with_transpose.choose(hnd_to_nhd), Staged)
    nhd_fan_in = [plan_pair(page(4, p, "NHD"), page(2, 0, "NHD")) for p in (0, 1)]
    assert isinstance(with_transpose.choose(nhd_fan_in), Unsupported)
    nhd_fan_out = [plan_pair(page(2, 0, "NHD"), page(4, 1, "NHD"))]
    assert isinstance(with_transpose.choose(nhd_fan_out), Unsupported)
    assert isinstance(default.choose([]), Unsupported)


def test_mla_pages_are_one_run():
    mla = [
        plan_pair(
            KVPage.attention(64, 1, 0, "HND", unit=656),
            KVPage.attention(64, 1, 0, "NHD", unit=656),
        )
    ]
    assert mla[0].runs == ((0, 0, 64 * 656),)
    assert isinstance(SingleRunStrategy().choose(mla), Direct)


# ---------------------------------------------------------- fix-up executor


def _hnd_pages(truth, heads_per_page, tokens_per_page):
    """Split a logical [H, N, D] block into HND pages of the given size."""
    pages = []
    for h in range(0, truth.shape[0], heads_per_page):
        for t in range(0, truth.shape[1], tokens_per_page):
            pages.append(
                truth[h : h + heads_per_page, t : t + tokens_per_page].contiguous()
            )
    return pages


def _receive(runs, pages, numel, head_dim):
    """Land each page whole at its staged offset (in head-token units)."""
    block = torch.full((numel,), -1, dtype=torch.int64)
    for (src, dst, n), page in zip(runs, pages):
        block[dst * head_dim : (dst + n) * head_dim] = page.reshape(-1)[
            src * head_dim : (src + n) * head_dim
        ]
    return block


def _caches(order, tokens, heads, head_dim, received):
    """A logical [B, H, N, D] view over a physical block that holds ``received``,
    plus the physical tensor, for either page order."""
    if order == "HND":
        phys = torch.empty(1, heads, tokens, head_dim, dtype=torch.int64)
        phys.reshape(-1)[:] = received
        return phys, phys
    phys = torch.empty(1, tokens, heads, head_dim, dtype=torch.int64)
    phys.reshape(-1)[:] = received
    return phys, phys.permute(0, 2, 1, 3)


def test_apply_local_permute_transpose_hn_on_logical_nhd_view():
    """Fan-in of HND pages into an NHD cache: the staged copy plus the
    planner's permute leaves the same bytes as direct runs, applied through
    the logical [B, H, N, D] view NIXL registers."""
    from vllm.distributed.kv_transfer.kv_connector.utils import apply_local_permute

    tokens, heads, head_dim = 16, 4, 8
    truth = torch.arange(heads * tokens * head_dim).reshape(heads, tokens, head_dim)
    pages = _hnd_pages(truth, 2, tokens)
    pairs = [
        plan_pair(
            KVPage.attention(tokens, 2, 2 * p, "HND"),
            KVPage.attention(tokens, heads, 0, "NHD"),
        )
        for p in (0, 1)
    ]
    plan = staged_plan(pairs)
    assert plan.fixup.kind == "transpose_hn"

    direct = torch.empty(truth.numel(), dtype=torch.int64)
    for pair, page in zip(pairs, pages):
        for src, dst, n in pair.runs:
            direct[dst * head_dim : (dst + n) * head_dim] = page.reshape(-1)[
                src * head_dim : (src + n) * head_dim
            ]
    received = _receive(plan.runs, pages, truth.numel(), head_dim)
    phys, cache = _caches("NHD", tokens, heads, head_dim, received)
    apply_local_permute(cache, torch.tensor([0]), plan.fixup)
    assert torch.equal(phys.reshape(-1), direct)
    assert torch.equal(cache[0], truth)


@pytest.mark.parametrize("dst_order", ["HND", "NHD"])
def test_apply_local_permute_block_ratio(dst_order):
    """A block of 16 tokens receives four HND pages of 4 tokens landed one
    after another; the planner's block-ratio permute puts every token of
    every head where the page expects it."""
    from vllm.distributed.kv_transfer.kv_connector.utils import apply_local_permute

    tokens, heads, head_dim, ratio = 16, 2, 8, 4
    truth = torch.arange(heads * tokens * head_dim).reshape(heads, tokens, head_dim)
    dst = KVPage.attention(tokens, heads, 0, dst_order)
    fixup = block_ratio_plan("HND", dst, ratio)
    assert fixup.kind == (
        "block_ratio" if dst_order == "HND" else "block_ratio_transpose_hn"
    )
    pages = _hnd_pages(truth, heads, tokens // ratio)
    chunk = heads * tokens // ratio
    received = _receive(
        [(0, i * chunk, chunk) for i in range(ratio)], pages, truth.numel(), head_dim
    )
    phys, cache = _caches(dst_order, tokens, heads, head_dim, received)
    apply_local_permute(cache, torch.tensor([0]), fixup)
    assert torch.equal(cache[0], truth)


def test_block_ratio_same_order_pages_need_no_permute():
    """NHD pages landed in token order already form an NHD block, and a
    single-head (MLA) block never needs a permute whatever the orders."""
    assert block_ratio_plan("NHD", KVPage.attention(16, 4, 0, "NHD"), 4).is_identity
    assert block_ratio_plan(
        "HND", KVPage.attention(64, 1, 0, "NHD", unit=656), 2
    ).is_identity
    assert not block_ratio_plan("HND", KVPage.attention(16, 4, 0, "HND"), 4).is_identity


def test_strategy_allows_only_executable_fixups():
    nixl_default = SingleRunStrategy(frozenset({"identity", "block_ratio"}))
    hnd_hnd = block_ratio_plan("HND", KVPage.attention(16, 2, 0, "HND"), 2)
    hnd_nhd = block_ratio_plan("HND", KVPage.attention(16, 2, 0, "NHD"), 2)
    assert nixl_default.allows(hnd_hnd)
    assert not nixl_default.allows(hnd_nhd)
    assert SingleRunStrategy(frozenset({"block_ratio_transpose_hn"})).allows(hnd_nhd)
