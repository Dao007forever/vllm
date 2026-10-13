# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan KV transfers between two engines from their page layouts.

The shared part answers one question: which bytes of a prefill (P) page hold the
same elements as which bytes of a decode (D) page. It stops at a byte
correspondence (:class:`PairPlan`) and two pure derivations of it: the direct
copies (:attr:`PairPlan.runs`) and the staged alternative
(:func:`staged_plan`), whole-page copies followed by a local permute. Which of
the two a connector uses is the connector's call, made through a
:class:`TransferStrategy`. Nothing here produces transport descriptors.

Pages are measured in head-token units unless a byte mode is given, so the
same plan serves any head size: a run of ``k`` units is ``k * unit`` bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Literal, Protocol

from vllm.distributed.kv_transfer.kv_layout.layout import (
    Layout,
    composition,
    is_contiguous,
    logical_divide,
    max_common_layout,
)

PageOrder = Literal["HND", "NHD"]


def head_range(rank: int, tp_size: int, total_heads: int) -> tuple[int, int]:
    """Global KV heads ``[lo, hi)`` held by ``rank`` of a ``tp_size`` engine.

    With more ranks than heads (GQA replication) every rank holds one head and
    ``tp_size // total_heads`` ranks share it, matching
    ``TransferTopology.local_physical_heads``.
    """
    per_rank = max(1, total_heads // tp_size)
    lo = rank * total_heads // tp_size
    return lo, lo + per_rank


@dataclass(frozen=True)
class KVPage:
    """One rank's page of one attention region.

    ``layout`` maps ``(unit byte, token, local head)`` to an offset inside the
    block. ``head_lo`` is the global index of local head 0. MLA and replicated
    KV are pages with one head that every rank holds, ``head_lo = 0``.
    """

    layout: Layout
    head_lo: int
    order: PageOrder | None = None

    @classmethod
    def attention(
        cls,
        tokens: int,
        local_heads: int,
        head_lo: int,
        order: PageOrder,
        unit: int = 1,
    ) -> KVPage:
        if order == "HND":
            stride = (1, unit, unit * tokens)
        elif order == "NHD":
            stride = (1, unit * local_heads, unit)
        else:
            raise ValueError(f"unknown page order {order!r}")
        return cls(Layout((unit, tokens, local_heads), stride), head_lo, order)

    @property
    def unit(self) -> int:
        return self.layout.shape[0]

    @property
    def tokens(self) -> int:
        return self.layout.shape[1]

    @property
    def local_heads(self) -> int:
        return self.layout.shape[2]

    @property
    def head_hi(self) -> int:
        return self.head_lo + self.local_heads

    @property
    def size(self) -> int:
        return self.layout.size

    def restrict(self, lo: int, hi: int) -> tuple[Layout, int]:
        """The page cut to global heads ``[lo, hi)`` plus the offset at which
        those heads start: ``composition`` with a picker that keeps the byte
        and token modes whole and takes ``hi - lo`` heads in a row."""
        if not (self.head_lo <= lo < hi <= self.head_hi):
            raise ValueError(f"heads [{lo}, {hi}) are not held by {self}")
        unit, tokens = self.unit, self.tokens
        picker = Layout((unit, tokens, hi - lo), (1, unit, unit * tokens))
        return composition(self.layout, picker), self.layout(0, 0, lo - self.head_lo)


@dataclass(frozen=True)
class PairPlan:
    """The byte correspondence between one P page and one D page.

    ``runs`` are ``(src, dst, length)`` triples relative to the start of each
    block, in head-token units, listed stamp by stamp. Every run is contiguous
    on both sides by construction; ``run_len`` is the largest length for which
    that holds. A connector that can issue one descriptor per run uses
    ``runs`` directly; one that cannot asks for :func:`staged_plan`.
    """

    src: KVPage
    dst: KVPage
    head_lo: int
    head_hi: int
    src_page: Layout
    dst_page: Layout
    src_off: int
    dst_off: int
    vec: Layout
    runs: tuple[tuple[int, int, int], ...]

    @property
    def num_heads(self) -> int:
        return self.head_hi - self.head_lo

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def run_len(self) -> int:
        return self.vec.size

    @property
    def src_contiguous(self) -> bool:
        """The shared heads are one run in the source page."""
        return is_contiguous(self.src_page)

    @property
    def dst_contiguous(self) -> bool:
        return is_contiguous(self.dst_page)


def plan_pair(src: KVPage, dst: KVPage) -> PairPlan | None:
    """Steps 2–5 of the procedure for one rank pair; ``None`` when the pages
    share no head and the pair exchanges nothing."""
    lo, hi = max(src.head_lo, dst.head_lo), min(src.head_hi, dst.head_hi)
    if lo >= hi:
        return None
    src_page, src_off = src.restrict(lo, hi)
    dst_page, dst_off = dst.restrict(lo, hi)
    vec = max_common_layout(src_page, dst_page)
    _, src_rest = logical_divide(src_page, vec)
    _, dst_rest = logical_divide(dst_page, vec)
    assert src_rest.size == dst_rest.size
    runs = tuple(
        (src_off + src_rest(i), dst_off + dst_rest(i), vec.size)
        for i in range(src_rest.size)
    )
    return PairPlan(src, dst, lo, hi, src_page, dst_page, src_off, dst_off, vec, runs)


@dataclass(frozen=True)
class LocalPermute:
    """A device-side reorder of one received block: the bytes arrived laid out
    as ``staging`` and the page wants them laid out as ``actual``. Both are over
    the same modes, so the permute is a transpose of the block viewed as a
    tensor in staging memory order."""

    shape: tuple[int, ...]
    staging: Layout
    actual: Layout
    tag: str = "general"

    @property
    def is_identity(self) -> bool:
        return self.staging.coalesce() == self.actual.coalesce()

    def _memory_order(self, layout: Layout) -> list[int]:
        """Modes above the unit mode, outermost first; size-1 modes dropped."""
        active = [i for i in range(1, len(self.shape)) if self.shape[i] > 1]
        return sorted(active, key=lambda i: -layout.stride[i])

    @property
    def view_dims(self) -> tuple[int, ...]:
        """The received block as a tensor, outermost mode first, with the unit
        (one head-token's payload) as an implicit innermost dimension."""
        return tuple(self.shape[i] for i in self._memory_order(self.staging))

    @property
    def perm(self) -> tuple[int, ...]:
        """Axis permutation of ``view_dims`` that yields the page's order; the
        payload dimension stays last."""
        staging_order = self._memory_order(self.staging)
        return tuple(staging_order.index(i) for i in self._memory_order(self.actual))

    @property
    def kind(self) -> str:
        """Which fix-up is needed: ``identity`` when the received order already
        is the page's order, else the ``tag`` the planner attached
        (``transpose_hn``, ``block_ratio``, ``block_ratio_transpose_hn`` or
        ``general``)."""
        return "identity" if self.is_identity else self.tag


def _is_transpose_hn(shape: tuple[int, ...], staging: Layout, actual: Layout) -> bool:
    """Staging is the whole block in HND order and the page wants NHD, over
    modes (unit, token, head modes...)."""
    unit, tokens, heads = shape[0], shape[1], shape[2:]
    hnd = [1, unit]
    nhd = [1, unit * prod(heads)]
    for i in range(len(heads)):
        hnd.append(unit * tokens * prod(heads[:i]))
        nhd.append(unit * prod(heads[:i]))
    return (
        staging.coalesce() == Layout(shape, tuple(hnd)).coalesce()
        and actual.coalesce() == Layout(shape, tuple(nhd)).coalesce()
    )


@dataclass(frozen=True)
class StagedPlan:
    """Whole-page copies into the destination block in source order, plus the
    permute that turns the block into the destination's layout."""

    pairs: tuple[PairPlan, ...]
    runs: tuple[tuple[int, int, int], ...]
    fixup: LocalPermute


def staged_plan(pairs: Sequence[PairPlan]) -> StagedPlan | None:
    """Stage every source's restricted page, concatenated in head order, at the
    destination block.

    Needs each source's shared heads to be one run on the source side, and the
    sources together to cover the destination's heads exactly once. Returns
    ``None`` otherwise: the scatter is on the source side, so staging at the
    destination cannot help.
    """
    if not pairs:
        return None
    dst = pairs[0].dst
    ordered = sorted(pairs, key=lambda p: p.head_lo)
    if any(p.dst != dst for p in ordered) or not all(p.src_contiguous for p in ordered):
        return None
    heads = [p.num_heads for p in ordered]
    if len(set(heads)) != 1 or sum(heads) != dst.local_heads:
        return None
    if any(p.head_lo != dst.head_lo + i * heads[0] for i, p in enumerate(ordered)):
        return None
    k, n_src = heads[0], len(ordered)
    unit, tokens = dst.unit, dst.tokens
    chunk = unit * tokens * k
    runs = tuple((p.src_off, i * chunk, chunk) for i, p in enumerate(ordered))
    # Modes (unit, token, head within source, which source): a source's
    # restricted page keeps its own strides and each source lands one chunk on.
    src_page = ordered[0].src_page
    staging = Layout(
        (unit, tokens, k, n_src),
        (src_page.stride[0], src_page.stride[1], src_page.stride[2], chunk),
    )
    dst_layout = dst.layout
    actual = Layout(
        (unit, tokens, k, n_src),
        (
            dst_layout.stride[0],
            dst_layout.stride[1],
            dst_layout.stride[2],
            dst_layout.stride[2] * k,
        ),
    )
    shape = (unit, tokens, k, n_src)
    tag = "transpose_hn" if _is_transpose_hn(shape, staging, actual) else "general"
    return StagedPlan(tuple(ordered), runs, LocalPermute(shape, staging, actual, tag))


def block_ratio_plan(src_order: PageOrder, dst: KVPage, ratio: int) -> LocalPermute:
    """The fix-up for a destination block that receives ``ratio`` source pages
    of ``tokens / ratio`` tokens each, holding the destination's heads in the
    source's page order, landed one after another in token order.

    Modes are (unit, token within a source page, head, which source page).
    """
    if dst.tokens % ratio:
        raise ValueError(f"{dst.tokens} tokens do not split into {ratio} pages")
    kb = dst.tokens // ratio
    chunk = KVPage.attention(kb, dst.local_heads, dst.head_lo, src_order, dst.unit)
    cs, ds = chunk.layout.stride, dst.layout.stride
    shape = (dst.unit, kb, dst.local_heads, ratio)
    staging = Layout(shape, (cs[0], cs[1], cs[2], chunk.size))
    actual = Layout(shape, (ds[0], ds[1], ds[2], ds[1] * kb))
    if dst.order == src_order:
        tag = "block_ratio"
    elif src_order == "HND" and dst.order == "NHD":
        tag = "block_ratio_transpose_hn"
    else:
        tag = "general"
    return LocalPermute(shape, staging, actual, tag)


@dataclass(frozen=True)
class Direct:
    """Issue every run of every pair as its own copy."""

    pairs: tuple[PairPlan, ...]


@dataclass(frozen=True)
class Staged:
    """Copy whole pages, then run ``plan.fixup`` on the received blocks."""

    plan: StagedPlan


@dataclass(frozen=True)
class Unsupported:
    reason: str


class TransferStrategy(Protocol):
    """How a connector realises a byte correspondence. Called once per remote
    engine at handshake with every (source rank → this rank) pair of one
    attention region."""

    def choose(self, pairs: Sequence[PairPlan]) -> Direct | Staged | Unsupported: ...


class SingleRunStrategy:
    """Policy for a transport that issues one descriptor per block per source.

    ``Direct`` needs every pair to be a single run. Otherwise the pairs are
    staged when their fix-up is one the receiver can execute
    (``supported_fixups``, by :attr:`LocalPermute.kind`).
    """

    def __init__(self, supported_fixups: frozenset[str] = frozenset({"identity"})):
        self.supported_fixups = supported_fixups

    def allows(self, fixup: LocalPermute) -> bool:
        """Whether the receiver can execute this fix-up after a staged copy."""
        return fixup.kind in self.supported_fixups

    def choose(self, pairs: Sequence[PairPlan]) -> Direct | Staged | Unsupported:
        if not pairs:
            return Unsupported("no source rank shares a KV head with this rank")
        if all(p.num_runs == 1 for p in pairs):
            return Direct(tuple(pairs))
        worst = max(pairs, key=lambda p: p.num_runs)
        plan = staged_plan(pairs)
        if plan is None:
            return Unsupported(
                f"shared heads are {worst.num_runs} runs of {worst.run_len} units in "
                f"{worst.dst.layout} and not one run in {worst.src.layout}: "
                "heads are not contiguous on either side"
            )
        if not self.allows(plan.fixup):
            return Unsupported(
                f"shared heads are {worst.num_runs} runs of {worst.run_len} units; "
                f"staging would need a {plan.fixup.kind!r} permute "
                f"{plan.fixup.staging} -> {plan.fixup.actual} on the receiver"
            )
        return Staged(plan)
