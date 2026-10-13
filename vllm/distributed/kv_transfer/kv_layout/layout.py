# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A torch-free port of the CuTe layout algebra used by the KV transfer planner.

A :class:`Layout` is a function from a logical index to an offset:
``L(i) = sum_k coord_k(i) * stride_k`` where ``coord(i)`` is the colexicographic
(leftmost mode fastest) coordinate of ``i`` in ``shape``. Only flat (non-nested)
layouts are supported; every operation the planner needs can be expressed on
flat modes. Semantics follow ``include/cute/layout.hpp`` in CUTLASS:
``coalesce``, ``composition``, ``complement``, ``right_inverse``,
``max_common_layout`` and ``logical_divide``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import prod


class LayoutError(ValueError):
    """Raised when an operation is undefined for its inputs, mirroring the
    static assertions CuTe raises at compile time (divisibility, injectivity)."""


@dataclass(frozen=True)
class Layout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __post_init__(self) -> None:
        shape = tuple(int(s) for s in self.shape)
        stride = tuple(int(d) for d in self.stride)
        if len(shape) != len(stride):
            raise LayoutError(f"shape {shape} and stride {stride} differ in rank")
        if not shape:
            raise LayoutError("a layout needs at least one mode")
        if any(s < 1 for s in shape) or any(d < 0 for d in stride):
            raise LayoutError(f"invalid layout {shape}:{stride}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)

    @classmethod
    def from_modes(cls, modes: Iterable[tuple[int, int]]) -> Layout:
        modes = tuple(modes)
        if not modes:
            return cls((1,), (0,))
        return cls(tuple(m[0] for m in modes), tuple(m[1] for m in modes))

    @classmethod
    def ordered(cls, shape: Sequence[int], order: Sequence[int]) -> Layout:
        """``make_ordered_layout``: the mode with the lowest rank in ``order``
        gets stride 1, every other mode's stride is the product of the sizes
        of the modes ranked below it."""
        if len(shape) != len(order):
            raise LayoutError("shape and order differ in rank")
        stride = [0] * len(shape)
        for i, rank in enumerate(order):
            stride[i] = prod(shape[j] for j in range(len(shape)) if order[j] < rank)
        return cls(tuple(shape), tuple(stride))

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def cosize(self) -> int:
        return self(self.size - 1) + 1

    @property
    def modes(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.shape, self.stride))

    def coord(self, idx: int) -> tuple[int, ...]:
        out = []
        for s in self.shape:
            out.append(idx % s)
            idx //= s
        return tuple(out)

    def __call__(self, *coord: int) -> int:
        if len(coord) == 1 and len(self.shape) != 1:
            coord = self.coord(coord[0])
        if len(coord) != len(self.shape):
            raise LayoutError(f"coordinate {coord} does not match {self}")
        return sum(c * d for c, d in zip(coord, self.stride))

    def coalesce(self) -> Layout:
        """Merge adjacent modes whose strides chain and drop size-1 modes."""
        out: list[list[int]] = []
        for s, d in self.modes:
            if s == 1:
                continue
            if out and out[-1][1] * out[-1][0] == d:
                out[-1][0] *= s
            else:
                out.append([s, d])
        if not out:
            return Layout((1,), (0,))
        return Layout.from_modes((s, d) for s, d in out)

    def __str__(self) -> str:
        def part(vals: tuple[int, ...]) -> str:
            return (
                str(vals[0]) if len(vals) == 1 else "(" + ",".join(map(str, vals)) + ")"
            )

        return f"{part(self.shape)}:{part(self.stride)}"

    __repr__ = __str__


def _compose_mode(a: Layout, s: int, d: int) -> list[tuple[int, int]]:
    """``composition(a, s:d)`` as a list of flat modes (CuTe ``composition_impl``)."""
    if d == 0:
        return [(s, 0)]
    # Strip the first ``d`` elements of ``a``.
    stripped: list[tuple[int, int]] = []
    rest = d
    for sh, st in a.modes:
        if rest == 1:
            stripped.append((sh, st))
        elif rest % sh == 0:
            rest //= sh
        elif sh % rest == 0:
            stripped.append((sh // rest, st * rest))
            rest = 1
        else:
            raise LayoutError(f"stride {d} does not divide {a}: shape divisibility")
    if rest != 1:
        raise LayoutError(f"stride {d} reaches past {a}")
    # Keep the first ``s`` elements of what is left.
    out: list[tuple[int, int]] = []
    need = s
    for sh, st in stripped:
        if need == 1:
            break
        if need < sh:
            if sh % need:
                raise LayoutError(f"size {s} does not divide {a}: shape divisibility")
            out.append((need, st))
            need = 1
        else:
            if need % sh:
                raise LayoutError(f"size {s} does not divide {a}: shape divisibility")
            out.append((sh, st))
            need //= sh
    if need != 1:
        raise LayoutError(f"size {s} reaches past {a}")
    return out or [(1, 0)]


def composition(a: Layout, b: Layout) -> Layout:
    """``R = a ∘ b``: ``R(i) = a(b(i))``. R has b's shape and a's strides along
    the path b walks. Each mode of b becomes a group of modes in R; groups
    are coalesced independently and concatenated."""
    modes: list[tuple[int, int]] = []
    for s, d in b.modes:
        modes.extend(Layout.from_modes(_compose_mode(a, s, d)).coalesce().modes)
    return Layout.from_modes(modes)


def complement(a: Layout, m: int) -> Layout:
    """Starting points at which copies of the stamp ``a`` tile ``[0, m)``.

    Sort a's modes by stride; each mode ``s:d`` leaves a hole of ``d / covered``
    cells at stride ``covered``, then ``covered = s * d``; finally repeat
    ``ceil(m / covered)`` times at stride ``covered``. Size-1 modes are dropped.
    """
    modes = sorted(((s, d) for s, d in a.modes if s > 1), key=lambda m: m[1])
    if any(d == 0 for _, d in modes):
        raise LayoutError(f"Non-injective Layout detected in complement: {a}")
    covered = 1
    out: list[tuple[int, int]] = []
    for s, d in modes:
        if d < covered or d % covered:
            raise LayoutError(f"Non-injective Layout detected in complement: {a}")
        if d // covered > 1:
            out.append((d // covered, covered))
        covered = s * d
    rest = -(-m // covered)
    if rest > 1:
        out.append((rest, covered))
    return Layout.from_modes(out)


def right_inverse(a: Layout) -> Layout:
    """Slot → logical index for the gap-free prefix of a's codomain.

    Coalesce, sort modes by stride, and keep each mode whose stride equals the
    span covered so far (starting at 1). A kept mode's stride in the result is
    its stride in index space: the product of the sizes listed before it.
    """
    flat = a.coalesce()
    index_strides = []
    pos = 1
    for s in flat.shape:
        index_strides.append(pos)
        pos *= s
    order = sorted(range(len(flat.shape)), key=lambda i: flat.stride[i])
    covered = 1
    out: list[tuple[int, int]] = []
    for i in order:
        if flat.stride[i] != covered:
            break
        out.append((flat.shape[i], index_strides[i]))
        covered *= flat.shape[i]
    return Layout.from_modes(out)


def max_common_layout(a: Layout, b: Layout) -> Layout:
    """The largest prefix of logical indices that is contiguous in both ``a``
    and ``b`` when walked in b's memory order, expressed in index space.

    ``inv = right_inverse(b)`` names the element in each b slot;
    ``common = coalesce(a ∘ inv)`` is the b-slot → a-slot map; its mode 0 is a
    run only when its stride is 1, and ``inv ∘ mode0`` turns that run back
    into logical indices, which both layouts understand.
    """
    inv = right_inverse(b)
    common = composition(a, inv).coalesce()
    if common.stride[0] == 1 and common.shape[0] > 1:
        return composition(inv, Layout((common.shape[0],), (1,)))
    return Layout((1,), (0,))


def logical_divide(a: Layout, tiler: Layout) -> tuple[Layout, Layout]:
    """``a ∘ (tiler, complement(tiler, size(a)))`` as ``(tile, rest)``.

    ``tile`` walks one stamp through ``a``; ``rest`` picks the stamp. Both are
    returned flat instead of nested in one two-mode layout.
    """
    rest = complement(tiler, a.size)
    return composition(a, tiler).coalesce(), composition(a, rest).coalesce()


def is_contiguous(a: Layout) -> bool:
    """True when ``a`` is one run of ``size`` consecutive offsets, whatever
    the order its modes are listed in: the gap-free prefix that
    ``right_inverse`` walks covers every element."""
    return right_inverse(a).size == a.size
