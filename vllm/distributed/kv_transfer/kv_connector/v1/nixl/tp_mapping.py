# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_layout import (
    KVPage,
    PairPlan,
    head_range,
    plan_pair,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds
    block_ids_by_region: bool = False


def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.
    """

    # Remote TP ranks that this local rank reads from, per group.
    # Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # FA head offset factor for hetero-TP (D_TP > P_TP).
    rank_offset_factor: int

    # Local ranks (in aggregate) that read from a given source rank. The producer frees
    # a request's blocks only once that many notifications have come in.
    local_consumers: int = 1


# ======================================================================
# TP mapping computation
# ======================================================================


def _attention_pairs(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    tp_rank: int,
    tp_size: int,
    total_num_kv_heads: int,
) -> list[tuple[int, PairPlan]]:
    """Plan this rank's page against every remote rank in its handshake window.

    Pages are built in head-token units with HND order on both sides: the
    source ranks and the offset of this rank's heads inside a source page do
    not depend on the page order or the head size, only on which global heads
    each rank holds. The order-dependent decision (direct runs or a staged
    copy) is made separately by the connector's transfer strategy.
    """
    local = KVPage.attention(
        1,
        max(1, total_num_kv_heads // tp_size),
        head_range(tp_rank, tp_size, total_num_kv_heads)[0],
        "HND",
    )
    pairs: list[tuple[int, PairPlan]] = []
    for remote_rank in transfer_topology.handshake_target_ranks(remote_tp_size):
        remote = KVPage.attention(
            1,
            max(1, total_num_kv_heads // remote_tp_size),
            head_range(remote_rank, remote_tp_size, total_num_kv_heads)[0],
            "HND",
        )
        pair = plan_pair(remote, local)
        if pair is not None:
            pairs.append((remote_rank, pair))
    return pairs


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    remote_dcp_size: int = 1,
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.

    DCP support is scoped to MLA only, with a side is either fully replicated or fully
    sharded. DCP-branch reuses the same rank set used at handshake selection.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    total_num_kv_heads = transfer_topology.total_num_kv_heads
    # --- Attention source ranks ---
    # Head-sharded attention: which remote ranks in the handshake window share
    # a KV head with this rank, from the transfer planner. MLA is replicated,
    # so every remote rank qualifies and no plan is needed.
    pairs: dict[int, PairPlan] = {}
    if not transfer_topology.is_mla:
        pairs = dict(
            _attention_pairs(
                transfer_topology, remote_tp_size, tp_rank, tp_size, total_num_kv_heads
            )
        )
    if transfer_topology.is_mla and remote_dcp_size > 1:
        attn_ranks = transfer_topology.dcp_source_ranks(remote_tp_size, remote_dcp_size)
    elif transfer_topology.is_mla or tp_size >= remote_tp_size:
        # Every remote rank in the handshake window holds all of this rank's
        # heads (or, for MLA, the replicated cache); read from the one the
        # window names so that local ranks spread over the remote ranks.
        attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): this rank reads from every remote rank
        # that shares a KV head with it, keeping one rank per head range when
        # several replicate the same head (GQA).
        attn_ranks = []
        seen_heads: set[tuple[int, int]] = set()
        for remote_rank, pair in pairs.items():
            heads = (pair.head_lo, pair.head_hi)
            if heads not in seen_heads:
                seen_heads.add(heads)
                attn_ranks.append(remote_rank)

    # --- SSM source ranks ---
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    if has_ssm:
        if tp_size < remote_tp_size:
            abs_tp = remote_tp_size // tp_size
            ssm_ranks = list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
        else:
            ssm_ranks = list(attn_ranks)
    else:
        ssm_ranks = []

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    # Where this rank's heads start inside the source page, in units of this
    # rank's own page: the plan's source offset divided by the local page size.
    # Zero whenever the source page starts at a shared head (P TP <= D TP, MLA).
    rank_offset_factor = 0
    if pairs:
        first = pairs[attn_ranks[0]]
        rank_offset_factor = first.src_off // first.dst.size

    local_consumers = transfer_topology.dcp_consumer_count(
        remote_tp_size, remote_dcp_size
    )

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
        local_consumers=local_consumers,
    )
