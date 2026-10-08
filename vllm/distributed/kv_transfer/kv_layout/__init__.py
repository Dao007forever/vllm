# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout algebra and transfer planning shared by KV connectors."""

from vllm.distributed.kv_transfer.kv_layout.layout import (
    Layout,
    LayoutError,
    complement,
    composition,
    is_contiguous,
    logical_divide,
    max_common_layout,
    right_inverse,
)
from vllm.distributed.kv_transfer.kv_layout.planner import (
    Direct,
    KVPage,
    LocalPermute,
    PageOrder,
    PairPlan,
    SingleRunStrategy,
    Staged,
    StagedPlan,
    TransferStrategy,
    Unsupported,
    head_range,
    plan_pair,
    staged_plan,
)

__all__ = [
    "Direct",
    "KVPage",
    "Layout",
    "LayoutError",
    "LocalPermute",
    "PageOrder",
    "PairPlan",
    "SingleRunStrategy",
    "Staged",
    "StagedPlan",
    "TransferStrategy",
    "Unsupported",
    "complement",
    "composition",
    "head_range",
    "is_contiguous",
    "logical_divide",
    "max_common_layout",
    "plan_pair",
    "right_inverse",
    "staged_plan",
]
