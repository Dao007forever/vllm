# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot mapping correctness under hybrid_blocks (per-group order > 0).

When kernel_block_size < block_size, the BlockTable expands one kv_manager
block id into ``blocks_per_kv_block = 2^order`` consecutive kernel block ids,
and the slot mapping kernel uses kernel_block_size as the row stride
(equivalently, tokens_per_row). For an allocated chunk at start row
``r0 = kv_manager_id * 2^order`` covering ``block_size`` tokens, the
expected slot for position ``p`` (within the request) is::

    lb     = p // block_size
    off    = p % block_size
    row    = block_table_kernel[req, lb*2^order + off // kbs]
    slot   = row * kernel_block_size + (off % kernel_block_size)

This file verifies that the existing _compute_slot_mapping_kernel through the
BlockTable hybrid_blocks path produces these slot ids exactly for several
orders. No new kernel is introduced — the test pins down the contract we lean
on when wiring buddy + per-group order on top.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _expected_slots(
    positions: np.ndarray,
    kv_manager_ids: list[int],
    block_size: int,
    kernel_block_size: int,
) -> np.ndarray:
    """Reference computation matching the hybrid_blocks contract."""
    blocks_per_kv_block = block_size // kernel_block_size
    kernel_ids = []
    for kv_id in kv_manager_ids:
        for k in range(blocks_per_kv_block):
            kernel_ids.append(kv_id * blocks_per_kv_block + k)
    kernel_ids_np = np.asarray(kernel_ids, dtype=np.int64)
    out = np.empty_like(positions, dtype=np.int64)
    for i, p in enumerate(positions):
        row_idx = p // kernel_block_size
        within = p % kernel_block_size
        out[i] = kernel_ids_np[row_idx] * kernel_block_size + within
    return out


def _make_block_table(
    block_size: int, kernel_block_size: int, max_num_blocks_per_req: int = 16
) -> BlockTable:
    return BlockTable(
        block_size=block_size,
        max_num_reqs=4,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=4096,
        pin_memory=False,
        device=torch.device(DEVICE),
        kernel_block_size=kernel_block_size,
        cp_kv_cache_interleave_size=1,
    )


@pytest.mark.skipif(DEVICE != "cuda", reason="Triton kernel requires CUDA")
@pytest.mark.parametrize(
    "block_size, kernel_block_size",
    [
        (16, 16),  # order = 0, uniform (no hybrid)
        (32, 16),  # order = 1
        (64, 16),  # order = 2
        (128, 16),  # order = 3
        (256, 16),  # order = 4
    ],
)
def test_slot_mapping_hybrid_order(block_size: int, kernel_block_size: int) -> None:
    bt = _make_block_table(block_size, kernel_block_size)

    # Single request, 3 kv_manager blocks → expanded to 3 * 2^order rows.
    kv_manager_ids = [4, 7, 12]
    bt.append_row(kv_manager_ids, row_idx=0)
    bt.commit_block_table(num_reqs=1)

    num_tokens = 3 * block_size
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    qsl = torch.tensor([0, num_tokens], device=DEVICE, dtype=torch.int32)

    bt.compute_slot_mapping(num_reqs=1, query_start_loc=qsl, positions=positions)
    got = bt.slot_mapping.gpu[:num_tokens].cpu().numpy()

    expected = _expected_slots(
        positions.cpu().numpy(),
        kv_manager_ids,
        block_size,
        kernel_block_size,
    )
    assert np.array_equal(got, expected), (
        f"Slot mismatch for block_size={block_size}, "
        f"kernel_block_size={kernel_block_size}: "
        f"expected[:8]={expected[:8]}, got[:8]={got[:8]}"
    )


@pytest.mark.skipif(DEVICE != "cuda", reason="Triton kernel requires CUDA")
def test_slot_mapping_hybrid_pad_region() -> None:
    """Tokens past num_tokens must be padded to PAD_SLOT_ID."""
    block_size, kernel_block_size = 64, 16
    bt = _make_block_table(block_size, kernel_block_size)
    bt.append_row([1, 2], row_idx=0)
    bt.commit_block_table(num_reqs=1)

    num_tokens = 5
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    qsl = torch.tensor([0, num_tokens], device=DEVICE, dtype=torch.int32)
    bt.compute_slot_mapping(num_reqs=1, query_start_loc=qsl, positions=positions)
    sm = bt.slot_mapping.gpu.cpu().numpy()
    # First num_tokens are real; the tail is padded.
    assert (sm[num_tokens:] == PAD_SLOT_ID).all()


@pytest.mark.skipif(DEVICE != "cuda", reason="Triton kernel requires CUDA")
def test_slot_mapping_hybrid_multi_request() -> None:
    """Multiple requests with different lengths land in correct slot ranges."""
    block_size, kernel_block_size = 64, 16  # order = 2
    bt = _make_block_table(block_size, kernel_block_size)
    bt.append_row([3, 9], row_idx=0)  # 128 tokens of capacity
    bt.append_row([5], row_idx=1)  # 64 tokens of capacity
    bt.commit_block_table(num_reqs=2)

    lens = [80, 40]
    positions_list = []
    for length in lens:
        positions_list.append(torch.arange(length, dtype=torch.int64))
    positions = torch.cat(positions_list).to(DEVICE)
    qsl = torch.tensor([0, lens[0], lens[0] + lens[1]],
                       device=DEVICE, dtype=torch.int32)

    bt.compute_slot_mapping(num_reqs=2, query_start_loc=qsl, positions=positions)
    got = bt.slot_mapping.gpu[: sum(lens)].cpu().numpy()

    expected0 = _expected_slots(
        np.arange(lens[0]), [3, 9], block_size, kernel_block_size
    )
    expected1 = _expected_slots(
        np.arange(lens[1]), [5], block_size, kernel_block_size
    )
    expected = np.concatenate([expected0, expected1])
    assert np.array_equal(got, expected), (
        f"Multi-req slot mismatch: expected[:8]={expected[:8]} "
        f"got[:8]={got[:8]}"
    )
