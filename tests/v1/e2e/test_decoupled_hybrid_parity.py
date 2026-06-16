# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end parity for decoupled hybrid paging on a real model (AC-10).

Runs a real attention + linear-attention/Mamba hybrid model — Kimi-Linear by
default — twice: once with the uniform-page baseline and once with the
decoupled buddy allocator enabled as a supported config, at the SAME attn
block size. The buddy changes allocation, not numerics, so greedy generations
must match the baseline token-for-token (allowing at most one prompt of
MoE/atomics nondeterminism), prefix-cache reuse must be exact (pass-2 ==
pass-1), and known-answer sanity must hold.

This is a GPU end-to-end test: it loads a large model with tensor parallelism,
so it is skipped unless GPUs and the model are available. Each configuration
runs in its own subprocess (clean TP engine lifecycle per config). Override the
model with DECOUPLED_E2E_MODEL, TP with DECOUPLED_E2E_TP, and attn block size
with DECOUPLED_E2E_BS (default 64; 64 keeps FlashInfer-MLA's tile-alignment
divisor at 2, avoiding the block_num%(128/block_size)==0 decode constraint).
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

torch = pytest.importorskip("torch")

MODEL = os.environ.get("DECOUPLED_E2E_MODEL", "moonshotai/Kimi-Linear-48B-A3B-Instruct")
TP = int(os.environ.get("DECOUPLED_E2E_TP", "4"))
BS = os.environ.get("DECOUPLED_E2E_BS", "64")
RUNNER = os.path.join(os.path.dirname(__file__), "decoupled_hybrid_runner.py")

# Known-answer checks: (prompt_index, accepted lowercase substrings).
SANITY = [(0, ["paris"]), (1, ["391"]), (4, ["1972"]), (5, ["twelve", "12"])]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < TP,
    reason=f"requires >= {TP} GPUs",
)


def _run(mode: str, out_path: str) -> dict:
    env = {
        **os.environ,
        "DECOUPLED_E2E_MODE": mode,
        "DECOUPLED_E2E_MODEL": MODEL,
        "DECOUPLED_E2E_TP": str(TP),
        "DECOUPLED_E2E_BS": BS,
        "DECOUPLED_E2E_OUT": out_path,
    }
    subprocess.run([sys.executable, RUNNER], env=env, check=True, timeout=3600)
    with open(out_path) as f:
        return json.load(f)


def _token_diffs(a: list[dict], b: list[dict]) -> list[int]:
    return [i for i, (x, y) in enumerate(zip(a, b)) if x["token_ids"] != y["token_ids"]]


@pytest.mark.timeout(7800)
def test_decoupled_hybrid_matches_uniform_baseline():
    with tempfile.TemporaryDirectory() as d:
        base = _run("baseline", os.path.join(d, "baseline.json"))
        dec = _run("decoupled", os.path.join(d, "decoupled.json"))

    # The two runs must differ in allocator as intended: baseline must not have
    # inherited a stray buddy enable from the environment, and decoupled must
    # have it on.
    assert base["info"]["buddy_enabled"] is False
    assert dec["info"]["buddy_enabled"] is True

    # The decoupled run must actually engage the decoupled layout, not silently
    # fall back. Reinterpreting the id space at the smaller base page yields
    # STRICTLY more base blocks than the uniform baseline (Mamba no longer
    # inflates the page). A fallback would leave the counts equal, so `>` — not
    # `>=` — is what proves activation.
    assert dec["info"]["num_gpu_blocks"] > base["info"]["num_gpu_blocks"], (
        f"decoupled num_gpu_blocks ({dec['info']['num_gpu_blocks']}) is not "
        f"greater than baseline ({base['info']['num_gpu_blocks']}); the "
        "decoupled layout likely did not activate (silent fallback)."
    )

    # Prefix-cache reuse must be exact under both allocators.
    assert not _token_diffs(base["pass1"], base["pass2"])
    assert not _token_diffs(dec["pass1"], dec["pass2"])

    # Decoupled generations match the uniform baseline token-for-token (allow
    # at most one prompt of MoE/atomics nondeterminism at this scale).
    diffs = _token_diffs(base["pass1"], dec["pass1"])
    assert len(diffs) <= 1, "decoupled diverged from baseline on prompts " + ", ".join(
        f"{i}: base={base['pass1'][i]['text']!r} dec={dec['pass1'][i]['text']!r}"
        for i in diffs
    )

    # Known-answer sanity on the decoupled run.
    bad = [
        i
        for i, accepts in SANITY
        if not any(s in dec["pass1"][i]["text"].lower() for s in accepts)
    ]
    assert not bad, f"decoupled failed known-answer sanity on prompts {bad}"
