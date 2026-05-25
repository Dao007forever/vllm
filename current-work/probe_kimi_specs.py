"""Standalone probe: just instantiate the engine and dump KV cache config —
no inference. Compares baseline vs buddy under mamba_cache_mode='align'.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python current-work/probe_kimi_specs.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

PROBE = r"""
import json, os, sys, torch
from vllm import LLM

torch.cuda.empty_cache()
llm = LLM(
    model=sys.argv[1],
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    trust_remote_code=True,
    enable_prefix_caching=True,
    attention_backend="TRITON_MLA",
)
cfg = llm.llm_engine.vllm_config
cc = cfg.cache_config
out = {
    "buddy": os.environ.get("VLLM_USE_BUDDY_BLOCK_POOL", "0"),
    "cache_config": {
        "block_size": cc.block_size,
        "mamba_block_size": getattr(cc, "mamba_block_size", None),
        "mamba_cache_mode": getattr(cc, "mamba_cache_mode", None),
        "mamba_page_size_padded": getattr(cc, "mamba_page_size_padded", None),
        "num_gpu_blocks": cc.num_gpu_blocks,
        "enable_prefix_caching": cc.enable_prefix_caching,
    },
}

# Try multiple paths to find kv_cache_config.
kv_cfg = None
for attr in ("kv_cache_config",):
    for parent in (llm.llm_engine, cfg, getattr(llm.llm_engine, "engine_core", None)):
        if parent is None:
            continue
        v = getattr(parent, attr, None)
        if v is not None:
            kv_cfg = v
            break
    if kv_cfg is not None:
        break

# Try scheduler path (v1)
if kv_cfg is None:
    sched = getattr(getattr(llm.llm_engine, "engine_core", None), "scheduler", None)
    if sched is None:
        sched = getattr(llm.llm_engine, "scheduler", None)
    if sched is not None:
        kv_cfg = getattr(sched, "kv_cache_config", None)

if kv_cfg is not None:
    out["kv_cache_config"] = {
        "base_page_bytes": getattr(kv_cfg, "base_page_bytes", None),
        "num_blocks": getattr(kv_cfg, "num_blocks", None),
        "groups": [],
    }
    for g in getattr(kv_cfg, "kv_cache_groups", []):
        s = g.kv_cache_spec
        out["kv_cache_config"]["groups"].append({
            "spec_type": type(s).__name__,
            "block_size": s.block_size,
            "page_size_bytes": s.page_size_bytes,
            "page_size_padded": getattr(s, "page_size_padded", None),
            "chunk_order": getattr(g, "chunk_order", 0),
            "num_layers": len(g.layer_names),
            "layer_names_preview": list(g.layer_names)[:3],
        })
else:
    out["kv_cache_config"] = "NOT_FOUND"

print("RESULT_JSON=" + json.dumps(out, default=str))
"""


def run_case(env_extra):
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.run(
        [".venv/bin/python", "-c", PROBE, MODEL],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"--- {env_extra} FAILED ---")
        print("STDERR:", proc.stderr[-3000:])
        print("STDOUT:", proc.stdout[-2000:])
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    print("no RESULT_JSON in:", proc.stdout[-2000:])
    return None


def main():
    cases = [
        ("baseline", {}),
        ("buddy", {"VLLM_USE_BUDDY_BLOCK_POOL": "1", "VLLM_BUDDY_MAX_ORDER": "8"}),
    ]
    for name, env in cases:
        print(f"\n=== {name} {env} ===")
        r = run_case(env)
        if r:
            print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
