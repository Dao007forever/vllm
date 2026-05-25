"""Probe per-group KV memory accounting under buddy ON/OFF.

For each KV cache group:
  - page_size_bytes (the spec's native per-block size for this group)
  - block_size (tokens per block at the spec level)
  - num_layers (number of layers in the group)

And at the pool level:
  - num_gpu_blocks (the buddy address space)
  - page_size_padded (the unified per-buddy-block memory across groups)
  - total_kv_bytes = num_gpu_blocks * page_size_padded
  - per-group active footprint = num_gpu_blocks * sum(group.page_size_bytes)
  - "padding overhead" = total_kv_bytes - sum(per-group footprint)

This is what would shrink under cross-group block sharing: each buddy
chunk would carry actual per-group data, not padding to the largest
group's page size.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = """
import json, os, sys
# Hook prepare_kernel_block_sizes (runs in the engine subprocess at init)
# to emit a sentinel JSON line with per-group page sizes. The parent grep
# for this line.
import vllm.v1.worker.utils as _U
_orig = _U.prepare_kernel_block_sizes
def _hooked(kv_cache_config, attn_groups):
    r = _orig(kv_cache_config, attn_groups)
    groups = []
    for i, g in enumerate(kv_cache_config.kv_cache_groups):
        spec = g.kv_cache_spec
        try:
            real = spec.real_page_size_bytes
        except AttributeError:
            real = spec.page_size_bytes
        groups.append({
            "i": i,
            "spec": type(spec).__name__,
            "block_size_tokens": getattr(spec, "block_size", None),
            "page_size_bytes": spec.page_size_bytes,
            "real_page_size_bytes": real,
            "page_size_padded": getattr(spec, "page_size_padded", None),
            "n_layers": len(g.layer_names),
        })
    sys.stderr.write("GROUPS_JSON=" + json.dumps(groups) + "\\n")
    # Also emit the actual kv_cache_tensors list — this is the unique-
    # tensor view that accounts for cross-group sharing via layer tuples.
    tensors = [
        {"size": t.size, "n_layers_sharing": len(t.shared_by)}
        for t in kv_cache_config.kv_cache_tensors
    ]
    sys.stderr.write("TENSORS_JSON=" + json.dumps(tensors) + "\\n")
    sys.stderr.flush()
    return r
_U.prepare_kernel_block_sizes = _hooked

from vllm import LLM

model = sys.argv[1]
max_num_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 0
enable_prefix_caching = os.environ.get("BENCH_PREFIX_CACHING", "0") == "1"
llm_kwargs = dict(
    model=model,
    max_model_len=4096,
    gpu_memory_utilization=0.45,
    enforce_eager=False,
    trust_remote_code=True,
    enable_prefix_caching=enable_prefix_caching,
)
if max_num_seqs > 0:
    llm_kwargs["max_num_seqs"] = max_num_seqs
llm = LLM(**llm_kwargs)

cfg = llm.llm_engine.vllm_config
cache_cfg = cfg.cache_config

# Reach into the engine for the post-init KV cache config.
worker = llm.llm_engine.engine_core.engine_core_proc.engine.model_executor.driver_worker.worker if False else None
# Easier path: introspect via the model_executor's known structure isn't always
# stable across versions. Use the scheduler's coordinator instead.
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)

# We rely on the engine populating num_gpu_blocks on cache_cfg.
num_gpu_blocks = cache_cfg.num_gpu_blocks
# page_size_padded is the unified per-buddy-block size in bytes.
# Pull it from kv_cache_config if it's reachable.
group_summary = []
total_per_group = 0
page_size_padded = 0
# group_summary is populated from the hook printed to stderr by the
# subprocess. The parent driver will pair them up.
pass

total_kv_bytes = num_gpu_blocks * page_size_padded if page_size_padded else 0
per_group_footprint = num_gpu_blocks * total_per_group

result = {
    "model": model,
    "buddy": os.environ.get("VLLM_USE_BUDDY_BLOCK_POOL", "0"),
    "prefix_caching": enable_prefix_caching,
    "num_gpu_blocks": num_gpu_blocks,
    "n_groups": len(group_summary),
    "page_size_padded_bytes": page_size_padded,
    "sum_group_page_size_bytes": total_per_group,
    "total_kv_bytes_padded": total_kv_bytes,
    "per_group_footprint_bytes": per_group_footprint,
    "padding_waste_bytes": total_kv_bytes - per_group_footprint,
    "groups": group_summary,
}
print("RESULT_JSON=" + json.dumps(result))
"""


def run(env_extra: dict[str, str], model: str, max_num_seqs: int | None) -> dict:
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.run(
        [".venv/bin/python", "-c", PROBE, model, str(max_num_seqs or 0)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("STDERR tail:", proc.stderr[-3000:])
        raise SystemExit(f"probe failed for {env_extra}")
    result: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            result = json.loads(line[len("RESULT_JSON="):])
            break
    if result is None:
        raise SystemExit("no RESULT_JSON in probe output")
    # The hook emits one GROUPS_JSON line on stderr per engine subprocess
    # init. Look for the most recent one (engine starts after the LLM ctor).
    groups: list[dict] = []
    tensors: list[dict] = []
    for raw in proc.stderr.splitlines():
        idx = raw.find("GROUPS_JSON=")
        if idx >= 0:
            try:
                groups = json.loads(raw[idx + len("GROUPS_JSON="):])
            except json.JSONDecodeError:
                continue
        idx = raw.find("TENSORS_JSON=")
        if idx >= 0:
            try:
                tensors = json.loads(raw[idx + len("TENSORS_JSON="):])
            except json.JSONDecodeError:
                continue
    result["kv_cache_tensors"] = tensors
    if tensors:
        result["unique_tensor_bytes"] = sum(t["size"] for t in tensors)
    if groups:
        N = result["num_gpu_blocks"]
        # Effective bytes for each group: page_size_bytes already reflects
        # padding (the property returns page_size_padded if set).
        per_group_eff = [
            g["n_layers"] * N * g["page_size_bytes"] for g in groups
        ]
        per_group_real = [
            g["n_layers"] * N * g["real_page_size_bytes"] for g in groups
        ]
        result["groups"] = groups
        result["n_groups"] = len(groups)
        result["total_kv_bytes"] = sum(per_group_eff)
        result["total_kv_bytes_unpadded"] = sum(per_group_real)
        result["padding_waste_bytes"] = (
            result["total_kv_bytes"] - result["total_kv_bytes_unpadded"]
        )
    return result


def gib(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def fmt(d: dict) -> str:
    return (
        f"buddy={d['buddy']!s:<3} prefix={int(d['prefix_caching'])} | "
        f"blocks={d['num_gpu_blocks']:<6} groups={d.get('n_groups', 0):<2} "
        f"tensors={len(d.get('kv_cache_tensors', [])):<3} | "
        f"actual_kv={gib(d.get('unique_tensor_bytes', 0))} "
        f"(naive sum {gib(d.get('total_kv_bytes', 0))}) "
        f"padding_waste={gib(d.get('padding_waste_bytes', 0))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    args = parser.parse_args()

    cases = [
        {"BENCH_PREFIX_CACHING": "0"},
        {"BENCH_PREFIX_CACHING": "1"},
        {"VLLM_USE_BUDDY_BLOCK_POOL": "1", "VLLM_BUDDY_MAX_ORDER": "4",
         "BENCH_PREFIX_CACHING": "0"},
        {"VLLM_USE_BUDDY_BLOCK_POOL": "1", "VLLM_BUDDY_MAX_ORDER": "4",
         "BENCH_PREFIX_CACHING": "1"},
    ]
    results = []
    for env in cases:
        print(f"\n=== {env} ===", flush=True)
        r = run(env, args.model, args.max_num_seqs)
        results.append(r)
        print(fmt(r))
        print("  groups:")
        for g in r.get("groups", []):
            ps = g["page_size_bytes"]
            real = g["real_page_size_bytes"]
            tag = " (padded)" if ps > real else ""
            print(
                f"    [{g['i']}] {g['spec']:<22} "
                f"layers={g['n_layers']:<3} "
                f"block_size={g['block_size_tokens']} "
                f"page={ps:>8}B (real {real}B){tag}"
            )

    print("\n=== SUMMARY ===")
    for r in results:
        print(fmt(r))
    Path(f"current-work/probe_memory_{args.model.replace('/', '_')}.json"
         ).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
