"""Compare KV-cache footprint and throughput between buddy OFF and ON for Zamba2.

Run in two passes (subprocess) so each LLM instance gets a clean GPU.

Usage:
    python current-work/bench_zamba2.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "Zyphra/Zamba2-1.2B-instruct"
PROMPTS_POOL = [
    "Describe an autumn forest at sunset in vivid detail.",
    "Write a Python function to compute fibonacci(n) iteratively.",
    "Explain how a transformer attention mechanism works step by step.",
    "List 10 creative names for a new coffee shop and explain each.",
    "Write a short story about a robot learning to paint.",
    "Compare cats and dogs as pets across 5 dimensions.",
    "Explain the difference between TCP and UDP with examples.",
    "Write a haiku about the ocean, then another about the desert.",
]
PROBE = """
import json, os, sys, time, torch
from vllm import LLM, SamplingParams

model = sys.argv[1]
prompts = json.loads(sys.argv[2])
max_new = int(sys.argv[3])
max_num_seqs = int(sys.argv[4]) if len(sys.argv) > 4 else 0

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
free_before, total = torch.cuda.mem_get_info()

enable_prefix_caching = os.environ.get("BENCH_PREFIX_CACHING", "0") == "1"
t0 = time.time()
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
load_s = time.time() - t0

# Pull engine stats
cfg = llm.llm_engine.vllm_config
cache_cfg = cfg.cache_config
num_gpu_blocks = cache_cfg.num_gpu_blocks
block_size = cache_cfg.block_size
kv_bytes = torch.cuda.max_memory_allocated() - free_before  # not reliable

free_after, _ = torch.cuda.mem_get_info()
used_alloc_bytes = free_before - free_after
peak_alloc_bytes = torch.cuda.max_memory_allocated()

sp = SamplingParams(temperature=0.0, max_tokens=max_new, seed=0)
tok = llm.get_tokenizer()
if getattr(tok, "chat_template", None):
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
# Warmup
llm.generate(prompts[:1], sp)

t0 = time.time()
outs = llm.generate(prompts, sp)
gen_s = time.time() - t0
total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
throughput = total_out_tokens / gen_s if gen_s > 0 else 0.0

result = {
    "model": model,
    "buddy": os.environ.get("VLLM_USE_BUDDY_BLOCK_POOL", "0"),
    "max_order": os.environ.get("VLLM_BUDDY_MAX_ORDER", ""),
    "group_orders": os.environ.get("VLLM_BUDDY_GROUP_ORDERS", ""),
    "prefix_caching": enable_prefix_caching,
    "load_s": load_s,
    "gen_s": gen_s,
    "num_gpu_blocks": num_gpu_blocks,
    "block_size": block_size,
    "used_alloc_gb": used_alloc_bytes / 1e9,
    "peak_alloc_gb": peak_alloc_bytes / 1e9,
    "num_prompts": len(prompts),
    "max_new_tokens": max_new,
    "total_out_tokens": total_out_tokens,
    "throughput_tok_per_s": throughput,
}
print("RESULT_JSON=" + json.dumps(result))
"""


def run_case(env_extra: dict[str, str], prompts: list[str], max_new: int,
             model: str, max_num_seqs: int | None) -> dict:
    env = os.environ.copy()
    env.update(env_extra)
    venv_py = ".venv/bin/python"
    proc = subprocess.run(
        [venv_py, "-c", PROBE, model, json.dumps(prompts), str(max_new),
         str(max_num_seqs or 0)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-3000:])
        print("STDOUT:", proc.stdout[-3000:])
        raise SystemExit(f"probe failed for {env_extra}")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    raise SystemExit("no RESULT_JSON line in probe output")


def fmt(d: dict) -> str:
    return (
        f"buddy={d['buddy']!s:<3} prefix={int(d.get('prefix_caching', False))} "
        f"order={d.get('max_order','')!s:<3} | "
        f"num_blocks={d['num_gpu_blocks']:<6} "
        f"used={d['used_alloc_gb']:.2f} GiB | "
        f"gen={d['gen_s']:.2f}s "
        f"thru={d['throughput_tok_per_s']:.1f} tok/s "
        f"(outs={d['total_out_tokens']})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    args = parser.parse_args()

    prompts = [PROMPTS_POOL[i % len(PROMPTS_POOL)] for i in range(args.batch)]
    cases = [
        {"BENCH_PREFIX_CACHING": "0"},
        {"BENCH_PREFIX_CACHING": "1"},
        {
            "VLLM_USE_BUDDY_BLOCK_POOL": "1",
            "VLLM_BUDDY_MAX_ORDER": "4",
            "BENCH_PREFIX_CACHING": "0",
        },
        {
            "VLLM_USE_BUDDY_BLOCK_POOL": "1",
            "VLLM_BUDDY_MAX_ORDER": "4",
            "BENCH_PREFIX_CACHING": "1",
        },
    ]
    results = []
    for env in cases:
        print(f"\n=== running env={env} ===", flush=True)
        r = run_case(env, prompts, args.max_new, args.model, args.max_num_seqs)
        results.append(r)
        print(fmt(r), flush=True)

    print("\n=== SUMMARY ===")
    for r in results:
        print(fmt(r))
    Path("current-work/bench_zamba2_result.json").write_text(
        json.dumps(results, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
