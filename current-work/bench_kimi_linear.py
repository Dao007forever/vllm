"""Compare KV-cache footprint and throughput between buddy OFF and ON for
Kimi-Linear-48B-A3B-Instruct under mamba_cache_mode='align' (hybrid KV
manager, prefix caching ON).

Mirrors bench_zamba2.py but tuned for the 48B MoE: longer max_model_len,
real generation work to amortize load time. Runs each case in a fresh
subprocess to get a clean GPU.

Usage:
    .venv/bin/python current-work/bench_kimi_linear.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
PROMPTS_POOL = [
    "Describe an autumn forest at sunset in vivid detail.",
    "Write a Python function to compute fibonacci(n) iteratively.",
    "Explain how a transformer attention mechanism works step by step.",
    "List 10 creative names for a new coffee shop and explain each.",
    "Write a short story about a robot learning to paint.",
    "Compare cats and dogs as pets across 5 dimensions.",
    "Explain the difference between TCP and UDP with examples.",
    "Write a haiku about the ocean, then another about the desert.",
    "Summarize the plot of Hamlet in three paragraphs.",
    "Outline a six-week training plan for a beginner runner.",
    "Walk through long division of 4567 by 23.",
    "Translate the following sentence into French and German: 'The weather is lovely today.'",
    "Describe the steps of photosynthesis at a high-school level.",
    "List five trade-offs of microservices vs. monoliths.",
    "Write a polite email declining a meeting invite.",
    "Explain the difference between supervised and unsupervised learning.",
]
PROBE = r"""
import json, os, sys, time, torch
from vllm import LLM, SamplingParams

model = sys.argv[1]
prompts = json.loads(sys.argv[2])
max_new = int(sys.argv[3])
max_num_seqs = int(sys.argv[4]) if len(sys.argv) > 4 else 0

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
free_before, total = torch.cuda.mem_get_info()

enable_prefix_caching = os.environ.get("BENCH_PREFIX_CACHING", "1") == "1"
t0 = time.time()
llm_kwargs = dict(
    model=model,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    # Disable CUDA graph capture: the capture path builds a fake page table
    # close to num_gpu_blocks, which triggers the FlashInfer-MLA block_num
    # alignment check and the buddy reshape bug at bs ≥ 128. Eager decode
    # is the cleaner test signal for the buddy work.
    enforce_eager=True,
    trust_remote_code=True,
    enable_prefix_caching=enable_prefix_caching,
    # Set via $BENCH_BLOCK_SIZE so we can sweep across configurations.
    # Default 64: FlashInfer-MLA's largest supported kernel block size →
    # no kernel split → both the FlashInfer alignment constraint and the
    # decouple-strided reshape stride are trivially satisfied.
    block_size=int(os.environ.get("BENCH_BLOCK_SIZE", "64")),
)
_override = os.environ.get("NUM_GPU_BLOCKS_OVERRIDE")
if _override:
    llm_kwargs["num_gpu_blocks_override"] = int(_override)
if max_num_seqs > 0:
    llm_kwargs["max_num_seqs"] = max_num_seqs
llm = LLM(**llm_kwargs)
load_s = time.time() - t0

cfg = llm.llm_engine.vllm_config
cache_cfg = cfg.cache_config
num_gpu_blocks = cache_cfg.num_gpu_blocks
block_size = cache_cfg.block_size
mamba_block_size = getattr(cache_cfg, "mamba_block_size", None)
mamba_cache_mode = getattr(cache_cfg, "mamba_cache_mode", None)
mamba_page_size_padded = getattr(cache_cfg, "mamba_page_size_padded", None)

# Per-group spec details (block_size, page_size_bytes, chunk_order).
kv_cfg = getattr(llm.llm_engine, "kv_cache_config", None) or getattr(
    cfg, "kv_cache_config", None
)
group_specs = []
base_page_bytes = None
if kv_cfg is not None:
    base_page_bytes = getattr(kv_cfg, "base_page_bytes", None)
    for g in getattr(kv_cfg, "kv_cache_groups", []):
        s = g.kv_cache_spec
        group_specs.append({
            "type": type(s).__name__,
            "block_size": s.block_size,
            "page_size_bytes": s.page_size_bytes,
            "page_size_padded": getattr(s, "page_size_padded", None),
            "chunk_order": getattr(g, "chunk_order", 0),
            "num_layers": len(g.layer_names),
        })

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
    "prefix_caching": enable_prefix_caching,
    "mamba_cache_mode": mamba_cache_mode,
    "mamba_block_size": mamba_block_size,
    "mamba_page_size_padded": mamba_page_size_padded,
    "base_page_bytes": base_page_bytes,
    "group_specs": group_specs,
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
        # FlashInfer-MLA tile alignment: block_num × block_size % 128 == 0.
        # If we hit it, parse the reported block_num, round down to the
        # nearest satisfying value, and retry once via num_gpu_blocks_override.
        m = re.search(
            r"got block_num=(\d+) and block_size=(\d+)", proc.stderr
        )
        if m and "NUM_GPU_BLOCKS_OVERRIDE" not in env:
            bn, bs = int(m.group(1)), int(m.group(2))
            align = max(1, 128 // bs)
            rounded = (bn // align) * align
            print(
                f"FlashInfer alignment failure: block_num={bn} block_size={bs} "
                f"(need % {align} == 0). Retrying with "
                f"num_gpu_blocks_override={rounded}.",
                flush=True,
            )
            env2 = env.copy()
            env2["NUM_GPU_BLOCKS_OVERRIDE"] = str(rounded)
            proc = subprocess.run(
                [venv_py, "-c", PROBE, model, json.dumps(prompts),
                 str(max_new), str(max_num_seqs or 0)],
                env=env2, capture_output=True, text=True,
            )
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-5000:])
            print("STDOUT:", proc.stdout[-3000:])
            raise SystemExit(f"probe failed for {env_extra}")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    print("STDOUT (no RESULT_JSON line):", proc.stdout[-3000:])
    raise SystemExit("no RESULT_JSON line in probe output")


def fmt(d: dict) -> str:
    groups = " ".join(
        f"[{g['type']}:bs={g['block_size']},page={g['page_size_bytes']//1024}KB,"
        f"order={g['chunk_order']}]"
        for g in d.get("group_specs", [])
    )
    base = d.get("base_page_bytes")
    base_str = f"base={base//1024}KB " if base else ""
    return (
        f"buddy={d['buddy']!s:<3} prefix={int(d.get('prefix_caching', False))} "
        f"mode={d.get('mamba_cache_mode'):<5} | "
        f"num_blocks={d['num_gpu_blocks']:<6} "
        f"attn_bs={d['block_size']} mamba_bs={d.get('mamba_block_size')} "
        f"{base_str}| "
        f"used={d['used_alloc_gb']:.2f} GiB peak={d['peak_alloc_gb']:.2f} GiB | "
        f"gen={d['gen_s']:.2f}s thru={d['throughput_tok_per_s']:.1f} tok/s\n"
        f"  groups: {groups}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    args = parser.parse_args()

    prompts = [PROMPTS_POOL[i % len(PROMPTS_POOL)] for i in range(args.batch)]
    cases = [
        # Baseline align (no buddy, prefix caching on → auto-selects align mode
        # because Kimi-Linear declares HasInnerState+IsHybrid but does NOT set
        # supports_mamba_prefix_caching).
        {"BENCH_PREFIX_CACHING": "1"},
        # Buddy align (decoupled hybrid pages).
        {
            "VLLM_USE_BUDDY_BLOCK_POOL": "1",
            "VLLM_BUDDY_MAX_ORDER": "8",
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

    # Quick delta print.
    if len(results) == 2:
        a, b = results
        print(
            f"\nnum_blocks: {a['num_gpu_blocks']} -> {b['num_gpu_blocks']} "
            f"(x{b['num_gpu_blocks']/max(a['num_gpu_blocks'],1):.2f})"
        )
        print(
            f"throughput: {a['throughput_tok_per_s']:.1f} -> "
            f"{b['throughput_tok_per_s']:.1f} tok/s "
            f"(x{b['throughput_tok_per_s']/max(a['throughput_tok_per_s'],1):.2f})"
        )

    Path("current-work/bench_kimi_linear_result.json").write_text(
        json.dumps(results, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
