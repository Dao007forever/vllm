"""Smoke-test correctness of a model under vLLM.

Usage:
    python test_correctness.py --model <hf_id> [--chat]
    python test_correctness.py --model <hf_id> --save-golden current-work/golden
    python test_correctness.py --model <hf_id> --check-golden current-work/golden

Run two models concurrently on different GPUs by setting CUDA_VISIBLE_DEVICES
in each invocation.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPTS = [
    # 1. Factual short answer
    "The capital of France is",
    # 2. Arithmetic
    "Compute 17 * 23. Show the result as a single integer.",
    # 3. Code completion
    "Write a Python function `is_prime(n)` that returns True iff n is prime.\n\n"
    "def is_prime(n):\n",
    # 4. Multi-step reasoning
    "Alice has 3 apples. Bob gives her 5 more, then she eats 2. "
    "How many apples does Alice have? Answer with just the number.",
    # 5. Long-context-ish recall (small)
    "The secret word is 'platypus'. Remember it. "
    "Now write one sentence about the weather. "
    "Then on a new line repeat the secret word.",
    # 6. Instruction following
    "List three primary colors, one per line, no other text.",
]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _record(outs) -> list[dict]:
    return [
        {
            "text": out.outputs[0].text,
            "token_ids": list(out.outputs[0].token_ids),
        }
        for out in outs
    ]


def _save_golden(dir_path: str, model: str, chat: bool, records: list[dict]) -> str:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    fname = f"{_sanitize(model)}{'_chat' if chat else ''}.json"
    path = os.path.join(dir_path, fname)
    payload = {
        "model": model,
        "chat": chat,
        "prompts": PROMPTS,
        "outputs": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _check_golden(
    dir_path: str, model: str, chat: bool, records: list[dict]
) -> tuple[int, str]:
    fname = f"{_sanitize(model)}{'_chat' if chat else ''}.json"
    path = os.path.join(dir_path, fname)
    if not os.path.exists(path):
        return 1, f"golden file not found: {path}"
    with open(path) as f:
        golden = json.load(f)
    g_outs = golden["outputs"]
    if len(g_outs) != len(records):
        return 1, f"length mismatch: golden={len(g_outs)} current={len(records)}"
    mismatches = 0
    for i, (g, c) in enumerate(zip(g_outs, records)):
        if g["token_ids"] != c["token_ids"]:
            mismatches += 1
            print(f"MISMATCH prompt {i + 1} (token_ids differ)")
            # Find first diverging position
            lim = min(len(g["token_ids"]), len(c["token_ids"]))
            div = next(
                (k for k in range(lim) if g["token_ids"][k] != c["token_ids"][k]),
                lim,
            )
            print(f"  first divergence at token #{div}")
            print(f"  golden text: {g['text']!r}")
            print(f"  current text: {c['text']!r}")
    return mismatches, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Cap context length to keep init fast.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Apply the tokenizer chat template (use for *-instruct models).",
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable prefix caching.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override max_num_seqs (useful when prefix caching tightens the cache pool).",
    )
    parser.add_argument(
        "--save-golden",
        metavar="DIR",
        help="Save outputs to DIR/<model>.json as a golden reference.",
    )
    parser.add_argument(
        "--check-golden",
        metavar="DIR",
        help="Diff outputs against DIR/<model>.json. Non-zero exit on mismatch.",
    )
    args = parser.parse_args()

    print(f"[{args.model}] loading...", flush=True)
    t0 = time.time()
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.45,
        enforce_eager=False,
        trust_remote_code=True,
        enable_prefix_caching=not args.no_prefix_cache,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    load_s = time.time() - t0
    print(f"[{args.model}] loaded in {load_s:.1f}s", flush=True)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=0,
    )

    if args.chat:
        tok = llm.get_tokenizer()
        formatted = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in PROMPTS
        ]
    else:
        formatted = PROMPTS

    t0 = time.time()
    outs = llm.generate(formatted, sp)
    gen_s = time.time() - t0

    print(f"\n[{args.model}] ===== outputs (gen {gen_s:.2f}s) =====", flush=True)
    failures = 0
    for i, out in enumerate(outs):
        text = out.outputs[0].text.strip()
        print(f"\n[{args.model}] --- prompt {i + 1} ---")
        print(f"PROMPT: {PROMPTS[i]!r}")
        print(f"OUTPUT: {text!r}")
        if not text:
            print(f"[{args.model}] FAIL: empty output for prompt {i + 1}")
            failures += 1

    p1 = outs[0].outputs[0].text.lower()
    if "paris" not in p1:
        print(f"\n[{args.model}] WARN: prompt 1 did not mention 'paris'")
    p4 = outs[3].outputs[0].text
    if "6" not in p4:
        print(f"[{args.model}] WARN: prompt 4 did not include '6'")
    p5 = outs[4].outputs[0].text.lower()
    if "platypus" not in p5:
        print(f"[{args.model}] WARN: prompt 5 lost the secret word")

    records = _record(outs)
    golden_exit = 0
    if args.save_golden:
        path = _save_golden(args.save_golden, args.model, args.chat, records)
        print(f"[{args.model}] saved golden to {path}", flush=True)
    if args.check_golden:
        mismatches, info = _check_golden(
            args.check_golden, args.model, args.chat, records
        )
        if mismatches == 0:
            print(f"[{args.model}] golden OK (matches {info})", flush=True)
        else:
            print(
                f"[{args.model}] golden FAIL: {mismatches} prompt(s) "
                f"differ from {info}",
                flush=True,
            )
            golden_exit = 2

    print(
        f"\n[{args.model}] DONE. failures={failures} "
        f"load={load_s:.1f}s gen={gen_s:.2f}s",
        flush=True,
    )
    if failures:
        return 1
    return golden_exit


if __name__ == "__main__":
    sys.exit(main())
