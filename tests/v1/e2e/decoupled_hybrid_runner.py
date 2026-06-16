# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker for the decoupled hybrid-paging end-to-end parity test (AC-10).

Run in its own process (the model is loaded with tensor parallelism, so each
configuration needs a clean engine lifecycle). Selected by env vars:

    DECOUPLED_E2E_MODE=baseline|decoupled
    DECOUPLED_E2E_MODEL=<hf id or local path>
    DECOUPLED_E2E_TP=<tensor parallel size>
    DECOUPLED_E2E_OUT=<json output path>
    DECOUPLED_E2E_BS=<attn block size, default 64>

In ``decoupled`` mode it sets VLLM_USE_BUDDY_BLOCK_POOL=1 *before* importing
vLLM, so the buddy allocator and decoupled-page layout are active (the buddy
order is derived from the layout, no manual knob). Both modes use the same attn
block_size so the comparison isolates the
allocator (the buddy changes allocation, not numerics). It generates greedily
on a fixed prompt set TWICE (pass-2 must reuse the prefix cache and match
pass-1) and writes outputs to JSON for the parent test to compare.

block_size=64 keeps FlashInfer-MLA's tile-alignment divisor at 2, avoiding the
``block_num % (128/block_size) == 0`` decode-kernel constraint that fires at
the default block_size=32 on SM10 (see current-work/FLASHINFER_BLOCK_NUM_CHECK.md).
"""

import json
import os

SHARED = (
    "You are a meticulous assistant. Read the following context carefully "
    "before answering.\nContext: The Apollo program ran from 1961 to 1972 and "
    "landed twelve astronauts on the Moon across six successful missions.\n\n"
)
PROMPTS = [
    "The capital of France is",
    "Compute 17 * 23. Show the result as a single integer.",
    "List three primary colors, one per line, no other text.",
    "The secret word is 'platypus'. Write one sentence about the weather, "
    "then on a new line repeat the secret word.",
    SHARED + "Question: In what year did the Apollo program end? "
    "Answer with just the year.",
    SHARED + "Question: How many astronauts landed on the Moon? "
    "Answer with a single word or number.",
]


def main() -> None:
    mode = os.environ["DECOUPLED_E2E_MODE"]
    model = os.environ["DECOUPLED_E2E_MODEL"]
    tp = int(os.environ.get("DECOUPLED_E2E_TP", "4"))
    out_path = os.environ["DECOUPLED_E2E_OUT"]
    block_size = int(os.environ.get("DECOUPLED_E2E_BS", "64"))

    if mode == "decoupled":
        # Enable the buddy allocator + decoupled-page layout. We deliberately do
        # NOT set VLLM_BUDDY_MAX_ORDER: the coordinator derives the buddy order
        # from the layout's per-group spans, so this exercises that
        # auto-derivation end to end (the test used to mask the missing
        # derivation by pinning the order here).
        os.environ["VLLM_USE_BUDDY_BLOCK_POOL"] = "1"
    else:
        # Force baseline explicitly OFF so it cannot inherit a stray enable from
        # the ambient environment, which would collapse the A/B.
        os.environ["VLLM_USE_BUDDY_BLOCK_POOL"] = "0"

    # Import only after env is set so the flags are read at startup.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        block_size=block_size,
    )

    cache_cfg = llm.llm_engine.vllm_config.cache_config
    info = {
        "mode": mode,
        # The allocator flag this run actually saw (captured after the mode
        # block sets it). The parent test asserts baseline=off / decoupled=on so
        # the A/B can't be silently collapsed by an ambient env var.
        "buddy_enabled": os.environ.get("VLLM_USE_BUDDY_BLOCK_POOL") == "1",
        "block_size": cache_cfg.block_size,
        # num_gpu_blocks is the cross-process-visible proof that the decoupled
        # layout engaged: reinterpreting the id space at the smaller base page
        # yields strictly more base blocks than the uniform baseline. The
        # kv_cache_config (and base_page_bytes) lives in the engine-core process
        # under TP and is not readable from this driver process.
        "num_gpu_blocks": cache_cfg.num_gpu_blocks,
        "mamba_block_size": getattr(cache_cfg, "mamba_block_size", None),
        "mamba_cache_mode": getattr(cache_cfg, "mamba_cache_mode", None),
    }

    tok = llm.get_tokenizer()
    prompts = PROMPTS
    if getattr(tok, "chat_template", None):
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in PROMPTS
        ]

    greedy = SamplingParams(temperature=0.0, max_tokens=48, seed=0)

    def run(engine):
        outs = engine.generate(prompts, greedy)
        return [
            {"text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)}
            for o in outs
        ]

    pass1 = run(llm)
    pass2 = run(llm)  # must hit the prefix cache and match pass-1

    with open(out_path, "w") as f:
        json.dump({"info": info, "pass1": pass1, "pass2": pass2}, f)
    print(f"[{mode}] wrote {out_path} (bs={info['block_size']})", flush=True)

    # Release GPU memory and the tensor-parallel workers before exit so a
    # subsequent run (the parent test loads baseline then decoupled
    # sequentially) does not start against a still-occupied device.
    del llm
    from vllm.distributed import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
