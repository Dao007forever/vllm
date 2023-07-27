# Chatbot Experiment (CacheFlow)

## Installation

```bash
pip install psutil numpy ray torch
pip install git+https://github.com/huggingface/transformers  # Required for LLaMA.
pip install sentencepiece  # Required for LlamaTokenizer.
pip install ninja  # To parallelize the compilation of flash-attn.
pip install flash-attn  # This may take up to 10 mins.
pip install -e .
```

## Run all experiments (CacheFlow)

```bash
python3 run_all_chatbot.py
```

The results will be saved to `~/exp`
