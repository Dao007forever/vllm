# Chatbot Experiment (Orca)

## Installation

```bash
pip install psutil numpy ray torch
pip install git+https://github.com/huggingface/transformers  # Required for LLaMA.
pip install sentencepiece  # Required for LlamaTokenizer.
pip install ninja  # To parallelize the compilation of flash-attn.
pip install flash-attn  # This may take up to 10 mins.
pip install -e .
```

## Run all experiments (Orca)

```bash
python3 run_all_chatbot.py --duration 3600 --len-estimator oracle
python3 run_all_chatbot.py --duration 3600 --len-estimator power2
python3 run_all_chatbot.py --duration 3600 --len-estimator constant
```

The results will be saved to `~/exp`
