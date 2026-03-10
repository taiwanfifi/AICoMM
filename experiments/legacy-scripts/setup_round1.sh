#!/bin/bash
# setup_round1.sh — Run once after vast.ai instance creation
# Installs dependencies and downloads models for Round 1 experiments
#
# Usage: bash setup_round1.sh
# Requirements: A100 80GB SXM4, ≥150GB disk, pytorch Docker image

set -e

echo "=== Round 1 Setup: $(date) ==="

# Install Python dependencies
pip install --upgrade pip
pip install transformers>=4.40 datasets rouge-score scipy accelerate sentencepiece protobuf tqdm

# Verify GPU
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

# Set HuggingFace cache
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

echo ""
echo "=== Downloading models ==="

# Download all models needed for Round 1
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

models = [
    'Qwen/Qwen2.5-3B',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-14B',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
]

for m in models:
    print(f'\nDownloading {m}...')
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, trust_remote_code=True)
    del model, tok
    import gc; gc.collect()
    print(f'  Done: {m}')
"

echo ""
echo "=== Downloading datasets ==="

python3 -c "
from datasets import load_dataset
print('Downloading SQuAD v2...')
load_dataset('rajpurkar/squad_v2', split='validation')
print('Downloading XSum...')
load_dataset('EdinburghNLP/xsum', split='test')
print('Done.')
"

echo ""
echo "=== Setup complete: $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Models cached in: $HF_HOME"
echo ""
echo "To run experiments:"
echo "  cd /workspace/AI-Comm/experiments/scripts"
echo "  tmux new -s exp"
echo "  bash run_round1.sh"
