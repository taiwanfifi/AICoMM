#!/bin/bash
# ============================================================================
# GPU Server Setup Script
# Re-run this script to rebuild the experiment environment on a new instance.
#
# Usage: ssh -p PORT root@HOST 'bash -s' < setup_gpu_server.sh
#   OR: scp this to server, then: bash setup_gpu_server.sh
#
# Server requirements: NVIDIA GPU, Python 3.10+, pip
# ============================================================================

set -e

echo "=== GPU Server Setup for KV-Cache Experiments ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python3 --version)"

# ---------- 1. Install Python packages ----------
echo ""
echo "--- Installing dependencies ---"
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install transformers>=4.45 datasets accelerate scipy numpy matplotlib

# Verify torch + CUDA
python3 -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---------- 2. Setup HF cache directory ----------
echo ""
echo "--- Setting up HuggingFace cache ---"
# Use /workspace if available (persistent), else /root
if [ -d /workspace ]; then
    HF_DIR=/workspace/hf_cache
else
    HF_DIR=/root/hf_cache
fi
mkdir -p $HF_DIR/datasets
export HF_HOME=$HF_DIR
export HF_DATASETS_CACHE=$HF_DIR/datasets

echo "HF_HOME=$HF_DIR"

# ---------- 3. Setup experiment directory ----------
echo ""
echo "--- Setting up experiment directory ---"
EXP_DIR=/workspace/experiments
mkdir -p $EXP_DIR/scripts $EXP_DIR/results $EXP_DIR/logs

echo "Experiment directory: $EXP_DIR"

# ---------- 4. Pre-download models ----------
echo ""
echo "--- Pre-downloading models (this may take a while) ---"
python3 -c "
import os
os.environ['HF_HOME'] = '$HF_DIR'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    'Qwen/Qwen2.5-3B',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-14B',
    'mistralai/Mistral-7B-v0.3',
    '01-ai/Yi-6B-Chat',
]

for name in models:
    print(f'Downloading {name}...')
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        # Download model files but don't load into GPU
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, torch_dtype='auto',
            device_map='cpu', low_cpu_mem_usage=True
        )
        del model
        print(f'  OK: {name}')
    except Exception as e:
        print(f'  FAILED: {name}: {e}')

# Pre-download datasets
from datasets import load_dataset
print('Downloading SQuAD v2...')
load_dataset('rajpurkar/squad_v2', split='validation')
print('Downloading WikiText-2...')
load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
print('Downloading HotpotQA...')
try:
    load_dataset('hotpot_qa', 'distractor', split='validation')
except: pass
print('Done.')
"

# ---------- 5. Create environment file ----------
cat > $EXP_DIR/.env << 'ENVEOF'
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
ENVEOF

echo ""
echo "=== Setup complete ==="
echo "To run experiments:"
echo "  source $EXP_DIR/.env"
echo "  cd $EXP_DIR/scripts"
echo "  python3 run_exp_q2c_ablation.py"
echo ""
echo "To sync results back:"
echo "  scp -P PORT root@HOST:$EXP_DIR/results/*.json ./results/"
