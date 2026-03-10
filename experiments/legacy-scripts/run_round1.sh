#!/bin/bash
# run_round1.sh — Run all Round 1 experiments sequentially
#
# Experiments:
#   E1: Summarization scout (XSum, 7B→14B, n=200, ROUGE)     ~3-4 hr
#   E2: Long context 4K (3B→7B full + 7B→14B scout-only)     ~4-6 hr
#   E3: Instruct alignment (7B-I→14B-I, SQuAD, n=200)        ~3-4 hr
#
# Total: ~10-14 hours on A100 80GB
#
# Usage:
#   tmux new -s exp
#   cd /workspace/AI-Comm/experiments/scripts
#   bash run_round1.sh 2>&1 | tee round1.log
#   # Ctrl+B, D to detach

set -e

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPTS_DIR"

echo "=================================================================="
echo "Round 1 Experiments — $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- E1: Summarization Scout ----
echo ""
echo "=================================================================="
echo "[E1] Summarization Scout (XSum, 7B→14B, n=200)"
echo "Start: $(date)"
echo "=================================================================="
python3 run_exp_summarization_scout.py
echo "[E1] Done: $(date)"

# ---- E2: Long Context 4K ----
echo ""
echo "=================================================================="
echo "[E2] Long Context 4K (3B→7B + 7B→14B)"
echo "Start: $(date)"
echo "=================================================================="
python3 run_exp_long_context_4k.py
echo "[E2] Done: $(date)"

# ---- E3: Instruct Alignment ----
echo ""
echo "=================================================================="
echo "[E3] Instruct Alignment (7B-I→14B-I, SQuAD, n=200)"
echo "Start: $(date)"
echo "=================================================================="
python3 run_exp_instruct_alignment.py
echo "[E3] Done: $(date)"

# ---- Summary ----
echo ""
echo "=================================================================="
echo "ALL ROUND 1 EXPERIMENTS COMPLETE — $(date)"
echo "=================================================================="
echo ""
echo "Results saved to: $(ls -t ../../results/exp_summarization_scout_*.json ../../results/exp_long_context_4k_*.json ../../results/exp_instruct_alignment_*.json 2>/dev/null | head -6)"
echo ""
echo "Next: sync results to local and run review iteration"
