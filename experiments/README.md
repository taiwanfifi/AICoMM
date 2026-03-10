# Experiment Reproducibility Guide

**Project**: KV-Cache Semantic Communication (Scout Protocol)
**Authors**: Wei-Lun Cheng, Wanjiun Liao — NTU EE
**Last verified**: 2026-02-27
**GPU status**: All instances destroyed. All data synced locally.

---

## Quick Start

```bash
# CPU-only experiments (no GPU needed)
python experiments/scripts/run_exp_bw_estimation.py
python experiments/scripts/run_exp_protocol_real_traces.py

# GPU experiments (requires NVIDIA GPU, 40-80GB VRAM)
python experiments/scripts/run_exp_scout_n200.py
python experiments/scripts/run_exp_q2c_ablation.py

# Generate figures
python papers/jsac/generate_figures.py
python papers/paper-A/generate_figures.py
```

## Dependencies

```
torch>=2.0
transformers>=4.40
numpy
matplotlib
tqdm
```

## Critical Technical Notes

1. **Attention extraction**: `output_attentions=True` requires `attn_implementation="eager"` (SDPA/FlashAttention don't support it)
2. **Transformers 5.x**: Use `cache.layers[i].keys` / `.values` (NOT `.key_cache`)
3. **BFloat16 → numpy**: `.float().cpu().numpy()` (NOT `.cpu().numpy()`)
4. **FP16 + Eager on Blackwell (sm_120)**: Produces garbage for 7B+ models. Always use `dtype=torch.bfloat16`
5. **Prompt format**: Qwen2.5 BASE models need `"Context: ...\nQuestion: ...\nAnswer:"` (NOT ChatML)
6. **Quantization**: Cannot use `past_key_values=` with `generate()` for quantized caches. Use manual greedy decode loop.
7. **14B + output_attentions @ 4K+**: OOM (~148GB needed). Workaround: use 3B→7B pair or quality-only metrics.

## GPU Requirements

| Experiment | Min VRAM | Models Loaded | Est. Time |
|-----------|----------|---------------|-----------|
| Scout n=200 (Qwen 3B/7B/14B) | 80 GB | 2 models simultaneous | ~20 min |
| Q2C ablation (3 models) | 40 GB | 1 model at a time | ~15 min |
| Cross-architecture (Llama/Gemma) | 40 GB | 2 models | ~30 min |
| Perplexity (WikiText-2) | 40 GB | 1 model | ~10 min |
| Long context 4K | 80 GB | 2 models | ~40 min |
| Instruct alignment | 80 GB | 2 instruct models | ~25 min |
| Summarization (XSum) | 80 GB | 2 models | ~95 min |
| Protocol simulation | 0 (CPU) | None | ~2 min |
| BW estimation | 0 (CPU) | None | ~5 sec |

**Tested on**: NVIDIA A100 SXM4 (40GB/80GB), RTX 3090, via vast.ai
**Models path on GPU server**: `/dev/shm/hf_7b/` (download from HuggingFace)

## Result Files (49 JSON)

### JSAC Core (cited in paper)

| File | Experiment | Paper Section | Key Result |
|------|-----------|---------------|------------|
| `exp_scout_n200_20260210_073907.json` | Scout validation | III, VII-B | 83.7% overlap 7B→14B |
| `exp_q2c_ablation_20260210_071938.json` | Q2C last vs all layer | III-B, VII-E | Pearson r>0.99 |
| `exp_attention_entropy_20260210_064020.json` | Entropy analysis | III-E | 3B:4.21, 7B:5.49, 14B:4.65 |
| `exp_protocol_real_traces_20260209_185457.json` | 5G protocol sim | VI, VII-D | 100% scout deadline compliance |
| `exp_perplexity_20260210_130438.json` | WikiText-2 PPL | V-D | INT4 catastrophic for 7B (80.3) |
| `exp_hybrid_mode_20260210_083814.json` | Hybrid mode | IV-E | Hybrid = pure scout quality |
| `exp_scout_multitask_20260210_132605.json` | SQuAD + HotpotQA | VII-C | Scout +0.088 @50% (p=0.026) |
| `exp_scout_long_ctx_7b14b_20260210_131505.json` | Long context 1-2K | VII-D | Stable 82-83% @75% |
| `exp_cross_family_overlap_20260223_125732.json` | Llama 3, Gemma 2 | III-C | Llama 91.8%, Gemma 86.1% |
| `exp_cross_family_scout_20260211_111947.json` | Qwen→Mistral | III-D | 73% cross-family overlap |
| `exp_paper_a_unified_qwen_7b_20260211_111947.json` | Q2C vs SnapKV vs H2O | V-A | Q2C ≈ SnapKV >> H2O |
| `exp_paper_a_quant_fix_20260211_135232.json` | Quantization fix | V-B | INT8=99.4%, INT4=68.7%, Mixed=93.6% |
| `exp_eager_overhead_20260211_111947.json` | Eager overhead | VII-E | +0.44% marginal cost |
| `exp_long_context_4k_20260226_095315.json` | 4K tokens | VII-D | 80.2% @4K (3B→7B) |
| `exp_instruct_alignment_20260226_103928.json` | Instruct models | VII-E | 85.8% overlap (instruct) |
| `exp_summarization_scout_20260226_084727.json` | XSum summarization | VII-C | 88.3% overlap, ROUGE-1 |
| `exp_perplexity_mistral_20260211_111947.json` | Mistral PPL | V-D | INT4 robust for Mistral |
| `exp_bw_estimation_20260227_035412.json` | BW estimation | VI-B | EWMA quality gap <1.5% |

### Legacy Results (in `legacy-results/`)

23 additional JSON files from batch experiments (batch2-batch30). These are exploratory results from Jan-Feb 2026. They are not cited in the current papers but preserved for historical reference.

## Directory Structure

```
experiments/
├── scripts/              # 22 core experiment scripts
│   ├── run_exp_*.py      # Primary experiments (15)
│   ├── run_fix_quant.py  # Quantization bug fix
│   ├── exp_utils.py      # Shared utilities
│   └── ...
├── results/              # 26 core JSON results (exp_*.json only)
├── legacy-scripts/       # 41 old batch scripts (superseded by run_exp_*)
├── legacy-results/       # 23 old batch results (exploratory, not in papers)
├── figures/              # Figure generation
├── poc/                  # CPU-only proof-of-concept (TinyLlama)
└── README.md
```

## Script Inventory

### Core (in `scripts/`, produce paper results)
```
run_exp_scout_n200.py          # Scout n=200 (3 Qwen pairs)
run_exp_q2c_ablation.py        # Q2C last-layer vs all-layer
run_exp_attention_entropy.py   # Attention entropy analysis
run_exp_protocol_real_traces.py # 5G protocol simulation (CPU)
run_exp_perplexity.py          # WikiText-2 perplexity
run_exp_hybrid_mode.py         # Hybrid scout + partial KV
run_exp_scout_multitask.py     # SQuAD + HotpotQA
run_exp_scout_long_context.py  # Long context 1-2K
run_exp_long_context_4k.py     # Long context 4K
run_exp_instruct_alignment.py  # Instruct model alignment
run_exp_cross_family_overlap.py # Llama 3, Gemma 2
run_exp_scout_cross_family.py  # Qwen→Mistral
run_exp_summarization_scout.py # XSum summarization
run_exp_bw_estimation.py       # BW estimation validation (CPU)
run_exp_attention_heatmap.py   # Attention visualization
```

### Support (in `scripts/`)
```
exp_utils.py                   # Shared utilities (model loading, Q2C, stats)
run_reviewer_fixes.py          # Reviewer-requested experiments
run_fix_quant.py               # Quantization bug fix (manual decode loop)
run_llama_eval.py              # Llama-3.1-8B evaluation
check_progress.sh              # Progress monitor
```

### Legacy (in `legacy-scripts/`, superseded)
41 scripts (`run_batch2.py` through `run_batch30_adaptive_protocol.py`) plus deployment scripts (`setup_gpu_server.sh`, `sync_server.sh`). These were exploratory experiments from Jan-Feb 2026, now replaced by `run_exp_*.py`.

## Proof-of-Concept (CPU, TinyLlama-1.1B)

```
experiments/poc/
├── exp1_kv_structure.py         # KV-cache structure exploration
├── exp2_attention_selection.py  # Attention-based position selection
├── exp3_cross_injection.py      # Cross-model KV injection
├── exp4_task_aware_selection.py # Task-specific Q2C
├── exp5_closed_loop_protocol.py # Protocol state machine
├── exp6_bandwidth_accuracy.py   # Bandwidth-accuracy tradeoffs
├── utils.py                     # PoC utilities
├── task_qa.py                   # QA task framework
└── requirements.txt             # torch, transformers, numpy, matplotlib
```

## Figure Generation

| Script | Paper | Figures |
|--------|-------|---------|
| `papers/paper-A/generate_figures.py` | Paper A | 7 figs (selection, quantization, scaling) |
| `experiments/figures/generate_paper_b_figures.py` | Paper B | 4 figs (scout, protocol, multi-agent) |
| `papers/jsac/generate_figures.py` | JSAC merged | 10 figs (all combined) |

## Data Lineage

```
JSAC Paper (15 pages)
├── Section III (Attention Alignment) ← exp_scout_n200, exp_cross_family_overlap
├── Section IV (Protocol Design)      ← exp_hybrid_mode, exp_protocol_real_traces
├── Section V (Compression)           ← exp_paper_a_unified, exp_paper_a_quant_fix, exp_perplexity
├── Section VI (Transport)            ← exp_protocol_real_traces, exp_bw_estimation
├── Section VII (Experiments)         ← ALL result files
└── Figures                           ← papers/jsac/generate_figures.py
```

## Re-deploying GPU Experiments

If you need to re-run experiments on a new GPU server:

```bash
# 1. Set up environment
pip install torch transformers numpy matplotlib tqdm

# 2. Download models (or copy from cache)
#    Models used: Qwen2.5-{3B,7B,14B}, Llama-3.2-3B, Llama-3.1-8B,
#                 Gemma-2-{2B,9B}, Mistral-7B-v0.3, Yi-1.5-6B
#    For Llama-3.2-3B: use unsloth/Llama-3.2-3B (ungated mirror)

# 3. Set environment variables
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

# 4. Run experiments
python experiments/scripts/run_exp_scout_n200.py
# Each script is self-contained and saves to experiments/results/
```

All scripts set `TRANSFORMERS_NO_TF=1` and `TOKENIZERS_PARALLELISM=false` internally. Seeds are fixed (42 for main, 123 for long-context) for exact reproducibility.
