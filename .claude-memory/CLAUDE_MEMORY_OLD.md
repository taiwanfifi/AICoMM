# Project Memory: KV-Cache Semantic Communication

## Key Technical Learnings

### HuggingFace Transformers (v4.40+)
- `output_attentions=True` requires `attn_implementation="eager"` (SDPA doesn't support it)
- Transformers 5.x: `cache.layers[i].keys` / `.values` (NOT `.key_cache`)
- Cannot create NEW DynamicCache for generate() — modify IN-PLACE then manual loop
- Set `TRANSFORMERS_NO_TF=1` to prevent TensorFlow deadlock on macOS
- Phi-3.x custom code incompatible with transformers 5.x. NOT worth fixing; skip Phi-3.5.

### CRITICAL: FP16 + Eager Overflow on Blackwell (sm_120)
- 7B+ models with `FP16 + eager` produce GARBAGE on Blackwell GPU
- **FIX**: Use `dtype=torch.bfloat16` for 7B+ on Blackwell

### BFloat16 + numpy
- `tensor.cpu().numpy()` FAILS for BFloat16 — always use `.float().cpu().numpy()`

### RoPE Sparse Mode Bug
- Physical removal of positions breaks RoPE encoding → use attention mask instead

### Prompt Format for Base Models
- Qwen2.5 BASE models don't understand ChatML — use simple "Context: ...\nQuestion: ...\nAnswer:" format
- ChatML (`<|im_start|>`) only works with -Instruct variants
- F1 drops from 0.73 to 0.08 with wrong prompt format!

### 14B + output_attentions OOM at Long Context
- 14B with `output_attentions=True` at 4096+ tokens: 48 layers × [1,40,4096,4096] × 4B ≈ 128GB → OOM
- Workaround: test 7B→14B long context at 1K and 2K only (sufficient to show stability)

## JSAC Experiments — ALL COMPLETE (2026-02-10)

| Exp | Name | Key Result | JSON File |
|-----|------|-----------|-----------|
| P1 | Protocol real traces | 100% deadline compliance scout mode | `exp_protocol_real_traces_20260209_185457.json` |
| F1 | Q2C ablation (3 models) | Pearson r=0.990-0.995 | `exp_q2c_ablation_20260210_071938.json` |
| A1 | Attention entropy | 3B:4.21, 7B:5.49, 14B:4.65 | `exp_attention_entropy_20260210_06*.json` + `_07*.json` |
| S1 | Scout n=200 (3 pairs) | 7B→14B improves 14B | `exp_scout_n200_20260210_073907.json` |
| S2 | Scout long context | Overlap stable 82-83% at 75% | `exp_scout_long_ctx_*.json` |
| F2 | Perplexity (7B/14B) | INT4 catastrophic for 7B (80), fine for 14B (5.9) | `exp_perplexity_20260210_130438.json` |
| P2 | Hybrid mode | Completed | `exp_hybrid_mode_20260210_083814.json` |
| S4 | Multitask (SQuAD+HotpotQA) | Scout +0.088 on SQuAD@50% (p=0.026) | `exp_scout_multitask_20260210_132605.json` |

### Cross-Architecture Overlap — COMPLETE (2026-02-23)
| # | Family | Pair | Overlap @75% | JSON File |
|---|--------|------|-------------|-----------|
| 1 | Llama 3 | 3B→8B | **91.8%** (p=0.060) | `exp_cross_family_overlap_20260223_125732.json` |
| 2 | Gemma 2 | 2B→9B | **86.1%** (p=0.359) | `exp_cross_family_overlap_20260223_125732.json` |

- Used `unsloth/Llama-3.2-3B` (ungated mirror) + `meta-llama/Llama-3.1-8B`
- Gemma-2 has 256 head_dim (not 128 like Qwen/Llama)
- All three families show 84-92% overlap → paper upgraded from Qwen-only to cross-architecture

### Reviewer Fix Experiments — ALL COMPLETE (2026-02-11)
| # | Experiment | Result | JSON File |
|---|-----------|--------|-----------|
| 1 | Cross-family scout (Qwen-7B→Mistral-7B) | overlap 73/59/41% @75/50/25%; scout helps @75% (p=0.013) | `exp_cross_family_scout_20260211_111947.json` |
| 2 | Paper A unified selection (Q2C vs SnapKV vs H2O, n=200) | Q2C≈SnapKV>>H2O; H2O p<1e-5 | `exp_paper_a_unified_qwen_7b_20260211_111947.json` |
| 2b | Paper A quantization FIX (n=200) | INT8=99.4%, INT4=68.7%, Mixed=93.6% | `exp_paper_a_quant_fix_20260211_135232.json` |
| 3 | Eager overhead benchmark | output_attentions only +0.44% overhead | `exp_eager_overhead_20260211_111947.json` |
| 4 | Yi-6B base vs Chat | Base F1=0.659 vs Chat F1=0.161 | `exp_yi_comparison_20260211_111947.json` |
| 5 | Perplexity Mistral-7B | BF16=6.49, INT8=6.49, INT4=6.51 | `exp_perplexity_mistral_20260211_111947.json` |
| 6 | Llama-3.1-8B eval (n=100) | F1=0.288±0.068 (verbose base model) | `exp_llama_eval_20260211_150948.json` |

### Quantization Bug Fix
- Original `run_reviewer_fixes.py` passed `past_key_values=None` to generate() → ignored quantized cache
- Fix: `run_fix_quant.py` uses manual greedy decode loop from quantized cache
- Qwen-7B results: INT8 lossless (99.4%), INT4 catastrophic (68.7%), Mixed-INT4 recovers (93.6%)

### Not Done
- TriviaQA in S4: dataset too large

### Key Consolidated Results

**Perplexity (WikiText-2, KV-cache quantization):**
| Model | BF16 | INT8 | INT4 | Mixed-INT4 |
|-------|------|------|------|------------|
| Qwen-7B | 8.63 | 8.85 | **80.27** | 8.97 |
| Qwen-14B | 5.73 | 5.73 | 5.87 | 5.83 |

**Multitask Scout (7B→14B):**
- SQuAD: scout +0.088 at 50% (p=0.026), +0.085 at 25% (p=0.039)
- HotpotQA: neutral (gap <0.015, p>0.6)

**Long Context Overlap (7B→14B):**
- @1024: 83.3% / 69.4% / 55.6% (75/50/25% ret)
- @2048: 82.7% / 68.2% / 53.9% — stable across lengths

**Q2C Ablation:** last-layer ≈ all-layer (r>0.99 all models)
**Attention Focusing:** 3B entropy 4.21 < 14B 4.65 < 7B 5.49

## GPU Server (vast.ai)
- ALL instances DESTROYED. All data synced locally. No remote experiments running.

## Repository Structure
```
papers/paper-A/main.tex    # KV-cache compression paper (7 pages)
papers/paper-B/main.tex    # Scout protocol paper (7 pages)
experiments/scripts/        # GPU experiment scripts (including 9 new JSAC scripts)
experiments/results/        # JSON result files (31+ files)
experiments/figures/        # Figure generation scripts
research/                   # Research directions
```

## Paper Status
- **Paper A**: 7 pages, COMPLETE — target INFOCOM 2027
- **Paper B**: 7 pages, COMPLETE — target ICC/JSAC
- **JSAC merged paper**: 15 pages (1493 lines), iterating toward top-journal quality
  - Review scores: Gemini R1: 89 → R2: 97 → Claude R3: 72.3 → Claude R4: 72.6
  - **R4 fixes DONE (2026-02-27)**:
    1. ✅ Added prompt-only baseline to Table 2 + new Table (scout_vs_prompt) quantifying decode savings
    2. ✅ BW estimation validation: new experiment (exp_bw_estimation), 2 new tables (bw_validation, alpha_sensitivity)
    3. ✅ Strengthened Prop 1 with bounded quality loss under estimation error
    4. ✅ Added Poisson arrival extension for multi-agent
    5. ✅ Updated fallback hierarchy to 7 modes (including prompt-only)
    6. ✅ Updated abstract, intro contributions, conclusion
  - **R4 remaining TODO** (need GPU or external tools):
    - CacheGen direct quality comparison (code available, needs GPU)
    - Random-k baseline for regularization control
    - 7B→14B at 4K+ tokens
  - Standing directive: "一直迭代到10+ round以上"
  - Review files: `reviewer/REVIEW_2026-02-27_claude_JSAC_round4.md`
  - ChatGPT/Gemini prompts: `reviewer/PROMPT_chatgpt_R4_reviewer.md`, `reviewer/PROMPT_gemini_R4_reviewer.md`
  - **Waiting for**: ChatGPT + Gemini R4 reviews to cross-compare
- **Authors**: Wei-Lun Cheng and Wanjiun Liao, NTU EE
