# Claude Code Memory — KV-Cache Semantic Communication Project

> **Purpose**: This file preserves accumulated knowledge from previous Claude Code sessions.
> When starting on a new computer, tell Claude Code: "Read `.claude-memory/MEMORY.md` and set up your memory."
> Claude will then copy the relevant content to its own memory directory.

---

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

### Quantization Bug Fix
- Original `run_reviewer_fixes.py` passed `past_key_values=None` to generate() → ignored quantized cache
- Fix: `run_fix_quant.py` uses manual greedy decode loop from quantized cache
- Qwen-7B results: INT8 lossless (99.4%), INT4 catastrophic (68.7%), Mixed-INT4 recovers (93.6%)

---

## Experiments — ALL COMPLETE

### Core JSAC Experiments (2026-02-10)

| ID | Name | Key Result | JSON File |
|----|------|-----------|-----------|
| P1 | Protocol real traces | 100% deadline compliance scout mode | `exp_protocol_real_traces_*.json` |
| F1 | Q2C ablation (3 models) | Pearson r=0.990-0.995 | `exp_q2c_ablation_*.json` |
| A1 | Attention entropy | 3B:4.21, 7B:5.49, 14B:4.65 | `exp_attention_entropy_*.json` |
| S1 | Scout n=200 (3 pairs) | 7B→14B improves 14B by 10.2% | `exp_scout_n200_*.json` |
| S2 | Scout long context | Overlap stable 82-83% at 75% | `exp_scout_long_ctx_*.json` |
| F2 | Perplexity (7B/14B) | INT4 catastrophic for 7B (80), fine for 14B (5.9) | `exp_perplexity_*.json` |
| P2 | Hybrid mode | Completed | `exp_hybrid_mode_*.json` |
| S4 | Multitask (SQuAD+HotpotQA) | Scout +0.088 on SQuAD@50% (p=0.026) | `exp_scout_multitask_*.json` |

### Cross-Architecture Overlap (2026-02-23)

| Family | Pair | Overlap @75% | JSON File |
|--------|------|-------------|-----------|
| Qwen 2.5 | 7B→14B | **83%** | `exp_scout_n200_*.json` |
| Llama 3 | 3B→8B | **91.8%** | `exp_cross_family_overlap_*.json` |
| Gemma 2 | 2B→9B | **86.1%** | `exp_cross_family_overlap_*.json` |

### Reviewer Fix Experiments (2026-02-11)

| # | Experiment | Result | JSON File |
|---|-----------|--------|-----------|
| 1 | Cross-family scout (Qwen→Mistral) | overlap 73% @75% | `exp_cross_family_scout_*.json` |
| 2 | Q2C vs SnapKV vs H2O (n=200) | Q2C≈SnapKV>>H2O | `exp_paper_a_unified_*.json` |
| 2b | Quantization FIX | INT8=99.4%, INT4=68.7%, Mixed=93.6% | `exp_paper_a_quant_fix_*.json` |
| 3 | Eager overhead | only +0.44% | `exp_eager_overhead_*.json` |
| 4 | Yi-6B base vs Chat | Base F1=0.659 vs Chat F1=0.161 | `exp_yi_comparison_*.json` |
| 5 | Perplexity Mistral-7B | BF16=6.49, INT8=6.49, INT4=6.51 | `exp_perplexity_mistral_*.json` |

---

## Key Consolidated Results

**Perplexity (WikiText-2, KV-cache quantization):**
| Model | BF16 | INT8 | INT4 | Mixed-INT4 |
|-------|------|------|------|------------|
| Qwen-7B | 8.63 | 8.85 | **80.27** | 8.97 |
| Qwen-14B | 5.73 | 5.73 | 5.87 | 5.83 |

**Multitask Scout (7B→14B):**
- SQuAD: scout +0.088 at 50% (p=0.026), +0.085 at 25% (p=0.039)
- HotpotQA: neutral (gap <0.015, p>0.6)

**Long Context Overlap (7B→14B):**
- @1024: 83.3% / @2048: 82.7% — stable across context lengths

**Q2C Ablation:** last-layer ≈ all-layer (r>0.99 all models)
**Attention Focusing:** 3B entropy 4.21 < 14B 4.65 < 7B 5.49

---

## Paper Status (as of March 2026)

- **Paper A**: 7 pages, COMPLETE — target INFOCOM 2027 (deadline ~Aug 2026)
- **Paper B**: 7 pages, COMPLETE — target ICC/JSAC
- **JSAC merged**: 15 pages, iterating reviews (standing directive: iterate 10+ rounds)
  - Review scores: Gemini R1: 89 → R2: 97 → Claude R3: 72.3 → Claude R4: 72.6
  - R4 fixes done, waiting for additional review rounds
  - Remaining TODO (need GPU): CacheGen comparison, Random-k baseline, 7B→14B @4K+
- **MLSys**: Formatted, shares figures with JSAC
- **Authors**: Wei-Lun Cheng and Wanjiun Liao, NTU EE

## GPU Server
- ALL vast.ai instances DESTROYED. All data synced locally. 49 JSON result files preserved.
