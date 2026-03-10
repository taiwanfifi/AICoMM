# 實驗結果索引 (Experiment Results Index)

**專案**: KV-Cache 語意通訊 (Scout Protocol)
**作者**: 鄭威倫、廖婉君 — 國立臺灣大學電機工程學系
**最後更新**: 2026-03-10
**檔案總數**: 26 個 JSON 結果檔

---

## 一、結果檔案總覽

| 檔案名稱 | 日期 | 模型 | 樣本數 | JSAC 章節 | 主要發現 |
|----------|------|------|--------|-----------|---------|
| `exp_protocol_real_traces_20260209_185457.json` | 2026-02-09 | 無（CPU 模擬） | 2000 requests | IV, VI, VII-D | 5G 協定模擬：Scout 模式 100% 達成時限要求 |
| `exp_q2c_ablation_7b_20260210_062832.json` | 2026-02-10 | Qwen2.5-7B | 200 | III-B, VII-E | Q2C 最後層 vs 全層消融：Pearson r=0.990 |
| `exp_attention_entropy_20260210_064020.json` | 2026-02-10 | Qwen2.5-3B | 200 | III-E | 注意力熵分析：3B 整體均值 2.61, Q2C 熵 4.21 |
| `exp_attention_entropy_20260210_073931.json` | 2026-02-10 | Qwen2.5-14B | 200 | III-E | 注意力熵分析：14B 整體均值 2.13, Q2C 熵 4.65 |
| `exp_scout_n200_20260210_073907.json` | 2026-02-10 | Qwen2.5-3B/7B/14B | 200 | III, VII-B | Scout 核心驗證：3 組 Qwen 配對，7B→14B overlap 81.5% @75% |
| `exp_q2c_ablation_20260210_071938.json` | 2026-02-10 | Qwen2.5-7B/3B/14B | 200 | III-B, VII-E | Q2C 最後層 vs 全層（3 模型）：Pearson r>0.99 全模型 |
| `exp_hybrid_mode_20260210_083814.json` | 2026-02-10 | Qwen2.5-7B→14B | 200 | IV-E | 混合模式（Scout + 部分 KV 傳輸）：品質等同純 Scout |
| `exp_perplexity_20260210_130438.json` | 2026-02-10 | Qwen2.5-7B, Qwen2.5-14B | 100 段 | V-D | WikiText-2 困惑度：INT4 對 7B 災難性（80.3），14B 穩健（5.87） |
| `exp_scout_long_ctx_7b14b_20260210_131505.json` | 2026-02-10 | Qwen2.5-7B→14B | 多樣本 | VII-D | 長上下文 1-2K：overlap 穩定 82-83% @75% retention |
| `exp_scout_multitask_20260210_132605.json` | 2026-02-10 | Qwen2.5-7B→14B | 100/task | VII-C | 多任務（SQuAD+HotpotQA）：Scout SQuAD +0.088 @50%（p=0.026） |
| `exp_cross_family_scout_20260211_111947.json` | 2026-02-11 | Qwen2.5-7B→Mistral-7B | 200 | III-D | 跨家族 Scout：73% overlap @75%，Scout 顯著改善（p=0.013） |
| `exp_eager_overhead_20260211_111947.json` | 2026-02-11 | Qwen2.5-7B | benchmark | VII-E | Eager 模式額外開銷：output_attentions 僅增加 +0.44% |
| `exp_paper_a_unified_qwen_7b_20260211_111947.json` | 2026-02-11 | Qwen2.5-7B | 200 | V-A | 選擇方法比較：Q2C ≈ SnapKV >> H2O（H2O p<0.056） |
| `exp_perplexity_mistral_20260211_111947.json` | 2026-02-11 | Mistral-7B-v0.3 | 100 段 | V-D | Mistral 困惑度：INT4 穩健（BF16=6.49, INT4=6.51） |
| `exp_yi_comparison_20260211_111947.json` | 2026-02-11 | Yi-6B-Chat, Yi-1.5-6B | 100 | VII-E | Base vs Chat 比較：Base F1=0.659 vs Chat F1=0.161 |
| `exp_yi_comparison_yi_1_5_6b_20260211_111947.json` | 2026-02-11 | Yi-1.5-6B | 100 | VII-E | Yi-1.5-6B 逐樣本結果（Base 模型表現良好） |
| `exp_yi_comparison_yi_6b_chat_20260211_111947.json` | 2026-02-11 | Yi-6B-Chat | 100 | VII-E | Yi-6B-Chat 逐樣本結果（Chat 模型表現差） |
| `exp_paper_a_quant_fix_20260211_135232.json` | 2026-02-11 | Qwen2.5-7B | 200 | V-B | 量化修復：INT8=99.4%, INT4=68.7%, Mixed=93.6%（手動解碼迴圈） |
| `exp_llama_eval_20260211_150948.json` | 2026-02-11 | Llama-3.1-8B | 100 | VII-E | Llama-3.1-8B SQuAD 評估：F1=0.288（Base 模型冗長輸出） |
| `exp_cross_family_overlap_20260223_125732.json` | 2026-02-23 | Llama-3.2-3B→3.1-8B, Gemma-2-2B→9B | 200 | III-C | 跨架構 overlap：Llama 91.8%, Gemma 86.1% @75% |
| `exp_summarization_scout_20260226_084727.json` | 2026-02-26 | Qwen2.5-7B→14B | 200 | VII-C | XSum 摘要任務：88.3% overlap，ROUGE-1 評估 |
| `exp_summarization_scout_summary_20260226_084727.json` | 2026-02-26 | Qwen2.5-7B→14B | 200 | VII-C | XSum 摘要（摘要版本，同上） |
| `exp_long_context_4k_20260226_095315.json` | 2026-02-26 | Qwen2.5-3B→7B | 100 | VII-D | 4096 token 長上下文：80.2% overlap @75%（3B→7B） |
| `exp_instruct_alignment_20260226_103928.json` | 2026-02-26 | Qwen2.5-7B/14B-Instruct | 200 | VII-E | Instruct 模型對齊：85.8% overlap（與 Base 模型一致） |
| `exp_bw_estimation_20260227_035355.json` | 2026-02-27 | Qwen-7B（CPU 模擬） | 5000 steps | VI-B | 頻寬估計驗證：EWMA 品質差距 <1.5% |
| `exp_bw_estimation_20260227_035412.json` | 2026-02-27 | Qwen-7B（CPU 模擬） | 5000 steps | VI-B | 頻寬估計驗證（第二次執行，結果一致） |

---

## 二、資料血統 (Data Lineage)

以下說明各實驗結果檔案如何對應到 JSAC 論文的各章節。

```
JSAC 論文 (15 頁)
│
├── Section III — 注意力對齊 (Attention Alignment)
│   ├── III-B  Q2C 消融 ← exp_q2c_ablation, exp_q2c_ablation_7b
│   ├── III-C  跨架構 overlap ← exp_cross_family_overlap (Llama 3, Gemma 2)
│   ├── III-D  跨家族 scout ← exp_cross_family_scout (Qwen→Mistral)
│   ├── III-E  注意力熵 ← exp_attention_entropy (×2, 3B+14B)
│   └── 核心驗證 ← exp_scout_n200 (3 組 Qwen 配對)
│
├── Section IV — 協定設計 (Protocol Design)
│   ├── IV-E  混合模式 ← exp_hybrid_mode
│   └── 協定模擬 ← exp_protocol_real_traces
│
├── Section V — 壓縮 (Compression)
│   ├── V-A   選擇方法比較 ← exp_paper_a_unified_qwen_7b (Q2C vs SnapKV vs H2O)
│   ├── V-B   量化修復 ← exp_paper_a_quant_fix (INT8/INT4/Mixed)
│   └── V-D   困惑度 ← exp_perplexity (Qwen), exp_perplexity_mistral (Mistral)
│
├── Section VI — 傳輸 (Transport)
│   ├── VI     協定模擬 ← exp_protocol_real_traces
│   └── VI-B   頻寬估計 ← exp_bw_estimation (×2)
│
└── Section VII — 實驗 (Experiments) ← 全部結果檔案
    ├── VII-B  Scout 核心 ← exp_scout_n200
    ├── VII-C  多任務 ← exp_scout_multitask (SQuAD+HotpotQA)
    │          摘要 ← exp_summarization_scout (XSum)
    ├── VII-D  長上下文 ← exp_scout_long_ctx_7b14b (1-2K)
    │                   ← exp_long_context_4k (4K tokens)
    │          協定效能 ← exp_protocol_real_traces
    ├── VII-E  消融與補充 ← exp_eager_overhead
    │                     ← exp_yi_comparison (Base vs Chat)
    │                     ← exp_llama_eval (Llama-3.1-8B)
    │                     ← exp_instruct_alignment (Instruct 模型)
    │                     ← exp_q2c_ablation
    └── 圖表生成 ← papers/jsac/generate_figures.py
```

---

## 三、模型清單

| 模型 | 參數量 | 用途 |
|------|--------|------|
| Qwen/Qwen2.5-3B | 3B | Edge 模型（Scout 發送端） |
| Qwen/Qwen2.5-7B | 7B | Edge/Cloud 雙用 |
| Qwen/Qwen2.5-14B | 14B | Cloud 模型（Scout 接收端） |
| Qwen/Qwen2.5-7B-Instruct | 7B | Instruct 對齊驗證 |
| Qwen/Qwen2.5-14B-Instruct | 14B | Instruct 對齊驗證 |
| meta-llama/Llama-3.1-8B | 8B | 跨架構驗證（Cloud） |
| unsloth/Llama-3.2-3B | 3B | 跨架構驗證（Edge，ungated mirror） |
| google/gemma-2-2b | 2B | 跨架構驗證（Edge） |
| google/gemma-2-9b | 9B | 跨架構驗證（Cloud） |
| mistralai/Mistral-7B-v0.3 | 7B | 跨家族 Scout 驗證 |
| 01-ai/Yi-1.5-6B | 6B | Base vs Chat 比較 |
| 01-ai/Yi-6B-Chat | 6B | Base vs Chat 比較 |

---

## 四、實驗時間軸

| 日期 | 實驗 | 說明 |
|------|------|------|
| 2026-02-09 | protocol_real_traces | 5G 協定模擬（CPU），首批核心實驗 |
| 2026-02-10 | scout_n200, q2c_ablation, attention_entropy, hybrid_mode, perplexity, scout_long_ctx, scout_multitask | GPU 主實驗批次（A100 80GB） |
| 2026-02-11 | cross_family_scout, eager_overhead, paper_a_unified, perplexity_mistral, yi_comparison, paper_a_quant_fix, llama_eval | Reviewer 修復實驗 |
| 2026-02-23 | cross_family_overlap | 跨架構 overlap（Llama 3 + Gemma 2），論文升級為跨架構 |
| 2026-02-26 | summarization_scout, long_context_4k, instruct_alignment | 補充實驗：摘要任務、4K 長上下文、Instruct 模型 |
| 2026-02-27 | bw_estimation | 頻寬估計驗證（CPU），R4 reviewer 修復 |

---

## 五、重現說明

所有實驗腳本位於 `experiments/scripts/`，每個腳本獨立執行並輸出帶時間戳的 JSON 至本目錄。隨機種子固定（主實驗 seed=42，長上下文 seed=123）以確保完全可重現。詳細重現步驟請參閱 `experiments/README.md`。
