# 全面盲點分析：所有 Reviewer 來源交叉比對

**日期**: 2026-02-26
**來源**: Gemini 2.5 Pro review × 3, 內部 reviewer_questions.md × 6, REVIEW_STATUS.md, REVIEW_2026-02-13.md

---

## 一、已完全解決的問題（不需任何行動）

| # | 問題 | 解決方式 | 驗證來源 |
|---|------|---------|---------|
| 1 | Q2C 公式兩篇不一致 | 統一為 last-layer + ablation (r>0.99) | Gemini 未提到 |
| 2 | Yi-6B 用 Chat 模型 | Base vs Chat 比較 + 改用 Base | Gemini 未提到 |
| 3 | n=50 不足 | 加到 n=200 | Gemini 稱讚 "commendable" |
| 4 | 缺統計檢定 | 加 Bonferroni + 95% CI + paired t-test | Gemini 稱讚 "model of rigor" |
| 5 | Table 2/5 不同 sample set | 統一 n=200 | Gemini 未提到 |
| 6 | Q2C overclaim (29-47%) | 改為 "matches SnapKV" | Gemini 認可 honest framing |
| 7 | Markov chain 不真實 | 換 Lumos5G real traces | Gemini 稱讚 "grounded in reality" |
| 8 | 只在 Qwen 驗證 | 加 Llama 3, Gemma 2, Mistral | Gemini 稱讚跨架構 |
| 9 | 缺 perplexity | 加 WikiText-2 perplexity | Gemini 未提到 |
| 10 | Attention focusing 矛盾 | 改為 capacity matching 解釋 | Gemini 未提到（已修好） |
| 11 | Eager overhead 未量化 | 加 benchmark (+0.44%) | Gemini 未提到 |
| 12 | 缺 multi-task | 加 HotpotQA | Gemini 認可 |
| 13 | Context length 太短 | 加 1K/2K 實驗 | Gemini 認可穩定性 |
| 14 | "Risk-free" overclaim | Paper B 有問題，JSAC 已修正 | Gemini 對 JSAC 未提到 |

---

## 二、需要論文層面修改的問題（不需新實驗）

| # | 問題 | 來源 | 建議修改 | 優先級 |
|---|------|------|---------|--------|
| P1 | Cross-family improvement 討論不足 | Gemini JSAC | Qwen→Mistral p=0.013 值得 1-2 句解釋 | Should Fix |
| P2 | Llama borderline p=0.06 應承認 | Gemini JSAC | 加一句 nuance | Should Fix |
| P3 | Q2C vs H2O p-value 誤報 | Internal (REVIEW_2026-02-13) | p=2.65e-5 不是 <1e-5，需修正 | Must Fix |
| P4 | Q2C vs SnapKV 相同 p-value 可疑 | Internal (REVIEW_2026-02-13) | 已加 footnote 解釋，確認足夠 | Done? |
| P5 | 加 attention heatmap visualization | Gemini JSAC + Internal | 一張 qualitative 圖 | Nice to Have |
| P6 | FlashAttention 相容性討論 | Gemini JSAC | 加一句提到 FA 不支援 output_attentions | Nice to Have |

---

## 三、需要新實驗的問題（★ 核心迭代目標）

### ★★★ 最高優先級

| # | 實驗 | 所有 reviewer 都提到? | 預計 GPU 時間 | 預計影響 |
|---|------|---------------------|-------------|---------|
| E1 | **Generation task (summarization)** | ✅ Gemini (Paper A + Paper B + JSAC) + Internal | ~2-3 hr A100 | 從 "QA-only" 升級為 "multi-task"；直接回應最常被問的問題 |
| E2 | **Longer context 4K-8K** | ✅ Gemini + Internal | ~2-4 hr A100 | 證明 scout 在真正 long-context 場景仍然有效 |

### ★★ 高優先級

| # | 實驗 | 來源 | 預計 GPU 時間 | 預計影響 |
|---|------|------|-------------|---------|
| E3 | **Instruction-tuned model alignment** | Gemini (Paper A), Internal | ~2 hr A100 | Production 場景更 relevant |
| E4 | **70B+ model** (若資源允許) | Gemini (JSAC limitations) | ~8-12 hr A100 80GB | 直接回應 "our largest model is 14B" limitation |

### ★ 中優先級

| # | 實驗 | 來源 | 預計 GPU 時間 | 預計影響 |
|---|------|------|-------------|---------|
| E5 | Cross-family alignment 改進 | Gemini JSAC 研究方向 | ~2 hr A100 | 73% → ? 可以用 tokenizer alignment fine-tuning |
| E6 | Dynamic retention tuning | Internal (REVIEW_2026-02-13 #6) | ~1 hr A100 | 根據 entropy 自動調整 retention ratio |
| E7 | Security/privacy analysis of scout indices | Gemini JSAC 研究方向 | CPU only | Information-theoretic bound |

---

## 四、「KV-cache transfer 是 strawman」的根本問題

### 狀態分析

這是內部 reviewer_questions.md 提出的最嚴重攻擊。目前 JSAC 論文用兩個段落回應：

1. Section IV-A "Why not simply retransmit the original text?" — 解釋 scout 的價值是 position mask（O(rn) decode attention）
2. Framing 已從 "bandwidth saving" 轉為 "quality improvement + computational saving"

### Gemini 的反應

**Gemini 2.5 Pro 完全沒有提出這個攻擊。** 這可能意味著：
- (a) 修改後的 framing 足夠好，fresh eyes 覺得沒問題
- (b) Gemini 不夠嚴格沒看到

### 建議

保持目前的 framing，但在 Round 2 用 GPT-5.2 review 時特別看是否仍被攻擊。若被攻擊，需要加一個實驗：量化 scout mask 在 decode 階段帶來的實際 latency saving（O(n) vs O(rn) attention）。

---

## 五、迭代計劃 Round 1 實驗清單

基於以上分析，Round 1 應執行：

### Experiment Bundle 1（預計 4-6 小時 A100 80GB）

```
E1: Summarization scout evaluation
    - Model: Qwen2.5-7B → Qwen2.5-14B
    - Task: XSum or CNN/DailyMail
    - Metric: ROUGE-1/2/L
    - n=200 samples
    - 比較: cloud-own Q2C selection vs 7B scout selection vs full KV
    - 同時測 overlap

E2: Long context 4K+
    - Model: Qwen2.5-7B → Qwen2.5-14B
    - Context: 4096 tokens (needle-in-haystack)
    - 繞過 OOM: 不用 output_attentions，改用 manual attention extraction per-layer
    - 或：只在 7B 上測（7B + output_attentions 不會 OOM at 4K）

E3: Instruction-tuned alignment
    - Model: Qwen2.5-7B-Instruct → Qwen2.5-14B-Instruct
    - Task: SQuAD v2 (same samples as base model)
    - 比較 base vs instruct overlap
    - n=200
```

### 預估成本
- A100 80GB on vast.ai: ~$1-2/hr
- 估計 6 小時 = ~$10-12
- 用 A100 SXM4 80GB 跑最快

---

## 六、超越 97 分的路徑

Gemini 給了 97/100。要從 "strong accept" 走向 "best paper"：

1. **補 generation task** → 消除 "QA-only" 攻擊面（+2-3 分）
2. **補 4K+ context** → 消除 "short context only" 攻擊面（+1 分）
3. **加 attention heatmap visualization** → 直觀 wow factor（+1 分）
4. **加 instruction-tuned alignment** → production relevance（+1 分）

目標：所有維度都 ≥97，消除所有已知攻擊面。
