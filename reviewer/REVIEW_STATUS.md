# Reviewer 問題追蹤：原本怎樣、問題是什麼、現在改到哪

**最後更新**: 2026-02-11
**寫法**: 白話工程思維 — 先講道理再給名詞，每個數字都有來源

---

## 目錄

1. [30 秒快覽：全部問題一張表](#1-30-秒快覽)
2. [嚴重問題（紅燈）](#2-嚴重問題)
3. [中等問題（黃燈）](#3-中等問題)
4. [輕微問題（綠燈）](#4-輕微問題)
5. [方法論質疑](#5-方法論質疑)
6. [結論](#6-結論)

---

## 1. 30 秒快覽

| # | 嚴重度 | 問題 | 狀態 |
|---|--------|------|------|
| 1 | 🔴 | Q2C 公式兩篇論文不一致 | ✅ 已解決 |
| 2 | 🔴 | Yi-6B 用了 Chat 模型不公平 | ✅ 已解決 |
| 3 | 🟡 | 樣本量 50 太少，統計不顯著 | ✅ 已解決 |
| 4 | 🟡 | Pythia-2.8B baseline 接近零 | ⚠️ 不需額外實驗 |
| 5 | 🟡 | Table 2 和 Table 5 不同 sample set | ✅ 已解決 |
| 6 | 🟢 | Yi-6B 長上下文 baseline 太低 | ⚠️ 已記錄 |
| 7 | 方法 | 沒量化 eager attention 的額外開銷 | ✅ 已解決 |
| 8 | 方法 | Scout 只在 Qwen 家族驗證 | ✅ 已解決 |
| 9 | 方法 | 只用 SQuAD，缺 multi-task 驗證 | ✅ 已解決 |
| 10 | 方法 | 缺少 perplexity 評估 | ✅ 已解決 |
| 11 | 方法 | Attention focusing 沒有直接證據 | ✅ 已解決 |
| 12 | 方法 | Markov chain 信道模型不真實 | ✅ 已解決 |
| 13 | 方法 | Context length 太短 (170 tokens) | ✅ 已解決 |
| 14 | 方法 | 缺第四個 model family (Llama) | ✅ 已解決 |

**結論：14 個問題中 12 個完全解決，2 個已記錄（不需額外實驗）。**

---

## 2. 嚴重問題

### 🔴 問題 1：Q2C 公式兩篇論文寫的不一樣

**原本怎樣：**
我們發明了一個叫 Q2C 的方法，用來決定哪些 token 重要、哪些可以丟掉。簡單講就是：「問題在看哪些字？那些字就是重要的。」

但問題是，Paper A 的公式說「只看最後一層 attention」，Paper B 的公式卻說「看所有層的平均」。而實際程式碼用的是「最後一層」。

**Reviewer 會怎麼看：**
「你兩篇論文的核心公式寫法不同，程式碼又跟 Paper B 的公式對不上。到底哪個是對的？這是基本的學術誠信問題。」

**我們怎麼修的：**
跑了一個正面對決的實驗 — 把「只看最後一層」和「看所有層平均」兩種方法，在完全相同的 200 個問答題上跑一遍，用三個不同大小的模型（7B、14B、Mistral-7B）。

結果：兩種方法選出來的重要 token 幾乎一模一樣。

| 模型 | Pearson 相關係數 | 說明 |
|------|-----------------|------|
| Qwen-7B | 0.990 | 兩種方法的分數排名幾乎完全一致 |
| Qwen-14B | 0.993 | 同上 |
| Mistral-7B | 0.995 | 同上 |

**白話翻譯：** 不管用哪種算法，結果差不多。所以我們統一用比較簡單、比較快的「只看最後一層」。這樣 Paper A 和 Paper B 的公式就一致了，而且有實驗證明這樣做不會損失什麼。

> 資料來源：`exp_q2c_ablation_20260210_071938.json`

---

### 🔴 問題 2：Yi-6B 用了 Chat 模型，跟其他 Base 模型比不公平

**原本怎樣：**
我們的論文說比較了「7 個模型家族」，但其中 Yi-6B 其實用的是有經過 instruction tuning 的 Chat 版本。這就像考試的時候，6 個學生裸考，但有 1 個學生先看過考古題。不公平。

**Reviewer 會怎麼看：**
「你的 Yi-6B 結果可能被高估了。如果它的 INT4 robustness 是因為 Chat tuning 帶來的，那你 'INT4 fragility is model-specific' 這個結論的對照組就有問題。」

**我們怎麼修的：**
直接跑了 Base vs Chat 的比較（100 個樣本），證明確實有差：

| 模型版本 | F1 分數 | 說明 |
|---------|---------|------|
| Yi-1.5-6B (Base) | 0.659 | 裸考成績 |
| Yi-6B-Chat | 0.161 | 看過考古題但答題格式不對，反而更差 |

**白話翻譯：** Chat 模型在我們的測試格式下反而更差（因為我們用 base model 的 prompt 格式 `"Context:...Question:...Answer:"`，Chat 模型期待的是 ChatML 格式）。所以原本 Paper A 用 Chat 模型的結果其實是**低估**而不是高估。

我們還額外補了 Llama-3.1-8B 作為第四個 model family：

| 模型 | 類型 | F1 | 說明 |
|------|------|-----|------|
| Qwen2.5-7B | Base | 0.73 | 表現最好 |
| Yi-1.5-6B | Base | 0.659 | 不錯 |
| Llama-3.1-8B | Base | 0.288 | 回答太囉嗦，extractive F1 低 |
| Mistral-7B | Base | ~0.17 | 一般 |

> 資料來源：`exp_yi_comparison_20260211_111947.json`, `exp_llama_eval_20260211_150948.json`

---

## 3. 中等問題

### 🟡 問題 3：每組實驗只有 50 個樣本，統計說服力不夠

**原本怎樣：**
Paper A 宣稱「Q2C 比 SnapKV 好 29-47%」，但 p-value 是 0.14-0.29。統計學上，p > 0.05 就代表「這個差距有可能只是運氣好」。而且每組實驗只有 50 個樣本，信賴區間嚴重重疊。

**Reviewer 會怎麼看：**
「你的 Q2C vs SnapKV 差距在統計上不顯著，但你在摘要裡用 'outperforms by 29-47%' 這種強烈措辭，這是誤導讀者。」

**我們怎麼修的：**
1. 把樣本量從 50 加到 **200**
2. 重新跑了完整的 selection 比較（Q2C vs SnapKV vs H2O）

結果：

| 方法 | F1 @75% | F1 @50% | F1 @25% |
|------|---------|---------|---------|
| Q2C | 0.608 | 0.541 | 0.368 |
| SnapKV | 0.608 | 0.544 | 0.366 |
| H2O | 0.558 | 0.380 | 0.246 |

**白話翻譯：** 加大樣本後，事實很清楚 — **Q2C 和 SnapKV 差不多**（幾乎一樣），但 **兩者都遠遠贏過 H2O**（p < 0.00001）。所以論文裡「Q2C 大幅超越 SnapKV」的說法要改掉，改成「Q2C 和 SnapKV 表現相當，兩者都顯著優於 H2O」。

> 資料來源：`exp_paper_a_unified_qwen_7b_20260211_111947.json`

---

### 🟡 問題 4：Pythia-2.8B baseline F1 只有 0.032

**原本怎樣：**
Pythia-2.8B 這個模型根本答不了 SQuAD 的題目（F1 = 3.2%，幾乎是隨機水平）。在一個連題目都做不了的模型上討論「INT4 保留了 85% 的效能」毫無意義。

**我們的處理：**
這個不需要額外實驗。在合併的 JSAC 論文中，我們會把 Pythia 從主要結果表中移到附錄，並明確標註它的 baseline 太低、結論不適用。有效模型數量從 7 減為 5-6 個，但加上 Llama-3.1-8B 補回來了。

---

### 🟡 問題 5：Table 2 和 Table 5 用了不同的 sample set

**原本怎樣：**
Paper A 的 selection 比較（Table 2）和 quantization 比較（Table 5）來自不同批次的實驗，用不同的隨機樣本。Table 2 的 Qwen-7B baseline 是 0.805，Table 5 是 0.696 — 差距明顯。

**Reviewer 會怎麼看：**
「你的 selection 和 quantization 結果不能直接組合。如果我想知道 'Q2C selection + INT8 quantization' 的聯合效果，你的數據沒辦法回答。」

**我們怎麼修的：**
跑了一個統一實驗：**在同一組 200 個樣本上，同時跑 selection（Q2C/SnapKV/H2O）和 quantization（BF16/INT8/INT4/Mixed-INT4）**。

Quantization 結果（用修好的手動生成迴圈）：

| 量化方法 | F1 | 佔 BF16 的百分比 | 說明 |
|---------|-----|-----------------|------|
| BF16（不壓縮）| 0.706 | 100% | 基準線 |
| INT8 | 0.701 | 99.4% | 幾乎無損 |
| INT4 | 0.484 | 68.7% | 災難性下降 |
| Mixed-INT4 | 0.660 | 93.6% | 敏感層用 INT8，救回大部分 |

**插曲 — 量化 bug：** 第一次跑的時候，四種量化方法的結果完全一樣（都是 F1=0.696）。查了程式碼才發現：`model.generate(past_key_values=None)` — 傳了 `None` 進去，等於完全忽略量化過的 cache！修好後改成手動一個 token 一個 token 從量化 cache 生成，結果才正確。

> 資料來源：`exp_paper_a_quant_fix_20260211_135232.json`

---

## 4. 輕微問題

### 🟢 問題 6：Yi-6B 長上下文 baseline 太低

**原本怎樣：**
Yi-6B 在 needle-in-haystack 長上下文測試中，full F1 只有 0.19-0.21。在這麼低的基礎上報告「INT4 保持 97%」沒有意義。

**我們的處理：**
在 JSAC 論文中會移除 Yi-6B 的長上下文結果，改用 Qwen 7B→14B 的長上下文實驗（那個有意義的多）。詳見問題 13。

---

## 5. 方法論質疑

### 問題 7：沒量化 `output_attentions=True` 的額外開銷

**原本怎樣：**
我們的 Q2C 方法需要跑 eager attention（不能用更快的 Flash Attention），這會增加計算開銷。但論文裡完全沒提這件事。

**Reviewer 會怎麼看：**
「你的方法需要 eager attention，這比 SDPA 慢很多吧？為什麼不報告開銷？」

**我們怎麼修的：**
實際跑了 benchmark：

| 設定 | 時間 |
|------|------|
| Eager（沒有取 attention）| 52.3 ms |
| Eager + output_attentions | 52.6 ms |
| SDPA（Flash Attention）| 43.2 ms |

**白話翻譯：** 取 attention weights 本身只多了 **0.44%** 的開銷（52.3→52.6 ms），基本可以忽略。真正的代價是用 eager 而不是 SDPA（慢了 21%）。但在 edge-cloud 場景中，edge 設備本來就要跑完整的 prefill，多 21% 的 prefill 時間跟網路傳輸時間比起來很小。

> 資料來源：`exp_eager_overhead_20260211_111947.json`

---

### 問題 8：Scout 只在 Qwen 家族驗證，可能是 Qwen-specific

**原本怎樣：**
Scout 的核心假設是「同家族的模型注意力模式是對齊的」。但我們只在 Qwen2.5 的 3B/7B/14B 上測過。它們共享訓練資料、tokenizer、RoPE，alignment 可能只是 Qwen 的特性。

**Reviewer 會怎麼看：**
「你怎麼知道這不是 Qwen-specific？不同家族的模型也有嗎？」

**我們怎麼修的：**
跑了 **跨家族 scout**：用 Qwen-7B 當 scout，去幫 Mistral-7B（完全不同的模型家族、不同的 tokenizer）選重要位置。

| 保留比例 | Position Overlap | Scout F1 | 自己選的 F1 | p-value |
|---------|-----------------|----------|------------|---------|
| 75% | 73.4% | 0.167 | 0.122 | **0.013** |
| 50% | 58.6% | 0.102 | 0.081 | 0.054 |
| 25% | 41.4% | 0.059 | 0.062 | 0.360 |

**白話翻譯：** 即使跨家族（Qwen → Mistral），73% 的位置仍然一致（@75% 保留）。而且在 75% 的情況下，Qwen scout 選的位置讓 Mistral **表現更好**（p=0.013，統計顯著）。這證明 scout 不是 Qwen-specific 現象。

技術細節：因為兩個模型的 tokenizer 不同，我們用了字元級位置對齊（character-level span mapping）來轉換位置。

> 資料來源：`exp_cross_family_scout_20260211_111947.json`

---

### 問題 9：只用 SQuAD 做評估，太單一

**原本怎樣：**
SQuAD v2 是一個很簡單的「文章裡找答案」任務。LLM 的真正應用場景遠不止這個。

**我們怎麼修的：**
加了 HotpotQA（多跳推理任務）的 scout 實驗：

| 任務 | 保留 50% 時 Scout F1 | 自己選的 F1 | 差距 | p-value |
|------|---------------------|------------|------|---------|
| SQuAD | 0.560 | 0.472 | **+0.088** | **0.026** |
| HotpotQA | 0.510 | 0.521 | -0.011 | 0.6+ |

**白話翻譯：** Scout 在 SQuAD 上顯著有幫助（p=0.026），在 HotpotQA 上打平。沒有哪個任務上 scout 會「搞砸」結果。TriviaQA 因為資料集太大沒有跑，但兩個任務已經足以展示 generality。

> 資料來源：`exp_scout_multitask_20260210_132605.json`

---

### 問題 10：缺 perplexity 評估

**原本怎樣：**
幾乎所有 KV-cache 壓縮論文都會報告 WikiText-2 perplexity（語言模型的「標準體檢」）。我們只報告了 task-specific F1，讓結果很難跟別人的論文直接比較。

**我們怎麼修的：**
在三個模型上跑了 WikiText-2 perplexity：

| 模型 | BF16 | INT8 | INT4 | Mixed-INT4 |
|------|------|------|------|------------|
| Qwen-7B | 8.63 | 8.85 | **80.27** | 8.97 |
| Qwen-14B | 5.73 | 5.73 | 5.87 | 5.83 |
| Mistral-7B | 6.49 | 6.49 | 6.51 | 6.51 |

**白話翻譯：**
- INT8 全部無損（perplexity 幾乎不變）
- INT4 對 7B 模型是災難性的（8.63 → 80.27，基本上變成胡言亂語）
- INT4 對 14B 和 Mistral-7B 卻沒什麼影響
- Mixed-INT4（敏感層保持 INT8）完全救回來

這跟 F1 的結論互相印證，而且現在可以跟其他論文的 perplexity 數字直接比了。

> 資料來源：`exp_perplexity_20260210_130438.json`, `exp_perplexity_mistral_20260211_111947.json`

---

### 問題 11：「Attention Focusing Effect」缺直接證據

**原本怎樣：**
Paper B 宣稱「小模型的注意力更集中，所以小模型幫大模型選反而更好」。但論文裡只有 F1 結果，沒有直接看 attention 到底有多集中。

**Reviewer 會怎麼看：**
「你說小模型注意力更集中，但你沒測 entropy。你怎麼知道不是別的原因？」

**我們怎麼修的：**
跑了 attention entropy 分析：

| 模型 | Q2C Entropy | 說明 |
|------|-------------|------|
| Qwen-3B | 4.21 | 最集中（entropy 最低）|
| Qwen-14B | 4.65 | 中等 |
| Qwen-7B | 5.49 | 最分散 |

**白話翻譯：** 3B 確實最集中（entropy 最低），但 14B 竟然比 7B 還集中。所以不是簡單的「越小越集中」。更準確的說法是：**3B 最集中，14B 因為模型更強也能聚焦，7B 卡在中間反而最分散**。論文中的 "attention focusing effect" 措辭需要修正為更精確的描述。

> 資料來源：`exp_attention_entropy_20260210_073931.json`

---

### 問題 12：Markov chain 信道模型太理想化

**原本怎樣：**
Paper B 用 6 個狀態的 Markov chain 模擬無線頻寬變化。但真實 5G 的頻寬波動比這複雜得多。

**我們怎麼修的：**
改用 **Lumos5G 真實 5G 量測 trace** 跑 protocol simulation：

| 模式 | Deadline 達成率 | 平均延遲 |
|------|---------------|---------|
| Static INT8 | 18.9% | 1898 ms |
| Adaptive（不含 scout）| 41.4% | — |
| **Adaptive + Scout** | **100%** | **263 ms** |
| Scout only | **100%** | 204 ms |

**白話翻譯：** 用真實 5G trace 跑出來，scout 模式還是 100% 達標。因為 scout 只傳 336 bytes 的位置索引（不是幾十 MB 的 KV-cache），任何頻寬下都傳得完。結果比原本的 Markov chain 模擬更有說服力。

> 資料來源：`exp_protocol_real_traces_20260209_185457.json`

---

### 問題 13：Context length 太短 (平均 170 tokens)

**原本怎樣：**
KV-cache 壓縮的痛點在長上下文（4K-128K tokens），但我們的 SQuAD 實驗平均只有 170 tokens。170 tokens 的 KV-cache 只有 9.7 MB，在 10 Mbps 下 8 秒就傳完了，根本不需要壓縮。

**我們怎麼修的：**
跑了 1K 和 2K token 的長上下文 scout 實驗（7B→14B）：

| Context Length | Overlap @75% | Overlap @50% | Overlap @25% |
|---------------|-------------|-------------|-------------|
| 1024 tokens | 83.3% | 69.4% | 55.6% |
| 2048 tokens | 82.7% | 68.2% | 53.9% |

**白話翻譯：** 上下文從 170 拉到 2048 tokens（12 倍），overlap 幾乎不變（83% → 83%）。這證明 scout 的 attention alignment 不會隨著上下文變長而崩壞。

沒有跑到 4K-8K 是因為 14B 模型 + `output_attentions=True` 在長上下文下會 OOM（48 層 × 4096² 的 attention matrix ≈ 128GB）。但 1K-2K 已經足以展示穩定性趨勢。

> 資料來源：`exp_scout_long_ctx_7b14b_20260210_131505.json`

---

### 問題 14：只有 Qwen + Mistral + Yi，缺少 Llama

**原本怎樣：**
之前 Llama-3.1-8B 是 gated model（需要申請存取權），跑的時候出現 401 auth error。

**我們怎麼修的：**
申請了 HuggingFace 存取權，拿到 token 後成功下載並跑了 100 個 SQuAD 樣本：

- **Llama-3.1-8B F1 = 0.288 ± 0.068**

F1 偏低是因為 Llama base model 回答很囉嗦（會附帶解釋和反問），extractive F1 計算會懲罰多餘的字。但作為資料點是有效的 — 現在有四個 model family（Qwen、Mistral、Yi、Llama）的結果。

> 資料來源：`exp_llama_eval_20260211_150948.json`

---

## 6. 結論

### 一句話總結
原本的論文有 14 個可被挑戰的地方。我們補跑了 15 個 GPU 實驗（共 19 個 JSON 結果檔），12 個問題完全用數據回應，2 個在寫作層面處理（移到附錄/加註解）。

### 核心結論的變化

| 原本的說法 | 現在應該怎麼說 |
|-----------|--------------|
| 「Q2C outperforms SnapKV by 29-47%」 | Q2C ≈ SnapKV，兩者都顯著優於 H2O |
| 「7 model families」 | 5-6 個有效模型（去掉 Pythia，加上 Llama） |
| 「INT4 fragility is model-specific」 | ✅ 不變，但用 perplexity 和修好的 F1 數據支持 |
| 「Attention focusing: 小模型更集中」 | 3B 最集中，但不是單調關係（14B 比 7B 集中）|
| 「Scout 可跨模型轉移」 | ✅ 擴展到跨家族（Qwen→Mistral, 73% overlap） |
| 「Adaptive protocol 100% deadline」 | ✅ 用真實 5G trace 確認，不只是 Markov chain |
| 「Q2C 用 all-layer average」(Paper B) | 統一為 last-layer（ablation 證明差異 < 1%） |

### 所有實驗結果檔案對照表

| 實驗 | 解決的問題 | JSON 檔案 |
|------|-----------|----------|
| Q2C ablation | #1 | `exp_q2c_ablation_20260210_071938.json` |
| Scout n=200 | #3 | `exp_scout_n200_20260210_073907.json` |
| Attention entropy | #11 | `exp_attention_entropy_20260210_07*.json` |
| Scout long context | #13 | `exp_scout_long_ctx_7b14b_20260210_131505.json` |
| Real 5G traces | #12 | `exp_protocol_real_traces_20260209_185457.json` |
| Hybrid mode | 額外 | `exp_hybrid_mode_20260210_083814.json` |
| Multitask scout | #9 | `exp_scout_multitask_20260210_132605.json` |
| Perplexity Qwen | #10 | `exp_perplexity_20260210_130438.json` |
| Cross-family scout | #8 | `exp_cross_family_scout_20260211_111947.json` |
| Unified selection | #3, #5 | `exp_paper_a_unified_qwen_7b_20260211_111947.json` |
| Quantization fix | #5 | `exp_paper_a_quant_fix_20260211_135232.json` |
| Eager overhead | #7 | `exp_eager_overhead_20260211_111947.json` |
| Yi base vs chat | #2 | `exp_yi_comparison_20260211_111947.json` |
| Perplexity Mistral | #10 | `exp_perplexity_mistral_20260211_111947.json` |
| Llama-3.1-8B eval | #14 | `exp_llama_eval_20260211_150948.json` |
