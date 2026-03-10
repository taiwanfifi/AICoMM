# 學術審查報告：KV-Cache 語意通訊論文雙篇深度審查

**審查日期**: 2026-02-10
**審查對象**: Paper A — *Task-Aware KV-Cache Compression for Bandwidth-Efficient Collaborative LLM Inference*
**審查對象**: Paper B — *Scout: Bandwidth-Adaptive KV-Cache Transport for Heterogeneous Edge-Cloud LLM Inference*
**作者**: Wei-Lun Cheng, Wanjiun Liao (NTU EE)
**目標刊物**: Paper A → INFOCOM 2027; Paper B → ICC or JSAC

---

## 目錄

1. [第一輪：數據驗證與根本性錯誤檢查](#1-數據驗證與根本性錯誤檢查)
2. [第二輪：方法論深度質疑](#2-方法論深度質疑)
3. [第三輪：同行文獻比對 (40+ 篇)](#3-同行文獻比對)
4. [第四輪：高層次審視](#4-高層次審視)
5. [第五輪：細節層次審視](#5-細節層次審視)
6. [第六輪：綜合評分與發表可能性](#6-綜合評分與發表可能性)
7. [第七輪：潛在新研究方向](#7-潛在新研究方向)

---

## 1. 數據驗證與根本性錯誤檢查

### 1.1 原始 JSON 數據 vs 論文數字交叉驗證

我逐一比對了 23 個 JSON 結果檔與論文中的表格數字：

| 論文聲稱 | JSON 原始值 | 匹配？ |
|---------|-----------|--------|
| Paper A Table 2: Q2C@75% Qwen-14B = 0.737 | `q2c_75_f1 = 0.737` | ✅ |
| Paper A Table 2: SnapKV@75% Qwen-14B = 0.662 | `snapkv_75_f1 = 0.662` | ✅ |
| Paper A Table 2: H2O@75% Qwen-14B = 0.529 | `h2o_75_f1 = 0.529` | ✅ |
| Paper A Table 2: Q2C@25% Mistral = 0.294 | `q2c_25_f1 = 0.294` | ✅ |
| Paper A Table 5: INT8 Qwen-7B = 99.6% | `f1_pct = 99.62%` | ✅ |
| Paper A Table 5: Mixed INT4 Qwen-7B = 107% | `f1_pct = 107.0%` | ✅ |
| Paper B Table 2: 7B→14B 75% overlap = 83.4% | `overlap_pct = 83.37%` | ✅ |
| Paper B Table 2: 7B→14B scout F1 = 0.714 | `scout_f1 = 0.714` | ✅ |
| Paper B Table 2: 3B→14B gap at 50% = -0.028 | `scout_vs_own_gap = -0.028` | ✅ |

**結論：論文數字與原始實驗數據完全一致，無捏造或計算錯誤。**

### 1.2 發現的根本性問題

#### 🔴 嚴重問題 1：Q2C 定義在兩篇論文中不一致

- **Paper A (Eq. 4)**: Q2C 只使用**最後一層**的 attention：`s_j = Σ_h Σ_i A^(L,h)_{i,j}`
- **Paper B (Eq. 2)**: Q2C 使用**所有層**的平均 attention：`s_j = (1/LH) Σ_ℓ Σ_h (1/|Q|) Σ_i A_{ℓ,h}[i,j]`
- **實際代碼** (`run_batch28_scout_model.py:112`): 使用 `out.attentions[-1]`，即**最後一層**

**影響**: Paper B 的公式與實際實驗不符。這意味著 Paper B Table 2 的所有 scout 結果是用 last-layer-only Q2C 得到的，但論文中聲稱的是 all-layer-averaged Q2C。如果審稿人發現這一點，Paper B 將可能被拒。

**修復建議**: 將 Paper B 的 Eq. 2 改為與代碼一致（last layer only），或重新跑一組 all-layer-averaged 的實驗來驗證差異。

#### 🔴 嚴重問題 2：Yi-6B 使用了 Chat 模型而非 Base 模型

Paper A 宣稱比較 7 個「model families」，但實驗數據顯示 Yi-6B 使用的是 `Yi-1.5-6B-Chat`（ChatML 格式），而其他模型（Qwen, Mistral）似乎使用的是 base 模型。這構成不公平比較：

- Chat 模型已經過 instruction tuning，天然對 extractive QA 更友好
- Yi-6B 的 INT4 robustness（100%）可能因此被高估
- 論文中的核心結論 "INT4 fragility is model-specific, not architecture-determined" 的對照組有問題

#### 🟡 中等問題 3：樣本量過小導致統計顯著性不足

- Paper A 明確承認 Q2C vs SnapKV 的差異不顯著（p=0.14-0.29），但仍在 abstract 和 conclusion 中使用 "outperforms by 29-47%" 的強烈措辭
- 每個配置只有 50 個樣本（delta encoding 實驗只有 30 個），考慮到 F1 的高方差（std 0.3-0.4），很多比較的 confidence interval 嚴重重疊
- Paper B 的 scout 實驗同樣只有 50 個樣本

#### 🟡 中等問題 4：Pythia-2.8B 的 baseline F1 接近零

Pythia-2.8B 的 baseline F1 只有 0.032（3.2%），在此基礎上討論 INT4 的 85% 或 103% 毫無意義。論文雖然加了 daggar 註記，但仍然把它列入 Table 3 並作為「7 model families」計數的一部分。這實質上是用一個無法完成任務的模型來填充實驗數量。

#### 🟡 中等問題 5：Paper A 的 Table 2 和 Table 5 使用了不同的 sample set

Paper A 的 selection 比較（Table 2）和 quantization 比較（Table 3）來自不同的實驗批次（不同的 JSON 檔），使用不同的隨機 sample set。雖然論文聲明 "within each table, all methods are compared on the same sample set"，但 Table 2 的 Qwen-7B baseline（0.805）和 Table 5 的 Qwen-7B baseline（0.696）明顯不同。

這意味著 **Q2C selection 和 quantization 的結果不能直接組合**來推導聯合壓縮效果，除非重新在同一 sample set 上跑全管線。

#### 🟢 輕微問題 6：Yi-6B 的 context length scaling 基線極低

Yi-6B 在 needle-in-haystack 實驗中的 full F1 只有 0.19-0.21，說明這個模型根本無法有效解決長序列 needle-in-haystack 任務。在此基礎上報告 INT4 保持 97.7-100% 是無意義的——兩者都接近隨機水平。

---

## 2. 方法論深度質疑

### 2.1 Q2C 方法的根本性局限

**問題**: Q2C 依賴於完整的 prefill forward pass 產生的 attention weights，這意味著：
1. 它**不能用於 streaming 場景**——必須等到所有 context + query 都被處理完
2. 它**增加了邊端的計算需求**——edge 必須跑完整的 prefill（包括 attention weight 提取），這在 eager mode 下比普通推理更慢
3. 論文沒有量化 `output_attentions=True` 帶來的額外開銷（需要 eager attention，不能用 Flash Attention / SDPA）

**Quest (ICML 2024) 的對比**: Quest 使用 query-aware page-level sparsity，也是 query-aware 的 selection，但它的操作在 decode 階段而非 prefill 階段，不需要完整 attention matrix，並且在 128K 上下文上實現了 2.23x speedup。Paper A 完全忽略了與 Quest 在 query-awareness 上的深度比較。

### 2.2 Scout 模型的核心假設可質疑

**假設**: 同 family 的 models 有 aligned attention patterns。

**問題**:
1. **只在 Qwen2.5 家族上驗證** — 沒有跨家族驗證。Qwen2.5-3B/7B/14B 共享相同的訓練數據分佈、tokenizer、RoPE base frequency，這使得 attention alignment 是 Qwen-specific 現象而非通用結論
2. **只在 SQuAD v2 上驗證** — 一個相對簡單的 extractive QA 任務。在 multi-hop reasoning、summarization、code generation 等任務上，attention patterns 可能完全不同
3. **"Attention focusing effect" 的解釋過於 ad hoc** — 聲稱 7B 的 selection 比 14B 自己的 selection 好是因為 "smaller model concentrates attention"，但沒有提供 attention entropy 分析或其他直接證據

### 2.3 Adaptive Protocol 的 simulation 不夠真實

**Markov chain bandwidth model 的問題**:
1. 6 state Markov chain（5/10/25/50/100/200 Mbps）是一個極度簡化的信道模型，與真實 5G/LTE 的快衰落、slow fading、MIMO scheduling 完全不同
2. 沒有考慮 **RTT、packet loss、jitter** — 實際上 KV-cache 傳輸需要可靠傳輸（TCP 或 QUIC），retransmission 會顯著增加延遲
3. **Scout mode 的 "100% deadline compliance" 是人為的** — 因為 scout payload 是 336 bytes，任何帶寬下都能在 deadline 內傳完。但實際上 cloud 必須重新跑 prefill（14B 需要 57ms），這在論文中被當成 "negligible"，但在 1s deadline 下佔了 5.7%

### 2.4 Delta Encoding 反駁的公平性問題

Paper A 聲稱反駁了 CacheGen 的 delta encoding，但：
1. **CacheGen 使用 arithmetic coding**，不是簡單的 fixed-point quantization。論文只實現了 delta + quantization，沒有 entropy coding
2. **CacheGen 使用 layer-wise graded bit allocation**，Paper A 使用 uniform bit allocation
3. 因此 "delta encoding is strictly inferior" 的結論可能只適用於作者自己的簡化實現，不適用於 CacheGen 的完整系統

---

## 3. 同行文獻比對

我搜索了 40+ 篇同行論文（2023-2026），分為以下類別進行比對：

### 3.1 Token Selection / Eviction 方法（直接競爭者）

| 論文 | 年份/刊物 | 核心差異 | 對 Paper A 的影響 |
|-----|---------|---------|-----------------|
| H2O (NeurIPS 2023) | 2023 | Cumulative attention eviction | 已作為 baseline ✅ |
| SnapKV (NeurIPS 2024) | 2024 | Observation window selection | 已作為 baseline ✅ |
| **Quest (ICML 2024)** | 2024 | **Query-aware** page-level sparsity, 128K context, 2.23x speedup | ⚠️ 最大威脅：同為 query-aware，但 Quest 在更大規模上驗證 |
| Scissorhands (NeurIPS 2023) | 2023 | Persistence of importance | 已引用但未實驗比較 |
| FastGen (ICLR 2024) | 2024 | Per-head adaptive policies | 已引用但未實驗比較 |
| Keyformer (MLSys 2024) | 2024 | Key token selection with discarded-token-aware scoring | 未引用 ❌ |
| **PyramidInfer (ACL 2024)** | 2024 | Layer-wise decreasing budget | 未引用 ❌ — 與 mixed-precision 的 layer-wise 思路類似 |
| **PyramidKV (TMLR 2025)** | 2024 | Pyramidal information funneling | 未引用 ❌ — 直接相關的 layer-wise 分析 |
| CAOTE (arXiv 2025) | 2025 | Attention output error-based eviction | 較新，可理解 |

### 3.2 KV-Cache Quantization（直接競爭者）

| 論文 | 年份/刊物 | 核心差異 | 對 Paper A 的影響 |
|-----|---------|---------|-----------------|
| KIVI (ICML 2024) | 2024 | Asymmetric 2-bit, per-channel key / per-token value | 已引用 ✅ |
| KVQuant (NeurIPS 2024) | 2024 | Non-uniform quantization, pre-RoPE | 已引用 ✅ |
| **ZipCache (NeurIPS 2024)** | 2024 | Salient-token-aware quantization, 4.98x compression | ⚠️ 未引用，直接相關 |
| **KVTuner (ICML 2025)** | 2025 | **Layer-wise mixed-precision** with sensitivity search | 🔴 最大威脅：已被 ICML 2025 接收，幾乎完全相同的 contribution — layer-wise mixed-precision quantization，且在 Qwen2.5-7B 上實現 4.0-bit |
| GEAR (NeurIPS 2024) | 2024 | Quantization + low-rank + sparse correction | 已引用 ✅ |
| QAQ (arXiv 2024) | 2024 | Attention-score-based bit allocation | 已引用 ✅ |
| ATOM (MLSys 2024) | 2024 | Mixed-precision serving | 未引用 ❌ |

### 3.3 低秩壓縮 & 架構級方法

| 論文 | 年份/刊物 | 核心差異 | 影響 |
|-----|---------|---------|-----|
| PALU (ICLR 2025) | 2025 | Low-rank KV-cache projection, 91.25% compression | 已引用 ✅ |
| MiniCache (NeurIPS 2024) | 2024 | Cross-layer KV merging in depth dimension | 未引用 ❌ |
| DMC (ICML 2024) | 2024 | Learned online compression ratios | 未引用 ❌ |
| X-EcoMLA (arXiv 2025) | 2025 | Upcycling attention into MLA | 較新 |
| DeepSeek-V2 MLA | 2024 | Architecture-level low-rank | 已引用 ✅ |

### 3.4 Edge-Cloud Collaborative Inference（Paper B 的競爭者）

| 論文 | 年份/刊物 | 核心差異 | 對 Paper B 的影響 |
|-----|---------|---------|-----------------|
| CacheGen (SIGCOMM 2024) | 2024 | KV-cache streaming with adaptive compression | 已引用 ✅ |
| Splitwise (ISCA 2024) | 2024 | Phase splitting for LLM serving | 已引用 ✅ |
| DistServe (OSDI 2024) | 2024 | Disaggregated prefill/decode | 已引用 ✅ |
| Mooncake (FAST 2025) | 2024 | KVCache-centric disaggregated architecture | 已引用 ✅ |
| **EdgeShard (IEEE IoT-J 2024)** | 2024 | Edge-cloud LLM sharding | 未引用 ❌ |
| **Adaptive Layer Splitting (FITEE 2024)** | 2024 | RL-based wireless LLM split | 未引用 ❌ — 直接相關場景 |
| **Hybrid SLM-LLM (MobiSys Wkshp 2024)** | 2024 | Small-large model collaboration at edge | 未引用 ❌ — 概念最接近 |
| LMCache (arXiv 2025) | 2025 | Enterprise KV cache management | 較新 |
| EAGLE / Medusa (ICML 2024) | 2024 | Speculative decoding | 已引用 speculative decoding 概念 ✅ |

### 3.5 Semantic Communication（論文定位）

| 論文 | 年份/刊物 | 影響 |
|-----|---------|-----|
| LLM-SemCom (IEEE, 2025) | 2025 | LLM-based semantic communication framework |
| Rethinking KV Cache Compression (MLSys 2025) | 2025 | 系統性重新評估 KV cache 壓縮 |

### 3.6 文獻比對總結

**嚴重遺漏**:
- **KVTuner**: 幾乎與 Paper A 的 mixed-precision contribution 完全重疊 — 都是 layer-wise sensitivity analysis + mixed-precision quantization。如果 KVTuner 先發表，Paper A 的 contribution 2 (diagnostic mixed-precision) 的新穎性大打折扣
- **PyramidKV / PyramidInfer**: layer-wise budget 的概念與 Paper A 的 bottleneck layer 分析直接相關
- **Hybrid SLM-LLM collaboration**: 概念上與 Paper B 的 scout model 非常接近

**已覆蓋的主要 baselines**: H2O ✅, SnapKV ✅, KIVI ✅, KVQuant ✅, CacheGen ✅, Quest (部分) ✅

---

## 4. 高層次審視

### 4.1 整體 Contribution 的原創性評估

**Paper A:**
- Contribution 1 (Q2C selection): **中等原創性**。Query-aware selection 的想法並非首創（Quest ICML 2024 已做過），但 Q2C 的具體實現（last-layer query-to-context attention）確實是不同的。問題是 Quest 已證明 query-aware 的有效性，Q2C 只是用不同 granularity 做了類似的事
- Contribution 2 (Mixed-precision): **低原創性**。KVTuner (Feb 2025) 已經做了幾乎相同的事——layer-wise sensitivity analysis + mixed-precision。Paper A 的 "bottleneck layer discovery" 雖然直觀好懂，但技術深度不如 KVTuner
- Contribution 3 (Cross-architecture characterization): **中等原創性**。7 個模型的系統評估確實有價值，但其中 Pythia 基本無用，Phi-3.5 與 transformers 5.x 不兼容（來自 MEMORY.md），Yi-6B 用了 Chat 版本。有效模型數量約 4-5 個
- Contribution 4 (Delta encoding counter-finding): **高原創性**。直接反駁 SIGCOMM paper 的核心技術，且用 entropy analysis 解釋了原因。但公平性存疑（見 2.4）
- Contribution 5 (Latency analysis): **低原創性**。只是簡單的 size / bandwidth 計算，不涉及真實網路實驗

**Paper B:**
- Contribution 1 (Scout protocol): **高原創性**。Cross-model attention alignment 用於消除 KV-cache 傳輸是全新的概念。336 bytes vs 33 MB 的壓縮比極其驚人。但只在 Qwen2.5 上驗證
- Contribution 2 (Adaptive policy): **中等原創性**。5-mode lookup table 的設計相當簡單，但 practical 價值高
- Contribution 3 (Multi-agent allocation): **低-中原創性**。Model-aware proportional allocation 是直觀的想法，quality-maximizing greedy 也不新穎
- Contribution 4 (End-to-end evaluation): **高價值**。GPU 實驗 + Markov chain simulation 的組合提供了合理的驗證

### 4.2 兩篇論文的關係問題

Paper B 大量依賴 Paper A 的結果（cite [paperA]），並且使用 Paper A 的 empirical quality-bandwidth data 作為 simulation 輸入。如果 Paper A 未被接受：
1. Paper B 的所有 operating point quality 數字失去引用基礎
2. Paper B 的 adaptive protocol simulation 變成基於未發表數據的模擬

**建議**: 考慮將兩篇合併為一篇 journal paper（如 JSAC），這樣 (a) 所有數據自洽，(b) 跨模型實驗 + 協議設計的組合更有份量投 journal。

### 4.3 與頂會/頂刊水準的差距

**INFOCOM 要求**:
- 強調 networking contribution，Paper A 的核心是 ML 實驗（quantization + selection），networking 部分只有簡單的 latency = size / bandwidth 計算
- 需要更真實的網路評估（real traces, ns-3 simulation, 或 testbed）
- 與 CacheGen (SIGCOMM 2024) 相比，Paper A 缺少系統實現和 end-to-end deployment

**ICC 要求**:
- 比 INFOCOM 門檻稍低，Paper B 的 scout + adaptive protocol 的組合可能足夠
- 但需要加強無線信道模型的真實性

**JSAC 要求**:
- 需要更深入的理論分析和更全面的實驗
- 兩篇合併可能達到門檻

---

## 5. 細節層次審視

### 5.1 寫作品質
- 英文品質良好，句法清晰
- IEEE 格式遵循正確
- 表格和圖表呈現專業
- 算法描述清晰

### 5.2 具體寫作問題

1. **Paper A Abstract 過度聲稱**: "the first comprehensive characterization of KV-cache compressibility" — Rethinking KV Cache Compression (MLSys 2025, Wei Gao et al.) 已正式發表並做了系統性評估，且其結論之一（value-cache 在 shallow layers 更敏感）與 Paper A 的 bottleneck layer 發現部分重疊
2. **Paper A Sec V.A**: "29-47% higher F1 than SnapKV" — 但 p=0.14-0.29，不顯著。應該改為 "consistently higher though not individually statistically significant"
3. **Paper B Eq. 2**: Q2C 公式與實際代碼不一致（見 1.2）
4. **Paper B Sec III.B "Attention Focusing Effect"**: 假設性解釋缺乏直接的 attention entropy / distribution 分析支持
5. **Paper A Sec V.E**: "delta encoding is strictly inferior" — 但在 Yi-6B 上，anchor delta 改善了 HotpotQA 9.2%。"Strictly" 的措辭不準確
6. **Paper B Table 3**: Adaptive policy 的 "deadline_success_rate" 與 static INT4 相同（0.749 at 1s），因為 adaptive 在帶寬不足時降級到 INT4，不能進一步降級到 scout（這應該在 adaptive policy 中被考慮）

### 5.3 實驗設計問題

1. **Context length 太短**: Paper A 主要在 100-500 tokens 上評估（"avg_context_tokens": 168.88），這遠低於 KV-cache 壓縮真正有價值的場景（4K-128K tokens）。在 170 tokens 上，KV-cache 只有 9.7 MB，即使在 10 Mbps 下也只需 7.8 秒。真正的痛點是長上下文場景
2. **只用 SQuAD v2 做主要評估**: SQuAD v2 是一個短上下文、單跳 extractive QA 任務，不代表 LLM 的主要應用場景（長上下文 QA、summarization、code generation、multi-turn dialogue）
3. **沒有 perplexity 評估**: 幾乎所有 KV-cache 壓縮的同行論文都報告 perplexity，這是一個更穩定、更通用的質量指標。只報告 task-specific F1 使得結果難以與其他論文直接比較
4. **Paper B 的 scout 只在同家族上測試**: 沒有 cross-family 實驗（Qwen → Mistral），這限制了 generalizability 的宣稱

---

## 6. 綜合評分與發表可能性

### 6.1 Paper A 評分

| 維度 | 分數 (0-100) | 說明 |
|------|-------------|------|
| 原創性 | 55 | Q2C 有一定新意但 Quest 已先行；mixed-precision 與 KVTuner 高度重疊 |
| 技術深度 | 50 | 方法簡單直觀（attention score ranking + per-layer sensitivity），缺乏理論分析 |
| 實驗充分性 | 55 | 7 models × 4 tasks 的 coverage 不錯，但 context length 太短、sample size 不足、缺 perplexity |
| 寫作品質 | 75 | 清晰流暢，表格專業 |
| 應用價值 | 65 | Practical guidelines 有用，但缺少系統實現 |
| 與頂刊/頂會的距離 | 45 | 對 INFOCOM 而言 networking contribution 不足；對 ML 會議（NeurIPS/ICML）則實驗規模不足 |

**Paper A 綜合分數: 55/100**

**發表可能性**:
- INFOCOM 2027: **30%**（networking 貢獻不足）
- ICC: **60%**（作為 short/workshop paper 可能性更高）
- Workshop (NeurIPS/ICML SysML): **70%**

### 6.2 Paper B 評分

| 維度 | 分數 (0-100) | 說明 |
|------|-------------|------|
| 原創性 | 70 | Scout = cross-model attention transfer for BW savings 是新穎概念；attention focusing effect 有趣 |
| 技術深度 | 45 | 5-mode lookup table 過於簡單；Markov chain BW model 過於理想化 |
| 實驗充分性 | 50 | 只在 Qwen2.5 上驗證 scout；只有 SQuAD v2；simulation 不夠真實 |
| 寫作品質 | 70 | 清晰但略長；算法描述好 |
| 應用價值 | 70 | Scout 概念在 edge-cloud 場景有高實用價值 |
| 與頂刊/頂會的距離 | 50 | ICC 可能性合理；JSAC 需要大幅擴展實驗和理論 |

**Paper B 綜合分數: 58/100**

**發表可能性**:
- JSAC: **25%**（需要更深理論 + 更廣實驗）
- ICC 2027: **55%**
- GLOBECOM: **65%**
- Workshop (MobiSys/MobiCom): **75%**

### 6.3 合併成 Journal Paper 的評分

如果將 Paper A + Paper B 合併為一篇 JSAC/TWC journal paper:
- **綜合分數: 65/100**
- **JSAC 發表可能性: 45%**（前提是補充 cross-family scout 實驗 + 更真實的信道模型 + perplexity 評估）

---

## 7. 潛在新研究方向

基於對兩篇論文和 40+ 篇同行文獻的深入分析，我識別出以下有價值的研究方向：

### 7.1 Cross-Family Scout（高價值，直接可行）
Paper B 的 scout 只在 Qwen2.5 家族內驗證。如果能在 Qwen → Mistral 或 Qwen → Yi 之間證明 attention alignment（即使需要 tokenizer remapping），這將是一個顯著的 contribution。這需要解決 tokenizer 不同導致的 position 對齊問題。

### 7.2 Learned Q2C Scoring（取代 heuristic attention-based scoring）
目前 Q2C 只用 raw attention weight 作為 importance score。可以訓練一個 tiny neural network（甚至 linear layer）在 attention patterns 上做 importance scoring，用 downstream task performance 作為 supervision。這可以超越 attention-weight-as-importance 的限制。

### 7.3 KV-Cache Compression with Information Bottleneck
CLAUDE.md 中提到 Information Bottleneck formulation `min I(X;Z) - β I(Z;Y)`。可以正式將 KV-cache compression 框架化為 IB 問題：Z = compressed KV, X = full KV, Y = task output。這提供理論上的 rate-distortion bound，填補目前所有 KV-cache 壓縮論文的理論空白。

### 7.4 Adaptive Scout with Quality Feedback
目前的 adaptive protocol 是 open-loop 的（根據帶寬選 mode，不考慮質量反饋）。可以設計 closed-loop protocol：cloud 在 decode 後估計 response quality（例如通過 logprob entropy），如果質量不達標就 request 額外的 KV 資訊（differential KV update）。

### 7.5 Scout for Multi-Turn Dialogue
Scout 目前只處理 single-turn inference。在 multi-turn 場景中，前幾輪的 KV-cache 已在 cloud 上，只需要傳輸 new context 的 position indices。這大幅降低了 incremental 成本，是 scout 的天然延伸。

### 7.6 Layer-Selective KV-Cache Transfer
結合 Paper A 的 bottleneck layer 發現和 Paper B 的 scout 思路：只傳輸 bottleneck layer 的 KV-cache（1 layer），其餘 layer 由 cloud 自己 prefill。這是 "partial KV transfer" 的中間地帶，介於 full KV transfer 和 scout-only 之間。

### 7.7 Real Network Testbed
用 5G/WiFi 6 testbed + real LLM deployment 驗證 adaptive protocol。這是系統論文的標配，CacheGen 在 SIGCOMM 上被接受部分原因就是有真實系統實現。

### 7.8 KV-Cache as Semantic State: Theoretical Framework
將 KV-cache 正式定義為 semantic state variable，用 rate-distortion theory 分析不同壓縮策略的理論 bound。這是 CLAUDE.md 中提到的 "semantic state synchronization" 的理論化，可以投 IEEE Trans. on Information Theory 或 JSAC。

---

## 附錄：比對的同行論文完整列表

### KV-Cache Compression
1. H2O (NeurIPS 2023) — cumulative attention eviction
2. Scissorhands (NeurIPS 2023) — persistence of importance
3. SnapKV (NeurIPS 2024) — observation window selection
4. Quest (ICML 2024) — query-aware page-level sparsity
5. FastGen (ICLR 2024) — per-head adaptive policies
6. Keyformer (MLSys 2024) — key token scoring
7. PyramidInfer (ACL 2024) — layer-wise decreasing budget
8. PyramidKV (TMLR 2025) — pyramidal information funneling
9. CAOTE (arXiv 2025) — attention output error-based eviction
10. KIVI (ICML 2024) — asymmetric 2-bit quantization
11. KVQuant (NeurIPS 2024) — non-uniform quantization
12. ZipCache (NeurIPS 2024) — salient-token-aware quantization
13. KVTuner (arXiv 2025) — layer-wise mixed-precision
14. GEAR (NeurIPS 2024) — quant + low-rank + sparse
15. QAQ (arXiv 2024) — attention-score-based bit allocation
16. ATOM (MLSys 2024) — mixed-precision low-bit serving
17. PALU (ICLR 2025) — low-rank projection
18. MiniCache (NeurIPS 2024) — cross-layer merging
19. DMC (ICML 2024) — learned online compression
20. X-EcoMLA (arXiv 2025) — upcycling into MLA
21. ReCalKV (arXiv 2025) — low-rank with head reordering
22. Rethinking KV Cache (MLSys 2025) — systematic re-evaluation

### Edge-Cloud / Collaborative Inference
23. CacheGen (SIGCOMM 2024) — KV-cache streaming
24. Splitwise (ISCA 2024) — phase splitting
25. DistServe (OSDI 2024) — disaggregated serving
26. Mooncake (FAST 2025) — KVCache-centric architecture
27. EdgeShard (IEEE IoT-J 2024) — edge LLM sharding
28. Adaptive Layer Splitting (FITEE 2024) — RL-based wireless split
29. Hybrid SLM-LLM (MobiSys Wkshp 2024) — small-large collaboration
30. LMCache (arXiv 2025) — enterprise KV cache layer
31. CROSS-SEC (NAIC Wkshp 2024) — cross-WAN security
32. Sarathi-Serve (OSDI 2024) — chunked prefills
33. vLLM (SOSP 2023) — paged attention

### Speculative Decoding (Draft Model Reference)
34. EAGLE (ICML 2024) — feature-level speculation
35. Medusa (ICML 2024) — multi-head parallel decoding
36. Draft & Verify (ACL 2024) — self-speculative decoding
37. Decoding Speculative Decoding (NAACL 2025)

### Knowledge Transfer / Attention Analysis
38. LLM Modules (arXiv 2025) — cross-attention transfer
39. LISA (arXiv 2024) — cross-layer attention sharing
40. Dual-Space KD (EMNLP 2024) — knowledge distillation

### Semantic Communication / Adaptive AI
41. LLM-SemCom (IEEE 2025) — LLM-based semantic communication
42. TORC (Computer Networks 2023) — bandwidth-adaptive multi-task AI
43. Active Inference Offloading (IEEE TMC 2024)

---

*審查完畢。以上意見僅代表匿名審查者的獨立判斷，供作者參考。*
