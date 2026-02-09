# Paper A: Submission Strategy / 投稿策略文件

**Paper**: "Task-Aware KV-Cache Compression for Bandwidth-Efficient Collaborative LLM Inference"
**Authors**: Wei-Lun Cheng and Wanjiun Liao, Dept. of EE, National Taiwan University
**Status**: 7 pages, IEEE format, complete, proofread, 23 references

---

## 1. Venue Positioning / 投稿定位

### Primary Target: IEEE INFOCOM 2027 / 首選目標：IEEE INFOCOM 2027

**Why INFOCOM is the best fit / 為何 INFOCOM 最適合**:

- INFOCOM is the premier IEEE venue for networking and communications systems. Our paper addresses a *networking bottleneck*---KV-cache transmission over bandwidth-constrained edge-cloud links---which places it squarely in INFOCOM's core scope.
- INFOCOM 2026 已收錄 CacheGen (SIGCOMM 2024) 的後續延伸工作，表示 KV-cache 傳輸在網路頂會中是被認可的新興研究方向。
- 我們的論文有完整的系統模型（edge-cloud split inference）、明確的頻寬最佳化目標（Eq. 2）、以及端到端延遲分析——這些都是 INFOCOM 審稿人重視的系統面貢獻。
- Acceptance rate: INFOCOM typically accepts 19-20% of submissions (~250/1250). Our paper's breadth (7 model families, 4 tasks, 75+ configurations) provides the systematic rigor INFOCOM values.
- **Deadline estimate**: August 2026 (based on recent years: INFOCOM 2026 deadline was July 2025).

**Fit analysis / 契合度分析**:

| Criterion / 評估項目 | Score / 評分 | Notes / 說明 |
|---|---|---|
| Topic relevance / 主題相關性 | 9/10 | Edge-cloud 協作推論是網路系統核心問題 |
| Technical depth / 技術深度 | 8/10 | 完整壓縮管道 + 跨架構實驗 |
| Novelty / 新穎性 | 8/10 | Q2C 選擇法 + delta encoding 反面發現 |
| Experimental rigor / 實驗嚴謹度 | 9/10 | 7 模型、4 任務、75+ 組態，遠超同類工作 |
| System relevance / 系統相關性 | 9/10 | 直接解決 5G/6G 邊緣運算頻寬瓶頸 |

### Secondary Target: IEEE ICC 2027 / 備選目標：IEEE ICC 2027

- ICC (International Conference on Communications) 接受率約 40-45%，是 INFOCOM 的穩健備選。
- ICC has dedicated tracks for "Communication QoS, Reliability and Modeling" and "Machine Learning for Communications" where this paper fits naturally.
- **Deadline estimate**: October 2026 (ICC 2027 deadline likely ~Oct 2026 based on ICC 2026 deadline of Oct 2025).
- ICC 的頁數限制也是 6-7 頁，目前版本無需大幅調整。

### Alternative Venues / 其他備選

| Venue | Fit | Acceptance Rate | Notes |
|---|---|---|---|
| **ACM SIGCOMM** | High | ~15% | CacheGen 在此發表；但 SIGCOMM 偏好完整系統實作 |
| **IEEE JSAC Special Issue** | High | ~30% | 若有 "LLM Systems" 或 "Edge AI" 特刊 |
| **USENIX NSDI** | Medium-High | ~18% | 偏好完整系統原型，我們缺少端對端部署 |
| **MLSys** | Medium | ~25% | 偏向 ML 訓練/推論最佳化，網路面向較弱 |
| **IEEE TWC** | Medium | ~20% | 若加入無線通道模型可投期刊 |

---

## 2. Paper Value & Significance / 論文價值與意義

### Gap Filled / 填補的研究空白

**English**: Prior to this work, no systematic study existed that characterized KV-cache compressibility across multiple model families, tasks, and context lengths simultaneously. CacheGen (SIGCOMM 2024) demonstrated the concept of KV-cache streaming but did not address task-aware selection and assumed delta encoding is always beneficial. H2O and SnapKV addressed token eviction but only for single-model inference, not cross-device transmission. Our paper is the first to:

1. Introduce **task-aware** token selection (Q2C) that uses query-to-context attention rather than generic attention patterns
2. Demonstrate that **INT4 fragility is model-specific**, not architecture-determined---a finding that contradicts implicit assumptions in KIVI and KVQuant
3. Provide a **counter-finding** that CacheGen's core delta encoding technique degrades quality
4. Characterize KV-cache compression across **7 model families** (vs. 2 in CacheGen, 1-2 in H2O/SnapKV)
5. Combine selection + quantization into a unified compression pipeline with per-stage latency analysis

**Traditional Chinese**:
在本研究之前，尚無任何系統性研究同時涵蓋多個模型家族、多種任務與多種上下文長度的 KV 快取壓縮特性分析。CacheGen（SIGCOMM 2024）展示了 KV 快取串流的概念，但未處理任務感知選擇，且假設 delta 編碼總是有益的。H2O 和 SnapKV 處理了 token 淘汰，但僅針對單一模型推論，而非跨裝置傳輸。

本論文的核心價值在於：
- **Q2C 選擇法**：首次利用 query-to-context 注意力分數進行任務感知的 token 選擇，在 25% 保留率下比 SnapKV 高 29-47%、比 H2O 高 92-128%
- **模型特異性發現**：INT4 量化脆弱性是模型訓練的新興特性，而非架構決定——Yi-6B 和 Qwen-7B 有相同的 GQA 架構（4 KV heads），但 INT4 品質分別為 100% 和 77%
- **反面發現**：Delta encoding（CacheGen 核心技術）在結合量化時降低任務品質 5.6-14.0 個百分點
- **混合精度方案**：診斷式瓶頸層發現，保護單一層即可達到 3.6x 無損壓縮

### Impact Statement / 影響力聲明

This paper changes how the community thinks about KV-cache compression:
- 從「所有層一視同仁」→「模型特異性診斷」
- 從「delta 編碼減少冗餘」→「delta 編碼可能有害」
- 從「通用注意力模式選擇 token」→「任務感知的 query-to-context 選擇」
- 從「壓縮只看壓縮率」→「壓縮必須考慮端到端延遲和任務品質」

---

## 3. Competitive Advantages / 競爭優勢

### vs. CacheGen (SIGCOMM 2024)

| Dimension / 面向 | CacheGen | Our Work / 本研究 | Advantage / 優勢 |
|---|---|---|---|
| Token selection / Token 選擇 | None (all tokens) | Q2C task-aware | Q2C at 25% retention: +29-47% over SnapKV |
| Delta encoding | Core technique | Counter-evidence: degrades quality | Saves researchers from wrong direction |
| Model coverage | LLaMA, Mistral (2 families) | 7 families (1.1B-14B) | 3.5x broader coverage |
| Task coverage | 3 long-context tasks | 4 diverse tasks (QA, multi-hop, MCQ) | More comprehensive evaluation |
| Context scaling | Not studied | 512-4096 tokens | First characterization of length effects |
| Layer-level analysis | Graded heuristic | Diagnostic bottleneck discovery | Lossless 3.6x compression |
| Latency analysis | CUDA kernel benchmarks | Per-stage timing (prefill/quant/TX) | Shows TX dominates 43x |

### vs. H2O / SnapKV

- H2O 和 SnapKV 是為**單機推論**設計的記憶體節省方法，不是為**跨裝置傳輸**設計的
- 它們使用累積注意力（H2O）或觀察窗口（SnapKV），而非查詢特定的注意力
- 在 25% 保留率（頻寬受限場景的關鍵操作點），Q2C 一致性地在所有 4 個模型上超越兩者

### vs. KIVI / KVQuant

- KIVI 和 KVQuant 只研究量化，不研究 token 選擇
- 它們在 1-2 個模型上實驗，我們在 7 個模型家族上系統性分析
- 我們揭示 INT4 脆弱性是模型特異性的（非架構決定），這是 KIVI/KVQuant 未探討的

### Unique Selling Points / 獨特賣點

1. **Counter-finding with evidence / 有證據支持的反面發現**: Delta encoding 降低品質，且我們用 entropy 分析解釋原因（delta 值更均勻分布於量化範圍，INT4+delta 達 7.94 bits/element vs. 直接 INT4 的 6.13）
2. **Actionable deployment guidelines / 可操作的部署指南**: Section VI 提供明確的決策流程——先 INT8、再診斷、再決定混合精度或 Q2C
3. **Reproducibility / 可重現性**: 38 個實驗腳本、23 個 JSON 結果檔案，完整的實驗基礎設施

---

## 4. Submission Strategy / 投稿策略

### Timeline / 時間線

| Date / 日期 | Milestone / 里程碑 | Details / 細節 |
|---|---|---|
| 2026-02 ~ 2026-03 | Additional experiments / 補充實驗 | 考慮加入更長上下文（8K, 16K tokens）實驗 |
| 2026-03 ~ 2026-04 | Strengthen paper / 強化論文 | 加入更多 statistical tests、視覺化、possible rebuttal points |
| 2026-05 ~ 2026-06 | Internal review / 內部審查 | 請 Liao 教授及實驗室同學審閱 |
| 2026-06 ~ 2026-07 | Camera-ready polish / 最終修改 | 確認格式、補充 related work |
| 2026-08 (est.) | **INFOCOM 2027 submission** | 首選目標 |
| 2026-10 (est.) | **ICC 2027 submission** (if needed) | 若 INFOCOM 未中 |

### Dual Submission Considerations / 雙重投稿考量

- INFOCOM 和 ICC **不可同時投稿**相同論文（IEEE policy）
- 建議策略：**先投 INFOCOM**（deadline ~Aug 2026），若被拒則修改後投 ICC（deadline ~Oct 2026）
- INFOCOM 審稿週期約 3 個月（Aug → Nov），結果出來後仍有時間修改投 ICC
- 注意：若投 INFOCOM，不能同時將高度重疊的 Paper B 投到 INFOCOM——但 Paper B 投 ICC 是完全可以的，因為它們是不同的貢獻

### Pre-submission Checklist / 投稿前檢查清單

- [ ] 確認頁數限制（INFOCOM: 9 pages + refs; ICC: 6 pages + refs）
- [ ] 目前版本為 7 頁，INFOCOM 可擴展至 9 頁（加入更多實驗和分析）
- [ ] 若投 ICC，需要壓縮至 6 頁
- [ ] 確認 IEEE 格式（目前已使用 IEEEtran conference 格式）
- [ ] 確認所有圖表的可讀性（黑白列印測試）
- [ ] 加入 reproducibility statement
- [ ] 更新 reference list（確認所有 arXiv 論文是否已正式發表）

### Reviewer Anticipation / 預期審稿意見

**Potential concerns and responses / 預期質疑與回應**:

1. **"Sample size is small (n=50)"**
   - Response: 我們報告 95% CIs 和 Wilcoxon p-values；跨 7 模型的一致性趨勢比單模型大樣本更有說服力
   - 可在修改時增加至 n=100 或 200

2. **"No real network evaluation"**
   - Response: 我們的貢獻在於壓縮品質分析，而非系統實作；延遲分析基於真實 GPU timing + simulated TX
   - 可加入 network emulation（tc/netem）實驗

3. **"Only up to 14B parameters"**
   - Response: 14B 是 edge-cloud 場景的現實上限（70B 不會跑在邊緣裝置）；且我們的發現（INT4 fragility 是模型特異性）在 7 個模型上已穩定

4. **"Q2C requires full prefill to compute attention"**
   - Response: 這是 split inference 的前提——edge 本來就要跑 prefill；Q2C 不增加額外運算

---

## 5. Future Work & Extensions / 未來工作與延伸

### Short-term (before submission) / 短期（投稿前）

1. **Longer context experiments / 更長上下文實驗**
   - 目前最長 4096 tokens，可擴展至 8K 和 16K
   - 預期 Q2C 的優勢會隨上下文長度增加而更大（更多無關 tokens 需要過濾）
   - 實驗設備已備妥（RTX PRO 6000, 102 GB VRAM 足以處理 16K context）

2. **More models / 更多模型**
   - LLaMA 3.1 8B/70B（Meta 最新系列）
   - Gemma 2 (Google)
   - 確認 INT4 fragility 的模型特異性在更多家族上成立

3. **Real network emulation / 真實網路模擬**
   - 使用 Linux tc/netem 模擬不同頻寬和延遲
   - 加入 packet loss 的影響分析

### Medium-term (post-submission extensions) / 中期（投稿後延伸）

4. **Entropy coding on top of quantization / 量化後加入熵編碼**
   - 目前只做量化，未加 arithmetic/Huffman coding
   - 可能額外節省 20-30% 頻寬
   - 但需注意 delta encoding 的教訓——不是所有看似合理的壓縮技術都有效

5. **Online Q2C for streaming contexts / 串流上下文的線上 Q2C**
   - 目前 Q2C 需要完整 prefill；可探索增量版本
   - 在每個新 chunk 到達時更新注意力分數

6. **Integration with architectural compression / 與架構壓縮整合**
   - 結合 MLA (DeepSeek-V2) 或 GQA 的架構壓縮
   - 理論分析：architectural compression + post-hoc compression 的壓縮率界限

### Long-term (journal extension) / 長期（期刊版）

7. **Full system prototype / 完整系統原型**
   - Edge device (Jetson Orin) + Cloud server (A100/H100)
   - 真實 5G 網路測試
   - 這會是 JSAC 或 TMC 期刊版的核心貢獻

8. **Theoretical analysis / 理論分析**
   - 資訊瓶頸框架下的最佳 token 選擇理論
   - Rate-distortion analysis of KV-cache compression
   - 證明 Q2C 在特定條件下是最優的

---

## 6. Collaboration Opportunities / 合作機會

### Academic Collaborators / 學術合作

1. **CacheGen team (University of Chicago / 芝加哥大學)**
   - Yuhan Liu et al. — CacheGen 作者群
   - 我們的 delta encoding 反面發現直接回應他們的工作，有潛力合作改進
   - 合作方向：結合他們的系統最佳化 + 我們的品質分析

2. **SnapKV / H2O teams**
   - SnapKV: University of Virginia
   - H2O: Carnegie Mellon University
   - 合作方向：將 Q2C 整合到他們的框架中，或共同開發混合策略

3. **Edge AI research groups in Taiwan / 台灣邊緣 AI 研究群**
   - 中研院資訊所 (Academia Sinica IIS) — 有 LLM 部署經驗
   - 交大電信所 — 無線通訊 + AI 交叉領域
   - 成大資工系 — 分散式系統

4. **Wireless/networking + LLM intersection**
   - Georgia Tech (Raghupathy Sivakumar group) — mobile computing + LLM
   - MIT CSAIL (Mohammad Alizadeh group) — networking systems, Pensieve ABR
   - UC Berkeley (Ion Stoica group) — vLLM, LLM serving systems

### Industry Partners / 產業合作

1. **Qualcomm Research**
   - 在 edge AI inference 有大量投資
   - 直接需要 KV-cache 壓縮技術用於 Snapdragon 裝置
   - 合作方向：在真實 mobile chipset 上驗證

2. **MediaTek (聯發科)**
   - 台灣最大 IC 設計公司，Dimensity 晶片支援 on-device AI
   - 與台大電機系有長期合作關係
   - 合作方向：KV-cache 壓縮硬體加速器設計

3. **NVIDIA**
   - TensorRT-LLM 已整合 KV-cache 量化
   - 合作方向：將 Q2C 和混合精度方案整合到 TensorRT-LLM

4. **Cloud providers (AWS, Azure, GCP)**
   - 邊緣推論服務需要頻寬最佳化
   - AWS Inferentia / Azure FPGA 等專用硬體
   - 合作方向：在真實雲端環境中部署和評估

### Funding Opportunities / 經費來源

- **科技部/國科會 (NSTC)** AI 研究計畫：可申請「邊緣 AI 推論最佳化」主題
- **MediaTek-NTU joint research / 聯發科-台大合作研究**：既有合作架構
- **IEEE ComSoc Student Travel Grant**: 若論文被接受，可申請會議旅費補助

---

## Summary / 總結

Paper A 的核心優勢在於**實驗的廣度和深度**（7 模型、4 任務、75+ 組態）以及**反直覺的發現**（delta encoding 有害、INT4 fragility 是模型特異性）。這些特質使其非常適合 INFOCOM 這類重視系統性和嚴謹性的頂會。

**建議投稿順序**：INFOCOM 2027 (Aug 2026) → ICC 2027 (Oct 2026) → JSAC/TMC special issue (擴展為期刊版)

**投稿前最關鍵的改進**：加入更長上下文的實驗（8K-16K tokens）和更多模型（LLaMA 3.1），將目前的 7 頁擴展至 INFOCOM 允許的 9 頁上限。
