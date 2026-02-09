# Paper B: Submission Strategy / 投稿策略文件

**Paper**: "Scout: Bandwidth-Adaptive KV-Cache Transport for Heterogeneous Edge-Cloud LLM Inference"
**Authors**: Wei-Lun Cheng and Wanjiun Liao, Dept. of EE, National Taiwan University
**Status**: 7 pages, IEEE format, complete, proofread, 14 references, 4 figures, 4 tables, 1 algorithm

---

## 1. Venue Positioning / 投稿定位

### Primary Target: IEEE ICC 2027 / 首選目標：IEEE ICC 2027

**Why ICC is the best fit / 為何 ICC 最適合**:

- ICC (International Conference on Communications) 是 IEEE 通訊領域的旗艦會議，每年接受約 1000-1200 篇論文。
- Paper B 的核心貢獻是**通訊協定設計**（adaptive protocol with 5 operating modes）和**多代理資源分配**（multi-agent bandwidth allocation），這些都是 ICC 的核心議題。
- ICC 有專門的 tracks："Communication QoS, Reliability and Modeling"、"Machine Learning for Communications"、"Wireless Communications" 都適合本論文。
- Scout 協定的 mode selection policy (Eq. 4) 和 multi-agent allocation (Eq. 5) 本質上是通訊系統最佳化問題，非常適合 ICC 審稿人的背景。
- **Deadline estimate**: October 2026 (ICC 2027, based on ICC 2026 deadline of Oct 2025).
- Acceptance rate: ~40-45%, 相對 INFOCOM 更穩健。

**Fit analysis / 契合度分析**:

| Criterion / 評估項目 | Score / 評分 | Notes / 說明 |
|---|---|---|
| Topic relevance / 主題相關性 | 9/10 | 自適應傳輸協定是通訊核心問題 |
| Technical depth / 技術深度 | 8/10 | 完整協定設計 + Markov chain 模擬 + 多代理分配 |
| Novelty / 新穎性 | 9/10 | Scout model paradigm 前所未有 |
| Experimental rigor / 實驗嚴謹度 | 8/10 | 3 model pairs, 36 configurations, 95% CIs |
| System relevance / 系統相關性 | 9/10 | 直接解決邊緣-雲端 LLM 推論的頻寬瓶頸 |

### Secondary Target: IEEE JSAC Special Issue / 備選目標：IEEE JSAC 特刊

- IEEE Journal on Selected Areas in Communications 是通訊領域最頂尖的期刊之一（IF ~13.8）。
- JSAC 定期有 "Edge Intelligence"、"Machine Learning for Communications"、"Next-Generation Networking" 相關特刊。
- 若將 Paper B 擴展至期刊版（14-16 頁），加入更多實驗（cross-family scout、longer contexts、real wireless experiments），可直接投 JSAC。
- **優勢**：期刊沒有頁數限制的壓力，可充分展開 attention focusing effect 的理論分析。

### Third Target: IEEE TWC / 第三選擇：IEEE TWC

- IEEE Transactions on Wireless Communications（IF ~8.9）適合強調無線通訊面向的版本。
- 需要加入更嚴謹的無線通道模型（3GPP TR 38.901, fading, MIMO）取代目前的 Markov chain 模型。
- 合作方向：若與無線通訊領域的研究者合作，可大幅強化通道建模部分。

### Alternative Venues / 其他備選

| Venue | Fit | Acceptance Rate | Notes |
|---|---|---|---|
| **IEEE INFOCOM 2027** | High | ~19% | 若 Paper A 不投 INFOCOM，可考慮 Paper B |
| **ACM MobiCom** | Medium-High | ~18% | 偏好有 real device 實驗的行動運算論文 |
| **ACM MobiSys** | Medium | ~20% | 需要完整系統實作 |
| **IEEE GLOBECOM** | High | ~45% | ICC 的姊妹會議，backup option |
| **IEEE/ACM Trans. Networking (ToN)** | Medium-High | ~15% | 需擴展為期刊長度 |

### Strategic Consideration: Paper A → INFOCOM, Paper B → ICC / 策略考量

**最佳投稿組合**：
- Paper A → **INFOCOM 2027** (Aug 2026 deadline): 著重壓縮品質的系統性分析
- Paper B → **ICC 2027** (Oct 2026 deadline): 著重協定設計和自適應傳輸

這樣兩篇論文有明確區分：Paper A 是 "compression characterization"，Paper B 是 "adaptive transport protocol"。Paper B cite Paper A 為 [11]，建立自然的引用關係。兩篇分別在不同會議發表不會違反任何 IEEE policy。

---

## 2. Paper Value & Significance / 論文價值與意義

### Gap Filled / 填補的研究空白

**English**: The existing literature on KV-cache compression treats compression level as a static design choice and evaluates quality in isolation from network conditions. No prior work has:

1. **Proposed cross-model attention transfer for bandwidth elimination**: The scout model paradigm---transmitting position indices instead of KV data---is entirely new. This reduces payload from 9.7-33.2 MB to 336 bytes (28,800-98,800x), a compression ratio unprecedented in LLM serving.

2. **Discovered the attention focusing effect**: The finding that a smaller model's attention selection can *improve* a larger model's quality (7B scout improves 14B by 10.2%, p=0.018) has no precedent. This transforms bandwidth optimization from a quality *tradeoff* into a quality *enhancement*.

3. **Designed a bandwidth-adaptive protocol for LLM inference**: While adaptive bitrate (ABR) is standard in video streaming, no prior work applies adaptive mode selection to KV-cache transport. Our protocol with 5 discrete operating modes achieves 98-107% quality with up to 100% deadline compliance.

4. **Addressed multi-agent LLM bandwidth allocation**: As multi-agent systems scale, multiple edge devices share uplink bandwidth. Our model-aware allocation converts 0% to 100% deadline compliance under congestion.

**Traditional Chinese**:
現有 KV 快取壓縮文獻將壓縮等級視為靜態設計選擇，且品質評估獨立於網路條件。本論文填補了四個重要空白：

- **Scout 模型範式**：首次提出跨模型注意力傳遞以消除 KV 快取傳輸。payload 從 9.7-33.2 MB 降至 336 bytes（28,800-98,800x 壓縮），這在 LLM 服務領域是前所未有的壓縮比。

- **注意力聚焦效應（Attention Focusing Effect）**：7B scout 模型的選擇**改善** 14B 雲端模型品質 10.2%（p=0.018）。這個發現將頻寬最佳化從品質**妥協**轉變為品質**提升**。這是一個違反直覺但有統計顯著性支持的結果。

- **自適應 KV 快取傳輸協定**：類似影片串流的 ABR，但應用於 LLM 推論。5 種操作模式的即時切換，在變動頻寬下達到 98-107% 品質和最高 100% 期限合規率。

- **多代理頻寬分配**：模型感知的分配策略在擁塞情況下將期限合規率從 0% 提升至 100%。

### Paradigm Significance / 範式意義

Scout 的核心洞察是：**在同一模型家族中，注意力模式比 KV-cache 數值更具可傳遞性**。KV-cache 的 cosine similarity 只有 ~0.22（不可傳遞），但 position overlap 達 82-83%（高度可傳遞）。

這意味著**傳輸「在哪裡看」比「看到什麼」更有效率**——從傳輸資料轉向傳輸決策，是一種根本性的通訊範式轉移。

---

## 3. Competitive Advantages / 競爭優勢

### vs. CacheGen (SIGCOMM 2024)

| Dimension / 面向 | CacheGen | Scout | Advantage / 優勢 |
|---|---|---|---|
| Compression ratio | 3.7-4.5x | 28,800-98,800x | Scout 高出 4 個數量級 |
| Adaptivity | Static | 5 adaptive modes | 即時適應頻寬變化 |
| Multi-model | Single model | Cross-model pairs | 支援異質邊緣-雲端模型 |
| Multi-agent | Not addressed | N=2,4,8 agents | 多代理資源分配 |
| Quality floor | ~96% (INT4) | 81-110% (model-dependent) | 7B→14B 可超越基線 |

### vs. Speculative Decoding

- Speculative decoding 使用小模型提議 tokens、大模型驗證——目標是**延遲降低**
- Scout 使用小模型提議 positions、大模型執行——目標是**頻寬降低**
- 核心差異：speculative decoding 在 decode 階段運作；Scout 在 prefill 階段運作
- 兩者可以**結合使用**：Scout 減少 prefill 傳輸頻寬，speculative decoding 加速 decode 延遲

### vs. Adaptive Bitrate (ABR) for Video

- ABR（如 Pensieve）在影片串流中自適應調整品質等級
- Scout 將類似概念應用於 LLM 推論，但有關鍵差異：
  - 品質以 task performance (F1) 而非 perceptual metrics (PSNR/SSIM) 衡量
  - 操作模式是離散的（5 modes vs. continuous quality ladder）
  - 有 scout fallback mode 作為零頻寬選項

### Unique Selling Points / 獨特賣點

1. **Attention focusing effect / 注意力聚焦效應**: 7B scout *改善* 14B 品質 10.2%，這是一個正面的、違反直覺的、有統計顯著性的發現——審稿人會對此印象深刻
2. **Extreme compression / 極端壓縮**: 9.7 MB → 336 bytes (28,800x)，數字本身就很有衝擊力
3. **Complete protocol design / 完整協定設計**: 從 GPU 實驗驗證到 Markov chain 模擬到多代理分配，full stack
4. **Practical operating points / 實用操作點**: 5 種模式的離散選擇，不需要複雜的連續最佳化
5. **Graceful degradation / 優雅退化**: 即使在最差頻寬下也有 scout fallback，不會服務失敗

---

## 4. Submission Strategy / 投稿策略

### Timeline / 時間線

| Date / 日期 | Milestone / 里程碑 | Details / 細節 |
|---|---|---|
| 2026-02 ~ 2026-04 | Cross-family experiments / 跨家族實驗 | Qwen→Mistral scout 測試（核心延伸） |
| 2026-04 ~ 2026-05 | Longer context / 更長上下文 | 4K-16K token 的 scout 實驗 |
| 2026-05 ~ 2026-06 | Strengthen wireless model / 強化無線模型 | 加入 3GPP channel model 或 real trace |
| 2026-06 ~ 2026-08 | Paper revision / 論文修改 | 根據新實驗更新數據和分析 |
| 2026-08 ~ 2026-09 | Internal review / 內部審查 | 請 Liao 教授及實驗室同學審閱 |
| 2026-10 (est.) | **ICC 2027 submission** | 首選目標 |
| 2026-11 (backup) | **JSAC special issue** (if applicable) | 若有適合的特刊 call |
| 2027-03 (backup) | **GLOBECOM 2027** | 若 ICC 未中的備選 |

### Relationship with Paper A / 與 Paper A 的關係

Paper B 引用 Paper A 為 [11]（reference #11 in the bibliography）。投稿策略上需要注意：

1. **若 Paper A 已投 INFOCOM**：Paper B 可引用為 "submitted" 或 "under review"——ICC 允許引用投稿中的論文
2. **若 Paper A 已被接受**：Paper B 可引用為 "accepted for publication at INFOCOM 2027"，大幅增加可信度
3. **若 Paper A 被拒**：Paper B 仍可引用為 "technical report" 或 arXiv preprint
4. **最佳情境**：Paper A 在 INFOCOM 被接受，Paper B 引用已接受的 Paper A，同時投 ICC——兩篇形成完整的研究故事

### Pre-submission Enhancements / 投稿前強化

**High priority / 高優先**:
- [ ] Cross-family scout 實驗（Qwen scout → Mistral cloud, 或反過來）——這是 Limitation 中最大的限制
- [ ] 更長上下文實驗（4K+ tokens）——目前只有 ~200 tokens
- [ ] 將 Markov chain 頻寬模型換成 real 5G trace（若能取得）

**Medium priority / 中等優先**:
- [ ] 加入 Latency CDF 圖（complementary to deadline compliance %）
- [ ] 理論分析 attention focusing effect 的充分條件
- [ ] Multi-agent 實驗增加到 N=16 或 32

**Low priority / 低優先**:
- [ ] 真實 mobile device (Jetson Orin) 上的 edge prefill benchmark
- [ ] Token-level analysis: 哪些 tokens 被 scout 「修正」了

### Reviewer Anticipation / 預期審稿意見

**Potential concerns and responses / 預期質疑與回應**:

1. **"Scout only works within the same model family"**
   - Response: 這是正確的限制，我們在 Section VII 明確承認。然而，同家族部署（如 Qwen2.5 3B edge + 14B cloud）是實際的部署模式。Cross-family 實驗是明確的 future work。
   - 強化方案：投稿前完成 cross-family 實驗（即使結果是負面的，也有分析價值）

2. **"Markov chain bandwidth model is simplistic"**
   - Response: 6-state Markov chain 已足以展示 adaptive policy 的優勢。我們在 limitation 中承認此點。
   - 強化方案：加入 real 5G traces（from public datasets like 5Gopher or Lumos5G）

3. **"Cloud needs to re-run prefill in scout mode — is this really saving anything?"**
   - Response: 是的，因為 prefill 延遲（18-57ms）遠小於 KV 傳輸延遲（775-2656ms at 100 Mbps）。TX 延遲在低頻寬下更嚴重（10 Mbps 下 Qwen-14B 需要 26.8 秒傳輸 KV，但 prefill 只需 57ms + scout 的 336 bytes 傳輸幾乎為零）。

4. **"Attention focusing effect might be task-specific"**
   - Response: 我們在 SQuAD v2 上觀察到此效應。需要在更多任務上驗證（TriviaQA, HotpotQA, MMLU）。
   - 強化方案：投稿前在至少 2 個額外任務上測試 7B→14B attention focusing

5. **"Sample size n=50 is small for the attention focusing claim"**
   - Response: p=0.018 < 0.05 with paired t-test at n=50 is statistically significant. 但可增加至 n=200 以增強說服力。

---

## 5. Future Work & Extensions / 未來工作與延伸

### Short-term (before submission) / 短期（投稿前）

1. **Cross-family scout experiments / 跨家族 scout 實驗**
   - 最重要的延伸：Qwen scout → Mistral cloud、Mistral scout → Qwen cloud
   - 預期結果：跨家族 overlap 會顯著降低（不同 tokenizer 和 RoPE 參數）
   - 即使是負面結果也有價值——明確界定 scout 的適用範圍
   - 可能需要 tokenizer alignment 技術（共同 vocabulary 映射）

2. **Multi-task attention focusing validation / 多任務注意力聚焦驗證**
   - 在 TriviaQA、HotpotQA、MMLU 上測試 7B→14B attention focusing
   - 確認此效應是否為通用現象或僅限於 extractive QA

3. **Longer context scaling / 更長上下文擴展**
   - 目前 scout 實驗的上下文長度約 170 tokens
   - 擴展至 1K、4K、16K tokens
   - 預期：更長上下文中 overlap 可能降低，但 bandwidth savings 也更大

### Medium-term (post-submission) / 中期（投稿後）

4. **Hierarchical scout chains / 層級式 scout 鏈**
   - 1B → 3B → 7B → 14B 的漸進式注意力傳遞
   - 每層 scout 逐步精化 position selection
   - 可能發現最佳的 scout chain 長度

5. **Bidirectional scout / 雙向 scout**
   - 不只 edge→cloud，也研究 cloud→edge 的注意力指引
   - Cloud 可以告訴 edge「下次請特別關注這些 positions」
   - 這連接到 semantic communication 的 feedback 機制

6. **Integration with speculative decoding / 與推測解碼整合**
   - Scout（prefill 階段頻寬最佳化）+ Speculative decoding（decode 階段延遲最佳化）
   - 完整的端到端 edge-cloud 推論最佳化方案

7. **Reinforcement learning for mode selection / 強化學習模式選擇**
   - 目前 mode selection 是 greedy enumeration
   - 可使用 RL（類似 Pensieve 的方法）學習最佳策略
   - 考慮 multi-step lookahead 和 multi-request 序列

### Long-term (journal version) / 長期（期刊版）

8. **Full system implementation / 完整系統實作**
   - Edge: Jetson Orin Nano/NX + mobile 5G modem
   - Cloud: A100/H100 伺服器
   - 真實 5G 網路上的端到端測試
   - 測量真實的 E2E latency、throughput、quality

9. **Theoretical foundations / 理論基礎**
   - 證明 attention focusing effect 的充分條件
   - 資訊論分析：position indices 的 channel capacity 需求
   - 最佳 scout model size 的理論界限（given edge compute budget and cloud model size）

10. **Extension to multi-modal models / 擴展至多模態模型**
    - Vision-Language Models (VLMs) 的 KV-cache 更大
    - 影像 tokens 的注意力分布可能有不同的 cross-model alignment 特性
    - 這是更長遠但具高影響力的方向

---

## 6. Collaboration Opportunities / 合作機會

### Academic Collaborators / 學術合作

1. **Speculative decoding researchers / 推測解碼研究者**
   - Google Research (Yaniv Leviathan et al.) — speculative decoding 原始作者
   - 合作方向：Scout + Speculative decoding 的聯合最佳化
   - 論文概念：「Prefill-aware Speculative Decoding with Scout Models」

2. **Adaptive bitrate / video streaming groups / 自適應位元率研究群**
   - MIT CSAIL (Mohammad Alizadeh) — Pensieve ABR 作者
   - CMU (Vyas Sekar) — video streaming + networking
   - 合作方向：將 ABR 的 RL 技術應用到 Scout 的 mode selection

3. **Wireless communication + AI groups / 無線通訊 + AI 研究群**
   - Georgia Tech (Raghupathy Sivakumar) — mobile computing + AI
   - KAIST (Dongsu Han) — networked systems + ML
   - 台大電信所 — 無線通道建模專長，可強化 Markov chain 模型
   - 合作方向：用 3GPP channel model 取代簡化的 Markov 模型

4. **Multi-agent LLM systems / 多代理 LLM 系統**
   - Tsinghua University (Zhiyuan Liu group) — Internet of Agents
   - Stanford NLP (Percy Liang group) — HELM benchmark, LLM evaluation
   - 合作方向：在大規模多代理場景中驗證 Scout

5. **Edge AI / on-device ML / 邊緣 AI**
   - Samsung AI Center — on-device LLM deployment
   - Apple MLR — efficient inference
   - 合作方向：在真實行動裝置上部署 scout model

### Industry Partners / 產業合作

1. **Qualcomm Research**
   - 在 on-device LLM inference 大量投資（AI Engine on Snapdragon）
   - Scout 的 edge model 直接對應到 Qualcomm 的邊緣推論需求
   - 合作方向：在 Snapdragon 8 Gen 3/4 上部署 3B scout model

2. **MediaTek (聯發科)**
   - Dimensity 9400 已支援 on-device LLM（最高 33B parameters with NeuroPilot）
   - 與台大電機系有長期合作
   - 合作方向：Scout protocol 在 MediaTek 晶片上的實作和最佳化

3. **NVIDIA**
   - TensorRT-LLM 是 LLM 服務的工業標準
   - NVIDIA Jetson 是 edge AI 的主要平台
   - 合作方向：在 Jetson + A100/H100 的真實 edge-cloud 環境中驗證

4. **Telecom operators / 電信營運商**
   - 中華電信、遠傳電信、台灣大哥大
   - 5G MEC (Multi-access Edge Computing) 平台
   - 合作方向：在真實 5G MEC 環境中部署 Scout 協定
   - 這是將論文轉化為實際系統的最佳路徑

5. **Cloud LLM providers / 雲端 LLM 提供者**
   - Together AI, Fireworks AI, Groq — disaggregated LLM serving
   - 合作方向：將 Scout 整合到 disaggregated serving 系統中

### Open Source Opportunities / 開源機會

- **vLLM integration**: Scout 可作為 vLLM 的 KV-cache transport plugin
- **HuggingFace integration**: 將 Q2C scoring + scout protocol 做成 HuggingFace 的 PR
- 開源有助於增加論文引用數和社群影響力

### Funding Opportunities / 經費來源

- **NSTC (國科會) 邊緣智慧計畫**: 台灣政府重點支持的研究方向
- **MediaTek-NTU 聯合研究中心**: 既有合作管道
- **Qualcomm Innovation Fellowship**: 年度獎學金，鼓勵 on-device AI 研究
- **IEEE ComSoc Student Travel Grant**: 會議旅費補助
- **Google Research Scholar Program**: 支持早期學術研究者

---

## Summary / 總結

Paper B 的核心賣點是 **Scout 模型範式**和**注意力聚焦效應**——這兩個都是全新的概念，沒有直接的先前工作可比較。28,800x 的壓縮比和 7B 改善 14B 品質 10.2% 這兩個數字具有極高的衝擊力。

**建議投稿順序**：ICC 2027 (Oct 2026) → JSAC special issue (if available) → GLOBECOM 2027 (backup)

**與 Paper A 的最佳配合**：Paper A → INFOCOM (Aug 2026)，Paper B → ICC (Oct 2026)。兩篇互相引用、互相支持，形成完整的「KV-cache compression + adaptive transport」研究故事。

**投稿前最關鍵的改進**：
1. Cross-family scout 實驗（即使結果是負面的也有價值）
2. 更長上下文（4K+ tokens）的 scout 驗證
3. 多任務的 attention focusing 驗證（確認 7B→14B improvement 非任務特定）

**Paper B 的長期願景**：Scout 不僅是一個壓縮技術，更是**邊緣-雲端 LLM 協作推論的通訊協定**。如果搭配真實 5G 環境驗證和系統實作，可以發展為 JSAC 或 TMC 的期刊論文，甚至成為博士論文的核心章節。
