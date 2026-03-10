# Round 0 初始審稿意見彙整

**建立日期**: 2026-03-10
**來源檔案**: reviewer_questions.md, reviewer2_comments.md, reviewer3_questions.md, reviewer4_questions.md, reviewer5_questions.md, reviewer6_questions.md
**說明**: 本文件將六位 reviewer / 討論者的初始回饋按主題分類整理，並標註後續 R1-R4 是否已處理。

---

## 一、基本架構與價值主張（Framing / Value Proposition）

### 1.1 KV-cache 傳輸是稻草人（Strawman Baseline）
- **提出者**: Reviewer 1（reviewer_questions.md）、Reviewer 2（reviewer2_comments.md）
- **內容**: 在無線邊緣場景（5-200 Mbps），直接傳文字（幾 KB）+ 雲端重新 prefill 永遠比傳 KV-cache（數十到數百 MB）快。把 200MB KV-cache 傳輸當 baseline 是假想敵。
- **數據佐證**: Qwen-14B, 1024 tokens, 100 Mbps 下，傳文字 + 重算只需 ~57ms，INT4 KV-cache 傳輸需 669ms，慢 12 倍。
- **R1-R4 處理狀態**: ✅ **已處理**
  - R3: 加入 C2C 引用、修正 payload reduction framing
  - R4: 加入 prompt-only baseline 到 Table 2 + 新增 scout_vs_prompt 表格量化 decode 節省

### 1.2 雲端反正要重算 Prefill，為什麼還需要 Edge？
- **提出者**: Reviewer 1（reviewer_questions.md）
- **內容**: Scout mode 中雲端本來就要自己做 prefill，那為什麼不讓雲端自己做 SnapKV/Q2C 選 token？Edge 跑 3B 是浪費電。
- **反駁**: 實驗數據顯示 7B scout selection 在 aggressive compression（25-50% retention）下比 14B 自己選顯著更好（SQuAD +8.8%, p=0.026）。Edge 跑 3B 是 local inference 的 sunk cost，scout index 是免費副產物。
- **R1-R4 處理狀態**: ✅ **已處理**
  - R4: 更新 fallback hierarchy 為 7 modes（含 prompt-only）；加入 local-first escalation 敘事

### 1.3 論文應從「省頻寬」改為「提升品質」的 Framing
- **提出者**: Reviewer 1（reviewer_questions.md）、Reviewer 2（reviewer2_comments.md）
- **內容**: 核心賣點不應是 28,800x payload reduction，而是 cross-model attention transfer 帶來的品質提升。
- **建議修正**: Baseline 改為「傳文字 + 雲端自己 prefill + 雲端自己做 Q2C/SnapKV」；Scout 的價值是品質提升而非頻寬節省。
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - R3-R4: 降低 28,800x 的 abstract 地位，但仍保留；加入 prompt-only 比較
  - R4 review 仍指出此為核心待解決問題（W1, W3）

---

## 二、實驗方法論（Methodology & Experiments）

### 2.1 Scout 品質提升的統計顯著性問題
- **提出者**: Reviewer 1（reviewer_questions.md 深度分析）
- **內容**: Paper B 宣稱「7B scout 讓 14B 品質提升 10.2% (p=0.018)」，但此為 n=50 的結果。n=200 實驗顯示 75% retention 下完全無效果（p=0.883），只有 25% retention 在 SQuAD 上一致顯著（p=0.039）。
- **影響**: 需將 claim 限縮為「在 aggressive compression（25-50% retention）下，extractive QA 有顯著效果」。
- **R1-R4 處理狀態**: ✅ **已處理**
  - R3-R4: 改用 n=200 數據，修正 claim 範圍

### 2.2 Scout 效果是 Task-Dependent
- **提出者**: Reviewer 1（reviewer_questions.md）、Reviewer 5（reviewer5_questions.md）
- **內容**: SQuAD（extractive QA）上有效，HotpotQA（multi-hop reasoning）上完全無效（所有 retention 都不顯著）。
- **意義**: 論文應明確界定 phenomenon 的適用範圍——shallow saliency 任務有效，deep reasoning 任務無效。
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - R4: 論文承認 limitation，但 R4 review 仍要求 random-k baseline 做 controlled regularization test（W7）

### 2.3 Context Length 太短，與 Motivation 不符
- **提出者**: Reviewer 1（reviewer_questions.md）、R3/R4 review
- **內容**: 論文以 1024-token context（201 MB KV-cache）為 motivation，但實驗平均 context 僅 ~170 tokens（~3.3 MB）。Long context 實驗只做到 3B→7B at 4K，flagship 7B→14B 只到 2K。
- **R1-R4 處理狀態**: ❌ **未完全處理**
  - R4: 仍列為 Tier 2 待修（需要 GPU，14B + output_attentions @ 4K+ tokens 會 OOM）

### 2.4 CacheGen 只比壓縮率，未比品質
- **提出者**: R3/R4 review（此處 Round 0 reviewer 也有提及）
- **內容**: Table 7 比較 Scout 28,800x vs CacheGen 3.5x，但從未實際執行 CacheGen 做 end-to-end 品質比較。
- **R1-R4 處理狀態**: ❌ **未處理**（需 GPU 執行 CacheGen code）

---

## 三、理論一致性（Internal Consistency）

### 3.1 Attention Focusing Effect 與 Entropy 數據矛盾
- **提出者**: Reviewer 1（reviewer_questions.md）、Reviewer 3（reviewer3_questions.md）
- **內容**: 論文宣稱「7B 模型容量較小，注意力更集中 → selection 更好」，但 entropy 數據顯示排序為 3B(4.21) < 14B(4.65) < 7B(5.49)。7B 是最分散的，非最集中。且 3B→14B 的 scout 反而比 14B 自己選差。
- **影響**: 「Attention focusing effect」假說與自身數據 internal inconsistency。
- **建議**: 改為「cross-model selection complementarity」或刪除機制解釋。
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - R3: 弱化 focusing 的用語
  - R4 review 仍指出 regularization 效果未做 controlled validation（需 random-k baseline）

### 3.2 「~100% 品質」的 Claim 混淆兩個 Baseline
- **提出者**: R3/R4 review（Round 0 討論中也觸及）
- **內容**: Table 2 中 Scout 品質標示「~100%」是相對於 cloud 自己做 eviction-based selection 的結果，而非相對於 full-KV（無 eviction）的 baseline。實際相對 full-KV 只有 ~90%。
- **R1-R4 處理狀態**: ✅ **已處理**
  - R3: 修正為 ~90%，加入 footnote 說明

---

## 四、協定設計深度（Protocol Design Depth）

### 4.1 Mode Selection 演算法過於簡單
- **提出者**: Reviewer 2（reviewer2_comments.md 間接提及）、R3/R4 review
- **內容**: 5-mode 選擇（Eq. 7）是暴力枚舉，Proposition 1 和 2 都是 trivially true。對 JSAC 等級期刊而言，networking 貢獻太淺。
- **建議**: 需加入 competitive ratio analysis、regret bounds、queuing model、或至少 BW estimation sensitivity analysis。
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - R4: 加入 Prop 1 bounded quality loss、Poisson arrival extension、BW estimation validation
  - R4 review 仍認為 protocol 貢獻對 JSAC 不夠深（W2）

### 4.2 頻寬估計（BW Estimation）未經驗證
- **提出者**: R3/R4 review（Round 0 未直接提及但相關）
- **內容**: EWMA α=0.3 + 0.8x conservative factor 只是宣稱「20% 誤差不影響」，未在 Lumos5G traces 上實際驗證。
- **R1-R4 處理狀態**: ✅ **已處理**
  - R4: 新增 exp_bw_estimation 實驗 + bw_validation 和 alpha_sensitivity 兩張表

### 4.3 Multi-Agent 模擬過於理想化
- **提出者**: R3/R4 review（Round 0 間接相關）
- **內容**: 500-round 模擬假設同步請求、靜態頻寬、同質 deadline。真實場景涉及 Poisson arrival、priority queuing、干擾驅動的頻寬波動。
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - R4: 加入 Poisson arrival extension 說明

---

## 五、論文定位與範疇（Scope & Positioning）

### 5.1 Paper A（KV-cache Compression）的 Motivation 需微調
- **提出者**: Reviewer 1（reviewer_questions.md）
- **內容**: Paper A 的核心貢獻是 compression characterization（Q2C selection、INT8 universally lossless、mixed-precision），但 motivation 只聚焦 wireless edge-cloud KV-cache transmission。應同時涵蓋 datacenter disaggregated serving 和 memory management（long context serving）場景。
- **R1-R4 處理狀態**: ✅ **已處理**（JSAC 合併版已涵蓋多場景 motivation）

### 5.2 工程論文 vs 模型行為分析論文的定位抉擇
- **提出者**: Reviewer 5（reviewer5_questions.md）
- **內容**: 如果定位為工程方法論文，reviewer 會攻「任務不泛化」；如果定位為模型行為分析論文，reviewer 會問「生成任務呢？」。需要做 strategic choice。
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - JSAC 版本定位為 systems/protocol paper，但 R4 review 認為對 JSAC 的 networking 深度不足

### 5.3 多裝置協同（Multi-Agent RAG Fusion）方向的評估
- **提出者**: Reviewer 2（reviewer2_comments.md）
- **內容**: 建議改為多台 Edge 做分散式特徵提取，各自傳 attention index 給 Cloud 做 fusion。
- **評估結果**: 技術上不可行——不同 edge 處理不同 context，attention index 跨 token space 無法對齊。且會丟棄所有現有實驗數據。改 framing 是更務實的修法。
- **R1-R4 處理狀態**: ❌ **未採用**（正確決定，改採 reframing 方案）

---

## 六、基礎概念釐清（Conceptual Clarifications）

### 6.1 82% Overlap 的實際意義
- **提出者**: Reviewer 4（reviewer4_questions.md）
- **內容**: 82% overlap 本身是否只是 observation？有什麼實際價值？
- **回答**:
  - 反直覺：不同 scale 模型的 early attention ranking 高度一致
  - 工程意義：enable cross-model proxy signal、cascade inference、context pruning
  - 理論意義：token importance 可能由 input structure（lexical overlap、discourse structure）主導，而非模型 capacity
  - 但單獨 overlap 不足以撐論文，需搭配品質影響數據
- **R1-R4 處理狀態**: ✅ **已充分闡述於論文中**

### 6.2 Overlap 僅在 Extractive 任務成立的局限性
- **提出者**: Reviewer 5（reviewer5_questions.md）
- **內容**: 82% overlap 主要在 shallow saliency（extractive QA）任務成立；multihop 失效；生成任務未驗證。這是「條件性成立的 observation」，非 universal principle。
- **正確 framing**: "Early token saliency across model scale is highly aligned for tasks dominated by lexical grounding."
- **R1-R4 處理狀態**: ⚠️ **部分處理**
  - 論文 Section VII-F 列出 limitation，但未在 abstract/intro 充分限定適用範圍

### 6.3 KV-cache / Prefill / Attention 的基礎解釋
- **提出者**: Reviewer 6（reviewer6_questions.md）
- **內容**: 詳細解釋 KV-cache 建立流程、prefill vs decode 階段、attention overlap 的量測時機（生成前 prefill attention）、小模型 vs 大模型的 capacity 差異如何影響 reasoning。
- **R1-R4 處理狀態**: N/A（背景知識討論，非論文修改項目）

---

## 七、Edge 裝置相關問題

### 7.1 Edge 信心判斷（Escalation Decision）機制
- **提出者**: Reviewer 1（reviewer_questions.md Q&A 補充）
- **內容**: 論文假設 edge 3B 「判斷我不行」後 escalate 到 cloud，但沒有具體的 confidence estimation 機制。
- **建議處理方式**: 不宣稱有自動 escalation 機制。寫明 escalation decision 是 system-level 設計選擇，非本論文 scope。Scout protocol 在 escalation decision 已做出後運作。
- **R1-R4 處理狀態**: ✅ **已處理**
  - 論文採用「orthogonal to our contribution」的寫法，列為 future work

---

## 八、待解決項目總覽（截至 R4）

| # | 問題 | 來源 | 狀態 | 需要資源 |
|---|------|------|------|---------|
| 1 | CacheGen 直接品質比較 | R3/R4 | ❌ 未處理 | GPU |
| 2 | Random-k baseline（regularization 控制實驗） | R4 W7 | ❌ 未處理 | GPU |
| 3 | 7B→14B at 4K+ tokens | R4 W5 | ❌ 未處理 | GPU（OOM 限制） |
| 4 | Protocol 深度不足（competitive ratio / regret bounds） | R3/R4 W2 | ⚠️ 部分處理 | 理論推導 |
| 5 | 28,800x claim 仍過於突出 | R4 W3 | ⚠️ 部分處理 | 文字修改 |
| 6 | Multi-agent 模擬加入 Poisson arrival | R4 W6 | ⚠️ 部分處理 | 模擬程式 |
| 7 | 生成任務的 overlap 驗證 | Reviewer 5 | ❌ 未處理 | GPU + 實驗設計 |

---

*本文件僅為彙整摘要，原始六份檔案保持不變。*
