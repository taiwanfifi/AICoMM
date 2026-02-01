# 論文寫作 Roadmap

**目標論文**：Token-Based Communication Protocol for Agent-Oriented 6G Networks
**投稿目標**：IEEE INFOCOM 2027 / IEEE ICC 2027
**指導教授**：廖婉君教授

---

## Milestone 概覽

```
2026/01 ────► 2026/03 ────► 2026/05 ────► 2026/07 ────► 2026/09 ────► 2026/11
   │              │              │              │              │              │
Problem      Framework      Technical    Implementation   Evaluation    Paper
Formulation   Design         Design        & Simulation     & Analysis   Writing
```

---

## Phase 1: Problem Formulation（2026/01/24 - 2026/02/28）

### 目標
明確定義研究問題，區分與現有工作的差異

### Deliverables

#### Week 1-2（1/24 - 2/7）
- [ ] **Research Question 正式化**
  - 核心問題陳述
  - 與 Semantic Communication 的差異
  - 與 JSCC/ISAC 的區別
  - 輸出：`01-problem-formulation/research-question.md`

- [ ] **Motivation 寫作**
  - 為什麼 Agent 通訊不同於人類通訊
  - 現有方法的限制（LangChain/AutoGen 忽略通訊成本）
  - 6G 時代的需求
  - 輸出：`01-problem-formulation/motivation.md`

#### Week 3-4（2/8 - 2/21）
- [ ] **Related Work 調研**
  - Semantic Communication (10 篇核心論文)
  - Task/Goal-oriented Communication (5 篇)
  - Agent Communication Protocols (MCP, A2A, etc.)
  - 輸出：`04-background/related-work/` 各子文件

- [ ] **Contribution 定位**
  - 與 SOTA 的差異表格
  - 本研究的 3-4 個核心貢獻
  - 輸出：`01-problem-formulation/contributions.md`

#### Week 5（2/22 - 2/28）
- [ ] **與教授 Meeting**
  - 準備一頁 summary（核心問題 + 貢獻）
  - 確認研究定位無誤
  - 取得繼續深入的許可

### Success Criteria
✅ 能用 3 句話說清楚研究問題
✅ 教授認可這個問題值得研究
✅ Related work 清楚定位我們的創新點

---

## Phase 2: Framework Design（2026/03/01 - 2026/04/15）

### 目標
建立 Semantic State Communication (SSC) 的理論框架

### Deliverables

#### Week 6-7（3/1 - 3/14）
- [ ] **Semantic State 數學定義**
  - State representation（KV-Cache, Latent vector）
  - State delta 的計算
  - 時序模型（token-based timeline）
  - 輸出：`02-core-framework/semantic-state-sync.md`

- [ ] **Communication Paradigm**
  - 傳統：Source → Encoder → Channel → Decoder
  - 提出：Shared World Model → State Δ → Sync
  - 輸出：`02-core-framework/communication-paradigm.md`

#### Week 8-9（3/15 - 3/28）
- [ ] **System Architecture**
  - Source (Edge Agent) 架構
  - Receiver (Cloud Agent) 架構
  - Control/Data Plane 分離
  - 輸出：`02-core-framework/architecture-overview.md`

- [ ] **Protocol Layer 設計**
  - Layer 5+ 的定義
  - 與 OSI 模型的關係
  - Signaling/Handshake 流程
  - 輸出：`02-core-framework/protocol-layers.md`

#### Week 10（3/29 - 4/15）
- [ ] **數學模型建立**
  - Optimization problem formulation
  - Objective function（Task success vs. Bandwidth）
  - Constraints（Latency, QoS）
  - 輸出：`02-core-framework/mathematical-model.md`

### Success Criteria
✅ 有完整的系統模型（Source, Channel, Receiver）
✅ 有明確的優化目標
✅ 能畫出清楚的架構圖

---

## Phase 3: Technical Design（2026/04/16 - 2026/05/31）

### 目標
設計具體的技術實現方案

### Deliverables

#### Week 11-12（4/16 - 4/30）
- [ ] **Attention-Based Filtering**
  - DeepSeek DSA 的應用
  - Semantic Indexer (DSA Lightning) → Semantic Pilot Channel
  - Top-k selection mechanism
  - 輸出：`03-technical-design/attention-filtering.md`

- [ ] **Token Representation**
  - Token encoding/decoding
  - Compression vs. State extraction
  - 與 Vector Quantization 的關係
  - 輸出：`03-technical-design/token-representation.md`

#### Week 13-14（5/1 - 5/15）
- [ ] **Control Plane Protocol**
  - Task negotiation
  - Model alignment
  - Schema synchronization
  - 輸出：`03-technical-design/control-plane.md`

- [ ] **Data Plane Protocol**
  - Token streaming mechanism
  - Flow control
  - Error handling
  - 輸出：`03-technical-design/data-plane.md`

#### Week 15（5/16 - 5/31）
- [ ] **Implementation Strategy**
  - 技術棧選擇（PyTorch, Ray, etc.）
  - 模擬環境設計
  - Baseline 實現
  - 輸出：`03-technical-design/implementation-notes.md`

### Success Criteria
✅ 技術設計足夠詳細，可以開始 coding
✅ 有清楚的算法 pseudocode
✅ 與教授確認技術路線可行

---

## Phase 4: Implementation & Simulation（2026/06/01 - 2026/07/31）

### 目標
實現原型系統，進行模擬實驗

### Deliverables

#### Week 16-18（6/1 - 6/21）
- [ ] **Baseline 實現**
  - Traditional video transmission (H.264)
  - JSCC-based semantic communication
  - 測試環境搭建

- [ ] **Proposed Method 實現**
  - Attention filtering module
  - Token encoding/decoding
  - Control plane simulator

#### Week 19-21（6/22 - 7/12）
- [ ] **實驗場景設計**
  - Scenario 1: Autonomous driving (video + control)
  - Scenario 2: Multi-agent collaboration
  - Scenario 3: Edge-cloud offloading

- [ ] **數據收集**
  - Task success rate under varying bandwidth
  - Latency analysis
  - Spectrum efficiency

#### Week 22（7/13 - 7/31）
- [ ] **結果分析**
  - 統計顯著性檢驗
  - 與 baseline 的對比
  - Ablation study（各組件的貢獻）

### Success Criteria
✅ 至少 3 組實驗結果
✅ 能證明在低頻寬下優於 baseline
✅ 有可視化的結果圖表

---

## Phase 5: Evaluation & Analysis（2026/08/01 - 2026/09/15）

### 目標
深入分析實驗結果，準備論文的 evaluation section

### Deliverables

#### Week 23-24（8/1 - 8/15）
- [ ] **Performance Evaluation**
  - Throughput vs. Task accuracy
  - Latency breakdown analysis
  - Bandwidth efficiency

- [ ] **Scalability Analysis**
  - 不同 agent 數量的影響
  - 不同網路條件（delay, loss）

#### Week 25-26（8/16 - 8/31）
- [ ] **Comparison Study**
  - vs. Traditional communication
  - vs. JSCC methods
  - vs. Existing agent frameworks (if applicable)

- [ ] **Discussion**
  - 為什麼 attention filtering 有效
  - Limitation 與未來改進
  - Real-world deployment challenges

#### Week 27（9/1 - 9/15）
- [ ] **Finalize Evaluation**
  - 所有圖表製作完成
  - 統計數據整理
  - 輸出：`05-evaluation/results.md`

### Success Criteria
✅ 實驗結果足以支撐論文宣稱的貢獻
✅ 有 honest discussion of limitations
✅ 教授認可實驗的完整性

---

## Phase 6: Paper Writing（2026/09/16 - 2026/11/15）

### 目標
組裝完整論文，準備投稿

### Deliverables

#### Week 28-30（9/16 - 10/7）
- [ ] **Introduction**（3-4 頁）
  - Motivation
  - Research gap
  - Contributions
  - Paper organization

- [ ] **Related Work**（2-3 頁）
  - Semantic communication
  - Agent communication
  - Edge computing
  - 與本研究的關係

#### Week 31-33（10/8 - 10/28）
- [ ] **System Model & Problem Formulation**（2-3 頁）
  - System architecture
  - Mathematical model
  - Problem statement

- [ ] **Proposed Method**（4-5 頁）
  - Semantic State Communication framework
  - Attention-based filtering
  - Protocol design
  - Algorithm pseudocode

#### Week 34-36（10/29 - 11/18）
- [ ] **Evaluation**（3-4 頁）
  - Experimental setup
  - Results & analysis
  - Comparison with baselines

- [ ] **Conclusion**（1 頁）
  - Summary of contributions
  - Future work

#### Week 37-38（11/19 - 11/30）
- [ ] **Polish & Review**
  - Abstract 寫作
  - 所有圖表編號與 caption
  - Reference 格式統一
  - 內部 review（實驗室同學）

#### Week 39（12/1 - 12/7）
- [ ] **Submit to Advisor**
  - 教授 review
  - 根據反饋修改

### Success Criteria
✅ 論文長度符合 conference 要求（通常 6-8 頁）
✅ 所有 claim 都有實驗支撐
✅ 通過教授的審核

---

## Phase 7: Submission & Response（2026/12 onwards）

### Target Conferences

| Conference | Deadline | Notification | Conference Date |
|------------|----------|--------------|----------------|
| IEEE INFOCOM 2027 | 2026/08 | 2026/12 | 2027/05 |
| IEEE ICC 2027 | 2026/10 | 2027/01 | 2027/06 |
| ACM MobiCom 2027 | 2027/03 | 2027/06 | 2027/10 |

### Submission Checklist
- [ ] 論文符合格式要求（IEEE template）
- [ ] Blind review 規則（移除作者資訊）
- [ ] Supplementary materials（如果允許）
- [ ] Copyright form

### If Rejected
- [ ] 仔細閱讀 reviewer comments
- [ ] 分類 comments（可接受 vs. 需辯護）
- [ ] 準備 rebuttal（如果有 rebuttal phase）
- [ ] 根據 feedback 改進後投下一個 venue

---

## 關鍵 Checkpoint（必須與教授討論）

### Checkpoint 1: Problem Formulation（2月底）
**問題**：
- 這個研究問題夠重要嗎？
- 與現有工作的區別清楚嗎？
- 投稿 venue 選對了嗎？

### Checkpoint 2: Framework Design（4月中）
**問題**：
- 理論框架完整嗎？
- 數學模型合理嗎？
- 有沒有明顯的 flaw？

### Checkpoint 3: Technical Design（5月底）
**問題**：
- 技術路線可行嗎？
- Baseline 選擇合適嗎？
- 實驗設計能證明貢獻嗎？

### Checkpoint 4: Initial Results（7月底）
**問題**：
- 實驗結果符合預期嗎？
- 如果不符合，需要調整什麼？
- 還需要哪些補充實驗？

### Checkpoint 5: Paper Draft（11月初）
**問題**：
- 論文邏輯清楚嗎？
- Contribution 的描述準確嗎？
- 需要加強哪些部分？

---

## 風險管理

### 潛在風險與對策

| 風險 | 可能性 | 影響 | 對策 |
|------|-------|------|------|
| 實驗結果不如預期 | 中 | 高 | 提早開始實驗，留時間調整 |
| Related work 發現類似研究 | 中 | 高 | 持續追蹤最新論文，及時調整定位 |
| 實現技術困難 | 中 | 中 | 簡化場景，focus on proof-of-concept |
| Reviewer 認為不夠 novel | 低 | 高 | 在 introduction 強調與 SOTA 的差異 |
| 時間不足 | 中 | 中 | 每月檢查進度，必要時調整 scope |

---

## 進度追蹤

### 每週回顧（建議格式）
```markdown
## Week X Progress Report

### Completed
- [ ] Task 1
- [ ] Task 2

### In Progress
- [ ] Task 3 (50% done)

### Blockers
- Issue 1: ...
- Solution: ...

### Next Week Plan
- [ ] Task 4
- [ ] Task 5
```

### 每月總結（與教授 meeting）
- 本月主要成果
- 遇到的挑戰與解決方案
- 下個月計畫
- 需要教授協助的事項

---

## Resource Tracking

### 計算資源需求
- GPU: NVIDIA A100 (for simulation) - 預估 100 GPU hours
- Storage: 500 GB (for datasets & results)
- Cloud credits: $500 (if using AWS/GCP)

### 文獻資源
- IEEE Xplore subscription ✓
- ACM Digital Library ✓
- arXiv.org ✓

### 工具與軟體
- LaTeX (Overleaf) ✓
- Python 3.10+ ✓
- PyTorch 2.0+ ✓
- Git/GitHub ✓

---

**Last Updated**: 2026-01-24
**Next Review**: 2026-02-28（Checkpoint 1）
