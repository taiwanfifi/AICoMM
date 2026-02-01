# Agent-Oriented Semantic Communication for 6G Networks

**博士論文研究 | 國立台灣大學電機工程學系**
**指導教授：廖婉君 教授（台大副校長）**

---

## 研究核心問題

> **當 AI Agents 成為未來網路的主要通訊實體時，網路應該如何演進？**

傳統通訊網路設計用於傳輸 **bits/packets**，目標是無損還原。但在 Agent-to-Agent 通訊場景中：
- 通訊實體不再是人，而是具備推理能力的 AI Agents
- 通訊目標不再是 bit recovery，而是 **task success**
- 通訊單位不應該是 packet，而應該是 **semantic state / token**

**本研究提出：Token-Based Communication Protocol for Agent Networks**

---

## 核心創新點（與現有研究的區隔）

### ❌ 不是傳統 Semantic Communication
- 傳統語義通訊：傳送 feature vector 取代 raw data（仍是 data transmission）
- **我們的方向**：傳送 **semantic state delta**（state synchronization）

### ❌ 不是 Agent Framework 應用
- LangChain/AutoGen：假設頻寬無限，傳送大量 JSON/Prompt（忽略通訊成本）
- **我們的方向**：設計 **通訊協定**，在頻寬受限下實現 Agent 協作

### ✅ 核心貢獻
1. **Token-Based Transmission**：傳輸單位從 Packet → Semantic Token
2. **Attention-Based Filtering**：Source 端用 Attention 機制決定「什麼值得傳」
3. **Task-Oriented Protocol**：傳輸決策由任務目標驅動，而非傳統的 QoS 指標
4. **Control/Data Plane 分離**：Control Plane 對齊任務與解碼格式，Data Plane 傳輸 Token

---

## 研究架構（三層抽象）

```
┌─────────────────────────────────────────────────┐
│  Layer 5+: Semantic Protocol Layer              │
│  - Control Plane (任務協商、模型對齊)                │
│  - Data Plane (Token 傳輸)                       │
└─────────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────┐
│  Agent Layer (Application)                      │
│  - Source: Edge Agent (感知、狀態生成)             │
│  - Receiver: Cloud Agent (任務執行、決策)          │
└─────────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────┐
│  Infrastructure (6G/O-RAN)                      │
│  - 物理層仍傳輸 0/1，但封裝的是 Semantic Token      │
└─────────────────────────────────────────────────┘
```

---

## 關鍵技術組件

### 1. Semantic State Representation
- **不傳送**：Raw pixels, compressed video
- **傳送**：Agent 內部的 KV-Cache / Latent State
- **時序**：不是 frame-based，而是 token-based

### 2. Attention-Based Source Filtering
- 借鑒 **DeepSeek-V3.2 的 Sparse Attention (DSA)** 機制
- Semantic Indexer（基於 DSA Lightning）：計算 Query (任務) 與 Key (狀態) 的匹配度
- **只傳 Top-k 的 task-critical tokens**

### 3. Control Plane (參考 Anthropic MCP 概念)
- **不是** Application Layer 的 API 呼叫
- **而是** 通訊協定層的 signaling/handshake
- 功能：任務對齊、模型參數協商、解碼格式同步

---

## 目標評估指標

### 傳統通訊指標 vs. Agent 通訊指標

| 維度 | 傳統網路 | Agent 網路 (本研究) |
|------|---------|-------------------|
| **傳輸單位** | Packet | Semantic Token |
| **成功標準** | BER, Throughput | Task Success Rate |
| **資源分配** | QoS (latency/bandwidth) | Goal-Oriented Scheduling |
| **評估場景** | File transfer, Video streaming | Multi-agent collaboration, Decision making |

### 實驗設計
- **Baseline**: H.264 video compression + traditional packet transmission
- **Proposed**: Attention-filtered token transmission
- **Metrics**:
  - Task accuracy under bandwidth constraint
  - Latency (time-to-decision)
  - Spectrum efficiency (bits per task success)

---

## 文獻定位（投稿目標）

### Target Venues
- **IEEE INFOCOM** (A*, 通訊網路頂會)
- **IEEE ICC** (通訊系統)
- **ACM SIGCOMM** (如果能凸顯網路協定創新)

### 研究角度
- **不是**：AI for Communications (用 AI 優化傳統通訊)
- **而是**：Communications for AI (為 AI Agent 重新設計通訊)

### 相關但不同的領域
- **Semantic Communication** (JSCC, Task-oriented comm): 我們是 state sync，不是 feature transmission
- **ISAC** (Integrated Sensing and Communication): 我們關注的是 agent reasoning，不是 sensing
- **Network Slicing/Edge Computing**: 我們提出新的協定層，不是資源分配

---

## Repository 結構說明

詳見各子目錄的 README：

- `00-advisor-feedback/`: 教授指導與溝通紀錄
- `01-problem-formulation/`: 研究問題定義、動機、挑戰
- `02-core-framework/`: Semantic State Communication 核心框架
- `03-technical-design/`: 協定設計、實現機制
- `04-background/`: 背景文獻、相關工作調研
- `05-evaluation/`: 實驗設計、評估指標
- `06-implementation/`: 實作規格
- `07-paper-drafts/`: 論文寫作（按章節組織）
- `08-code/`: 模擬、原型、評估程式碼
- `09-project-logs/`: 階段完成報告、狀態紀錄
- `tools/`: AI Agent 工具與方法論
- `archive/`: 已歸檔的舊版本想法與原始文件

---

## 當前進度與 Next Steps

### ✅ 已完成
1. 核心概念收斂：Token-based transmission + Attention filtering
2. 理論框架確立：Semantic State Communication (SSC)
3. 技術路徑選定：DeepSeek DSA 作為實現機制
4. 問題定位清晰：與教授反饋對齊

### 🚧 進行中
1. Problem formulation 正式化
2. 系統架構詳細設計
3. 模擬實驗環境搭建（MobileVLM + custom scheduler）

### 📋 待完成
1. Related work 完整調研（JSCC, Task-oriented comm, MCP protocols）
2. 數學模型建立（optimization problem formulation）
3. 實驗數據收集與分析
4. 論文初稿撰寫

---

## 重要文件索引

- **CLAUDE.md**: 給 Claude Code 的 context（如何協助本研究）
- **ROADMAP.md**: 論文寫作時間表與 milestone
- `01-problem-formulation/research-question.md`: 核心研究問題的正式定義
- `02-core-framework/semantic-state-sync.md`: SSC 框架的數學描述
- `03-technical-design/attention-filtering.md`: Attention-based filtering 的設計細節

---

## Citation（暫定格式）

```
@phdthesis{your-thesis-2026,
  title={Token-Based Communication Protocols for Agent-Oriented 6G Networks},
  author={Your Name},
  school={National Taiwan University, Department of Electrical Engineering},
  year={2026},
  advisor={Wan-Chun Liao}
}
```

---

**Last Updated**: 2026-01-24
**Contact**: [Your Email] | [GitHub/Lab Page]
