# Repository 重構計劃

**目標**：將研究文檔從「發散探索」整理為「收斂論文」的清晰結構

---

## 一、診斷：現有文件的問題

### 現況分析

| 文件 | 內容性質 | 問題 | 處理建議 |
|------|---------|------|---------|
| `professor_concepts.md` | ⭐ 核心指導 | 中英混雜，是對話紀錄而非結構化文檔 | **保留原檔**，提取關鍵洞察到新文件 |
| `t1.md` | 舊方向 | 6G/O-RAN automation，教授已明確說「太 application」 | **歸檔**（錯誤方向，但可作為 related work） |
| `t2.md` | 舊方向 | Edge-cloud RAG，格局太小（只是壓縮） | **歸檔**（技術可用，但不是核心貢獻） |
| `t3.md` | ⭐⭐⭐ 核心 | Semantic State Communication 框架 | **重構為正式文檔**，這是論文核心 |
| `t4.md` | 診斷報告 | 分析什麼對/什麼錯 | **提取判決結果**，刪除過程討論 |
| `t5.md` | 確認收斂 | 確認研究方向正確 | **提取關鍵論點**，刪除重複內容 |
| `t6.md` | ⭐ 技術實現 | DeepSeek DSA → Attention-based Protocol | **重構為技術設計文檔** |
| `t7.md` | 演進分析 | 分析不同版本的差異 | **歸檔**（過程紀錄，不需要在論文中） |
| `agent.md` | 背景文獻 | FM-powered agent services 調研 | **移至 background/**，作為 related work |
| `IOA.md` | 背景文獻 | Internet of Agents 架構 | **移至 background/**，作為 related work |
| `deepseek.md` | 背景文獻 | DeepSeek 技術細節 | **移至 background/**，作為技術參考 |

---

## 二、建議的新目錄結構

```
AI-Comm/
│
├── README.md                          # 主要說明（已創建）
├── CLAUDE.md                          # AI assistant context（已更新）
├── ROADMAP.md                         # 論文寫作時間表（待創建）
├── .gitignore                         # Git 忽略規則（建議新增）
│
├── 00-advisor-feedback/               # 教授指導紀錄
│   └── professor-concepts-raw.md      # 原始對話紀錄（改名自 professor_concepts.md）
│
├── 01-problem-formulation/            # 問題定義（論文 Section 1）
│   ├── README.md                      # 本目錄說明
│   ├── research-question.md           # 核心研究問題
│   ├── motivation.md                  # 研究動機（從 professor_concepts 提取）
│   ├── challenges.md                  # 技術挑戰
│   └── contributions.md               # 本研究的貢獻（與 SOTA 的區別）
│
├── 02-core-framework/                 # 核心框架（論文 Section 3-4）
│   ├── README.md
│   ├── semantic-state-sync.md         # SSC 框架（從 t3.md 重構）
│   ├── communication-paradigm.md      # 新的通訊範式定義
│   ├── architecture-overview.md       # 系統架構總覽
│   └── protocol-layers.md             # 協定層設計（Layer 5+）
│
├── 03-technical-design/               # 技術設計（論文 Section 4-5）
│   ├── README.md
│   ├── token-representation.md        # Token 如何表示 semantic state
│   ├── attention-filtering.md         # Attention-based source filtering（從 t6.md）
│   ├── control-plane.md               # Control Plane 設計（MCP-like）
│   ├── data-plane.md                  # Data Plane 設計（Token streaming）
│   └── implementation-notes.md        # 實現細節（DeepSeek DSA 應用）
│
├── 04-background/                     # 背景文獻（論文 Section 2）
│   ├── README.md
│   ├── papers/                        # PDF 文獻
│   │   ├── foundation-model-survey.pdf
│   │   ├── deepseek-v32.pdf
│   │   └── (其他 PDF)
│   ├── related-work/
│   │   ├── semantic-communication.md  # JSCC, Task-oriented comm
│   │   ├── agent-communication.md     # MCP, A2A protocols
│   │   ├── edge-computing.md          # Edge-cloud collaboration
│   │   └── 6g-networks.md             # O-RAN, network slicing
│   └── technical-background/
│       ├── agent-services.md          # 從 agent.md 移動
│       ├── internet-of-agents.md      # 從 IOA.md 移動
│       └── deepseek-architecture.md   # 從 deepseek.md 移動
│
├── 05-evaluation/                     # 評估設計（論文 Section 6）
│   ├── README.md
│   ├── metrics.md                     # 評估指標定義
│   ├── baselines.md                   # 對比基準（H.264, JSCC 等）
│   ├── scenarios.md                   # 實驗場景
│   └── expected-results.md            # 預期結果與分析
│
├── 06-paper-drafts/                   # 論文寫作
│   ├── README.md
│   ├── outline.md                     # 論文大綱
│   ├── abstract.md                    # 摘要
│   ├── introduction.md                # Introduction
│   ├── related-work.md                # Related Work
│   ├── problem-formulation.md         # Problem Statement
│   ├── methodology.md                 # Proposed Method
│   ├── evaluation.md                  # Evaluation
│   ├── discussion.md                  # Discussion
│   ├── conclusion.md                  # Conclusion
│   └── figures/                       # 論文圖表
│
├── 07-code/                           # 程式碼（未來實作）
│   ├── README.md
│   ├── simulation/                    # 模擬實驗
│   ├── prototype/                     # 原型實現
│   └── evaluation/                    # 評估腳本
│
└── archive/                           # 歸檔（已廢棄的方向）
    ├── README.md                      # 說明為何歸檔
    ├── old-directions/
    │   ├── t1-oran-automation.md      # 從 t1.md 移動
    │   ├── t2-edge-rag.md             # 從 t2.md 移動
    │   └── t4-diagnosis.md            # 從 t4.md 移動（過程紀錄）
    └── evolution-logs/
        ├── t5-convergence.md          # 從 t5.md 移動
        └── t7-version-comparison.md   # 從 t7.md 移動
```

---

## 三、具體執行步驟

### Phase 1: 備份與清理（第 1 天）

```bash
# 1. 創建備份
mkdir -p backup
cp *.md backup/

# 2. 創建新目錄結構
mkdir -p 00-advisor-feedback
mkdir -p 01-problem-formulation
mkdir -p 02-core-framework
mkdir -p 03-technical-design
mkdir -p 04-background/{papers,related-work,technical-background}
mkdir -p 05-evaluation
mkdir -p 06-paper-drafts/figures
mkdir -p 07-code/{simulation,prototype,evaluation}
mkdir -p archive/{old-directions,evolution-logs}

# 3. 移動 PDF 文件
mv *.pdf 04-background/papers/
```

### Phase 2: 文件重構（第 2-3 天）

#### 優先級 1：核心框架（最重要）

**從 t3.md 提取並重構**：
- `02-core-framework/semantic-state-sync.md`
  - 提取「Semantic State Communication」的定義
  - 提取「Token vs Packet」的對比
  - 提取「World State Synchronization」概念

**從 t6.md 提取並重構**：
- `03-technical-design/attention-filtering.md`
  - 提取 DeepSeek DSA 的應用
  - Lightning Indexer → Communication Protocol 的映射
  - Top-k selection 機制

#### 優先級 2：問題定義

**從 professor_concepts.md 提取**：
- `01-problem-formulation/motivation.md`
  - 教授的核心洞察：「傳的是 Token，不是 Packet」
  - 現有 Agent 框架忽略通訊成本的問題
  - 6G 時代的需求

**新創建**：
- `01-problem-formulation/research-question.md`
  - 正式定義研究問題
  - 與傳統 Semantic Communication 的差異
  - 與 ISAC/JSCC 的區別

#### 優先級 3：背景文獻整理

**移動現有文件**：
```bash
mv agent.md 04-background/technical-background/agent-services.md
mv IOA.md 04-background/technical-background/internet-of-agents.md
mv deepseek.md 04-background/technical-background/deepseek-architecture.md
```

**新創建 Related Work**：
- `04-background/related-work/semantic-communication.md`
  - JSCC (Joint Source-Channel Coding)
  - Task-oriented communication
  - Goal-oriented networking

#### 優先級 4：歸檔舊方向

```bash
mv t1.md archive/old-directions/t1-oran-automation.md
mv t2.md archive/old-directions/t2-edge-rag.md
mv t4.md archive/evolution-logs/t4-diagnosis.md
mv t5.md archive/evolution-logs/t5-convergence.md
mv t7.md archive/evolution-logs/t7-version-comparison.md
mv professor_concepts.md 00-advisor-feedback/professor-concepts-raw.md
```

### Phase 3: 創建論文大綱（第 4 天）

在 `06-paper-drafts/outline.md` 創建完整的論文結構，基於：
- INFOCOM/ICC 的論文格式
- 核心貢獻的邏輯順序
- Related work 的定位

---

## 四、關鍵文件創建指南

### 1. `01-problem-formulation/research-question.md`

**必須回答的問題**：
- What: 我們研究什麼問題？（Token-based agent communication）
- Why: 為什麼現有方法不夠？（忽略通訊成本 / 仍是 data transmission）
- How: 我們的方法與眾不同在哪？（State synchronization + Attention filtering）

**結構**：
```markdown
# Core Research Question

## Problem Statement
When AI agents become the primary communication entities...

## Research Gap
- Traditional networks: Bit recovery
- Semantic communication: Feature transmission
- **Our focus**: State synchronization

## Research Question
How to design a communication protocol that...

## Scope
- In scope: ...
- Out of scope: ...
```

### 2. `02-core-framework/semantic-state-sync.md`

**從 t3.md 提取的核心內容**：
- Semantic State 的數學定義
- Synchronization 機制
- 與傳統通訊的對比表格

**新增內容**：
- 正式化的符號系統
- 系統模型（Source, Channel, Receiver）
- 優化目標函數

### 3. `03-technical-design/attention-filtering.md`

**從 t6.md 提取**：
- DSA 的 Lightning Indexer 機制
- Score 計算公式
- Top-k selection

**改寫角度**：
- 不是「用 DeepSeek 做壓縮」
- 而是「Attention 作為 Task-oriented Filter 的理論基礎」

---

## 五、寫作優先級與時間分配

### Week 1-2：Problem Formulation（基礎）
- [ ] `research-question.md` - 2 天
- [ ] `motivation.md` - 1 天
- [ ] `contributions.md` - 1 天
- [ ] `06-paper-drafts/introduction.md` 初稿 - 3 天

### Week 3-4：Core Framework（核心）
- [ ] `semantic-state-sync.md` - 3 天
- [ ] `architecture-overview.md` - 2 天
- [ ] `protocol-layers.md` - 2 天

### Week 5-6：Technical Design（實現）
- [ ] `attention-filtering.md` - 3 天
- [ ] `control-plane.md` - 2 天
- [ ] `data-plane.md` - 2 天

### Week 7-8：Related Work & Evaluation（定位）
- [ ] Related work 調研與寫作 - 5 天
- [ ] Evaluation 設計 - 2 天

### Week 9-10：Paper Assembly（組裝）
- [ ] 論文初稿整合
- [ ] 圖表製作
- [ ] 數學推導檢查

---

## 六、每個目錄需要的 README.md

### `01-problem-formulation/README.md`
```markdown
# Problem Formulation

本目錄定義核心研究問題。

## 核心問題
當 AI Agents 成為主要通訊實體，傳統的 bit-oriented 網路應如何演進？

## 文件說明
- `research-question.md`: 正式的問題定義
- `motivation.md`: 為什麼這個問題重要
- `challenges.md`: 技術挑戰
- `contributions.md`: 我們的貢獻

## 與論文的對應
→ Paper Section 1 (Introduction)
→ Paper Section 3 (Problem Statement)
```

### `02-core-framework/README.md`
```markdown
# Core Framework: Semantic State Communication

本目錄包含本研究的核心理論框架。

## 核心概念
Semantic State Communication (SSC): 通訊的目的是同步 semantic state，而非傳輸 data。

## 文件說明
- `semantic-state-sync.md`: SSC 的數學定義與機制
- `communication-paradigm.md`: 新舊範式對比
- `architecture-overview.md`: 系統架構
- `protocol-layers.md`: Layer 5+ 協定設計

## 與論文的對應
→ Paper Section 3-4 (Methodology)
```

---

## 七、Git 版本控制建議

創建 `.gitignore`:
```
# System files
.DS_Store
*.swp
*~

# Backup
backup/
*.backup

# IDE
.vscode/
.idea/

# Temporary
temp/
tmp/

# Large PDFs (optional, 可考慮用 Git LFS)
# *.pdf
```

建議的 commit message 格式：
```
[SECTION] Brief description

- Detailed change 1
- Detailed change 2

Related: #issue-number
```

---

## 八、質量檢查清單

### 每個新文件創建時：
- [ ] 有清楚的標題與目的說明
- [ ] 使用一致的術語（參考 CLAUDE.md 的定義）
- [ ] 有與論文 section 的對應說明
- [ ] 數學符號使用一致（建立符號表）
- [ ] 引用文獻格式統一

### 論文章節寫作時：
- [ ] 每段都有 topic sentence
- [ ] 圖表都有編號與 caption
- [ ] 所有縮寫都在第一次出現時定義
- [ ] 與教授反饋對齊（不提 MCP 作為 application）
- [ ] 強調通訊貢獻，而非 AI 應用

---

## 九、下一步行動（立即執行）

1. **今天**：執行 Phase 1（備份與目錄創建）
2. **明天**：開始 Phase 2 - 從 t3.md 重構 `semantic-state-sync.md`
3. **本週內**：完成 Problem Formulation 的 3 個核心文件
4. **下週**：與教授 meeting 前準備一頁 summary

---

**重構負責人**: [Your Name]
**預計完成**: 2026-02-15
**Review by**: 廖婉君教授
