# 研究導讀：KV-Cache 語義通訊協議

**給新成員的完整指南 — 從零開始理解這個研究**

作者：鄭維倫、廖婉君，台大電機系
最後更新：2026-03-10

---

## 目錄

1. [你需要先知道的背景知識](#1-你需要先知道的背景知識)
2. [我們要解決什麼問題？](#2-我們要解決什麼問題)
3. [研究的演化歷程（含失敗方向）](#3-研究的演化歷程)
4. [最終研究方向：完整說明](#4-最終研究方向)
5. [三篇論文各自在講什麼](#5-三篇論文各自在講什麼)
6. [實驗：我們做了什麼、怎麼做的](#6-實驗)
7. [關鍵結果一覽](#7-關鍵結果)
8. [資料夾結構完整說明](#8-資料夾結構)
9. [如何在新電腦上復現](#9-如何在新電腦上復現)
10. [常見問題與陷阱](#10-常見問題與陷阱)
11. [你可以怎麼開始](#11-你可以怎麼開始)

---

## 1. 你需要先知道的背景知識

### 1.1 什麼是 LLM（大型語言模型）？

你應該知道 ChatGPT、Claude 這些 AI。它們背後的技術叫 **Transformer**，一個由很多「層」堆疊的神經網路。

**關鍵概念**：LLM 在回答問題時，會先「讀」你給它的文字（context），在每一層產生一組叫做 **KV-cache** 的中間計算結果。

### 1.2 什麼是 KV-cache？

想像你在讀一本書，邊讀邊做筆記。KV-cache 就是 LLM 的「筆記」：

```
你給 LLM 一段 1000 字的文章
→ LLM 每一層都會產生 Key（索引）和 Value（內容）
→ 這些 K、V 合起來就是 KV-cache
→ 後續生成答案時，LLM 靠這些「筆記」來回憶文章內容
```

**大小**：一個 7B 模型讀 1000 個 token 的 KV-cache ≈ **9.7 MB**（BF16 精度）。

### 1.3 什麼是 Edge-Cloud（邊緣-雲端）架構？

```
[手機/邊緣裝置]  ←── 無線網路 ──→  [雲端伺服器]
  小模型 (3B)       頻寬有限          大模型 (14B)
  反應快但不夠聰明                     很聰明但很遠
```

現實問題：你想讓邊緣的小 AI 和雲端的大 AI **合作**，但網路頻寬是有限的。傳整個 KV-cache（9.7 MB）在低頻寬下要好幾秒，任務可能就超時了。

### 1.4 什麼是 Attention（注意力機制）？

LLM 在讀文章時，不是每個字都一樣重要。Attention 機制會給每個位置一個「重要性分數」：

```
問題：「火災發生在哪裡？」
文章：「今天天氣很好。[台北市信義區發生火災]。股市上漲。」
                      ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                      Attention 分數很高！
```

**我們的核心發現**：不同大小的模型（3B, 7B, 14B），它們認為「重要」的位置竟然 **高度重合**（82-92%）！

---

## 2. 我們要解決什麼問題？

### 2.1 一句話版本

> 在頻寬有限的情況下，如何讓邊緣小模型和雲端大模型高效合作，而且不犧牲任務品質？

### 2.2 具體場景

想像一個緊急救災場景：

```
現場無人機 (邊緣)                         指揮中心 (雲端)
┌─────────────────┐                    ┌─────────────────┐
│ 小型 AI (3B)     │ ──── 4G/5G ────→ │ 大型 AI (14B)    │
│ 快速初步分析      │    頻寬: 50Mbps   │ 精確決策          │
│ 讀取現場報告      │    延遲: 要求<2秒  │ 綜合判斷          │
└─────────────────┘                    └─────────────────┘
```

邊緣 AI 已經讀了現場報告（有了 KV-cache），雲端 AI 也需要讀同一份報告才能做決策。

**傳統做法**：把原始文字重新傳給雲端，雲端重新算 → 浪費時間
**我們的做法**：把邊緣 AI 的「筆記」（KV-cache）壓縮後傳過去 → 雲端直接用

### 2.3 這不是什麼

| 容易搞混的概念 | 區別 |
|--------------|------|
| 傳統語義通訊 (semantic comm) | 他們傳「資料的語義表示」；我們傳「AI 的認知狀態」 |
| ISAC（感知通訊整合） | 他們整合感測資料；我們整合推理狀態 |
| JSCC（聯合編碼） | 他們優化重建品質；我們優化任務成功率 |
| LangChain/AutoGen | 他們假設頻寬無限；我們解決頻寬有限的問題 |
| MCP（Model Context Protocol） | 那是應用層工具呼叫；我們是傳輸層協議 |

---

## 3. 研究的演化歷程

這個研究不是一開始就想到最終方向的。我們經歷了多次 pivot（方向轉換）：

### 3.1 時間線

```
2025-12 ─── T1: O-RAN 網路自動化    ──→ ✗ 放棄（只是用 AI 管網路，不是 AI 的通訊）
  │
  ├──── T2: Edge-Cloud RAG 系統    ──→ ✗ 放棄（本質是影片壓縮，不是通訊協議）
  │
  ├──── T3: Agent 通訊協議         ──→ △ 有方向但太抽象
  │
2026-01 ─── 老師回饋：「你們在做 AI 管理網路，不是設計 AI 自己的通訊」
  │         ↓ 關鍵轉折點
  │
  ├──── T4-T5: 收斂到 KV-cache 語義狀態同步
  │         發現 KV-cache = 語義傳輸單元
  │         建立 Information Bottleneck 理論框架
  │
  ├──── T6: 技術方向確定 — Q2C 選擇 + Scout 協議
  │
2026-02 ─── Paper A 完成（壓縮）+ Paper B 完成（協議）
  │         49 個 GPU 實驗全部完成
  │         JSAC 合併論文完成，進入審稿迭代
  │
2026-03 ─── 目前：JSAC 論文 Round 4 審稿，持續改進中
```

### 3.2 每次 Pivot 的原因

**T1 (O-RAN) → 放棄**：老師指出這只是「用 AI 管理網路」，不是我們要做的「設計 AI 需要的通訊」。就像設計電話和用電話打給修電話的人是不同的事。

**T2 (Edge RAG) → 放棄**：雖然是對的問題（邊緣裝置頻寬有限），但解法還是在做「影片/感測資料壓縮」，沒有抓到 LLM 的本質。

**T3 (Agent 通訊) → 保留核心**：「AI 之間的通訊應該不是傳 bits 而是傳 tokens」這個直覺是對的，但缺乏具體技術載體。

**T4-T5 (收斂)**：關鍵發現 — **KV-cache 就是那個載體**！它是 LLM 的「認知狀態」，可以被選擇、壓縮、傳輸。這不是在傳資料，是在傳理解。

### 3.3 從失敗中學到的

每次失敗都有價值：
- T1 讓我們理解「AI-for-network ≠ network-for-AI」
- T2 讓我們確定問題空間（edge-cloud, 頻寬限制）
- T3 給了我們語義 token 的直覺
- 老師的回饋讓我們跳出「應用層」思維，進入「協議層」設計

---

## 4. 最終研究方向

### 4.1 核心想法（用比喻解釋）

想像兩個學生合作寫報告：

**傳統方式**：A 讀完資料 → 把整份資料傳給 B → B 從頭讀一遍 → 浪費時間

**我們的方式**：A 讀完資料 → 做了重點筆記 → 只傳「重點標記的位置」給 B → B 自己讀同樣的資料，但特別注意 A 標記的重點 → 省時又可能更好

更神奇的是：**A 是成績普通的學生（3B），B 是學霸（14B），但 A 的重點標記竟然讓 B 的回答品質提升了 10%**。因為小模型的注意力更「聚焦」，幫大模型過濾了雜訊。

### 4.2 技術框架

```
邊緣 (Edge)                              雲端 (Cloud)
┌──────────────────────┐               ┌──────────────────────┐
│ 1. 小模型讀取 context  │               │                      │
│ 2. 產生 KV-cache       │               │                      │
│ 3. 用 Attention 分數   │               │                      │
│    選出最重要的 k%     │               │                      │
│    (Q2C Selection)    │               │                      │
│ 4. 壓縮 (量化)         │               │                      │
│    BF16→INT8 (2x)     │   ────────→   │ 5. 解壓縮              │
│    或 Mixed-INT4 (4x) │   網路傳輸     │ 6. 注入到大模型         │
│                       │               │ 7. 生成高品質答案       │
└──────────────────────┘               └──────────────────────┘

                    Scout 模式（極低頻寬）:
                    只傳位置索引 (336 bytes vs 9.7 MB)
                    ≈ 28,800x 壓縮！
```

### 4.3 三個核心技術貢獻

**貢獻 1：Q2C (Query-to-Context) Token Selection**
- 用 Query 的注意力分數來決定 Context 中哪些 token 最重要
- 比現有方法 SnapKV 好 29-47%，比 H2O 好 92-128%
- 只用最後一層的注意力就夠了（不需要所有層，Pearson r > 0.99）

**貢獻 2：Scout Protocol（偵查兵協議）**
- 發現：同家族不同大小的模型，attention 重合度高達 82-92%
- 所以可以只傳「位置編號」（例如 "token 15, 42, 78, 103..."）而不是傳 KV 數值
- 336 bytes vs 9.7 MB = 壓縮 28,800 倍
- 跨架構也成立：Llama-3 (91.8%)、Gemma-2 (86.1%)、Qwen (83%)

**貢獻 3：Adaptive Transport Protocol（自適應傳輸）**
- 根據即時頻寬自動選擇最佳傳輸模式：
  - 頻寬充足 → Full BF16（完整傳輸）
  - 頻寬中等 → INT8 量化（品質不變，大小減半）
  - 頻寬不足 → Mixed-INT4（保護關鍵層，4x 壓縮）
  - 頻寬極低 → Scout 模式（只傳位置，28,800x 壓縮）
  - 頻寬不夠 → Prompt-only（只傳文字，雲端重算）
- 在真實 5G 頻寬 trace 下達到 100% deadline compliance

### 4.4 數學基礎

我們用 **Information Bottleneck** 理論來建模：

```
最小化 I(X; Z)     ← 壓縮傳輸量（傳越少越好）
滿足 I(Z; Y) ≥ η   ← 保證任務品質（重要資訊不能丟）
      E[|Z|] ≤ R   ← 頻寬限制
```

其中：
- X = 原始輸入（文章）
- Z = 傳輸的 KV-cache 片段
- Y = 任務結果（答案品質）

白話：**在頻寬限制內，找到一個壓縮方式，傳最少的資料，但保證任務做得好。**

---

## 5. 三篇論文各自在講什麼

### Paper A: KV-Cache 壓縮（7 頁，目標 INFOCOM 2027）

**問題**：KV-cache 太大，怎麼壓縮才不會損失品質？

**方法**：
1. Q2C selection — 只保留最重要的 token
2. 量化 — BF16→INT8（無損）或 Mixed-INT4（近無損）

**實驗規模**：7 個模型 × 4 個任務 × 多種壓縮比

**核心發現**：
- INT8 量化對所有模型都是無損的
- INT4 量化的效果取決於模型，不是架構（Yi-6B 和 Qwen-7B 都是 GQA，但 Yi 沒問題、Qwen 崩壞）
- Mixed-INT4（保護瓶頸層）可以恢復大部分品質

**檔案**：`papers/paper-A/main.tex`

### Paper B: Scout 協議（7 頁，目標 ICC/JSAC）

**問題**：能不能完全不傳 KV 數值，只傳位置？

**方法**：
1. 觀察到 cross-model attention 重合度 82%+
2. 設計 Scout 模式：edge 只傳 position indices
3. 設計自適應策略：根據頻寬選擇最佳模式

**核心發現**：
- 7B→14B 的 scout 反而讓 14B 品質提升 10.2%（p=0.018）
- 比傳完整 KV-cache 還好！因為小模型的注意力更 focused

**檔案**：`papers/paper-B/main.tex`

### JSAC 合併論文（15 頁，目標 IEEE JSAC）

合併 Paper A + B，加上：
- 跨架構實驗（Llama-3, Gemma-2）
- 完整協議設計（7 種模式的 fallback hierarchy）
- 頻寬估計與理論分析
- 多 agent 場景（Poisson arrival model）

**檔案**：`papers/jsac/main.tex`

---

## 6. 實驗

### 6.1 實驗環境

- **GPU**: NVIDIA RTX PRO 6000 (Blackwell, 102 GB VRAM)
- **租用**: vast.ai（已銷毀，所有資料已同步本地）
- **精度**: BF16（重要！FP16 在 Blackwell 上會出問題）
- **框架**: PyTorch + HuggingFace Transformers

### 6.2 主要實驗列表

| 編號 | 實驗名稱 | 在回答什麼問題 | JSON 檔案 |
|------|---------|--------------|----------|
| S1 | Scout n=200 | 3 種 Qwen 配對的 attention overlap 和品質 | `exp_scout_n200_*.json` |
| S2 | Scout long context | overlap 在長文本下是否穩定 | `exp_scout_long_ctx_*.json` |
| S4 | Multitask | scout 在不同任務上的效果 | `exp_scout_multitask_*.json` |
| F1 | Q2C ablation | 最後一層 vs 所有層的 attention 有差嗎 | `exp_q2c_ablation_*.json` |
| F2 | Perplexity | 量化後 perplexity 變化多少 | `exp_perplexity_*.json` |
| A1 | Attention entropy | 不同大小模型的注意力集中度 | `exp_attention_entropy_*.json` |
| P1 | Protocol traces | 在真實 5G 頻寬下協議表現 | `exp_protocol_real_traces_*.json` |
| P2 | Hybrid mode | 混合模式的品質-頻寬 tradeoff | `exp_hybrid_mode_*.json` |
| C1 | Cross-family scout | Qwen→Mistral 跨家族 | `exp_cross_family_scout_*.json` |
| C2 | Cross-architecture | Llama-3, Gemma-2 | `exp_cross_family_overlap_*.json` |
| Q1 | Quant fix | 量化 KV-cache 的正確評估 | `exp_paper_a_quant_fix_*.json` |
| B1 | BW estimation | 頻寬估計驗證 | `exp_bw_estimation_*.json` |

**總計 49 個 JSON 檔案**，每個都包含逐樣本的原始數據。

### 6.3 實驗怎麼跑的

每個實驗都是一個獨立 Python 腳本：

```bash
# 例如跑 scout 實驗
python experiments/scripts/run_exp_scout_n200.py
# → 自動下載模型、跑 200 個樣本、存結果到 experiments/results/
```

所有腳本特點：
- 固定隨機種子 (seed=42) → 完全可復現
- 自動存時間戳 JSON → 不會覆蓋舊結果
- 環境變數自動設定 → 不需要手動配置

---

## 7. 關鍵結果

### 7.1 Token Selection（Q2C vs 競爭方法）

在 SQuAD v2, 25% token 保留率下：

| 模型 | Q2C (我們) | SnapKV | H2O | Random |
|------|-----------|--------|-----|--------|
| Qwen-7B | **0.428** | 0.292 | 0.205 | 0.193 |
| Qwen-14B | **0.360** | 0.279 | 0.160 | 0.192 |
| Mistral-7B | **0.294** | 0.205 | 0.129 | 0.104 |

→ Q2C 在所有模型上都顯著勝出

### 7.2 Scout 跨模型 Attention 重合

| 配對 | 75% 保留 | 50% 保留 | 25% 保留 |
|------|---------|---------|---------|
| Qwen 3B→7B | 82% | 64% | 49% |
| Qwen 7B→14B | 83% | 68% | 55% |
| Llama 3B→8B | **92%** | — | — |
| Gemma 2B→9B | **86%** | — | — |

→ 跨架構都成立，不只是 Qwen 專屬現象

### 7.3 量化效果

| 精度 | 壓縮比 | 品質保留 |
|------|-------|---------|
| INT8 | 2x | 99.4%（無損） |
| INT4 | 4x | 68.7%（崩壞） |
| Mixed-INT4 | ~3.6x | 93.6%（恢復） |

### 7.4 傳輸模式比較

| 模式 | Payload 大小 | 品質 | 傳輸時間 @100Mbps |
|------|-------------|------|-------------------|
| Full BF16 | 9.7 MB | 100% | 775 ms |
| INT8 | 4.7 MB | 99.6% | 388 ms |
| Scout | **336 B** | 81-110% | **0.03 ms** |

→ Scout 模式 payload 壓縮 28,800 倍，而且品質可能更好！

### 7.5 協議在真實 5G 頻寬下

- 使用 Lumos5G dataset 的真實 trace（urban, suburban, vehicular）
- Markov 6-state bandwidth model
- **100% deadline compliance** in scout mode

---

## 8. 資料夾結構

```
AI-Comm/
│
├── papers/                          # 論文源碼
│   ├── paper-A/main.tex            # Paper A: KV-cache 壓縮 (7頁)
│   ├── paper-B/main.tex            # Paper B: Scout 協議 (7頁)
│   ├── jsac/main.tex               # JSAC 合併論文 (15頁)
│   └── mlsys/main.tex              # MLSys 格式版本
│
├── experiments/
│   ├── scripts/                    # 22 個核心實驗腳本
│   │   ├── run_exp_*.py            # 主要實驗 (15 個)
│   │   ├── run_fix_quant.py        # 量化 bug 修正
│   │   └── exp_utils.py            # 共用工具
│   ├── results/                    # 26 個核心 JSON 結果檔
│   │   └── exp_*.json              # 每個都是可復現的獨立結果
│   ├── legacy-scripts/             # 41 個舊版 batch 腳本（歷史參考）
│   ├── legacy-results/             # 23 個舊版結果（探索性實驗）
│   ├── figures/                    # 圖表生成腳本
│   └── poc/                        # 概念驗證 (TinyLlama, CPU-only)
│
├── research/                       # 研究方向探索
│   ├── 01-19_*.md                  # 19 個研究方向文件
│   ├── survey/                     # 文獻調查
│   └── future-ideas/              # 未來論文規劃
│
├── reviewer/                       # 審稿意見（按輪次組織）
│   ├── round0-initial/             # 內部自我檢查
│   ├── round1/                     # Gemini R1 (89分)
│   ├── round2/                     # 早期多源審稿
│   ├── round3/                     # Claude R3 + Gemini 分論文審
│   ├── round4/                     # 三方交叉 (Claude/ChatGPT/Gemini)
│   ├── prompts/                    # 審稿 prompt 模板
│   └── analysis/                   # 跨輪分析
│
├── docs/                           # 專案文件與內部筆記
│   ├── PROJECT_STATUS.md           # 專案狀態總覽
│   ├── advisor-notes/              # 給老師的信件
│   ├── notes/                      # 研究筆記
│   └── reference/                  # 工具操作手冊
│
├── .claude-memory/                 # Claude Code 記憶（給新電腦用）
│   ├── MEMORY.md                   # 累積知識
│   └── README.md                   # 使用說明
│
├── archive/                        # 歷史檔案（舊方向、演化紀錄）
│
├── CLAUDE.md                       # Claude Code 的指令檔（自動載入）
├── README.md                       # GitHub README (English)
├── README_zh.md                    # 中文 README (28頁)
├── ONBOARDING.md                   # ← 你現在在讀的這份文件
├── .gitignore                      # Git 忽略規則
└── LICENSE                         # MIT License
```

### 哪些檔案最重要？

| 優先級 | 檔案 | 為什麼重要 |
|-------|------|-----------|
| ⭐⭐⭐ | `papers/jsac/main.tex` | 最完整的論文，讀這篇就理解全部 |
| ⭐⭐⭐ | `CLAUDE.md` | Claude Code 的設定，包含所有技術約束 |
| ⭐⭐ | `papers/paper-A/main.tex` | 壓縮方法的獨立論文 |
| ⭐⭐ | `papers/paper-B/main.tex` | Scout 協議的獨立論文 |
| ⭐⭐ | `experiments/results/` | 所有原始實驗數據 |
| ⭐ | `README_zh.md` | 非常詳細的中文說明 |
| ⭐ | `archive/evolution-logs/` | 了解研究演化歷程 |

---

## 9. 如何在新電腦上復現

### 9.1 基本設置

```bash
# 1. Clone repo
git clone git@github.com:taiwanfifi/AICoMM.git
cd AICoMM

# 2. 安裝 Python 依賴
pip install torch transformers matplotlib numpy tqdm

# 3. 編譯論文 (需要 LaTeX)
cd papers/jsac && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 4. 生成圖表 (不需 GPU)
python papers/jsac/generate_figures.py
```

### 9.2 Claude Code 設置

Claude Code 會自動讀取 `CLAUDE.md`，所以 clone 完就有基本指令了。

要取得完整的技術記憶（所有踩過的坑、實驗結果、debug 經驗）：
1. 在新電腦 clone repo 後啟動 Claude Code
2. Claude Code 會自動讀取 CLAUDE.md
3. 第一次對話時說：「請讀 `.claude-memory/MEMORY.md` 並設定到你的 memory 裡，然後讀 `ONBOARDING.md` 了解專案全貌」
4. Claude 就會有完整的專案知識，包括所有技術陷阱和實驗結果

### 9.3 跑實驗（需要 GPU）

```bash
# CPU-only 實驗（可以在任何電腦上跑）
python experiments/scripts/run_exp_bw_estimation.py
python experiments/scripts/run_exp_protocol_real_traces.py

# GPU 實驗（需要 NVIDIA GPU, 40-80GB VRAM）
# 建議租 vast.ai 或 Lambda Labs
python experiments/scripts/run_exp_scout_n200.py
python experiments/scripts/run_exp_q2c_ablation.py
```

**重要**：GPU 實驗必須用 `dtype=torch.bfloat16`，FP16 在某些 GPU 上會出問題。

---

## 10. 常見問題與陷阱

### Q: 為什麼不用 FP16？
A: 在 NVIDIA Blackwell (sm_120) GPU 上，7B+ 模型用 FP16 + eager attention 會產生垃圾輸出。一定要用 BF16。

### Q: 為什麼要用 eager attention 而不是 FlashAttention？
A: 因為我們需要 `output_attentions=True` 來取得 attention 分數，FlashAttention 和 SDPA 不支持這個功能。

### Q: 量化後的 KV-cache 怎麼用？
A: 不能用 `model.generate(past_key_values=quantized_cache)`！必須自己寫手動 greedy decode loop。這是我們踩過的一個大坑（`run_fix_quant.py` 裡有修正）。

### Q: 為什麼 Qwen-7B 的 INT4 崩壞但 Yi-6B 沒事？
A: 這是我們論文的一個核心發現：量化耐受度是模型特定的，不是架構決定的。兩個模型都用 GQA，但行為完全不同。可能與訓練數據和權重分布有關。

### Q: Scout 為什麼能讓品質提升？
A: 小模型（7B）的 attention 比大模型（14B）更「集中」（entropy 更低）。當你告訴 14B 「注意這些位置」，等於幫它過濾了雜訊。就像一個專注的學生幫學霸畫重點。

### Q: 跨架構（Llama→Llama, Gemma→Gemma）為什麼 attention 會重合？
A: 我們猜測是因為同一家族的模型使用相同的 tokenizer 和類似的訓練數據，所以對「什麼重要」有相似的判斷。跨家族（例如 Qwen→Mistral）的重合度就低很多（73%）。

### Q: 所有實驗資料都在本地嗎？
A: 是的。GPU server（vast.ai）已經銷毀，所有 49 個 JSON 結果檔、實驗腳本、論文都已同步到本地 repo。

---

## 11. 你可以怎麼開始

### 第一步：理解大方向（1-2 天）

1. 讀這份文件（你正在做了）
2. 讀 `README.md` 和 `README_zh.md`
3. 讀 JSAC 論文的 Abstract 和 Introduction：`papers/jsac/main.tex` 的前 200 行

### 第二步：理解技術細節（2-3 天）

1. 讀 Paper A（`papers/paper-A/main.tex`）— 理解壓縮方法
2. 讀 Paper B（`papers/paper-B/main.tex`）— 理解 Scout 協議
3. 看實驗腳本：從 `experiments/scripts/run_exp_scout_n200.py` 開始

### 第三步：看實驗結果（1 天）

1. 用 Python 讀幾個 JSON 結果檔：
   ```python
   import json
   with open('experiments/results/exp_scout_n200_20260210_073907.json') as f:
       data = json.load(f)
   print(json.dumps(data, indent=2)[:2000])  # 看前 2000 字
   ```
2. 跑圖表生成腳本：`python papers/jsac/generate_figures.py`

### 第四步：理解研究演化（0.5 天）

1. 讀 `archive/old-directions/t1-oran-automation.md` — 理解為什麼放棄
2. 讀 `archive/evolution-logs/t5-convergence.md` — 理解怎麼收斂

### 第五步：準備向老師報告

#### 建議報告大綱（30 分鐘版本）

**Slide 1 — 問題 (3 min)**
- 邊緣 AI（手機/無人機）和雲端 AI 需要合作
- 頻寬是瓶頸：傳完整 KV-cache 要 9.7 MB，在 50 Mbps 下要 1.5 秒
- 現有方案都假設頻寬充足（LangChain、AutoGen）或只做壓縮不做協議（CacheGen）

**Slide 2 — 核心發現 (5 min)**
- 不同大小 LLM 的 attention 分布高度重合（82-92%）
- 這代表小模型「認為重要的位置」和大模型幾乎一樣
- 跨架構也成立：Qwen 83%, Llama 92%, Gemma 86%
- 展示圖：attention heatmap 或 overlap bar chart

**Slide 3 — 方法一：Q2C 壓縮 (5 min)**
- 用 Query 的 attention 選出最重要的 token
- 比 SnapKV 好 29-47%，比 H2O 好 92-128%
- 只用最後一層就夠（Pearson r > 0.99）
- 展示表格：Q2C vs SnapKV vs H2O vs Random

**Slide 4 — 方法二：Scout 協議 (5 min)**
- 既然 attention 重合度 82%+，那就只傳位置索引
- 336 bytes vs 9.7 MB = 壓縮 28,800 倍
- 7B scout 反而讓 14B 品質提升 10.2%（小模型更 focused）
- 展示圖：payload size vs quality 的 tradeoff

**Slide 5 — 方法三：自適應傳輸 (3 min)**
- 7 種 fallback 模式：Full → INT8 → Mixed → Scout → Prompt-only
- 根據即時頻寬自動切換
- 在真實 5G trace 下 100% deadline compliance

**Slide 6 — 跟別人的區別 (3 min)**
- 我們傳的是「AI 的認知狀態」，不是資料
- CacheGen (SIGCOMM'24)：只做壓縮，沒有 task-aware selection
- SnapKV (NeurIPS'24)：只做 selection，沒有跨模型和協議
- 我們是第一個把 KV-cache 當作跨模型語義傳輸單元

**Slide 7 — 實驗規模和可信度 (3 min)**
- 7 個模型家族（1.1B-14B），4 個 NLP 任務
- 49 個實驗，26 個核心結果，每個有逐樣本數據
- 統計檢驗：paired t-test，p-values 報告
- 所有腳本開源、種子固定、完全可復現

**Slide 8 — 投稿計畫與下一步 (3 min)**
- Paper A → INFOCOM 2027 (Aug 2026 deadline)
- Paper B → ICC / JSAC
- JSAC 合併 → 目前 R4，繼續迭代
- 剩餘 TODO：CacheGen 對比、Random-k baseline（需 GPU）

#### 老師可能的問題與準備好的回答

| 問題 | 回答要點 |
|------|---------|
| 這跟語義通訊有什麼關係？ | 我們不是傳統語義通訊（傳資料的語義表示），是傳 AI 的認知狀態（KV-cache）|
| Scout 為什麼品質能提升？ | 小模型 attention entropy 更低（4.21 vs 5.49），更 focused，像噪音過濾器 |
| 只在 Qwen 上驗證夠嗎？ | 不只 Qwen，Llama-3 (92%) 和 Gemma-2 (86%) 也驗證了 |
| INT4 崩壞怎麼解決？ | Mixed-precision：保護 bottleneck 層用 FP16，其他 INT4，恢復 93.6% |
| 實際部署可行嗎？ | Scout 模式只需傳 336 bytes，比一個 HTTP header 還小 |
| 跟 CacheGen 的差異？ | CacheGen 是壓縮工具，我們是完整協議（選擇+壓縮+自適應+理論）|

---

## 附錄 A：術語對照表

| 中文 | 英文 | 論文中的用法 |
|------|------|-------------|
| 語義狀態同步 | Semantic State Synchronization | ✅ 用這個 |
| 語義通訊 | Semantic Communication | ❌ 避免（我們不是傳統語義通訊） |
| Token 傳輸 | Token-based Transmission | ✅ 用這個 |
| Token 串流 | Token Streaming | ❌ 避免 |
| KV-cache 共享 | KV-cache Sharing | ✅ 用這個 |
| 任務成功率 | Task Success Rate | ✅ 用這個 |
| 準確率 | Accuracy | ❌ 避免（改用 task success rate） |

## 附錄 B：相關文獻（最重要的 5 篇）

1. **CacheGen** (SIGCOMM 2024) — KV-cache 的網路傳輸與壓縮
2. **SnapKV** (NeurIPS 2024) — Attention-based KV selection
3. **KVCOMM** (NeurIPS 2025) — 跨 context 的 KV-cache 重用 + RoPE 修正
4. **CacheBlend** (EuroSys 2025) — 接收端品質修復
5. **C2C** (清華大學) — 證明 KV-cache 可用於 agent 通訊

## 附錄 C：投稿計畫

| 論文 | 目標會議/期刊 | Deadline | 頁數 | 狀態 |
|------|-------------|----------|------|------|
| Paper A | INFOCOM 2027 | ~Aug 2026 | 7 | 完成，待投 |
| Paper B | ICC / JSAC | TBD | 7 | 完成，待投 |
| JSAC Merged | IEEE JSAC | TBD | 15 | 審稿迭代中 (R4) |
| MLSys | MLSys | TBD | -- | 已排版 |
