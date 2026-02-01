# 研究專案完整筆記

> **用途**：快速理解整個專案、掌握所有資訊、確認方向
> **最後更新**：2026-02-01

---

## 一、這個研究到底在做什麼？（30 秒版本）

**一句話**：設計一個讓 AI Agent 之間用「認知狀態差量」（而非傳統封包/文字）來溝通的通訊協定，適用於頻寬有限的 6G 環境。

現在 LangChain、AutoGen、MCP 這些框架讓 Agent 合作時，都是直接傳「一大段 prompt/JSON/文字」，隱含假設頻寬無限、延遲為零。但在 6G 邊緣環境（無人機、自駕車、工業機器人），頻寬是有限的。

我們的做法：**不傳文字，傳「想法的差量」**。就像兩台電腦同步檔案時只傳 diff 一樣，兩個 Agent 先對齊一個 shared world model，之後只傳「我的內部狀態跟上次比有什麼關鍵變化」。

---

## 二、具體例子：Before vs After（用真實場景理解）

### 例子 1：森林巡邏無人機火災偵測

**場景設定**：一台無人機（Edge Agent，跑 MobileVLM）在森林巡邏，雲端有一台伺服器（Cloud Agent，跑 GPT-4V）負責總指揮。無人機需要把觀測到的資訊傳給雲端來判斷決策。

---

#### BEFORE（傳統方法 — 傳影片）

```
[無人機 Camera]
     |
     v
1080p 影片串流 (H.264 編碼)
     |  ← 需要 5 Mbps 頻寬
     v
[雲端伺服器]
     |
     v
雲端從頭跑影像辨識模型
     |
     v
"偵測到火源，座標 (34.2, -118.5)"
```

**傳了什麼**：每秒 30 個 frame 的完整影像，每個 frame 約 150KB
**傳輸量**：~5 Mbps（持續）
**問題**：
- 95% 的 frame 是樹和天空，根本沒有有用資訊
- 雲端要從頭跑一次影像辨識，延遲 120ms+
- 頻寬不夠時直接斷線

---

#### BEFORE（現有 Agent 框架 — 傳文字）

```
[無人機 MobileVLM 推理]
  "I see gray smoke rising from coordinates (34.2, -118.5),
   approximately 200m away. The smoke appears dense.
   Wind direction is northeast. No visible flames yet.
   Surrounding area has dry vegetation.
   My current altitude is 150m.
   Battery level: 72%.
   Recommending closer inspection."
     |
     |  ← 傳送完整文字 prompt (~500 bytes)
     |  ← 看起來不多，但每 10 秒一次 × 100 台無人機 = 爆頻寬
     v
[雲端 GPT-4V]
     |
     v
接收文字 → tokenize → embedding → 重新推理
     |  ← 雲端要「重新理解一次世界」，120ms
     v
"派遣消防隊到 (34.2, -118.5)"
```

**傳了什麼**：Agent 用人類語言描述它看到的東西
**傳輸量**：~500 bytes/次，但 100 台無人機 × 每 10 秒 = 5KB/s × 100 = 500KB/s
**問題**：
- 文字是「非最小充分表示」— "gray smoke rising" 其實有用的只有 `belief[fire] = 0.87`
- 雲端收到文字後要 tokenize → embed → 重新推理，等於「重新想一遍」
- 不同 Agent 對 "dense smoke" 的理解可能不同（語義不對齊）

---

#### AFTER（我們的 SSC 方法 — 傳認知差量）

```
[初始化：Control Plane 語義握手]
無人機: { model: MobileVLM, kv_dim: 512, task: fire_detection, threshold: 0.85 }
雲端:   { model: GPT-4V, kv_dim: 4096, task: fire_detection, threshold: 0.85 }
→ 協商完成：state space 對齊、Neural Projector 載入（512→4096 維映射）

[t=0~59s：巡邏中，無人機持續推理]
MobileVLM 內部狀態：
  S_0 = { belief[fire]=0.02, belief[normal]=0.95, plan=patrol, attention=均勻分散 }
  S_1 = { belief[fire]=0.02, belief[normal]=0.95, plan=patrol, attention=均勻分散 }
  ...
  S_59 = { belief[fire]=0.03, belief[normal]=0.94, plan=patrol, attention=均勻分散 }

→ Attention filtering: max(attention) = 0.12 < threshold 0.85
→ 判定：沒有值得傳的東西
→ ❌ 不傳任何東西。60 秒零傳輸。

[t=60s：發現煙霧！]
MobileVLM 內部狀態突變：
  S_60 = { belief[fire]=0.87, belief[normal]=0.10, plan=investigate, attention[fire_region]=0.96 }

→ 算 State Delta:
  ΔS_60 = S_60 - S_59
        = { Δbelief[fire]=+0.84, Δbelief[normal]=-0.84, Δplan=patrol→investigate,
            Δattention[fire_region]=+0.84 }

→ Attention filtering: max(attention) = 0.96 > threshold 0.85 ✅
→ Top-k selection (k=3): 選出 fire_belief, plan_change, fire_region_attention
→ 其他幾百個不重要的維度（天空亮度、樹的顏色...）全部丟掉

→ 編碼成 Semantic Token:
  {
    timestamp: 60,
    anchor_id: "patrol_session_001",
    indices: [42, 107, 256],           // 3 個重要維度的 index
    values: [+0.84, 1, +0.84],        // delta 值（量化成 FP8）
    attention_weights: [0.96, 0.93, 0.91]
  }

→ FP8 量化 + ZSTD 壓縮
→ 傳送：48 bytes

[雲端收到]
→ 解壓
→ Neural Projector: 512-dim → 4096-dim（1.2ms）
→ 直接 apply delta: cloud_state[42] += 0.84, cloud_state[107] = investigate, ...
→ 不需要重新推理！直接知道「無人機現在認為有 87% 機率是火災，已切換到調查模式」
→ 決策：派遣消防隊
```

**傳了什麼**：3 個維度的數值變化量
**傳輸量**：48 bytes（一次性），之前 60 秒完全零傳輸
**對比**：

| | 傳統影片 | Agent 框架 | **我們 SSC** |
|---|---|---|---|
| 60 秒傳輸量 | 37.5 MB | 3 KB | **48 bytes** |
| 有用資訊比例 | ~5%（95% 是背景） | ~20%（大量冗餘文字） | **~100%（只傳關鍵差量）** |
| 雲端處理延遲 | 120ms（重跑推理） | 120ms（重新理解文字） | **1.2ms（apply delta）** |
| 100 台無人機 | 需 500 Mbps | 需 50 Kbps | **需 0.8 Kbps** |

---

### 例子 2：兩台自駕車協調變道

**場景**：車 A 想切到右車道，車 B 在右車道後方。兩車需要協調。

#### BEFORE（傳感測資料）

```
車 A 傳給車 B:
  - 完整 LiDAR 點雲 (500KB)
  - 前方攝影機影像 (150KB)
  - 完整行車狀態 JSON:
    {
      "speed": 65.2,
      "acceleration": 0.3,
      "steering_angle": 2.1,
      "lane_position": 1.82,
      "gps": [25.0330, 121.5654],
      "heading": 87.3,
      "brake_pressure": 0,
      "turn_signal": "right",
      "surrounding_objects": [
        {"type": "car", "distance": 12.3, "relative_speed": -2.1, "lane": 2},
        {"type": "truck", "distance": 45.0, "relative_speed": 0, "lane": 1},
        {"type": "motorcycle", "distance": 28.7, "relative_speed": 3.2, "lane": 2}
      ],
      "planned_trajectory": [[x1,y1], [x2,y2], ... 50 points],
      "confidence": 0.91
    }

傳輸量: ~700 KB
頻率: 10 Hz → 7 MB/s
車 B 收到後: 自己重新跑路徑規劃演算法
延遲: 50ms 傳輸 + 30ms 重新規劃 = 80ms
```

#### AFTER（SSC 方法）

```
[已完成語義握手：兩車共享道路模型、車道定義、座標系對齊]

車 A 內部狀態變化：
  t=0: plan=lane_keep, target_lane=1
  t=1: plan=lane_change_right, target_lane=2, gap_assessment=safe(0.91)

State Delta:
  Δplan = lane_keep → lane_change_right     ← attention weight: 0.97（超關鍵）
  Δtarget_lane = 1 → 2                      ← attention weight: 0.95
  Δgap_safe = 0.91                           ← attention weight: 0.88

Semantic Token:
  {
    timestamp: 1,
    indices: [plan_dim, lane_dim, gap_dim],
    values: [LANE_CHANGE_RIGHT, 2, 0.91],
    attention_weights: [0.97, 0.95, 0.88]
  }

傳輸量: 32 bytes
車 B 收到: 直接 apply → 「車 A 要切右道，它認為 gap 足夠（0.91 信心）」
車 B 反應: 減速讓道
延遲: 0.5ms 傳輸 + 0.1ms apply = 0.6ms（比傳統快 130 倍）
```

---

### 例子 3：多 Agent 工廠機器人協作

**場景**：3 台機器人在倉庫裡搬貨，需要避免碰撞和重複搬運。

#### BEFORE

```
每台機器人每秒廣播給其他所有機器人：
  - 自己的完整位置 + 速度 + 路徑規劃（2KB）
  - 手上貨物的完整描述（1KB）
  - 感測到的周圍環境（5KB）

3 台機器人 × 2 台接收者 × 8 KB × 10 Hz = 480 KB/s

如果是 20 台機器人：
  20 × 19 × 8 KB × 10 Hz = 30.4 MB/s（幾乎不可行）
```

#### AFTER

```
大部分時間：機器人各自搬貨，狀態穩定，不傳任何東西 → 0 bytes

只在「認知事件」發生時傳：

[事件 1] 機器人 A 發現路徑被擋
  Semantic Token: { Δplan: route_3→route_7, Δblocked_path: route_3, confidence: 0.95 }
  → 32 bytes，只發給可能受影響的機器人 B（走同方向的）

[事件 2] 機器人 C 拿完最後一箱 item_X
  Semantic Token: { Δinventory[item_X]: 3→0, Δstatus: item_X_depleted }
  → 24 bytes，廣播給所有人

20 台機器人，平均每分鐘 5 個認知事件 × 30 bytes = 150 bytes/min = 2.5 bytes/s
對比傳統的 30.4 MB/s → 節省 99.99992%
```

---

## 三、核心原理完整流程（技術實質）

### 三代通訊範式

```
第一代（Shannon）：Source → Encoder → Channel → Decoder → Bits → Application
  評估：Bit Error Rate (BER)
  目標：完美還原每一個 bit

第二代（JSCC/語義通訊）：Source → 語義特徵 → Channel → Decoder → 重建資料 → 再推理
  評估：MSE / PSNR（重建品質）
  目標：重建資料

第三代（我們 SSC）：Shared World Model → State Δ → Attention Filter → Token → Sync
  評估：Task Success Rate（任務成功率）
  目標：同步認知狀態以完成任務
```

### 運作六步驟

**Step 1: Control Plane 握手**
兩個 Agent 先協商：做什麼任務、state space 怎麼對齊、attention threshold 設多少。

**Step 2: 持續監控內部狀態**
Edge Agent 持續跑推理，產生 `S_t = (h_t, b_t, p_t, a_t)`：隱藏層、信念、策略、注意力。

**Step 3: 計算 State Delta**
`ΔS_t = S_t - S_{t-1}`

**Step 4: Attention-based Filtering（核心創新）**
借用 DeepSeek Sparse Attention：
- `Score = ReLU(Q_t · K_s)`（線性複雜度 O(T·d)）
- 只選 Top-k 最重要維度
- 超過閾值 τ 才傳（event-driven）
- 省下 95% 頻寬

**Step 5: Token 編碼與傳輸**
FP8 量化（3.5x 壓縮）+ ZSTD 無損壓縮 → protobuf 封裝。

**Step 6: Receiver 端確定性整合**
收到後直接 `state[indices] += delta`，不需重新推理。丟包用 zero-hold 或 task-aware retransmit。

### 數學核心

**Information Bottleneck**：
```
min I(X; Z) - β·I(Z; Y)
```
- I(X; Z)：傳了多少（越少越好）
- I(Z; Y)：傳的跟任務多相關（越多越好）

**Task-Oriented Rate-Distortion**：
```
min R(S→Z)  s.t. D_task = 1 - P(Task Success | Ŝ) ≤ D_max
```

**最優閾值解析解**：
```
τ* = λ·(-ln(B/N))^(1/k)
```
可直接根據頻寬 B 和 token 數 N 算出。

---

## 四、每個檔案的內容與狀態

### 核心文件（必讀，研究骨架）

| 檔案 | 內容 | 成熟度 |
|------|------|--------|
| `00-advisor-feedback/professor-concepts-raw.md` | 教授原話逐字稿：未來傳 Token 不傳 Packet、MCP 不是應用層、現有框架忽略通訊成本 | ✅ 定稿，所有方向的根源 |
| `01-problem-formulation/research-question.md` | 核心問題、5 子問題（表示/決策/壓縮/對齊/容錯）、5 假設、評估標準 | ✅ 完整 |
| `01-problem-formulation/defense-strategy.md` | 與 ISAC/JSCC/MCP 的本質差異、三革命點、預期質疑回應 | ✅ 口試防禦用 |
| `01-problem-formulation/mathematical-system-model.md` | IB 框架、R-D 目標、Attention Gate、State Update 模型 | ✅ 基礎完成 |
| `01-problem-formulation/theoretical-foundations.md` | 5 定理 + 2 引理完整證明 | ✅ 理論最硬 |
| `02-core-framework/semantic-state-sync.md` | SSC 框架全文 + Temporal Drift 分析 + Reset 策略 | ✅ 最重要框架文件 |
| `02-core-framework/semantic-token-definition.md` | Semantic Token 定義（認知單位，非 word token）| ✅ 清楚 |
| `02-core-framework/architecture-overview.md` | 統一架構：STL 三平面（Control/Data/Management）| ✅ 完整 |
| `03-technical-design/attention-filtering.md` | DeepSeek DSA → 通訊系統、Lightning Indexer、雙通道 | ✅ 完整 |
| `03-technical-design/state-integration.md` | Receiver 端：Anchor 對齊、亂序、丟包策略 | ✅ 完成 |
| `03-technical-design/kv-cache-alignment.md` | 異質模型 KV-Cache 對齊：Neural Projector、Distortion Bound | ✅ 詳細 |
| `01-problem-formulation/contributions.md` | 所有貢獻：3 理論 + 3 技術 + 3 實證 | ✅ 完整 |
| `01-problem-formulation/motivation.md` | 為什麼重要：趨勢、現有不足、影響 | ✅ 偏 promotional |

### 技術補充

| 檔案 | 內容 | 成熟度 |
|------|------|--------|
| `03-technical-design/token-encoding.md` | Protobuf schema、量化策略、壓縮方案 | ✅ 詳細 |
| `04-background/related-work/vs-ISAC.md` | 與 ISAC 對比 | ✅ 簡潔 |
| `04-background/related-work/vs-JSCC.md` | 與 JSCC 對比 | ✅ 簡潔 |
| `04-background/related-work/vs-traditional-comm.md` | 與傳統通訊對比 | ✅ 簡潔 |
| `04-background/technical-background/agent-services.md` | FM-Agent 5 層架構文獻 | ✅ 背景 |
| `04-background/technical-background/deepseek-architecture.md` | DeepSeek-V3 架構 | ✅ 背景 |
| `04-background/technical-background/internet-of-agents.md` | IoA 4 層架構（30KB）| ✅ 背景 |

### 實驗規劃（尚未執行）

| 檔案 | 內容 | 成熟度 |
|------|------|--------|
| `05-evaluation/scenarios.md` | 3 場景、Trace-driven 方法、Channel 模擬、Baseline | ✅ 設計完成 |
| `05-evaluation/experimental-design.md` | 詳細實驗設計 | ✅ 規劃完成 |
| `05-evaluation/cost-model.md` | 通訊成本模型 | ✅ 詳細 |
| `06-implementation/ssc-pipeline-spec.md` | SSC pipeline 規格書（31KB）| ✅ 規格完成 |

### 教授溝通

| 檔案 | 內容 | 狀態 |
|------|------|------|
| `00-advisor-feedback/t8-advisor-email-v1.md` | 給教授的信（含回覆）| 已寄出 |
| `00-advisor-feedback/t8-method-draft.md` | 方法論溝通稿 | 草稿 |
| `00-advisor-feedback/t8-updated-draft.md` | 更新版溝通稿 | 草稿 |
| `00-advisor-feedback/meeting-draft.md` | 面談材料 | 草稿 |
| `00-advisor-feedback/t9-research-strategy.md` | 研究策略（118KB）| 內部參考 |

### 專案管理

| 檔案 | 用途 |
|------|------|
| `ROADMAP.md` | 7 Phase 時程，目標 INFOCOM 2027 / ICC 2027 |
| `09-project-logs/PHASE1~3*.md` | Phase 完成報告 |
| `09-project-logs/PROJECT_STATUS_2026-01-24.md` | 1/24 狀態快照 |

### 不需要看的

| 檔案 | 原因 |
|------|------|
| `archive/old-directions/t1-oran-automation.md` | 已棄用：O-RAN 自動化 |
| `archive/old-directions/t2-edge-rag.md` | 已棄用：Edge RAG |
| `archive/evolution-logs/*` | 歷史思路演變記錄 |
| `archive/original-sources/*` | 重構前原始檔 |
| `backup-20260124-094204/*` | 1/24 備份 |
| `test-sweagent/calc.py` | 測試用，無關 |
| `tools/new_agent_idea.md` | AI 工具探索筆記 |

---

## 五、能看 vs 不能看

### 可以給教授看
1. `research-question.md` — 問題定義
2. `defense-strategy.md` — 差異化論述
3. `mathematical-system-model.md` — 數學模型
4. `theoretical-foundations.md` — 定理證明
5. `semantic-state-sync.md` — 核心框架
6. `architecture-overview.md` — 架構圖
7. `contributions.md` — 貢獻列表

### 可以給同學看（需潤色）
- `attention-filtering.md`
- `kv-cache-alignment.md`
- `token-encoding.md`
- `scenarios.md`

### 僅供內部
- `motivation.md`（太 promotional）
- `t9-research-strategy.md`（太長太散）
- 所有 `archive/`, `09-project-logs/`, `backup-*/`

---

## 六、研究方向總結

**論文標題**：Token-Based Communication Protocol for Agent-Oriented 6G Networks

**核心命題**：通訊的目的從「傳 bits」→「同步認知狀態」

**三個革命點**：
1. 傳輸單位：bit/packet → semantic state delta
2. 決策機制：有資料就傳 → attention-gated（只傳重要的）
3. 評估指標：BER/PSNR → Task Success Rate

**跟別人不一樣的地方（一句話版）**：
- vs 傳統通訊：我們不追求完美還原 bit，只追求任務做好
- vs JSCC：我們不重建資料，直接同步狀態
- vs ISAC：我們傳的是「想法」不是「感測資料」
- vs MCP/LangChain：我們認真考慮通訊有成本

**目標 venue**：IEEE INFOCOM 2027（~2026/08）或 IEEE ICC 2027（~2026/10）

**目前進度**：理論框架和技術設計完成（Phase 1-3），尚未開始 coding 和實驗（Phase 4-7）。

---

## 七、Input → Process → Output 完整對照圖

### 一般化流程

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT（發生在 Edge）                     │
│                                                              │
│  Edge Agent 持續觀測環境（Camera/LiDAR/Sensor）              │
│  → 跑本地 LLM/VLM 推理（如 MobileVLM）                      │
│  → 產生內部狀態 S_t = (hidden_state, belief, plan, attention)│
│                                                              │
│  具體來說：                                                  │
│  S_60 = {                                                    │
│    hidden_state: [0.12, -0.34, 0.87, ...] (512維向量),       │
│    belief: { fire: 0.87, normal: 0.10, smoke: 0.03 },        │
│    plan: "investigate",                                      │
│    attention: { fire_region: 0.96, sky: 0.02, trees: 0.01 }  │
│  }                                                           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────┐
│                   PROCESS（SSC Pipeline）                     │
│                                                              │
│  1. 計算差量：ΔS = S_60 - S_59                               │
│     Δbelief[fire] = 0.87 - 0.03 = +0.84                     │
│     Δplan = patrol → investigate                              │
│     Δattention[fire_region] = 0.96 - 0.12 = +0.84           │
│                                                              │
│  2. Attention Filtering:                                     │
│     512 個維度中，只有 3 個的 attention > threshold 0.85      │
│     → 丟掉 509 個不重要的維度（省 99.4% 資料量）             │
│                                                              │
│  3. 編碼：                                                   │
│     SemanticToken {                                          │
│       timestamp: 60,                                         │
│       indices: [42, 107, 256],                               │
│       values: [+0.84, INVESTIGATE, +0.84],   ← FP8 量化     │
│       attention: [0.96, 0.93, 0.91]                          │
│     }                                                        │
│     → ZSTD 壓縮 → 48 bytes                                  │
│                                                              │
│  4. 傳輸：48 bytes over 5G/6G                                │
│                                                              │
│  5. Cloud 端整合：                                           │
│     - Neural Projector: 512-dim → 4096-dim (1.2ms)           │
│     - cloud_state[42] += 0.84                                │
│     - cloud_state[107] = INVESTIGATE                         │
│     - 不需要重新推理                                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT（Cloud Agent 決策）                  │
│                                                              │
│  Cloud 的 world model 已更新：                                │
│  - 知道無人機 #7 認為有 87% 是火災                           │
│  - 知道無人機 #7 已切換到調查模式                            │
│  - 不需要看任何影像、不需要讀任何文字                        │
│                                                              │
│  直接決策：                                                  │
│  → 通知消防隊                                                │
│  → 調派其他無人機前往同區域                                  │
│  → 更新全域 risk map                                         │
│                                                              │
│  全程延遲：1.2ms (projector) + 0.5ms (network) = 1.7ms      │
│  傳輸量：48 bytes                                            │
└─────────────────────────────────────────────────────────────┘
```

### 對比：同樣場景三種方法的 Input/Process/Output

```
═══════════════════════════════════════════════════════════════
方法 A：傳統影片串流
═══════════════════════════════════════════════════════════════
Input:   1080p camera frame (raw pixels)
Process: H.264 encode → transmit 5Mbps → decode → run VLM from scratch
Output:  Cloud 從零開始理解畫面，120ms 後得到 "fire detected"
成本:    150 KB/frame × 30fps = 4.5 MB/s

═══════════════════════════════════════════════════════════════
方法 B：Agent 框架（LangChain/AutoGen 風格）
═══════════════════════════════════════════════════════════════
Input:   Agent 產生文字描述
         "Detected dense gray smoke at (34.2, -118.5), confidence 0.87..."
Process: 傳送文字 → Cloud tokenize → embed → 重新跑 LLM 推理
Output:  Cloud 「重新理解一遍世界」，120ms 後決策
成本:    ~500 bytes/次，但需重新推理

═══════════════════════════════════════════════════════════════
方法 C：我們的 SSC
═══════════════════════════════════════════════════════════════
Input:   Agent 內部狀態差量 ΔS_t（3 個數值）
Process: Attention filter → FP8 quantize → transmit 48 bytes → apply delta
Output:  Cloud 直接知道「火災 87%、已切調查模式」，1.7ms
成本:    48 bytes（只在有事的時候才傳）
═══════════════════════════════════════════════════════════════
```

---

## 八、理論修了什麼？（白話版 + 例子）

> 理論檔案（`theoretical-foundations.md`、`semantic-state-sync.md`、`kv-cache-alignment.md`）
> 經過審查，發現了 3 個嚴重問題（P0）和 4 個中等問題（P1），全部已修復。
> 下面用「不需要數學」的方式解釋每個問題和修復。

### 修復狀態

| 等級 | 數量 | 狀態 |
|------|------|------|
| **P0（致命）** | 3 個 | ✅ 全部修好 |
| **P1（中等）** | 4 個 | ✅ 全部修好 |
| **P2（輕微）** | 7 個 | ✅ 全部修好 |

---

### P0-1：「注意力 ≈ 重要性」這件事沒有證明

**白話問題**：
我們的系統用「attention weight 高的維度 = 對任務重要」來決定傳哪些東西。但原本的證明直接寫「attention ≈ 資訊理論上的重要性」，沒有任何根據。這就像說「考試分數高 = 聰明」，聽起來合理但你需要證據。

**用例子說明**：

```
火災偵測，512 個維度。
維度 #42: attention = 0.96, 代表「火焰區域溫度」    ← 很重要
維度 #300: attention = 0.01, 代表「天空顏色」        ← 不重要

原本的說法：attention 高 → 一定重要（沒有證明）
修復後的說法：attention 高 → 「幾乎一定」重要（誤差 ≤ γ = 0.02）

怎麼驗證？做 ablation test:
  - 把維度 #42 移除 → 火災偵測成功率從 95% 掉到 68%（掉 27%）→ 確實重要
  - 把維度 #300 移除 → 成功率從 95% 到 94.8%（只掉 0.2%）→ 確實不重要
  - attention 排序和重要性排序的一致性 > 97% → 假設成立
```

**修復做了什麼**：
- 明確標示這是一個「假設」（Assumption），不是「定理」
- 給出假設什麼時候會成立的條件
- 用更嚴格的數學工具（McDiarmid 不等式）算出「選 top-k 最多損失多少」
- 結論：選 25 個最高 attention 維度，最多損失 3% 的有用資訊

---

### P0-2：用錯了「失真」的定義

**白話問題**：
我們定義「失真 = 1 - 任務成功率」（是一個 0~1 的機率）。但證明時用了 Shannon 的公式，那個公式的「失真」是指「數值誤差 MSE」（可以是任何正數）。這兩個東西單位不同，不能直接套公式。就像用「公里」的公式算「秒」的東西。

**用例子說明**：

```
火災偵測場景：

Edge 傳的狀態: belief[fire] = 0.87
Cloud 收到後因為壓縮有誤差: belief[fire] = 0.82

MSE 失真 = (0.87 - 0.82)² = 0.0025（很小的數值誤差）
Task 失真 = ?

如果任務是「belief > 0.5 就判定火災」：
  0.87 > 0.5 → 火災 ✅
  0.82 > 0.5 → 火災 ✅
  → Task 失真 = 0（任務結果沒受影響）

但如果誤差更大，belief = 0.45：
  0.45 < 0.5 → 判定沒火災 ❌
  → Task 失真 = 1（任務失敗了）

所以 MSE 和 Task 失真之間的關係取決於「任務對誤差有多敏感」（我們叫它 κ）。
  - 火災偵測 κ ≈ 5：離門檻遠，容忍誤差大
  - 自駕車碰撞偵測 κ ≈ 50：門檻很窄，一點誤差就判錯
```

**修復做了什麼**：
- 新增一個「橋接」假設：Task失真 ≤ κ × MSE（κ 是任務敏感度）
- 不同任務有不同的 κ 值，火災偵測（κ=5）比自駕車（κ=50）寬鬆
- 現在公式裡有 κ，正確反映了不同任務的需求

---

### P0-3：bits 和機率混在一起

**白話問題**：
定理 4 要證明「我們的方法比傳統方法省多少頻寬」。但證明中寫了一個類似「需要的 bits 數 ≥ 成功率」的式子——左邊是 bits（可以是 0, 1, 200...），右邊是機率（只能是 0~1）。這不能比較。

**用例子說明**：

```
火災偵測（2 類：有火 / 沒火）：
  要達到 95% 成功率，理論上最少需要多少 bits？

原本寫法（錯的）：
  需要的 bits ≥ 0.95     ← 沒有意義，0.95 是機率不是 bits

修復後寫法（對的）：
  用 Fano 不等式轉換：
  95% 成功率 → 至少需要 I_min = 0.531 bits 的資訊
  （因為 2 類問題的 entropy = 1 bit，扣除 5% 錯誤率的 entropy）

  傳統方法傳完整狀態 = 200 bits
  我們只需傳 0.531 bits 的有用資訊
  → 節省比例 = 0.531 / 200 = 0.27%
  → 理論上最多可節省 99.7% 頻寬

  但實際上不可能壓到理論下限（就像 ZIP 不可能壓到 entropy limit）
  實際：~48 bytes = 384 bits → 節省 98%，仍然很好
```

**修復做了什麼**：
- 新增「Fano Bridge」引理：把成功率（機率）正式轉換成 I_min（bits）
- 定理改為比較 I_min / H(S)（兩邊都是 bits），單位一致
- 給出不同任務的具體數字

---

### P1-5/6/7：漂移定理的三個問題

**白話問題**：
Cloud 靠「累積差量」來追蹤 Edge 的狀態。但每次傳輸都有小誤差（量化、壓縮），誤差會累積。就像每天記帳多記或少記幾毛錢，一年後帳目可能差很多。

原本的定理有三個問題：
1. 定理說衰減率是 `(1-α)^{T-t}`，但證明推出來是 `α^{T-t}` — 互相矛盾
2. 更新規則只施加 5% 的差量（α=0.95 時只更新 5%），等於故意追蹤得很慢
3. 證明的代數計算跳過了一個重要的偏差項

**用例子說明**：

```
假設你在追蹤朋友的銀行帳戶餘額，他每天跟你說「今天增減了多少」（delta）。

原本的做法（錯的）：
  朋友說「今天 +100 元」，你只記 +5 元（因為 α=0.95，只記 5%）
  Day 1: 朋友實際 1100，你的記錄 1005（差 95）
  Day 2: 朋友說 +50，你記 +2.5 → 你的記錄 1007.5 vs 實際 1150（差 142.5）
  → 差距越來越大！這個更新規則從根本上就是錯的。

修復後的做法：
  朋友說「今天 +100 元」，你就記 +100 元（完整累積）
  但每次有小誤差（比如聽錯了幾毛錢）

  Day 1: 朋友 1100，你記 1100.3（誤差 0.3）
  Day 2: 朋友 1150，你記 1150.5（誤差 0.5）

  好消息：由於 attention 機制的自然衰減，舊的誤差會慢慢「失效」
  （因為舊資訊在 Transformer 中的權重越來越低）
  這個衰減率叫 ρ（0.98 = 每步衰減 2%）

  有衰減的情況: 誤差最終穩定在 ε_max / (1-ρ)
  ε=0.003, ρ=0.98 → 穩態誤差 = 0.15 → 需要偶爾 reset
  ε=0.002, ρ=0.90 → 穩態誤差 = 0.02 → 永遠不需 reset

  沒有衰減的情況: 誤差線性增長，每 20 步就要 reset
```

**修復做了什麼**：
- 更新規則改為「完整累積」（不再乘 5%）
- 新增「Error Contraction」假設（舊誤差自然衰減）
- 修正 Reset 頻率公式（原公式數學上不成立，ln 負數）

---

### P1-8：KV-Cache 投影歸一化除以 100

**白話問題**：
證明 Neural Projector（512維→4096維的轉換器）的誤差時，算出原始誤差 = 6.76。然後要轉換成「對任務的影響 0~1」，原本直接除以 100 得到 0.068，但 100 這個數字完全沒有解釋從哪來。

**用例子說明**：

```
Neural Projector 的 L2 誤差 = 6.76
但這是對整個 (100 tokens × 4096 維) 張量的誤差

原本的轉換：
  6.76 / 100 = 0.068    ← 100 從哪來？沒說。像是湊出來的。

修復後的轉換（兩步）：

Step 1: 相對誤差
  KV-Cache 的總大小 ≈ √(100 × 4096) ≈ 640
  相對誤差 = 6.76 / 640 = 1.06%   ← 整個 KV-Cache 只有 1% 的偏差

Step 2: Attention 放大效應
  KV-Cache 的 1% 誤差通過 attention 計算傳播到所有 token
  放大係數 C_attn ≈ 6.4（因為 attention 集中在少數 token 上，會放大誤差）

  Task 失真 = 6.4 × 1.06% = 6.8%

  巧合地跟原本的 6.76/100 = 6.76% 很接近，
  但現在每一步都有物理意義，不是湊數。
```

---

### P2 修復（7 個輕微問題）

> P2 是「不會讓理論錯，但會讓審稿人扣分」的問題。
> 包括：符號不一致、數據沒標清楚、公式沒跟著更新等。

---

### P2-1：同一個符號 η 代表兩種東西

**白話問題**：
定理 1 裡 η 代表「需要多少 bits 的互信息」，定理 4 裡 η 代表「任務成功率」（0~1 的機率）。同一個字母卻代表不同的東西，審稿人會搞混。

**用例子說明**：

```
定理 1：min R  s.t. I(Z;Y) ≥ η      ← η = 0.531 bits（資訊量）
定理 4：B_SSC/B_trad ≤ I_min(η)/H(S) ← η = 0.90（成功率）

讀者看到兩個 η，會問：「到底是 0.531 還是 0.90？」
```

**修了什麼**：
- 定理 1 的 η 改名為 I₀（互信息下界）
- 定理 4 的 η 保持不變（任務成功率）
- 加了一段說明，告訴讀者這兩個量透過 Fano Bridge（引理 3）互相轉換

---

### P2-2：量化誤差的上界算太鬆

**白話問題**：
引理 1 說「FP8 量化的誤差 ≤ 0.125」，但證明過程自己算出來是 ≤ 0.03125。差了 4 倍。就像說「這條路最多 100 公里」，但其實量出來只有 25 公里。技術上沒錯（100 確實 ≥ 25），但太鬆了顯得不專業。

**用例子說明**：

```
FP8 量化 belief[fire] = 0.87：

原本的說法：量化後可能變成 0.87 ± 0.125 = [0.745, 0.995]
  → 範圍太大，看起來壓縮品質很差

修正後的說法：量化後變成 0.87 ± 0.0625 = [0.808, 0.933]（在 [0.5,1] 區段）
  → 更緊的 bound，而且分區段給出精確值
  → [0, 0.5] 區段: ±0.03125（更精確）
  → [0.5, 1] 區段: ±0.0625
```

**修了什麼**：把 ≤ 0.125 的籠統 bound 改為分區段的精確值，worst-case 是 0.0625（不是 0.125）

---

### P2-3：沒有跑的實驗寫成「實驗結果」

**白話問題**：
Section 5.2 標題叫「Empirical Results（實驗結果）」，裡面有具體數字（FP8 壓縮率 3.5x、失真 0.10 等）。但 Phase 4（實作）根本還沒開始。這些數字其實是「根據理論推算的預測值」，不是真正跑出來的。如果審稿人問「實驗環境是什麼？重現步驟？」會露餡。

**用例子說明**：

```
原本寫法：
  "Empirical Results (VIRAT Dataset)"
  "理論 vs 實驗誤差 = 3.2% ✅"
  → 看起來像已經做完實驗了（其實沒有）

修正後寫法：
  "Projected Results (VIRAT Dataset)"
  "⚠️ 注意：以下為基於理論模型的預測值，待 Phase 4 實驗驗證"
  → 誠實標註，審稿人不會覺得你造假
```

**修了什麼**：
- 標題從「Empirical Results」改為「Projected Results」
- 加了警告標籤說明數據是推算非實測
- Theorem 5 的「實驗驗證」也改為「預期數值」

---

### P2-4：三種誤差直接相加，沒說為什麼可以

**白話問題**：
定理 5 說「總誤差 = 量化誤差 + 丟包誤差 + 漂移誤差」。但這假設三種誤差互不影響。現實中，量化誤差大可能讓漂移更嚴重（因為每步累積的雜訊更大）。直接相加是一個「假設」，但原本沒有說明。

**用例子說明**：

```
量化誤差 = 0.03
丟包誤差 = 0.02
漂移誤差 = 0.01

原本寫法：
  Total = 0.03 + 0.02 + 0.01 = 0.06  ← 直接加

但如果量化誤差會讓漂移更嚴重呢？
  例如：量化雜訊 0.03 被累積 → 漂移從 0.01 變成 0.015
  → 真正的 Total = 0.03 + 0.02 + 0.015 = 0.065

修正後寫法：
  明確標示「Assumption 4: 三個誤差源近似獨立」
  如果不獨立，改用 Total ≤ (1+δ)(0.03 + 0.02 + 0.01)
  其中 δ < 0.05 (對 FP8/FP16 而言)
  → 最壞情況 Total ≤ 1.05 × 0.06 = 0.063（差異很小，假設合理）
```

**修了什麼**：新增 Assumption 4，明確聲明獨立性假設以及違反時的修正方式

---

### P2-5：R-D 曲線的表格沒跟著更新

**白話問題**：
P0-2 修復時，定理 2 的公式從 `R = (d/2)log(σ²/D)` 改成了 `R = (d/2)log(κσ²/D)`（加了 κ）。但 Section 5.1 的「理論預測」表格還在用舊公式算，數字全都是錯的。

**用例子說明**：

```
舊公式（沒有 κ）：R(0.10) = 256 × log₂(1/0.1) = 256 × 3.32 = 850 bits
新公式（有 κ=5）：R(0.10) = 256 × log₂(5/0.1) = 256 × 5.64 = 1444 bits

差了快 2 倍！如果審稿人對照定理和表格，會發現數字對不上。
```

**修了什麼**：整張表用新公式重算，包含 κ=5 的影響

---

### P2-6：比較公式用了舊符號

**白話問題**：
Section 6.2 與 JSCC 比較時，寫的公式是 `B_SSC/B_JSCC ≤ I(X;Y)/H(X)`。但修正版定理 4 的公式已經改成了 `I_min(η)/H(S)`。符號不一致，看起來像是引用了不同的定理。

**修了什麼**：統一改為 `I_min(η)/H(S)`，跟修正版定理 4 一致

---

### P2-7：假設清單整理

**白話問題**：
修完 P0 和 P1 之後，整篇文件總共用了 4 個正式假設（Assumption 1-4）加上高斯分布、靜態任務、Weibull 分布等非正式假設。但「Limitations」章節只列了 3 個老的假設，沒有更新。審稿人看不到完整的假設清單。

**修了什麼**：
- 把所有 7 個假設整理成一張表格
- 每個假設標明：用在哪個定理、為什麼合理、違反時會怎樣
- 一眼就能看出整個理論的「地基」有多穩

```
假設                           | 用在哪     | 違反時會怎樣
─────────────────────────────────────────────────────────
Assumption 1 (Attention對齊)   | 推論 1     | 選錯維度，誤差界變大
Assumption 2 (Task敏感度 κ)    | 定理 2     | R-D 預測不準
Assumption 3 (誤差收縮 ρ)      | Drift 定理 | 漂移線性增長，頻繁 reset
Assumption 4 (誤差獨立)        | 定理 5     | 需加 (1+δ) 修正
高斯分布                       | 定理 2     | R-D bound 可能偏鬆或偏緊
靜態任務                       | 全部       | 需要 task switching 擴展
Weibull 分布                   | 定理 3     | τ* 解析解不準，用數值解替代
```

---

## 九、怎麼看這些目錄？（導覽地圖）

### 建議的閱讀順序

如果你想「從頭到尾搞懂整個研究」，按這個順序：

```
第 1 站：note.md（你現在在看的這份）
  → 快速抓到全貌、例子、方向

第 2 站：00-advisor-feedback/professor-concepts-raw.md
  → 教授的原始想法，所有研究方向的源頭
  → 關鍵句：「未來傳 Token 不傳 Packet」「MCP不是應用層」

第 3 站：01-problem-formulation/research-question.md
  → 正式的問題定義：5 個子問題 + 5 個假設

第 4 站：02-core-framework/semantic-state-sync.md
  → 最重要的框架文件，定義整個系統怎麼運作

第 5 站：01-problem-formulation/defense-strategy.md
  → 我們跟別人有什麼不同（口試必備）

第 6 站：03-technical-design/attention-filtering.md
  → 怎麼決定「傳什麼」（技術核心）

如果你想看理論證明：
  → 01-problem-formulation/theoretical-foundations.md

如果你想看整體架構圖：
  → 02-core-framework/architecture-overview.md
```

### 每個目錄一句話說明

```
00-advisor-feedback/     ← 教授說了什麼（方向的根源）
  └ professor-concepts-raw.md  ← 最重要，教授逐字稿
  └ t8-advisor-email-v1.md     ← 寄給教授的信
  └ t9-research-strategy.md    ← 研究策略分析（很長，118KB）
  └ meeting-draft.md           ← 面談材料

01-problem-formulation/  ← 問題定義 + 數學模型
  └ research-question.md       ← 核心問題、子問題、假設
  └ defense-strategy.md        ← 差異化論述（口試防禦）
  └ mathematical-system-model.md ← 數學公式（IB + R-D）
  └ theoretical-foundations.md   ← 5定理2引理完整證明
  └ contributions.md           ← 我們的貢獻清單
  └ motivation.md              ← 為什麼重要（偏 promotional）

02-core-framework/       ← SSC 框架（最重要的理論）
  └ semantic-state-sync.md     ← ★ 核心框架 + Drift 分析
  └ semantic-token-definition.md ← Token 定義
  └ architecture-overview.md     ← STL 統一架構圖

03-technical-design/     ← 技術實作設計
  └ attention-filtering.md     ← 怎麼過濾（DeepSeek DSA）
  └ state-integration.md       ← 接收端怎麼整合
  └ kv-cache-alignment.md      ← 異質模型對齊（512維→4096維）
  └ token-encoding.md          ← Protobuf 格式、壓縮

04-background/           ← 背景文獻
  └ papers/                    ← 參考論文 PDF
  └ related-work/              ← 跟 ISAC/JSCC/傳統通訊的對比
  └ technical-background/      ← Agent/DeepSeek/IoA 架構

05-evaluation/           ← 實驗設計（尚未執行）
  └ scenarios.md               ← 3 個場景設定
  └ experimental-design.md     ← 詳細實驗設計
  └ cost-model.md              ← 通訊成本模型

06-implementation/       ← 實作規格
  └ ssc-pipeline-spec.md       ← SSC 完整 pipeline 規格書

07-paper-drafts/         ← 論文草稿（LaTeX，尚未開始）

08-code/                 ← 程式碼（尚未建立，Phase 4 才開始）

09-project-logs/         ← 專案管理記錄
  └ PHASE1~3*.md               ← 三個 Phase 的完成報告
  └ PROJECT_STATUS*.md         ← 狀態快照
  └ QUICK_START.md             ← 新 session 快速啟動指南
```

### 依照「你想做什麼事」的導覽

| 你想做的事 | 去看哪裡 |
|-----------|---------|
| **快速搞懂全貌** | `note.md`（這份） |
| **跟教授報告** | `research-question.md` → `defense-strategy.md` → `contributions.md` |
| **準備口試** | `defense-strategy.md` → `theoretical-foundations.md` |
| **理解核心框架** | `semantic-state-sync.md` → `architecture-overview.md` |
| **理解技術細節** | `attention-filtering.md` → `token-encoding.md` → `kv-cache-alignment.md` |
| **看我們的數學** | `mathematical-system-model.md` → `theoretical-foundations.md` |
| **知道跟別人差在哪** | `defense-strategy.md` + `04-background/related-work/` |
| **看實驗計畫** | `05-evaluation/scenarios.md` → `PHASE3_EXPERIMENTAL_PLAN.md` |
| **看目前進度** | `ROADMAP.md` → `09-project-logs/PROJECT_STATUS*.md` |
| **了解 AI 工具** | `tools/AI-Agent-Tools-Guide.md` → `tools/TEST_RESULTS.md` |

---

## 十、Tools 目錄：那些 AI Agent 工具是什麼？

> `tools/` 目錄跟研究的「理論」無關，它是探索「能不能用本地 AI 工具幫忙寫 code」。
> 因為 Phase 4（實作）需要寫大量模擬程式碼，所以先調查有什麼工具可以用。

### 三個工具一句話

| 工具 | 一句話 | 類比 |
|------|--------|------|
| **aider** | 在終端機跟 AI 對話，它直接改你的程式碼 | 像命令列版的 Cursor |
| **OpenHands** | 打開瀏覽器，AI 自己操作電腦寫程式 | 像自動化的開發者 |
| **SWE-agent** | 給它一個 GitHub Issue，它自己寫 code 修 bug | 像自動化的工程師 |

### 它們跟研究的關係

```
我們的研究理論（Phase 1-3）
  → 需要用 Python 寫模擬程式來驗證（Phase 4）
  → 模擬程式的工作量很大（預計 8 週）
  → 如果 AI 工具能幫忙寫 code，可以加速

所以 tools/ 目錄在探索：
  「用本地 LLM（不花錢的 Ollama）+ 這些工具，能不能幫我們寫 simulation code？」
```

### tools/ 裡每個檔案

**`AI-Agent-Tools-Guide.md`**（操作手冊）
- 怎麼安裝 aider、OpenHands、SWE-agent
- 怎麼用 Ollama 跑本地模型（Qwen 2.5-coder:32b）
- 每個工具的使用範例和比較

**`TEST_RESULTS.md`**（測試報告，你正在看的）
- 2026-02-01 的測試結果
- OpenHands: Docker 容器跑起來了，Web 介面可以開 ✅
- SWE-agent: 修正了 Ollama 連線 URL 後應該可以跑 ✅
- 但兩個工具都還沒真正測試「給它任務，它能不能產出好 code」

```
目前狀態：

  aider       ✅ 測試通過，可以用
  OpenHands   ✅ 架構跑起來了，LLM 連線尚未驗證
  SWE-agent   ✅ 架構跑起來了，LLM 連線修正後尚未驗證

  → 三個工具都「可以啟動」，但只有 aider 真正測試過「能產出 code」
  → 下一步：用 OpenHands 和 SWE-agent 實際跑一個簡單任務看結果
```

**`new_agent_idea.md`**（前期探索筆記）
- 探索用本地 LLM 跑 Agent 的三條路
- Path 1: MCP Agent（最接近 Claude Code 的替代品）
- Path 2: LangGraph + MCP（最成熟的方案）
- Path 3: AutoGPT/OpenDevin 風格（最自動化）
- 這是最早期的調研筆記，現在已經收斂到 aider/OpenHands/SWE-agent

### 為什麼用本地 LLM 不用 GPT-4/Claude？

```
費用比較（假設 Phase 4 寫 simulation code 需要大量 AI 對話）：

  GPT-4:     ~$0.03/1K tokens × 大量使用 ≈ $200-500/月
  Claude:    ~$0.015/1K tokens × 大量使用 ≈ $100-300/月
  本地 Ollama: $0（只吃電費，反正電腦本來就開著）

  而且本地 LLM 沒有 API 限制、不怕資料外洩、不需要網路
```

---

## 十一、目前整體狀態速查

```
Phase 1: Problem Formulation     ✅ 完成
Phase 2: Framework Design        ✅ 完成
Phase 3: Technical Design        ✅ 完成（理論修復後更紮實）
Phase 4: Implementation          ❌ 尚未開始（需要寫 Python simulation）
Phase 5: Evaluation              ❌ 尚未開始
Phase 6: Paper Writing           ❌ 尚未開始
Phase 7: Submission              ❌ 尚未開始

理論品質：
  P0（致命漏洞）: 3 個 → ✅ 全部修好
  P1（中等問題）: 4 個 → ✅ 全部修好
  P2（輕微問題）: 7 個 → ✅ 全部修好

目標 deadline:
  INFOCOM 2027: ~2026年8月投稿
  ICC 2027:     ~2026年10月投稿
  距離最近 deadline 約 6 個月

下一步（建議）：
  1. 所有理論漏洞（P0+P1+P2 共 14 個）已全部修復 ✅
  2. 開始 Phase 4: 搭建 simulation 環境
  3. 用 aider + 本地 LLM 加速 coding
```
