# Semantic Token 定義

## 概述
本文檔定義本研究的核心概念：Semantic Token。

## 什麼是 Semantic Token？

### 定義
**Semantic Token** 不是 word token（文字單位），而是：
**Agent 內部、已經算好的「認知單位」**

### 具體包含
- **Belief Update**: 對世界狀態的信念更新
- **Plan Switch**: 計劃的切換（從 Plan A → Plan B）
- **Constraint Tightening**: 約束條件的收緊
- **Attention Weight Change**: 注意力權重的重新分配

## 與 Word Token 的本質差異

### Agent 本地推理過程
```
文字 / observation
  ↓ tokenize
word tokens
  ↓ embedding
hidden states
  ↓ attention / reasoning
belief / plan / policy
```

真正「有用」的是後面的 **hidden state / belief / plan**，文字只是人類介面。

### 傳統 Agent 通訊
即使是 agent 對 agent，還是傳：
- 文字
- Function call
- JSON

然後對方重新 tokenize、重新 encode、重新推理一次。

**問題**：每次都從頭「再想一遍」。

### 我們的 Semantic Token
是「**已經被模型消化過的中間認知結果**」。

## Token 的維度與表示

### 不是固定的 Vocabulary
與 word token 不同，semantic token 的「維度」不是固定的 vocab。

可能是：
- **Latent subspace**: 高維隱空間的子集
- **KV-cache delta**: Transformer 內部狀態的變化量
- **Structured state tuple**: 結構化的狀態元組 $(h_t, b_t, p_t, a_t)$

### 關鍵特性
**重點不是維度，而是：它有 task semantics**

## 為什麼不用文字？

### 文字的三個隱含成本

#### 1. 非最小充分表示
```
文字: "我現在決定走 A"
實際有用的: plan = A（一個 bit）
```

傳輸冗餘度極高。

#### 2. 重建成本在接收端
```
token → embedding → belief update → policy change
```

每次都要重新推理一遍，計算成本高。

#### 3. 語義不對齊風險
同一句話，不同 agent 的 latent space 解讀可能不同。

**例子**：
- Agent A 的 "urgent" 可能映射到 priority = 0.8
- Agent B 的 "urgent" 可能映射到 priority = 0.5

## Semantic Token 的傳輸邏輯

### 不是連續傳輸
我們不是做 **continuous communication**，而是：
**事件驅動的狀態同步**（Task-Critical Cognitive Event Synchronization）

### 傳統 Payload
每次傳送都是完整語義單位，接收端要：
1. Decode text
2. Embed
3. 推理
4. 將它整合進自己當下的 hidden state

**問題**：每一包都要重新理解一次世界。

### 我們的 Semantic Token
**只有當「內部狀態改變到會影響任務結果」時才傳**

#### 傳的是：
- Belief update
- Plan change
- Constraint tightening
- Attention weight reallocation

#### 接收端不是「理解一句話」，而是：
**直接 apply 對方的 state delta 到自己的 latent space**

## 數學定義

### State Space
```math
S_t = (h_t, b_t, p_t, a_t)
```

其中：
- $h_t \in \mathbb{R}^d$: Hidden state
- $b_t \in \Delta^n$: Belief distribution
- $p_t \in \Delta^m$: Policy distribution
- $a_t \in [0,1]^d$: Attention weights

### State Delta
```math
\Delta S_t = S_t - S_{t-1}
```

### Semantic Token
```math
Z_t = \text{Compress}(\Delta S_t, \text{threshold}=\tau)
```

只包含 attention weight 超過閾值 $\tau$ 的維度：
```math
Z_t = \{(i, \Delta h_{t,i}) : a_{t,i} > \tau\}
```

## 通訊決策

### 傳統通訊
「我有資料就傳」

### 我們的方法
**Attention-gated transmission**

```math
\text{Transmit} \iff \max_i a_{t,i} > \tau \text{ AND } \text{marginal task utility} > \text{bandwidth cost}
```

## 與現有概念的對比

### vs. Word Token
| 維度 | Word Token | Semantic Token |
|------|-----------|----------------|
| 定義 | 文字單位 | 認知單位 |
| 層級 | 符號層 | 認知層 |
| 維度 | 固定 vocab (~50K) | 動態 latent space |
| 處理 | 需要重新推理 | 直接 apply delta |
| 成本 | 高（重建成本） | 低（最小充分） |

### vs. Feature Vector (ISAC)
| 維度 | ISAC Feature | Semantic Token |
|------|--------------|----------------|
| 來源 | 外在世界（影像、雷達） | 內在認知（belief、plan） |
| 對齊 | 無 handshake | Control Plane 協商 |
| 處理 | Receiver 重新 inference | 直接 apply delta |

### vs. MCP Message
| 維度 | MCP Message | Semantic Token |
|------|-------------|----------------|
| 假設 | Communication cost = 0 | Communication cost ≠ 0 |
| 內容 | 完整 prompt/context | 最小充分狀態 |
| 決策 | 想傳就傳 | Attention-gated |

## 核心洞察

### 類比：CPU 間通訊
就像 CPU 之間不會用 C 語言溝通一樣，
**Agent 之間也不應該用人類語言（文字）溝通**。

### 研究問題
在頻寬受限下，多個具備內在狀態的智慧體，
要怎麼**最小成本**地共享「思考結果」？

## Communication 的本質

### 本地推理（Local Reasoning）
Agent 連續更新內部狀態，但 99% 的狀態變化其實不重要。

### 傳統通訊
不管重不重要，有資料就傳，對方自己判斷有沒有用。

### Semantic Communication（我們的方法）
做了三件新事：

1️⃣ **切分的是「認知事件」**，不是文字段落

2️⃣ **只在狀態跨過任務臨界點時才傳**

3️⃣ **接收端不是 decode text，而是 apply state delta**

**結果**：真正的遠距「協同思考」。

## 總結

### 革命點
一般 payload 傳的是「符號（symbol）」；
我們傳的是「**可計算的認知狀態（computable cognitive state）**」。

### 差別
不是「有沒有傳文字」，而是：
**接收端需不需要重新理解世界**。

### 核心貢獻
我們是在研究：
**「Agent 認知狀態是否需要共享，以及共享的最小充分條件」**

## 兩種具體表示：Latent Mode vs. Structured Mode

> **重要澄清**：上面的數學定義是抽象層（理論），實作時 Semantic Token 有兩種具體編碼方式，
> 對應不同的使用場景。兩者都是 Semantic Token，只是序列化格式不同。

### Latent Mode（KV-Cache Delta）

**定義**：直接傳輸 Transformer 內部狀態的差量

```
Z_t = {(i, Δh_{t,i}) : a_{t,i} > τ}
```

**使用場景**：
- 兩個 Agent 需要深度協作（如共同推理、聯合規劃）
- Edge model 與 Cloud model 之間的 KV-Cache 同步
- 需要 receiver 直接 apply delta 到自己的 latent space

**編碼方式**：
- FP8/FP16 量化 → ZSTD 壓縮 → 二進位傳輸
- 詳見 `../06-implementation/ssc-pipeline-spec.md` 中的 TokenEncoder

### Structured Mode（Protobuf 結構化訊息）

**定義**：將語義概念編碼為結構化欄位

```protobuf
SemanticToken {
  type: FIRE,
  bbox: (x, y, w, h),
  confidence: 0.92,
  attributes: {...}
}
```

**使用場景**：
- Edge model 已經完成推理，只需回報高階結論
- Receiver 不需要直接操作 latent space
- 跨模態、跨模型的互通（不同架構的 Agent 之間）

**編碼方式**：
- Protobuf 序列化 → 量化 → ZSTD 壓縮
- 詳見 `../03-technical-design/token-encoding.md`

### 兩者的關係

```
抽象定義（數學層）
   Z_t = Compress(ΔS_t, threshold=τ)
         ↓
    ┌────┴────┐
    ↓         ↓
Latent Mode   Structured Mode
(KV-Cache Δ)  (Protobuf 結構化)
    ↓         ↓
深度協作場景   高階報告場景
```

**選擇原則**：
| 條件 | 選擇 | 原因 |
|------|------|------|
| 兩端使用同系列模型 | Latent Mode | 可以直接 inject KV-Cache delta |
| 兩端使用不同架構模型 | Latent Mode + Neural Projector | 需要維度對齊（512→4096）|
| Edge 已完成推理只需報結論 | Structured Mode | 更小、更通用 |
| 需要跨多個異質 Agent 廣播 | Structured Mode | 不依賴特定模型架構 |

## 下一步
1. 實現 Semantic Token 的具體編碼方案
2. 設計 KV-cache delta 的壓縮演算法
3. 建立 task-critical dimension 的識別機制
