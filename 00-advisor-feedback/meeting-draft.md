# 給教授的報告草稿

## 郵件草稿

尊敬的老師好，

我是煒倫，與您更新近期的研究進度，以下有三點：

### 1️⃣ 研究概念
延續老師之前提到的「Token-based Agent Communication」方向，目前有初步想法方向。

**核心概念**：把通訊從「還原資料」變成「同步 Agent 腦中狀態」，讓 Agent 在有限頻寬下也能高效協作。

- **傳輸單位**從 Packet → Semantic Token（Agent 內部狀態）
- 用 **Attention 機制**當作 Filter，只傳 Task-Critical 的資訊
- 參考 **MCP 概念**設計 Control Plane，解決異 Agent 對齊問題

### 2️⃣ AI 技術變化研究
為了對 AI 溝通有更深入理解，以便完成上面的研究，於是對多種 AI 進程技術廣度與彼此關聯性進行整理，包含新的 AI Agent 框架資料（DeepSeek-V3、Agent 架構等），有搜集資料，準備與老師分享說明。

### 3️⃣ Claude Code 使用心得
上次與老師討論到 Agent，主要展現自動寫程式的 cursor。老師有問到的怎麼不使用 Claude Code，後來有聽取建議開始使用。的確更佳優異，跟 Cursor 比起來，Claude Code 更適合做系統性的程式開發，而且 Anthropic 最近推出了「MCP」和「Skills」兩個新標準機制，讓 AI Agent 可以標準化地呼叫外部工具，這跟我們在研究 Agent 通訊，到逐漸建立 Agent 溝通機制蠻有關聯。

想請問老師有方便的時間向您報告：
- **1️⃣ 研究概念**：還在草稿，想法、架構都還沒有收斂的那麼完整。
- **2️⃣ AI 技術變化研究簡報**、**3️⃣ Claude Code 使用心得**：已經可以向老師報告

---

## 核心概念簡述

### 標題
**「從 Bit Transmission → Semantic State Synchronization」**

### 簡單說明

#### 傳什麼
不傳 raw data / feature，而是傳 Agent 內部的「**狀態差分**」（類似 Transformer 的 KV Cache delta）

#### 怎麼決定傳不傳
用 **Attention 機制**當 Filter，只傳對任務有影響的部分（Task-Critical）

#### 跟 ISAC 的差別
ISAC 是 sensing + comm 共用頻譜，我們是改變「**傳輸的單位與決策機制**」

### Layer 定位
不是傳統 L1-L7 的某一層，而是定義一個新的「**Semantic Transport Layer**」，介於傳統通訊層和 Application 之間。物理層還是 0101，但 payload 變成 semantic token。

### MCP 的角色
把它當成 **Control Plane**（類似 SIP/RRC），用來做 semantic handshake——協商雙方的 goal、embedding format、attention threshold，確保傳過去的 token 能被正確理解。

---

## 可能問題與回答

### 1. 跟 ISAC 差在哪？
→ 改變的是**傳輸單位和決策機制**，不是 sensing+comm

### 2. Layer 是什麼？
→ 新的 **Semantic Transport Layer**，不取代物理層

### 3. MCP 怎麼用？
→ **Control Plane**，做 semantic handshake

---

## 詳細版報告

### 先 Recap 老師之前的想法

老師之前提到幾個關鍵概念：
- 「未來 Agent Communication 會很重要，傳的可能不是 Packet，而是 Token」
- 「現在 LangChain / AutoGen 做 Agent，都假設頻寬無限、延遲為零，沒考慮通訊成本」
- 「要從 Task / Goal-oriented 角度來看：我應該傳什麼？怎麼傳？」
- 「Agent 跟 Agent 之間會產生什麼行為？這是大家現在最關心的問題」

### 我們怎麼把這些想法收斂成具體方向

| 老師的問題 | 我們的回應 |
|-----------|-----------|
| 「傳 Token 不是 Packet，但怎麼做？」 | → 傳 **Semantic Token**（Agent 內部狀態的差分，類似 KV Cache delta） |
| 「怎麼決定該傳什麼？」 | → 用 **Attention 機制**當 Filter，只傳對任務有影響的部分 |
| 「Agent 間怎麼協調？」 | → **MCP 當 Control Plane**，做 semantic handshake（協商 goal、格式） |
| 「通訊成本怎麼考慮？」 | → 目標是**最小化傳輸量**，同時保證 Task Success Rate |

### Layer 定位
這不是改 L1-L7 某一層，而是定義一個新的「**Semantic Transport Layer**」。物理層還是傳 0101，但 payload 變成 semantic token，決策機制變成 task-oriented。

### 跟 ISAC 的差別
ISAC 是 sensing + comm 共用頻譜；我們是改變「**傳輸的單位**」（從 bit → semantic state）和「**決策機制**」（從重建資料 → 同步認知狀態）。

### AI 工具更新
整理了一些新的 AI Agent 框架資料，之後可以跟老師分享。

### Claude Code
上次老師問的我開始用了，Anthropic 的 MCP 機制跟我們研究方向有關聯。

**老師方便的話可以約時間當面報告！**

---

## 技術簡報補充

### 2025 大模型技術深度剖析

我針對《2025 AI 生態》製作了一份深度技術簡報。不同於一般的趨勢報告，我深入拆解了技術背後的「數學原理」與「系統工程」。

內容包含：

#### 演算法層
- DeepSeek MLA 的低秩矩陣壓縮原理
- MoE 的 Router 機制與專家坍塌問題

#### 系統層
- 訓練端的 3D 平行通訊瓶頸
- ZeRO-3 機制
- vLLM (PagedAttention) 與 Flash Attention 的 IO 感知原理

#### 應用層
- 從 LangGraph 的狀態機架構談到邊緣端 QLoRA 微調
- 結合 Triton Compiler 提出戰略分析

之後可以跟老師詳細報告這些底層技術細節。

---

## 備註

### 報告重點
1. ✅ 研究方向與老師想法一致
2. ✅ 有具體的技術路徑（Semantic Token + Attention Filter + MCP Control Plane）
3. ✅ 明確與 ISAC 的差異
4. ✅ 展現技術深度（AI 技術變化研究）

### 待討論
- 研究問題的精確定義
- 實驗設計的可行性
- 論文投稿的目標會議
- 時間規劃
