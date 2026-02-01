# Defense Strategy: 與現有研究的本質差異

## 概述
本文檔明確說明我們的研究與現有 ISAC、JSCC、MCP、Agent Frameworks 的本質差異，
建立清晰的理論護城河。

## 核心差異總覽

### 與 ISAC 的本質差異
**ISAC**: Integrated Sensing and Communication（頻譜共享）
**我們**: 改變傳輸單位（bit → semantic state）

### 與 JSCC 的本質差異
**JSCC**: Joint Source-Channel Coding（資料重建）
**我們**: Task-oriented state synchronization

### 與 MCP/Agent Frameworks 的本質差異
**MCP等**: 假設 communication cost = 0
**我們**: 在 communication 有成本時，agent 如何協作

## 三個革命點

### 1. 傳輸單位改變
```
傳統：Symbol / Bit / Packet
我們：Δhidden_state / Δbelief / Δpolicy manifold
```

**意義**：從 source coding → task-oriented representation coding

### 2. 決策機制改變
```
傳統：有資料就傳
我們：Attention-gated transmission
     Only transmit if marginal task utility > bandwidth cost
```

**意義**：這是經典 Shannon communication 做不到的東西

### 3. 評估指標改變
```
傳統：BER、Latency、Throughput
我們：Task Success Rate under Bandwidth Constraint
```

## 核心論述

### 問題定義
現有 agent communication（包含 MCP、ISAC）都是在假設 communication 是免費的前提下，
傳資料或 feature，讓對方重新 inference。

我們關心的是另一個問題：
**在 communication 有成本時，agent 能不能只同步「足夠完成任務的認知狀態」？**

### 技術定位
不是在優化「怎麼傳得更快」，是在改「該不該傳、傳什麼」。

## 詳細對比

### vs. Traditional Communication
傳統通訊關心的是：壓縮、編碼、error rate、頻譜效率
**目標**：重建訊息

我們問的是：「我是不是一定要讓對方『知道我知道的全部』？」
**目標**：完成任務

### vs. ISAC
ISAC 的三個假設：
1. 傳的是「外在世界的描述」（影像特徵、雷達回波）
2. Receiver 要自己「重新想一次」
3. 沒有認知對齊的 handshake

我們的差異：
1. 傳的是「內在認知狀態」（belief update, plan switch）
2. Receiver 直接 apply state delta
3. 有 Control Plane 協商（goal 對齊、state space 對齊）

### vs. JSCC
JSCC 目標：min D_MSE(X, X̂)
我們目標：min D_task(S, Ŝ) = 1 - Task_Success_Rate

重點不是「重建資料」，而是「同步認知狀態」。

### vs. MCP/Agent Frameworks
現在所有 Agent framework（MCP、LangGraph、AutoGen）都假設：
- 傳訊息很便宜
- 想傳就傳
- 延遲、頻寬、次數都不用算

我們的問題：
如果 agent 之間頻寬有限、延遲不可忽略、傳太多會影響任務，
還能不能合作？要怎麼合作？

## Semantic Token 定義

### 什麼是 Semantic Token？
不是 word token。

我們指的是：Agent 內部、已經算好的「認知單位」。
- belief update
- plan switch
- constraint tightening
- attention weight change

### 與 Word Token 的差異

| 維度 | Word Token | Semantic Token |
|------|-----------|----------------|
| 定義 | 文字單位 | 認知單位 |
| 維度 | 固定 vocab | Latent subspace |
| 處理 | 需要重新推理 | 直接 apply delta |
| 成本 | 重建成本在接收端 | 最小充分表示 |

### 為什麼不用文字？
文字有三個隱含成本：
1. **非最小充分表示**：「我現在決定走 A」實際有用的可能只有一個 bit：plan = A
2. **重建成本在接收端**：token → embedding → belief update → policy change
3. **語義不對齊風險**：同一句話，不同 agent latent space 解讀不同

## 技術創新點

### 1. 事件驅動的狀態同步
不是 continuous communication，而是：
**Task-Critical Cognitive Event Synchronization**

### 2. Attention-based Filtering
只在狀態跨過任務臨界點時才傳：
- 切分的是「認知事件」，不是文字段落
- 只在狀態跨過任務臨界點時才傳
- 接收端不是 decode text，而是 apply state delta

### 3. Control Plane 協商
與 ISAC 最大的差異：
在傳之前，先協商「我們要不要共享腦中的哪一部分」
- goal 對齊
- state space 對齊
- attention threshold 對齊

## 預期質疑與回應

### Q: 這不就是壓縮嗎？
A: 不是。壓縮是為了重建，我們是為了任務。評估指標不同。

### Q: ISAC 也是 task-oriented，有什麼不同？
A: ISAC 傳外在世界描述，我們傳內在認知狀態。ISAC 接收端要重新推理，我們直接 apply delta。

### Q: 為什麼不用 MCP？
A: MCP 是 application-layer 協定，假設通訊成本為零。我們要處理的是通訊有成本時的協作問題。

### Q: 實驗怎麼做？
A: Trace-driven simulation，這是通訊領域的標準方法。

## 總結
我們不是改 physical layer，而是定義一個 **Semantic Transport Layer**，
並引入 handshake 與 attention-based transmission decision。

這是一個全新的通訊範式：
從「Bit Transmission」→「Semantic State Synchronization」
