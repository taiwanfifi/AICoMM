# vs. ISAC: 本質差異分析

## ISAC 定義
**Integrated Sensing and Communication (ISAC)**: 在同一頻譜資源上同時進行感知和通訊。

## 核心區別

### 我們與 ISAC 的差異
| 維度 | ISAC | 我們的方法 |
|------|------|-----------|
| **傳輸單位** | Sensing data + Communication data | Semantic state delta |
| **傳輸內容** | 外在世界的描述（影像、雷達） | 內在認知狀態（belief, plan） |
| **接收端處理** | 重新 inference | 直接 apply delta |
| **對齊機制** | 無認知對齊 | Control Plane handshake |
| **目標** | 頻譜效率 | 任務成功率 |

## ISAC 的三個隱含假設

### 1. 傳的是「外在世界的描述」
例如：
- 影像特徵
- 雷達回波
- Sensor embedding

**不是** agent 的內在認知狀態。

### 2. Receiver 要自己「重新想一次」
```
收到 embedding → 自己 inference → 自己更新 belief
```
Sender 不知道 Receiver 怎麼用這些資訊。

### 3. 沒有認知對齊的 handshake
不會先協商：
- 你現在在做什麼任務？
- 你關心哪些 latent？
- 哪些 state 是 critical？

## 我們的創新

### 1. Control Plane 協商
在傳之前，先協商「我們要不要共享腦中的哪一部分」：
- Goal 對齊
- State space 對齊
- Attention threshold 對齊

### 2. State Delta 直接應用
```
接收端不是 decode → inference → update
而是：直接 apply state delta
```

### 3. Task-Oriented 評估
```
ISAC: min BER, max Spectrum Efficiency
我們: max Task Success Rate under Bandwidth Constraint
```

## 數學對比

### ISAC 優化目標
```math
\max \alpha \cdot R_{\text{comm}} + (1-\alpha) \cdot R_{\text{sensing}}
\text{subject to: } P_{\text{total}} \leq P_{\max}
```

### 我們的優化目標
```math
\max P(\text{Task Success})
\text{subject to: } R \leq B, T \leq D_{\max}, D_{\text{task}} \leq \epsilon
```

## 為什麼 ISAC 不夠？

### ISAC 的假設
「我有 data，我要盡量完整地送到 receiver」

### 我們的問題
「我是不是一定要讓對方『知道我知道的全部』？」
還是只要讓他「做出正確決策」就好？

## 總結
ISAC 是 sensing + comm 共用頻譜；
我們是改變「傳輸的單位」（從 bit → semantic state）和「決策機制」（從重建資料 → 同步認知狀態）。
