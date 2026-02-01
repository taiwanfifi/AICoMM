# vs. Traditional Communication: 範式轉移

## Shannon 範式

### 經典通訊模型
```
Source → Encoder → Channel → Decoder → Destination
```

### 核心假設
1. 目標是「bit-perfect transmission」
2. Channel capacity $C = B \log_2(1 + \text{SNR})$
3. 評估指標：BER, throughput, latency

## 我們的範式

### Semantic State Communication
```
Shared World Model → State Δ → Semantic Token → Sync → Task Execution
```

### 核心假設
1. 目標是「task-sufficient synchronization」
2. Semantic capacity $C_{\text{sem}} = f(\text{attention}, \text{task relevance})$
3. 評估指標：Task Success Rate, spectrum efficiency (bits per successful task)

## 範式轉移的三個層次

### 1. 傳輸單位
| 範式 | 傳輸單位 | 語義層級 |
|------|---------|---------|
| Shannon | Bit | 無語義 |
| Semantic Comm | Symbol | 低階語義（文字） |
| **Ours** | Cognitive State | 高階語義（認知） |

### 2. 通訊目標
```
Shannon: min P(error)
Semantic: min D(message)
Ours: max P(task success)
```

### 3. 評估標準
```
Shannon: BER, SNR, Capacity
Semantic: PSNR, SSIM, Perceptual Quality
Ours: Task Success Rate, Decision Latency
```

## 為什麼 Shannon 範式不夠？

### Shannon 的隱含假設
「只要 bit 傳對了，communication 就成功了」

### 在 AI Agent 場景下的問題
1. **冗餘傳輸**：傳了很多對任務無用的資訊
2. **重複推理**：每次都要重新理解
3. **缺乏對齊**：不知道對方需要什麼

## 具體對比

### 場景：兩個 Agent 協作導航

#### Traditional Communication
```
Agent A: 傳送完整地圖影像 (1 MB)
Agent B: 接收 → 解碼 → 物體偵測 → 路徑規劃
```

#### Our Approach
```
Agent A: "Plan changed: obstacle at (x,y), switch to route B" (100 bytes)
Agent B: 接收 → 直接更新 plan state
```

### 頻寬對比
Traditional: 1 MB
Ours: 100 bytes
**節省**: 10,000x

### 延遲對比
Traditional: T_encode + T_transmit + T_decode + T_inference
Ours: T_filter + T_transmit + T_integrate
**減少**: ~50%

## 理論貢獻

### Shannon 的 Rate-Distortion Theory
```math
R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X; \hat{X})
```

### 我們的 Task-Oriented Rate-Distortion
```math
R_{\text{task}}(D) = \min_{p(z|s): D_{\text{task}}(S,\hat{S}) \leq D} I(S; Z)
```

關鍵創新：
- $d(X, \hat{X})$: pixel-level distortion
- $D_{\text{task}}(S, \hat{S})$: task-level distortion

## 可能的質疑

### Q: 你們還是在用 bit 傳輸啊，哪裡不同？
A: Physical layer 是 bit，但 payload 變成 semantic token，決策機制變成 task-oriented。

### Q: Shannon capacity 還適用嗎？
A: Physical layer 的 capacity 還是適用，但我們定義了 Semantic capacity。

### Q: 這是 cross-layer 設計嗎？
A: 不完全是。我們定義了新的 Semantic Transport Layer，介於傳統通訊層和 Application 之間。

## 總結

### 範式轉移
```
Shannon: Bit Transmission
  ↓
Semantic Communication: Symbol Transmission
  ↓
Ours: Cognitive State Synchronization
```

### 核心創新
不是改善「怎麼傳」，而是改變「傳什麼」和「為什麼傳」。
