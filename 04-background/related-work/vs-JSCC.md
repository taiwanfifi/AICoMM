# vs. JSCC: 本質差異分析

## JSCC 定義
**Joint Source-Channel Coding (JSCC)**: 聯合優化 source coding 和 channel coding，
以最小化 end-to-end 失真。

## 核心區別

### 數學目標對比
| 框架 | 目標函數 | 失真度量 | 約束 |
|------|---------|---------|------|
| **JSCC** | $\min D(X, \hat{X})$ | MSE / PSNR | $R < C$ |
| **我們** | $\min D_{\text{task}}(S, \hat{S})$ | $1 - P(\text{Task Success})$ | $R < B, T < D_{\max}$ |

## JSCC 的假設

### 1. 目標是「重建資料」
```math
\min E[\|X - \hat{X}\|^2]
```
希望 $\hat{X}$ 儘可能接近 $X$。

### 2. 失真是 pixel-level 或 feature-level
評估用 MSE、PSNR、SSIM 等指標。

### 3. 不考慮 task semantics
不管資料是用來做什麼任務，只管重建準確度。

## 我們的創新

### 1. 目標是「完成任務」
```math
\min D_{\text{task}}(S, \hat{S}) = 1 - P(\text{Task Success} | \hat{S})
```

### 2. 失真是 task-level
不需要 bit-perfect recovery，只需要足夠完成任務。

### 3. Task-aware compression
根據任務的 attention weights 決定傳什麼。

## 具體例子

### JSCC 的做法
```
影像 X → Encoder → 壓縮表示 Z → Decoder → 重建影像 X̂
評估: PSNR(X, X̂)
```

### 我們的做法
```
Agent 狀態 S → Attention Filter → Semantic Token Z → State Integration → 更新狀態 Ŝ
評估: P(Task Success | Ŝ)
```

## 為什麼不用 JSCC？

### 場景：自駕車協作
**JSCC 會**：
- 壓縮整張影像
- 傳輸壓縮表示
- 重建影像
- Receiver 自己做物體偵測

**我們會**：
- 只傳 "前方有障礙物，建議切換到 Plan B"
- Receiver 直接更新 plan state

### 頻寬對比
JSCC: ~100 KB (壓縮影像)
我們: ~100 Bytes (semantic token)

### 延遲對比
JSCC: Encode + Transmit + Decode + Inference
我們: Filter + Transmit + Integrate

## 數學嚴謹性對比

### JSCC 有嚴格的理論保證
```math
R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X; \hat{X})
```

### 我們也有理論框架
```math
R_{\text{task}}(D) = \min_{p(z|s): E[d_{\text{task}}(S,\hat{S})] \leq D} I(S; Z)
```

關鍵差異：$d_{\text{task}} \neq d_{\text{MSE}}$

## 可能的質疑與回應

### Q: 這不就是 task-oriented JSCC 嗎？
A: 不是。JSCC 的 task-oriented extension 還是在「重建 feature」，我們是「同步 state」。

### Q: 你們的失真度量怎麼定義？
A: $D_{\text{task}} = 1 - P(\text{Task Success})$，可以通過實驗測量。

### Q: 有理論最優解嗎？
A: 我們有 Information Bottleneck 框架下的 rate-distortion bound。

## 總結
JSCC 追求「重建資料」，我們追求「完成任務」。
評估指標從 MSE → Task Success Rate。
