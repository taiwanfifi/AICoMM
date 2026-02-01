# Mathematical System Model

## 系統模型定義

### 基本元素
- **Source (Edge Agent)**: 具有內部狀態 $S_t \in \mathcal{S}$
- **Channel**: 帶寬受限 $B$，延迟 $D$
- **Receiver (Cloud Agent)**: 目標狀態 $\hat{S}_t$
- **Task**: 目標函數 $\mathcal{T}(S_t, \hat{S}_t)$

### 狀態空間定義
```math
S_t = (h_t, b_t, p_t, a_t)
```
其中：
- $h_t$: Hidden state (latent representation)
- $b_t$: Belief state (probabilistic world model)
- $p_t$: Policy state (action distribution)
- $a_t$: Attention weights (task-critical dimensions)

## Information Bottleneck 框架

### 目標函數
```math
\min I(X; Z) - \beta I(Z; Y)
```

其中：
- $X$: 原始狀態空間
- $Z$: 傳輸的 semantic token
- $Y$: 任務相關信息
- $\beta$: 權衡參數（bandwidth cost vs. task performance）

### 物理意義
- $I(X; Z)$: 傳輸率（越小越好）
- $I(Z; Y)$: 任務相關信息（越大越好）
- $\beta$: 根據頻寬成本動態調整

## Rate-Distortion 目標

### 優化問題
```math
\min R(S_t \to Z_t)
\text{subject to: } D_{\text{task}}(S_t, \hat{S}_t) \leq D_{\max}
```

其中：
- $R$: 傳輸率（bits per token）
- $D_{\text{task}}$: 任務失真（不是 MSE，是 task success rate）

### 任務失真定義
```math
D_{\text{task}}(S_t, \hat{S}_t) = 1 - P(\text{Task Success} | \hat{S}_t)
```

## 完整優化問題

### 主要目標
```math
\max \text{Task Success Rate}
```

### 約束條件
1. **帶寬約束**：$R \leq B$
2. **延遲約束**：$T_{\text{total}} \leq D_{\max}$
3. **狀態一致性**：$\|S_t - \hat{S}_t\|_{\text{task}} \leq \epsilon$

### 決策變數
- $\delta_t \in \{0,1\}$: 是否在時間 $t$ 傳輸
- $Z_t$: 傳輸的 semantic token
- $\tau$: Attention threshold

## Attention-Based Filtering

### Attention Gate 函數
```math
\delta_t = \mathbb{1}[\max_i a_{t,i} > \tau]
```

其中 $a_{t,i}$ 是第 $i$ 個 latent dimension 的 attention weight。

### Token Selection
```math
Z_t = \{h_{t,i} : a_{t,i} > \tau\}
```
只傳輸 attention weight 超過閾值的維度。

## State Update 模型

### Source 端（發送前）
```math
S_{t+1} = f_{\text{local}}(S_t, o_t)
```
其中 $o_t$ 是新的 observation。

### Receiver 端（接收後）
```math
\hat{S}_{t+1} = f_{\text{integrate}}(\hat{S}_t, Z_t, \text{Anchor}_t)
```

### State Delta
```math
\Delta S_t = S_t - S_{t-1}
Z_t = \text{Compress}(\Delta S_t)
```

## 與傳統通訊的數學對比

| 框架 | 目標函數 | 約束 | 評估 |
|------|---------|------|------|
| Shannon | $\max I(X;Y)$ | $R < C$ | BER |
| JSCC | $\min D(X,\hat{X})$ | $R < C$ | MSE |
| **Ours** | $\max P(\text{Task Success})$ | $R < B, T < D_{\max}$ | Task Success Rate |

## 關鍵數學創新

### 1. Task-Oriented Distortion
不是 $\|X - \hat{X}\|^2$，而是 $1 - P(\text{Task Success})$

### 2. Attention-Weighted Rate
```math
R_{\text{eff}} = \sum_{i: a_i > \tau} \log_2 |\mathcal{H}_i|
```
只計算被傳輸的維度的熵。

### 3. Semantic Consistency Constraint
```math
\text{KL}(p(a|S_t) \| p(a|\hat{S}_t)) \leq \epsilon
```
確保兩端的 policy 分布接近。

## 核心假設（Assumptions）

> 以下假設在 `theoretical-foundations.md` 的定理證明中被使用。
> 這裡統一列出，以便交叉驗證。

### Assumption 1: Attention-Importance Alignment

**陳述**：Attention weight 與 task-relevant importance 正相關。

```math
\text{Corr}(a_{t,i}, \text{Importance}(h_{t,i} \to Y)) \geq \rho_{\min} > 0
```

**依據**：
- DeepSeek-V3 實驗顯示 top-5% attention tokens 恢復 99%+ 輸出品質
- Attention 被廣泛用作 feature importance 的 proxy（LIME, SHAP 等方法佐證）

**使用於**：Theorem 1 (最優語義通信率), Theorem 3 (最優 threshold)

### Assumption 2: Lipschitz Task Sensitivity

**陳述**：Task distortion 對 state error 滿足 Lipschitz 條件。

```math
D_{\text{task}}(S, \hat{S}) \leq \kappa \cdot \|S - \hat{S}\|^2
```

其中 $\kappa > 0$ 為 task sensitivity 常數。

**意義**：state 的小誤差只會造成有限的 task 影響，不會無限放大。

**使用於**：Theorem 2 (R-D 函數), Theorem 4 (帶寬優勢), Section 5.1 (R-D 曲線)

### Assumption 3: Error Contraction Property

**陳述**：在接收端的整合過程中，歷史累積 error 有自然衰減。

```math
\|e_t\|_{\text{eff}} \leq \rho \cdot \|e_{t-1}\|_{\text{eff}} + \|\varepsilon_t\|
```

其中 $\rho \in [0, 1)$ 為 contraction parameter。

**依據**：
- Attention 機制自然遺忘舊 token（新 token 會搶佔 attention budget）
- KV-Cache 固定長度時，舊 entry 被覆蓋

**使用於**：Drift Theorem (semantic-state-sync.md), Reset Frequency 計算

### Assumption 4: Error Source Independence

**陳述**：不同來源的誤差（量化、投影、漂移）之間近似獨立。

```math
D_{\text{task}} \leq D_{\text{quant}} + D_{\text{proj}} + D_{\text{drift}}
```

**注意**：嚴格來說不完全成立（FP8 量化可能放大投影誤差），實際需要加 $(1+\delta)$ 修正係數。

**使用於**：Theorem 5 (端到端誤差界)

### 假設總覽

| Assumption | 名稱 | 使用的定理 | 合理性 | 違反時影響 |
|-----------|------|-----------|-------|-----------|
| 1 | Attention-Importance Alignment | Theorem 1, 3 | 中高（DeepSeek 實驗支持） | Top-k 選擇效率下降 |
| 2 | Lipschitz Task Sensitivity | Theorem 2, 4 | 高（平滑任務普遍成立） | R-D bound 不成立 |
| 3 | Error Contraction | Drift Theorem | 中（依賴 attention 機制） | 需要更頻繁 reset |
| 4 | Error Independence | Theorem 5 | 中（FP8 下近似成立） | 需加 $(1+\delta)$ 修正 |

## 下一步
1. ✅ 推導 optimal threshold $\tau^*$（已完成，見 `theoretical-foundations.md` Theorem 3）
2. ✅ 分析 rate-distortion trade-off（已完成，見 `theoretical-foundations.md` Theorem 2）
3. 設計具體的 compression algorithm → Phase 4 (Implementation)
