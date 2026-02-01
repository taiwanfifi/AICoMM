# State Integration: Receiver 端機制

## 概述
平衡 Source/Receiver 設計，補充 Receiver 端的狀態整合機制。

## Receiver 端狀態更新公式

### 基本更新方程
```math
S_{t+1} = f(S_t, \Delta_{\text{token}}, \text{Anchor})
```

其中：
- $S_t$: 當前狀態
- $\Delta_{\text{token}}$: 接收到的 semantic token
- $\text{Anchor}$: MCP 協商好的對齊點
- $f$: 確定性整合函數（Deterministic Integration）

## 關鍵機制

### 1. Anchor-based Alignment

#### Anchor 定義
Anchor 是雙方協商好的「語義錨點」，用於確保 token 插入正確的 latent slot。

#### 協商過程
```
1. Sender: "我的 anchor 是 [goal=delivery, location=A]"
2. Receiver: "我的 anchor 是 [goal=delivery, location=B]"
3. Control Plane: "Align on goal=delivery, location divergence noted"
```

#### 整合公式
```math
\hat{S}_{t+1} = \hat{S}_t + \alpha \cdot (\Delta_{\text{token}} - \text{ProjectionError}(\text{Anchor}))
```

### 2. Out-of-Order 處理

#### 問題
網路可能導致 token 亂序到達。

#### 解決方案
- Token 帶有時間戳 $t$
- 維護 buffer，按序整合
- 如果 $t_{\text{recv}} > t_{\text{expected}} + \delta$，觸發重傳請求

#### Buffer 管理
```python
def integrate_token(buffer, token):
    buffer.insert(token, key=token.timestamp)
    while buffer.is_continuous():
        S_t = buffer.pop_oldest()
        apply_delta(S_t)
```

### 3. 丟包處理

#### 不是 RAG 腦補
與傳統 agent 不同，我們不會「腦補」丟失的訊息。

#### 策略
1. **Zero-Hold**: 保持上一個狀態
   ```math
   \hat{S}_t = \hat{S}_{t-1} \quad \text{if packet lost}
   ```

2. **Kalman Filter**: 基於歷史趨勢預測
   ```math
   \hat{S}_t = \hat{S}_{t-1} + K(\Delta S_{\text{predicted}} - \Delta S_{\text{observed}})
   ```

3. **Task-aware Retransmission**: 根據任務重要性決定是否重傳
   ```python
   if task_criticality(lost_token) > threshold:
       request_retransmission(lost_token.id)
   else:
       use_zero_hold()
   ```

## 與 Traditional Communication 的對比

| 維度 | 傳統通信 | 我們的方法 |
|------|---------|-----------|
| 丟包處理 | ARQ/FEC | Task-aware retransmission |
| 順序保證 | TCP sequence number | Token timestamp + anchor |
| 狀態還原 | Bit-perfect recovery | Task-sufficient reconstruction |
| 錯誤處理 | CRC/Checksum | Semantic consistency check |

## Deterministic Integration 的優勢

### 1. 可預測性
給定相同的 $(S_t, \Delta_{\text{token}}, \text{Anchor})$，結果是確定的。

### 2. 可驗證性
可以用數學證明：
```math
\|S_{\text{sender}} - S_{\text{receiver}}\|_{\text{task}} \leq \epsilon
```

### 3. 可調試性
不像 RAG 會產生 hallucination，我們的整合過程是透明的。

## 實現細節

### 狀態整合算法
```python
class StateIntegrator:
    def __init__(self, anchor):
        self.anchor = anchor
        self.buffer = PriorityQueue()

    def integrate(self, token):
        # 1. 檢查 anchor 對齊
        if not self.check_alignment(token.anchor, self.anchor):
            raise SemanticMismatchError

        # 2. 插入 buffer
        self.buffer.put((token.timestamp, token))

        # 3. 按序處理
        while self.buffer.has_continuous_sequence():
            t, tok = self.buffer.get()
            self.apply_delta(tok.delta)

    def apply_delta(self, delta):
        # 4. 更新狀態
        self.state = self.state + self.alpha * delta

    def handle_loss(self, expected_t):
        # 5. 丟包處理
        if self.is_critical(expected_t):
            self.request_retransmit(expected_t)
        else:
            self.state = self.state  # Zero-hold
```

## 評估指標

### 1. 整合準確度
```math
\text{Accuracy} = 1 - \frac{\|S_{\text{true}} - \hat{S}\|_{\text{task}}}{\|S_{\text{true}}\|_{\text{task}}}
```

### 2. 延遲
```math
T_{\text{total}} = T_{\text{transmission}} + T_{\text{integration}}
```

### 3. 容錯性
```math
\text{Robustness} = P(\text{Task Success} | \text{packet loss rate} = p)
```

## Receiver 端兩層對齊機制

> **重要澄清**：Receiver 端的狀態整合包含兩個不同層次的對齊，散見於不同文件中。
> 此處統一說明它們的關係。

### 層次 1：語義對齊（Anchor-Based Alignment）

**描述於**：本文件上方的 Anchor-based Alignment 機制

**功能**：確保接收到的 token 插入正確的語義位置
- 檢查 anchor 是否匹配（任務一致性）
- 計算 ProjectionError（語義偏差）
- 決定是否接受或丟棄 token

**公式**：
```math
\hat{S}_{t+1} = \hat{S}_t + \alpha \cdot (\Delta_{\text{token}} - \text{ProjectionError}(\text{Anchor}))
```

### 層次 2：維度對齊（Neural Projector）

**描述於**：`kv-cache-alignment.md` 和 `../06-implementation/ssc-pipeline-spec.md`

**功能**：將 Edge 模型的 KV-Cache 維度映射到 Cloud 模型的維度空間
- Edge (MobileVLM): 512-dim KV-Cache
- Cloud (GPT-4V): 4096-dim KV-Cache
- 使用 Neural Projector 做線性投影 + LayerNorm

**公式**：
```math
K_{\text{cloud}} = \text{LayerNorm}(W \cdot K_{\text{edge}} + b)
```

### 兩層如何協作

```
接收到 Semantic Token
    ↓
[層次 1] Anchor 檢查：語義對齊
    - 這個 token 是否屬於當前任務上下文？
    - anchor 不匹配 → 拒絕 / 要求重新同步
    ↓
[層次 2] Neural Projector：維度對齊
    - 512-dim → 4096-dim 映射（僅 Latent Mode 需要）
    - Structured Mode 不需要此步驟
    ↓
Apply Delta：確定性整合
    - 插入到正確的 latent slot
    - 更新 Receiver 的世界模型
```

**注意**：
- 如果兩端使用相同模型（同維度），跳過層次 2
- 如果使用 Structured Mode token，跳過層次 2（不涉及 KV-Cache）
- 兩層對齊是正交的：語義對齊確保「放對位置」，維度對齊確保「格式相容」

## 下一步
1. 實現 StateIntegrator 原型
2. 測試不同丟包率下的性能
3. 優化 Kalman Filter 參數
