# Evaluation Scenarios and Methodology

## 實驗方法：Trace-driven Simulation

### 為什麼不用真實網路？

#### 實際限制
1. **6G 網路尚未部署**：我們的研究針對未來網路，現有 4G/5G 不支援 semantic transmission
2. **Agent 協作場景難以在現有硬件上測試**：需要大規模 agent deployment
3. **通訊論文接受的標準方法**：Trace-driven simulation 是通訊領域（INFOCOM、ICC）的標準評估方法

#### 學術先例
- IEEE INFOCOM 論文中 >70% 使用 simulation
- 5G NR 標準也是先用 system-level simulation 驗證

### Trace-driven Simulation 的優勢
1. **可重現性**：相同 trace 可以重複實驗
2. **可控性**：可以精確控制 channel conditions
3. **可擴展性**：可以測試大規模場景
4. **可對比性**：可以公平對比不同方法

---

## 具體實現

### 1. Trace 生成

#### 使用的模型
- **LLaVA**（Large Language and Vision Assistant）：用於 vision-language tasks
- **MobileVLM**：用於 edge deployment scenarios
- **DeepSeek-V3**：用於複雜推理任務

#### Trace 內容
```python
trace = {
    "timestamp": t,
    "hidden_state": h_t,          # Shape: (seq_len, hidden_dim)
    "attention_weights": a_t,     # Shape: (num_heads, seq_len, seq_len)
    "kv_cache": (K_t, V_t),       # Key-Value cache
    "task_state": {
        "belief": b_t,
        "plan": p_t,
        "policy": π_t
    },
    "observation": o_t
}
```

#### 生成流程
```
1. 準備 task scenarios（導航、物體偵測、協作決策）
2. 運行 agent 模型（LLaVA/MobileVLM）
3. 記錄每個 timestep 的：
   - Hidden states
   - Attention weights
   - KV cache changes
   - Task state updates
4. 保存為 trace files
```

### 2. Channel 模擬

#### Packet Loss
使用 **Gilbert-Elliott Model**（兩狀態 Markov chain）

```
States: {Good, Bad}
P(Good → Bad) = p_gb
P(Bad → Good) = p_bg
P(loss | Good) = p_loss_good
P(loss | Bad) = p_loss_bad
```

#### Delay
使用 **Exponential Distribution**

```
D ~ Exp(λ)
其中 λ 根據網路條件設定：
- WiFi: λ = 1/10ms
- 5G: λ = 1/1ms
- Satellite: λ = 1/200ms
```

#### Bandwidth Constraint
```
B_available(t) = B_max * (1 - α * interference(t))
其中 α ∈ [0,1] 是干擾係數
```

### 3. Baseline 對比

#### Baseline 1: H.264 Video Transmission
傳統方法：直接傳送視訊編碼

```
Source: Camera → H.264 Encoder → Channel → H.264 Decoder → Task Processing
```

#### Baseline 2: JSCC-based Semantic Communication
Joint Source-Channel Coding 方法

```
Source: Image → Deep JSCC → Channel → Deep JSCC Decoder → Reconstructed Image
```

#### Baseline 3: Full State Transmission（無 Filtering）
傳送完整的 hidden state，不做 attention filtering

```
Source: Agent → Full KV Cache → Channel → Receiver → State Integration
```

#### Our Method: Attention-Filtered Semantic Token
```
Source: Agent → Attention Filter → Semantic Token → Channel → State Integration
```

---

## 評估指標

### 主要指標

#### 1. Task Success Rate (TSR)
```math
\text{TSR} = \frac{\text{Number of successful tasks}}{\text{Total number of tasks}}
```

**定義「成功」**：
- 導航任務：到達目標點，誤差 < 1m
- 物體偵測：IoU > 0.5
- 協作決策：雙方 policy 一致性 > 0.9

#### 2. Bandwidth Efficiency
```math
\text{BE} = \frac{\text{TSR}}{\text{Average bits transmitted per task}}
```

單位：Successful tasks per Megabit

#### 3. Latency
```math
T_{\text{total}} = T_{\text{process}} + T_{\text{transmission}} + T_{\text{integration}}
```

測量從 event trigger 到 receiver state update 的時間。

### 次要指標

#### 4. Spectrum Efficiency
```math
\text{SE} = \frac{\text{Successful tasks}}{\text{Total bandwidth * time}} \quad [\text{tasks/Hz/s}]
```

#### 5. Robustness
```math
\text{Robustness} = \text{TSR}(p_{\text{loss}} = 0.2) / \text{TSR}(p_{\text{loss}} = 0)
```

#### 6. State Consistency
```math
\text{SC} = 1 - \frac{1}{T} \sum_{t=1}^T \|S_{\text{sender}}(t) - S_{\text{receiver}}(t)\|_{\text{task}}
```

---

## 實驗場景

### Scenario 1: Edge-Cloud Collaborative Navigation
**設定**：
- Edge agent：MobileVLM 在移動設備上
- Cloud agent：LLaVA 在雲端
- Task：協作導航，避開動態障礙物

**變數**：
- Bandwidth: 1 Mbps ~ 100 Mbps
- Latency: 10ms ~ 500ms
- Packet loss: 0% ~ 20%

**評估**：
- Navigation success rate
- Time to reach destination
- Bandwidth consumption

### Scenario 2: Multi-Agent Object Detection
**設定**：
- 3 個 edge agents，各自觀察不同視角
- 1 個 central coordinator
- Task：整合多視角信息，進行物體偵測

**變數**：
- Agent 數量：2 ~ 10
- Observation overlap：0% ~ 80%
- Communication topology：Star, Mesh, Tree

**評估**：
- Detection accuracy (mAP)
- Communication overhead
- Scalability

### Scenario 3: Autonomous Driving Coordination
**設定**：
- 2 個自駕車 agents
- Task：協調變道、避免碰撞

**變數**：
- Relative velocity：10 km/h ~ 100 km/h
- Distance：10m ~ 100m
- Channel condition：Good / Moderate / Bad

**評估**：
- Collision avoidance success rate
- Maneuver completion time
- Communication frequency

---

## 實驗參數

### Network Parameters
```python
network_config = {
    "bandwidth": [1e6, 10e6, 100e6],        # 1~100 Mbps
    "latency_mean": [0.01, 0.1, 0.5],       # 10ms, 100ms, 500ms
    "packet_loss_rate": [0, 0.05, 0.1, 0.2],
    "gilbert_elliott": {
        "p_gb": 0.1,
        "p_bg": 0.3,
        "p_loss_good": 0.01,
        "p_loss_bad": 0.5
    }
}
```

### Agent Parameters
```python
agent_config = {
    "model": "LLaVA-7B / MobileVLM",
    "hidden_dim": 4096,
    "num_layers": 32,
    "attention_threshold": [0.1, 0.3, 0.5, 0.7],
    "update_frequency": [1, 5, 10],  # Hz
}
```

### Task Parameters
```python
task_config = {
    "navigation": {
        "map_size": (100, 100),
        "num_obstacles": [5, 10, 20],
        "success_threshold": 1.0  # meters
    },
    "detection": {
        "num_objects": [1, 5, 10],
        "iou_threshold": 0.5
    }
}
```

---

## 數據收集與分析

### Trace Collection
```
每個 scenario 執行：
- 100 次獨立實驗
- 10 種不同的隨機種子
- 記錄完整的 state trajectory
```

### Statistical Analysis
```
報告：
- Mean ± Std
- 95% Confidence Interval
- Statistical significance test (t-test, ANOVA)
```

### Visualization
```
圖表：
- TSR vs. Bandwidth (line plot)
- Latency CDF (cumulative distribution)
- Bandwidth efficiency heatmap
- Robustness under packet loss (bar chart)
```

---

## 預期結果

### Hypothesis
**H1**: 我們的方法在低頻寬下（<10 Mbps）優於 H.264 和 JSCC
**H2**: Attention filtering 可減少 50%+ 的傳輸量，同時維持 >90% TSR
**H3**: 在高 packet loss（>10%）情況下，我們的方法更 robust

### 目標性能
```
Task Success Rate: >90%
Bandwidth Reduction: >50% vs. Full State Transmission
Latency: <100ms (90th percentile)
Robustness: TSR degradation <10% at 20% packet loss
```

---

## 實驗可行性

### 為什麼這個方法可行？

#### 1. Trace-driven 是標準方法
- INFOCOM/ICC 論文廣泛使用
- 可以引用先例（如 5G NR evaluation methodology）

#### 2. 可以用真實模型生成 Trace
- LLaVA/MobileVLM 是開源的
- 可以實際運行並記錄內部狀態

#### 3. Channel 模擬有成熟工具
- NS-3, MATLAB Communication Toolbox
- 可以精確模擬各種網路條件

#### 4. 對比公平
- 所有 baseline 使用相同的 trace 和 channel
- 差異只在傳輸方法

---

## 實現計劃

### Phase 1: Trace Generation（1 個月）
- 建立 LLaVA/MobileVLM 測試環境
- 設計 task scenarios
- 生成並驗證 traces

### Phase 2: Simulator 開發（1 個月）
- 實現 channel simulator
- 實現 semantic token encoder/decoder
- 實現 baselines

### Phase 3: 實驗執行（1 個月）
- 運行所有 scenarios
- 收集數據
- Statistical analysis

### Phase 4: 論文撰寫（1 個月）
- 整理結果
- 繪製圖表
- 撰寫 evaluation section

---

## 總結

### 評估策略
**Trace-driven Simulation** 是通訊論文的標準方法，可以：
- 公平對比不同方法
- 精確控制實驗條件
- 重現實驗結果

### 關鍵指標
- **Task Success Rate**：核心指標
- **Bandwidth Efficiency**：展示優勢
- **Robustness**：證明實用性

### 預期貢獻
證明 Semantic State Communication 在低頻寬、高延遲、高丟包環境下，
優於傳統方法（H.264）和現有 semantic communication（JSCC）。
