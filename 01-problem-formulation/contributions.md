# Contributions: 我們的貢獻

## 核心貢獻總覽

### 理論貢獻
1. **新的通訊範式**：從 Data Transmission → Semantic State Synchronization
2. **Task-oriented Rate-Distortion Theory**：擴展經典 R-D 理論到任務層級
3. **Attention-based Transmission Framework**：理論化 attention 機制在通訊中的應用

### 技術貢獻
1. **Semantic State Communication (SSC) Protocol**：完整的協定設計與實現
2. **Semantic Indexer for Communication**：將 DeepSeek DSA Lightning 機制應用於通訊決策
3. **Deterministic State Integration Algorithm**：Receiver 端的狀態整合機制

### 實證貢獻
1. **Trace-driven Evaluation Framework**：使用真實 LLM/VLM 模型的評估方法
2. **Comprehensive Baseline Comparison**：與 H.264, JSCC, Full Transmission 的全面對比
3. **Scalability & Robustness Analysis**：大規模、高丟包環境的性能分析

---

## 理論貢獻詳述

### 1. 新的通訊範式：Semantic State Synchronization

#### 傳統範式（Shannon）
```
Source → Encoder → Channel → Decoder → Destination
目標：Bit-perfect transmission
評估：BER, SNR
```

#### 第一代語義通訊（JSCC）
```
Source → Semantic Encoder → Channel → Semantic Decoder → Reconstruction
目標：最小化重建失真
評估：MSE, PSNR
```

#### 我們的範式（SSC）
```
Shared World Model → State Δ → Semantic Token → Sync → Task Execution
目標：最大化任務成功率
評估：Task Success Rate
```

#### 創新點
- **傳輸單位**：從 bit/symbol → semantic token/state delta
- **通訊目標**：從重建資料 → 同步認知狀態
- **評估標準**：從 BER/PSNR → Task Success Rate

#### 理論意義
這是第一個將通訊目標定義為「認知狀態同步」而非「資料傳輸」的工作，
為 Agent-oriented communication 建立了理論基礎。

---

### 2. Task-oriented Rate-Distortion Theory

#### 經典 Rate-Distortion
```math
R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X; \hat{X})
```

其中 $d(X, \hat{X})$ 通常是 MSE 或 Hamming distance。

#### 我們的擴展：Task-oriented R-D
```math
R_{\text{task}}(D) = \min_{p(z|s): E[d_{\text{task}}(S,\hat{S})] \leq D} I(S; Z)
```

其中：
```math
d_{\text{task}}(S, \hat{S}) = 1 - P(\text{Task Success} | \hat{S})
```

#### 創新點
1. **失真度量改變**：從 pixel-level → task-level
2. **優化目標改變**：從重建準確度 → 任務成功率
3. **實用意義**：不需要 bit-perfect，只需 task-sufficient

#### 理論結果
我們證明（詳見 `theoretical-foundations.md`）：
- **Theorem 3**（最優 Attention Threshold）：存在 optimal threshold $\tau^*$ 使得 task success rate 最大
- **Theorem 2**（Task-Oriented R-D）：在 Information Bottleneck 框架下有 rate-distortion bound
- **Theorem 1**（最優語義通信率）：Attention-based filtering 可以逼近 optimal solution

---

### 3. Attention-based Transmission Framework

#### 問題定義
何時應該傳輸狀態更新？

#### 我們的答案
```math
\text{Transmit}(t) \iff \max_i a_{t,i} > \tau \text{ AND } U_{\text{marginal}} > C_{\text{transmission}}
```

其中：
- $a_{t,i}$: Attention weight for dimension $i$
- $\tau$: Threshold（可學習）
- $U_{\text{marginal}}$: Marginal utility for task
- $C_{\text{transmission}}$: Communication cost

#### 理論框架
我們建立了 attention 機制與通訊決策的理論聯繫：

1. **Attention as Importance Indicator**
   ```math
   a_{t,i} = \text{softmax}(Q_t \cdot K_{t,i})
   ```
   表示 dimension $i$ 對當前任務的重要性。

2. **Theorem 4: SSC Bandwidth Advantage**（原稱 "Sparse Transmission Theorem"）
   在 Assumption 1-2 下，只傳輸 top-k attention dimensions 可以保證：
   ```math
   P(\text{Task Success} | Z_{\text{top-k}}) \geq (1-\epsilon) \cdot P(\text{Task Success} | Z_{\text{full}})
   ```
   > 詳見 `theoretical-foundations.md` Theorem 4，正式名稱為「SSC vs. Traditional Communication」

3. **Theorem 3: Optimal Threshold → Optimal k Selection**
   存在 optimal $k^*$ 使得：
   ```math
   k^* = \arg\max_k \left[ P(\text{Success} | k) - \lambda \cdot \frac{k}{T} \cdot C \right]
   ```
   > 詳見 `theoretical-foundations.md` Theorem 3，正式名稱為「Optimal Attention Threshold」

#### 創新點
第一個將 attention mechanism 理論化為通訊決策工具的工作。

---

## 技術貢獻詳述

### 1. Semantic State Communication (SSC) Protocol

#### Protocol Stack
```
┌──────────────────────────────┐
│   Application (Task Logic)   │
├──────────────────────────────┤
│   SSC Layer (我們的貢獻)      │
│   ├─ Control Plane           │
│   │  └─ Semantic Handshake   │
│   └─ Data Plane              │
│      ├─ Attention Filtering  │
│      ├─ Delta Compression    │
│      └─ State Integration    │
├──────────────────────────────┤
│   Transport (TCP/UDP/QUIC)   │
└──────────────────────────────┘
```

#### Control Plane 設計

**功能**：
1. Goal Negotiation
2. State Space Alignment
3. Threshold Setting
4. Anchor Establishment

**協定流程**：
```
1. HELLO: Initiate connection
2. CAPABILITY: Exchange agent capabilities
3. GOAL_SYNC: Negotiate task goals
4. STATE_MAP: Align state spaces
5. THRESHOLD_SET: Set attention threshold τ
6. ACK: Confirm handshake
```

**訊息格式**：
```protobuf
message ControlPlaneMessage {
  enum Type {
    HELLO, CAPABILITY, GOAL_SYNC, STATE_MAP, THRESHOLD_SET, ACK
  }
  Type type = 1;
  map<string, string> parameters = 2;
  bytes payload = 3;
}
```

#### Data Plane 設計

**功能**：
1. State Monitoring
2. Delta Calculation
3. Attention Filtering
4. Token Transmission
5. State Integration

**訊息格式**：
```protobuf
message SemanticToken {
  uint64 timestamp = 1;
  bytes anchor_id = 2;
  repeated uint32 indices = 3;  // Selected dimensions
  repeated float values = 4;    // Delta values
  repeated float attention_weights = 5;
  bytes checksum = 6;
}
```

**傳輸流程**：
```
Sender:
1. Monitor: ΔS_t = S_t - S_{t-1}
2. Filter: indices = Top-k(attention(ΔS_t))
3. Compress: Z_t = Quantize(ΔS_t[indices])
4. Send: Transmit(Z_t, timestamp, anchor)

Receiver:
1. Receive: (Z_t, timestamp, anchor)
2. Buffer: Insert to ordered buffer
3. Integrate: S_t = S_{t-1} + Decode(Z_t)
4. Verify: Check consistency
```

#### 創新點
- **雙層設計**：Control Plane 處理對齊，Data Plane 處理傳輸
- **Event-driven**：不是連續傳輸，而是狀態變化時觸發
- **Self-adaptive**：根據網路條件動態調整 threshold

---

### 2. Semantic Indexer for Communication

#### 從 DeepSeek DSA 到通訊系統

**DeepSeek 原始用途**：
在長 context 中快速找到相關 tokens。

**我們的創新應用**：
在大量歷史數據中快速決定哪些值得傳輸。

#### 技術實現

**Dual-Cache 架構**：
```python
class DualCacheManager:
    def __init__(self):
        self.L1_index = {}   # Light keys (64-dim, INT8)
        self.L2_payload = {} # Heavy values (full data)

    def write(self, timestamp, data):
        """寫入新數據"""
        key = self.light_encoder(data)  # 64-dim
        value = self.heavy_encoder(data)  # Full resolution

        self.L1_index[timestamp] = key
        self.L2_payload[timestamp] = value

    def query(self, intent, top_k):
        """根據 intent 查詢 top-k"""
        query_vec = self.intent_to_vector(intent)

        # Lightning indexer (只在 L1 運算)
        scores = {}
        for t, key in self.L1_index.items():
            scores[t] = ReLU(query_vec @ key)

        # Top-k selection
        selected = sorted(scores, key=scores.get, reverse=True)[:top_k]

        # 只傳輸 selected values
        return [self.L2_payload[t] for t in selected]
```

#### 複雜度分析
- **L1 Index 查詢**：$O(T \cdot d)$，其中 $d=64$，極快
- **Full Attention**：$O(T^2 \cdot d)$，慢 $T$ 倍
- **對於 $T=10000$，加速 $10000$ 倍**

#### 創新點
- 將 AI 模型的 attention 機制應用於通訊決策
- Dual-cache 設計兼顧速度與品質
- 線性複雜度，支援 real-time 處理

---

### 3. Deterministic State Integration Algorithm

#### 問題
Receiver 如何從收到的 semantic tokens 重建 Sender 的狀態？

#### 傳統方法的問題
- **RAG**: 會 hallucinate，不確定性高
- **Full retransmission**: 頻寬浪費
- **Interpolation**: 無法保證語義正確

#### 我們的解決方案：Deterministic Integration

**核心思想**：
給定相同的 (initial state, token sequence, anchor points)，
結果是**確定性的**。

**演算法**：
```python
class StateIntegrator:
    def __init__(self, initial_state, anchor):
        self.state = initial_state
        self.anchor = anchor
        self.buffer = PriorityQueue()  # Ordered by timestamp

    def integrate(self, token):
        """整合新的 semantic token"""
        # 1. 檢查 anchor 對齊
        if not self.verify_anchor(token.anchor):
            raise AlignmentError

        # 2. 插入 buffer（保持時序）
        self.buffer.put((token.timestamp, token))

        # 3. 按序處理
        while self.buffer.is_continuous():
            t, tok = self.buffer.get()
            self.apply_delta(tok)

    def apply_delta(self, token):
        """確定性地應用 state delta"""
        # 4. 解壓縮
        delta = self.decompress(token.values, token.indices)

        # 5. 更新狀態
        self.state[token.indices] += delta

        # 6. 驗證一致性
        self.verify_consistency()
```

#### 處理異常情況

**Out-of-Order**:
```python
def handle_out_of_order(self, token):
    # 使用 priority queue，自動排序
    self.buffer.put((token.timestamp, token))
```

**Packet Loss**:
```python
def handle_loss(self, expected_t):
    if self.is_critical(expected_t):
        self.request_retransmit(expected_t)
    else:
        self.state[expected_t] = self.state[expected_t - 1]  # Zero-hold
```

#### 創新點
- **確定性**：可預測、可驗證
- **容錯性**：處理丟包、亂序
- **效率**：$O(T \log T)$ 複雜度

---

## 實證貢獻詳述

### 1. Trace-driven Evaluation Framework

#### 為什麼用 Trace-driven？
- 6G 網路尚未部署
- 真實 agent 協作難以大規模測試
- Trace-driven 是通訊領域標準方法（INFOCOM >70% 論文使用）

#### Trace 生成流程
```
1. 準備場景（導航、偵測、協作）
2. 運行 LLaVA/MobileVLM
3. 記錄：
   - Hidden states
   - Attention weights
   - KV cache
   - Task decisions
4. 保存為標準格式（HDF5）
```

#### Trace 格式
```python
trace = {
    'metadata': {
        'model': 'LLaVA-7B',
        'task': 'navigation',
        'duration': 300,  # seconds
    },
    'states': [
        {
            't': 0.0,
            'hidden': np.array(...),  # (seq_len, 4096)
            'attention': np.array(...),  # (32, seq_len, seq_len)
            'kv_cache': (K, V),
            'task_state': {...}
        },
        ...
    ]
}
```

#### 創新點
- 第一個使用真實 FM 模型生成 trace 的 semantic communication 研究
- 提供標準化的評估框架
- 可重現、可對比

---

### 2. Comprehensive Baseline Comparison

#### Baseline 1: H.264 Video Transmission
**方法**：傳統影片編碼
**參數**：Bitrate 1-10 Mbps
**評估**：重建 PSNR + Task Success Rate

#### Baseline 2: JSCC (DeepJSCC)
**方法**：Deep learning-based joint source-channel coding
**參數**：Channel SNR 0-20 dB
**評估**：重建 MSE + Task Success Rate

#### Baseline 3: Full State Transmission
**方法**：傳輸完整 KV cache（無 filtering）
**參數**：Quantization (FP32/FP16/INT8)
**評估**：Bandwidth + Task Success Rate

#### Our Method: SSC with Attention Filtering
**方法**：Attention-based sparse transmission
**參數**：Threshold τ, Top-k
**評估**：全部指標

#### 對比維度
| 維度 | H.264 | JSCC | Full State | **Ours** |
|------|-------|------|-----------|---------|
| Bandwidth | 中 | 低 | 高 | **最低** |
| Latency | 高 | 中 | 低 | **最低** |
| TSR | 中 | 低 | 高 | **最高** |
| Robustness | 低 | 中 | 中 | **高** |

#### 創新點
- 第一個與多種 baseline 公平對比的工作
- 使用相同 trace 和 channel，對比公平
- 全面評估（頻寬、延遲、成功率、魯棒性）

---

### 3. Scalability & Robustness Analysis

#### Scalability 測試

**場景**：Multi-agent collaboration
**變數**：Agent 數量 N = 2, 5, 10, 20

**測試指標**：
1. **Bandwidth Scaling**：總頻寬是否隨 N 線性增長？
2. **Latency Scaling**：延遲是否可控？
3. **Success Rate**：多 agent 時是否保持高成功率？

**結果**（預期）：
```
N=2:  Bandwidth = 100 kbps, TSR = 95%
N=5:  Bandwidth = 250 kbps, TSR = 93%
N=10: Bandwidth = 500 kbps, TSR = 90%
N=20: Bandwidth = 1 Mbps,   TSR = 88%
```

**結論**：Sub-linear scaling（優於 full transmission 的 linear scaling）

#### Robustness 測試

**場景**：High packet loss environment
**變數**：Packet loss rate = 0%, 5%, 10%, 20%

**對比方法**：
- TCP (with retransmission)
- UDP (no retransmission)
- Our method (task-aware retransmission)

**結果**（預期）：
```
Loss=0%:  TCP=95%, UDP=95%, Ours=95%
Loss=5%:  TCP=90%, UDP=80%, Ours=92%
Loss=10%: TCP=85%, UDP=60%, Ours=88%
Loss=20%: TCP=75%, UDP=30%, Ours=82%
```

**結論**：我們的方法在高丟包下最 robust

#### 創新點
- 首次系統性測試 semantic communication 的可擴展性
- 首次分析不同丟包率下的魯棒性
- 提供實用性證明

---

## 與現有工作的對比總覽

### vs. Shannon Communication
| 貢獻維度 | Shannon | Ours |
|---------|---------|------|
| 範式 | Bit transmission | State synchronization |
| 單位 | Bit | Semantic token |
| 目標 | BER minimization | Task success maximization |
| 評估 | SNR, Capacity | TSR, Bandwidth efficiency |

### vs. Semantic Communication (JSCC)
| 貢獻維度 | JSCC | Ours |
|---------|------|------|
| 目標 | 重建資料 | 完成任務 |
| 失真 | MSE/PSNR | Task distortion |
| 決策 | 固定壓縮率 | Attention-gated |
| 接收端 | 重新推理 | 直接整合 delta |

### vs. ISAC
| 貢獻維度 | ISAC | Ours |
|---------|------|------|
| 內容 | 外在感知（radar, camera） | 內在認知（belief, plan） |
| 對齊 | 無 | Control Plane handshake |
| 處理 | Receiver 重新 inference | Apply delta |
| 評估 | 頻譜效率 | 任務成功率 |

### vs. Agent Frameworks (MCP)
| 貢獻維度 | MCP | Ours |
|---------|-----|------|
| 層級 | Application | Transport |
| 假設 | Comm cost = 0 | Comm cost ≠ 0 |
| 決策 | 想傳就傳 | Attention-based |
| 優化 | 無 | Rate-distortion optimal |

---

## 總結

### 核心創新
1. **範式轉移**：Data Transmission → State Synchronization
2. **理論擴展**：Task-oriented R-D, Attention-based transmission
3. **系統設計**：SSC Protocol, Semantic Indexer, Deterministic Integration
4. **實證驗證**：Trace-driven, Multi-baseline, Scalability & Robustness

### 學術價值
- **開創新方向**：Agent-oriented communication
- **理論嚴謹**：有數學證明和 bound
- **實用性強**：解決真實問題

### 預期影響
- **INFOCOM/ICC**：有機會 Best Paper（創新性高）
- **後續工作**：開創 Agent communication 研究領域
- **產業應用**：為 6G 標準化提供參考

### 最重要的一點
**這不只是「改進現有方法」，而是「定義新範式」**。

我們不是讓 semantic communication 更好一點，
而是重新定義：**通訊的目的是什麼？傳輸的單位是什麼？如何評估？**

這是 paradigm shift，不是 incremental improvement。
