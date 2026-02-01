# Research Question

## 核心研究問題

### 主要問題
**當 AI Agents 成為主要通訊實體時，傳統的 bit-oriented 網路應如何演進？**

### 細化問題
在頻寬受限、延遲敏感的環境下，多個具備內部認知狀態的 AI Agents 要如何**最小成本**地同步「思考結果」以完成協作任務？

## 問題背景

### 現狀
當前的 AI Agent frameworks（LangChain、AutoGen、MCP）都假設：
- 通訊頻寬 ≈ 無限
- 延遲 ≈ 0
- 傳訊息的成本可以忽略

### 問題
在真實的網路環境中：
- 頻寬有限（特別是邊緣、移動環境）
- 延遲不可忽略（5G: 1-10ms, Satellite: 200-500ms）
- 傳輸有成本（能耗、頻譜資源）

### 矛盾
現有 Agent 通訊方式無法適應受限網路環境，
但未來 6G 網路中 Agent-to-Agent 通訊將是主流。

## 研究範圍

### 在範圍內 ✅
1. **Agent 間的通訊協定**：如何設計新的通訊協定以支援 semantic state synchronization
2. **頻寬受限下的協作**：在有限資源下如何保證任務成功率
3. **通訊決策機制**：何時傳、傳什麼、傳多少
4. **狀態表示與壓縮**：如何表示和壓縮 semantic state
5. **評估指標**：Task Success Rate, Bandwidth Efficiency

### 不在範圍內 ❌
1. **Agent 推理演算法**：我們不改進 LLM 本身
2. **任務規劃演算法**：我們專注通訊，不是規劃
3. **單一 Agent 優化**：我們關心的是多 Agent 協作
4. **Application-layer frameworks**：我們是通訊層，不是應用層
5. **Physical layer 改進**：我們假設底層通訊已有（5G/6G）

## 具體研究子問題

### Q1: 表示問題
**如何將 Agent 的內部認知狀態表示為可傳輸的 semantic token？**

- Hidden states, beliefs, plans, policies 如何編碼？
- 如何確保不同 Agent 能理解相同的 token？
- 如何處理異構 Agent（不同模型架構）？

### Q2: 決策問題
**在什麼條件下，Agent 應該傳輸狀態更新？**

- 如何評估狀態變化的「重要性」？
- 如何平衡通訊成本與任務性能？
- 如何處理多個並發的狀態變化？

### Q3: 壓縮問題
**如何在保證任務成功率的前提下，最小化傳輸量？**

- Attention-based filtering 的理論基礎？
- Rate-distortion trade-off 如何量化？
- 如何根據網路條件動態調整壓縮率？

### Q4: 對齊問題
**如何確保 Sender 和 Receiver 的狀態空間對齊？**

- Control Plane 如何協商 semantic handshake？
- 如何處理 state drift（狀態漂移）？
- 如何驗證狀態一致性？

### Q5: 容錯問題
**在丟包、亂序、延遲的網路環境下，如何維持狀態同步？**

- Out-of-order token 如何處理？
- Packet loss 如何恢復？
- 何時應該重傳，何時應該放棄？

## 研究假設

### H1: Shared World Model
Agents 可以通過 Control Plane 建立共享的 world model，
使得只需傳輸 state delta 而非完整 state。

### H2: Task-Critical Sparsity
在大多數時間點，Agent 內部狀態的變化對任務結果影響很小，
可以通過 attention mechanism 識別並過濾。

### H3: Deterministic Integration
給定相同的 (initial state, delta sequence, anchor points)，
Receiver 可以確定性地重建 Sender 的狀態，誤差在任務可接受範圍內。

### H4: Bandwidth-Performance Trade-off
存在一個 optimal threshold $\tau^*$，使得：
```math
\max_{\tau} \text{Task Success Rate}(\tau) - \lambda \cdot \text{Bandwidth}(\tau)
```

### H5: Superiority over Baselines
在低頻寬環境（< 10 Mbps）下，我們的方法優於：
- 傳統影片傳輸（H.264）
- JSCC-based semantic communication
- Full state transmission

## 預期貢獻

### 理論貢獻
1. **新的通訊範式**：從 Data Transmission → State Synchronization
2. **Rate-Distortion 理論擴展**：從 pixel-level → task-level distortion
3. **Information Bottleneck 應用**：在通訊協定設計中的應用

### 技術貢獻
1. **Semantic State Communication Protocol**：完整的協定設計
2. **Attention-based Filtering**：基於 DeepSeek DSA 的通訊決策機制
3. **State Integration Algorithm**：Receiver 端的狀態整合演算法

### 實證貢獻
1. **Trace-driven Evaluation**：真實 LLM/VLM 模型的 trace
2. **Comprehensive Comparison**：與多個 baseline 的公平對比
3. **Scalability Analysis**：從 2 agents 到 10+ agents 的擴展性

## 評估標準

### 主要指標
1. **Task Success Rate (TSR)**：任務成功率
2. **Bandwidth Efficiency**：單位頻寬的任務成功率
3. **Latency**：從 event trigger 到 state update 的時間

### 次要指標
4. **Spectrum Efficiency**：單位頻譜資源的任務吞吐量
5. **Robustness**：在不同丟包率下的性能
6. **Scalability**：支援的 agent 數量

### 成功標準
```
TSR > 90%
Bandwidth Saving > 50% (vs. Full State Transmission)
Latency < 100ms (90th percentile)
Robustness: TSR degradation < 10% at 20% packet loss
```

## 與現有研究的差異

### vs. Traditional Communication
| 維度 | 傳統通訊 | 我們的研究 |
|------|---------|-----------|
| 傳輸單位 | Bit | Semantic Token |
| 目標 | Bit-perfect | Task-sufficient |
| 評估 | BER | Task Success Rate |

### vs. Semantic Communication (JSCC)
| 維度 | JSCC | 我們的研究 |
|------|------|-----------|
| 目標 | 重建資料 | 完成任務 |
| 失真度量 | MSE/PSNR | Task distortion |
| 接收端 | 重新推理 | 直接整合 delta |

### vs. ISAC
| 維度 | ISAC | 我們的研究 |
|------|------|-----------|
| 傳輸內容 | 外在感知 | 內在認知 |
| 對齊機制 | 無 | Control Plane handshake |
| 評估 | 頻譜效率 | 任務成功率 |

### vs. Agent Frameworks (MCP)
| 維度 | MCP | 我們的研究 |
|------|-----|-----------|
| 層級 | Application | Transport |
| 假設 | Comm cost = 0 | Comm cost ≠ 0 |
| 決策 | 想傳就傳 | Attention-gated |

## 研究挑戰

### 技術挑戰
1. **State Representation**：如何表示複雜的 Agent 內部狀態
2. **Real-time Processing**：如何在毫秒級做 attention filtering
3. **Heterogeneity**：如何處理不同架構的 Agent

### 理論挑戰
1. **Optimality**：如何證明 threshold 的最優性
2. **Convergence**：如何保證狀態同步最終收斂
3. **Robustness**：如何理論分析容錯性

### 實驗挑戰
1. **Trace Generation**：如何生成真實的 Agent trace
2. **Fair Comparison**：如何設計公平的 baseline
3. **Scalability Test**：如何測試大規模場景

## 潛在影響

### 學術影響
1. 開創新的研究方向：Agent-oriented communication
2. 連接 AI 和通訊兩個領域
3. 提供新的理論框架和評估標準

### 產業影響
1. 為 6G 標準化提供參考
2. 支援邊緣智能的大規模部署
3. 降低 Agent 協作的通訊成本

### 社會影響
1. 使能更多 AI 協作應用（自駕、工業自動化）
2. 降低能耗（減少不必要的傳輸）
3. 提升頻譜利用率

## 總結

### 核心問題
在頻寬受限下，AI Agents 如何高效協作？

### 核心思想
從傳輸資料 → 同步認知狀態

### 核心方法
Attention-based semantic state synchronization

### 預期成果
一個完整的、理論嚴謹的、實驗驗證的通訊協定，
能夠在低頻寬下支援 AI Agent 高效協作。
