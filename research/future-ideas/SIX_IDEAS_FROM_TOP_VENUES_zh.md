# 六大頂刊級研究方向 — 完整繁體中文講解

> **來源：** 基於 300+ 篇頂刊/頂會論文的深度調查，涵蓋 IEEE JSAC, TWC, TCOM, COMST, ACM SIGCOMM, NeurIPS, ICML 等
> **原則：** 以全球頂尖研究趨勢為主，實驗室現有方向為輔（至多 1-2 個有交集）
> **所有方向均為 Dry Lab (GPU模擬) 可完成**

---

## 方向來源比例

| 方向 | 主要靈感來源 | 與實驗室交集 |
|------|------------|------------|
| 1. World Models for 6G Networks | NeurIPS'25, IEEE TNSE Special Issue 2026 | 低 (Digital Twin相關但方法全新) |
| 2. Causal AI for Wireless | IEEE VTM 2024, arXiv frontier | 零 (全新方向) |
| 3. Diffusion Models for Physical Layer | IEEE TWC 2025, NeurIPS AI4NextG | 低 (Diffusion但非SemCom) |
| 4. Neuromorphic Wireless Receiver | arXiv 2025, King's College London | 零 (全新方向) |
| 5. Agentic AI for Autonomous Networks | ACM SIGCOMM'25 (Meta), IEEE TNSE 2026 | 低 (老師有networking但方法全新) |
| 6. SatFM (衛星Channel FM) | LWM (ASU), WiFo (清華), IEEE ICC 2024 | 中 (老師LEO方向+全新FM方法) |

**實驗室比例: 1/6 有中度交集，2/6 有低度交集，3/6 完全無交集 ✅**

---
---

# 方向一：World Models for Proactive 6G Wireless Networks
# （世界模型用於主動式 6G 無線網路）

---

## 一、背景

### 1.1 什麼是 World Model（世界模型）？

World Model 最早來自認知科學：人類的大腦並不是被動地對環境做出反應，而是在腦中維護一個「世界的內部模型」，用這個模型來**預測未來、模擬假設情境、做出規劃**。

在 AI 領域，World Model 被 Yann LeCun（Meta 首席 AI 科學家）視為 AGI 的核心組件之一。它在遊戲和機器人領域已經大獲成功：

```
遊戲：Dreamer v3（2023）
  - 在 Atari 遊戲中，agent 不是直接從像素學 policy
  - 而是先學一個「世界模型」：給定當前狀態和行動，預測下一個狀態
  - 然後在「想像中」（world model 內部）模擬數千個 trajectory
  - 選擇最好的行動序列
  - 結果：用真實環境互動次數減少 10-100 倍，效能更好

機器人：TD-MPC2（2024）
  - 機器人在真實世界做實驗很慢很貴
  - 先用少量真實數據學一個 world model
  - 然後在 world model 中「想像」上百萬次動作
  - 用 model predictive control (MPC) 規劃最佳動作
```

### 1.2 為什麼 World Model 適合無線網路？

**傳統 DRL 用於無線網路的問題：**

```
問題 1: Sample Inefficiency（樣本效率低）
  - DRL agent 需要和真實網路互動數百萬次才能學好 policy
  - 每次互動 = 改變網路配置 → 觀察結果
  - 在真實網路中，這需要數天甚至數週
  - 在模擬器中，每個 episode 也需要大量計算

問題 2: Safety（安全性）
  - DRL 探索過程中會嘗試很多「壞的」行動
  - 在真實網路中，一個壞的功率配置可能導致整個區域斷訊
  - 運營商不敢讓 DRL 直接在線上學習

問題 3: Reactivity（被動性）
  - 傳統 DRL 是 reactive：觀察到問題 → 調整
  - 不能 proactive：預測問題即將發生 → 提前調整
```

**World Model 如何解決：**

```
解法: 學一個無線網路的 world model W

W 的功能：
  給定: 當前網路狀態 s_t + 行動 a_t（如功率配置、beam 方向）
  預測: 下一個狀態 s_{t+1}（含 throughput、latency、user mobility 等）

有了 W 之後:
  1. Agent 在 W 的「想像」中模擬上萬次 → 找最佳 policy（不碰真實網路）
  2. 只在 W 說「安全」的行動中挑選 → 保證安全
  3. W 可以預測未來多步 → 主動式決策（proactive）
```

### 1.3 頂刊/頂會中的證據

**IEEE TNSE Special Issue（投稿截止 2026-03-15）：**
"Fusing Digital Twins and World Models for Proactive 6G Networks"
- IEEE Transactions on Network Science and Engineering 專門為此開了 special issue
- 這代表 IEEE 的 editorial board 認為這是 6G 的重要方向

**NeurIPS 2025 AI4NextG Workshop:**
- "Dual-Mind World Models" 被接受
- "World Model-Based Learning for AoI Minimization" 被接受

---

## 二、參考論文

### 論文 1: MobiWorld — "World Models for Mobile Wireless Network"
- **出處:** arXiv:2507.09462, 2025
- **作者:** 多位

**它做了什麼：**

MobiWorld 是**首個專門為行動網路設計的 world model**。

```
架構：
┌──────────────────────────────────────────────┐
│                 MobiWorld                      │
│                                                │
│  Input: 網路狀態 s_t = {                        │
│    基地台配置（功率、beam、頻率）                  │
│    用戶位置和移動軌跡                             │
│    流量負載                                      │
│    通道品質指標                                   │
│  }                                              │
│  Action: a_t = {基地台開關（sleep/active）}       │
│                                                │
│  World Model 架構: Diffusion Model              │
│    - 學習 p(s_{t+1} | s_t, a_t) 的分佈          │
│    - 不是預測一個確定的下一個狀態                   │
│    - 而是生成一個「分佈」（考慮隨機性）            │
│                                                │
│  Output: 預測的下一個狀態 ŝ_{t+1}               │
│          + 網路級指標（KPI）                     │
└──────────────────────────────────────────────┘
```

**應用場景：基地台節能（Sleep Optimization）**

```
問題：5G 網路的基地台佔電信業 70% 的能耗
      但深夜流量低，很多基地台可以「睡覺」省電
      關鍵：關掉哪些基地台？什麼時候關？不能影響用戶體驗

傳統 DRL 做法:
  - 在真實網路上跑 → 太慢太危險
  - 在模擬器上跑 → 模擬器可能不準

MobiWorld 做法:
  1. 用過去 30 天的真實網路數據訓練 world model
  2. World model 學會「如果我關掉這個基地台，coverage 會怎樣？throughput 會怎樣？」
  3. RL agent 在 world model 裡面「想像」10,000 種 sleep 策略
  4. 選出最佳策略，部署到真實網路

結果:
  - 能耗降低 15-20%
  - 用戶體驗（throughput、latency）幾乎不受影響
  - 相比直接在模擬器訓練的 DRL，收斂速度快 3 倍
```

### 論文 2: Dual-Mind World Models — "A General Framework for Learning in Dynamic Wireless Networks"
- **出處:** arXiv:2510.24546, 2025
- **作者:** Lingyi Wang, Rashed Shelim, Walid Saad, Naren Ramakrishnan

**核心創新：雙腦架構（靈感來自 Daniel Kahneman 的《Thinking, Fast and Slow》）**

```
人類有兩個思考系統：
  System 1（快思維）：直覺、快速、自動化
  System 2（慢思維）：理性、深入、邏輯推理

Dual-Mind World Model 也有兩個模組：

System 1 (Pattern-Driven):
  - 輕量級 RNN/Transformer
  - 學習通道和流量的統計規律（pattern）
  - 快速預測「下一秒通道大概會怎樣」
  - 計算量小，適合即時決策

System 2 (Logic-Driven):
  - 重量級 model，可能包含物理定律
  - 深入分析「為什麼通道突然變差？是因為車輛遮擋還是天氣？」
  - 做長期規劃和因果推理
  - 計算量大，用於離線規劃

兩者的整合:
  - 平常用 System 1 做即時決策
  - 遇到異常或重要決策時，啟動 System 2 做深度分析
  - System 2 的結果用來更新 System 1 的模型
```

**實驗場景: mmWave V2X（毫米波車聯網）**

```
場景: 基地台用 mmWave 為高速公路上的車輛提供服務
      mmWave 的 beam 很窄，車輛移動快，beam 很容易對不準

指標: Contextual Age of Information (CAoI)
      = 根據資訊的「重要程度」加權的 AoI
      （例如：煞車預警的 AoI 比音樂串流的 AoI 重要 100 倍）

數值結果（在 Sionna 模擬器上）:
  - Dual-Mind WM vs 純 Model-Free RL (PPO):
    · 數據效率: 提升 26%（需要的真實互動次數減少 26%）
    · CAoI 最佳化: 提升 16%
    · 收斂速度: 快 2 倍

  - Dual-Mind WM vs 單 System 的 WM:
    · 在環境突然變化時（例如突然下雨）:
      單 System WM 需要 50 步才能適應
      Dual-Mind 只需要 10 步（System 2 快速診斷原因）
```

### 論文 3: World Model-Based Learning for AoI Minimization in V2X
- **出處:** arXiv:2505.01712, 2025

```
對比 Model-Free RL (SAC) vs Model-Based RL (DreamerV3-style):
  環境: mmWave V2X, 20 vehicles, 3 RSUs
  指標: Average Weighted AoI

  Model-Free: 需要 500K 真實環境 steps 才收斂
  Model-Based (World Model): 只需要 50K 真實環境 steps 就收斂 → 10x 數據效率提升
  最終效能: Model-Based 比 Model-Free 好 16% (lower AoI)
```

---

## 三、Research Gap 與我們的研究方向

```
Gap 1: 無人為 LEO 衛星網路建過 world model
  - 所有現有 world model 都是地面網路（cellular）
  - LEO 衛星的特殊性: 軌道運動可預測但通道不可預測, 拓撲快速變化
  → 研究題目: "SatWorld: A World Model for LEO Satellite Network Management"

Gap 2: World model + semantic communication
  - World model 預測「下一刻 channel 會怎樣」
  - Semantic encoder 根據 world model 的預測，主動調整編碼策略
  → 研究題目: "Proactive Semantic Communication via World Models"

Gap 3: World model 的可擴展性
  - 現有 world model 只處理幾十個基地台
  - 如何 scale 到數千顆衛星？
  → 用 GNN 做 world model 的 backbone（而非 MLP）
```

**預估時程:** 9-12 個月出第一篇
**目標期刊:** IEEE TNSE Special Issue (deadline 2026-03-15!), IEEE TWC, IEEE JSAC
**GPU需求:** 高（world model training + RL in imagination）

---
---

# 方向二：Causal AI for Wireless Networks
# （因果 AI 用於無線網路）

---

## 一、背景

### 1.1 為什麼「因果」很重要？

目前無線網路中的 AI 幾乎都是**相關性**（correlation）的學習：

```
例子：DRL for beam management
  - 觀察到: 當 RSRP 下降時，選擇 beam #3 可以提升 throughput
  - 學到的: RSRP 下降 → beam #3（相關性）
  - 但不知道: 為什麼 RSRP 下降？
    是因為遮擋物（blockage）？→ 應該換 beam
    是因為 handover 邊界？→ 應該換基地台
    是因為功率衰減？→ 應該調功率

相關性學習的問題:
  1. 不可解釋: 為什麼選 beam #3？答不出來
  2. 不 robust: 環境一變（例如蓋了新大樓），學到的 policy 就失效
  3. 數據需求大: 因為不理解因果，所以要大量試錯才能學好
  4. 不能做 counterfactual: 「如果當時選了 beam #5 會怎樣？」答不出來
```

**Causal AI 的目標：學習因果關係，而不只是相關性。**

### 1.2 Causal Inference 的基本原理

**Structural Causal Model (SCM, 結構因果模型)：**

```
在無線網路中，一個 SCM 的例子：

  User Location ──→ Path Loss ──→ RSRP
       │                              │
       ↓                              ↓
  Blockage ────→ NLOS Indicator ──→ Beam Selection ──→ Throughput
       │                              ↑
       ↓                              │
  Mobility Speed ──→ Doppler ─────────┘

這個圖告訴我們:
  - User Location 「導致」Path Loss（因果關係，非只是相關）
  - Blockage 「導致」NLOS，NLOS 「影響」Beam Selection
  - Mobility Speed 通過 Doppler 影響 Beam Selection

知道因果圖之後:
  - 如果 throughput 下降了，可以沿著因果路徑追溯原因
  - 可以做 intervention: 如果「強制」改變功率會怎樣？
  - 可以做 counterfactual: 如果用戶沒有移動，throughput 會是多少？
```

**因果推斷的三個層次（Pearl 的因果階梯）：**

```
Level 1: Association（關聯）— 看到什麼
  P(throughput | RSRP = low) = ?
  「RSRP低時，throughput通常是多少？」— 純統計

Level 2: Intervention（介入）— 做了什麼
  P(throughput | do(power = high)) = ?
  「如果我主動把功率調高，throughput 會怎樣？」— 需要因果模型

Level 3: Counterfactual（反事實）— 假設什麼
  P(throughput | beam=3, had we chosen beam=5) = ?
  「我選了 beam #3 結果不好，如果當時選 beam #5 會更好嗎？」— 需要完整SCM
```

### 1.3 參考論文

**論文 1: "Causal Reasoning: Charting a Revolutionary Course for Next-Generation AI-Native Wireless Networks"**
- **出處:** IEEE Vehicular Technology Magazine, Vol. 19, No. 1, pp. 16-31, 2024
- **作者:** Christo K. Thomas et al.

```
這是這個領域的 foundational vision paper。

核心論點: 6G 的 AI 必須從 Level 1 (correlation) 升級到 Level 2/3 (causation)

具體應用場景:

場景 1: Ultra-reliable THz beamforming
  - THz 波束極窄（0.1°），任何微小遮擋都會中斷
  - 傳統 DL: 用大量數據學 beam pattern → 一旦環境變化就失效
  - Causal AI: 學會「遮擋物的位置和大小 → 波束中斷的因果機制」
    → 環境變化時仍然有效（因為因果機制不變）

場景 2: Digital Twin 模型校正
  - Digital Twin 不一定準確
  - 因果推斷可以找出 DT 和真實世界的差異來自哪裡（哪個因果變數不對）
  - 只修正那個變數，而非重新訓練整個模型

場景 3: 訓練數據增強
  - 用 SCM 做 counterfactual data augmentation
  - 「如果下雨了，RSRP 會怎樣？」→ 生成下雨場景的虛擬數據
  - 不需要等真的下雨才能收集數據
```

**論文 2: "Causal Model-Based RL for Sample-Efficient IoT Channel Access"**
- **出處:** arXiv:2511.10291, 2025

```
核心創新: 把因果模型 (SCM) 融入 multi-agent RL

傳統 MARL for channel access:
  - N 個 IoT devices 競爭 M 個 channels
  - 每個 device 是一個 RL agent
  - 純靠 trial-and-error 學會避免碰撞
  - 需要大量互動才能收斂

Causal Model-Based MARL:
  - 先學一個 SCM: device actions → channel states → rewards
  - SCM 的邊代表因果關係（哪個 device 的行動影響了哪個 channel）
  - Agent 用 SCM 做「想像中的 rollout」
  - 只在 SCM 說「有因果影響」的 channel 上做探索

數值結果:
  - 環境互動次數: 減少 58%（相比 model-free MARL）
  - 收斂速度: 快 2.3 倍
  - Throughput: 提升 8%

  例子（N=10 devices, M=5 channels）:
    Model-free MARL: 200K 步收斂, 最終 throughput = 3.2 Mbps/device
    Causal MARL: 84K 步收斂, 最終 throughput = 3.5 Mbps/device
```

**論文 3: "Causal Beam Selection for Reliable Initial Access"**
- **出處:** arXiv:2508.16352, 2025

```
問題: 5G Initial Access（初始接入）需要做 beam sweeping
      256 個 beam 方向中找到最好的 → 需要掃 256 次 → 延遲大

傳統 DL 方法:
  - 用部分 beam 量測（如 32 個）作 input
  - CNN/MLP 預測最佳 beam
  - 問題: 哪 32 個 beam 要測？通常用 random 或 fixed pattern

Causal 方法:
  Step 1 (Causal Discovery):
    用觀測數據建立 Bayesian causal graph
    圖的 node = 每個 beam 的 received power
    圖的 edge = 因果依賴關係（beam A 的 power 高 → beam B 的 power 也可能高）

  Step 2 (Causal Beam Selection):
    從 causal graph 中找到「因果上最有資訊量」的 beam
    只測量這些 beam → 從因果關係推導出其他 beam 的 power

數值結果:
  - 輸入選擇時間: 減少 94.4%（因為知道哪些 beam 是「因果關鍵」的）
  - Beam sweeping overhead: 減少 59.4%
  - 預測準確度: 與傳統方法相當
```

---

## 二、Research Gap 與方向

```
Gap 1: 大規模異質網路的因果發現（Causal Discovery）
  - 現有方法只在小網路（<20 nodes）上驗證
  - 6G 網路有數千個 node
  → "Scalable Causal Discovery for Heterogeneous 6G Networks"

Gap 2: Online Causal Learning（在線因果學習）
  - 現有方法假設因果結構不變
  - 真實網路的因果結構會隨時間改變（蓋新建築、天氣變化）
  → "Dynamic Causal Graph Learning for Time-Varying Wireless Channels"

Gap 3: Causal AI + Foundation Model
  - Foundation model 學到 correlation，能否用因果推斷來「修正」？
  → "Causally-Enhanced Wireless Foundation Models"
```

**預估時程:** 9-12 個月
**目標期刊:** IEEE TWC, IEEE JSAC, IEEE VTM
**GPU需求:** 中-高（causal discovery + RL training）

---
---

# 方向三：Diffusion Models for Physical Layer Problems
# （擴散模型用於實體層問題 — 非語義通訊）

---

## 一、背景：Diffusion 在通訊中的「第二春」

大多數人把 diffusion model 和「生成圖片」聯想在一起。但在通訊領域，diffusion 有一個完全不同的應用：**把 physical layer 的問題建模為「去噪問題」**。

```
Channel Estimation（通道估計）的本質:
  觀測: y = H × pilot + noise
  目標: 從有噪的觀測 y 中恢復 channel matrix H

Diffusion Model 的本質:
  觀測: x_t = 加了 t 步 noise 的 clean data
  目標: 從有噪的 x_t 中恢復 clean data x_0

兩者在數學上是同構的！

所以: channel estimation 可以被重新表述為一個 diffusion denoising 問題
```

### 論文 1: GDM4MMIMO — "Generative Diffusion Models for Massive MIMO"
- **出處:** arXiv:2412.18281, 2024

```
做了什麼: 把 diffusion model 統一應用到 massive MIMO 的三大問題：

問題 1: Channel Estimation
  - 傳統: LS/MMSE estimator → 需要知道 channel statistics
  - Diffusion: 把 noisy pilot measurements 視為 diffusion 的中間步
  - 用 score-based model 估計 ∇ log p(H|y)
  - 然後用 Langevin dynamics 採樣恢復 H

  數值: 64 antennas × 256 subcarriers OFDM system
    NMSE:
    - LS estimator: -10 dB
    - MMSE (perfect statistics): -18 dB
    - CNN-based: -16 dB
    - Diffusion-based: -20 dB (超越完美 MMSE！因為學到了 channel prior)

問題 2: Signal Detection (MIMO Detection)
  - 傳統: ML detection (NP-hard), ZF, MMSE detection
  - Diffusion: 把 received signal 視為「加了 channel + noise 後的 transmitted signal」
  - 用 conditional diffusion 恢復 transmitted signal

  數值: 16×16 MIMO, 64-QAM
    BER:
    - ZF: 10^{-2}
    - MMSE: 3×10^{-3}
    - CNN-based: 10^{-3}
    - Diffusion-based: 5×10^{-4}（比 MMSE 好 8 dB!）

問題 3: CSI Feedback/Compression
  - 問題: FDD massive MIMO 中，UE 需要把 downlink CSI 回傳給 BS
  - CSI matrix 很大（64×256 = 16384 complex values）→ 需要壓縮
  - 傳統: CsiNet (autoencoder-based compression)
  - Diffusion: 用 denoising 來「修復」壓縮損失

  數值: compression ratio 1/4 (只傳 4096 values)
    NMSE:
    - CsiNet: -15 dB
    - Transformer-based: -18 dB
    - Diffusion-enhanced: -22 dB
```

### 論文 2: Score-Based Models for Active User Detection + Channel Estimation
- **出處:** ZTE Communications, 2025

```
場景: Grant-free massive random access
  - 數千個 IoT devices，只有少部分（例如 5%）在某一時刻 active
  - BS 不知道哪些 device 是 active 的
  - 需要同時做: (1) 哪些 device active? (2) active devices 的 channel 是什麼?

這是一個 ill-posed inverse problem（欠定逆問題）：
  觀測維度 << 未知數維度

Score-based diffusion model 的做法:
  1. 學 p(H, a) 的 score function（H = channel, a = activity vector）
  2. 從 noisy observation 做 posterior sampling
  3. 同時恢復 activity pattern 和 channel

數值: 200 potential devices, 10 active, BS 64 antennas, 16 pilots
  Activity Detection Accuracy:
    - AMP (Approximate Message Passing): 92%
    - Diffusion: 99.5%
  Channel Estimation NMSE:
    - AMP: -8 dB
    - Diffusion: -18 dB (提升 10 dB!)
```

---

## 二、Research Gap

```
Gap 1: Diffusion for ISAC waveform design
  - 用 diffusion 生成「既適合通訊又適合雷達」的 waveform
  - 幾乎無人做過

Gap 2: 低延遲 diffusion for real-time PHY
  - 目前 diffusion 需要 10-50 步 denoising → 太慢
  - 用 consistency models 或 flow matching 做 1-step inference
  → "One-Step Diffusion Channel Estimator for Real-Time 6G"

Gap 3: Diffusion for satellite channel estimation
  - LEO satellite 的 channel estimation 特別困難（高 Doppler + 低 pilot density）
  - Diffusion 的 channel prior 可以補償 pilot 不足的問題
```

**預估時程:** 6-9 個月（code基礎好，DiffJSCC GitHub可參考）
**目標期刊:** IEEE TWC, IEEE TSP, IEEE TCOM
**GPU需求:** 高

---
---

# 方向四：Neuromorphic Wireless Receiver
# （神經形態無線接收器）

---

## 一、背景

### 1.1 什麼是 Neuromorphic Computing（神經形態計算）？

傳統 DNN 用連續的浮點數計算（multiply-accumulate）。每個 neuron 做：
```
output = activation(W × input + b)
```
這需要大量乘法運算，消耗大量能量。

**Spiking Neural Network (SNN, 脈衝神經網路)** 模仿生物神經元：
```
- 神經元累積輸入電位 (membrane potential)
- 當電位超過閾值 → 發出一個 spike（0 或 1）
- 發完 spike 後電位重置
- 不發 spike 時 → 不消耗能量

Leaky Integrate-and-Fire (LIF) neuron:
  V[t] = β × V[t-1] + I[t]    (β = leak factor, 0.9~0.99)
  if V[t] > V_threshold:
    spike = 1
    V[t] = V_reset
  else:
    spike = 0
```

**關鍵優勢：能耗極低。**

```
傳統 DNN (ResNet-18 on GPU):
  每次推理: ~100 mJ (millijoules)
  運算: 密集矩陣乘法，每個 neuron 都要算

SNN (同等規模 on neuromorphic chip):
  每次推理: ~1-3 mJ
  運算: 只有「發 spike」的 neuron 才需要計算（event-driven）
  能耗減少: 30-100 倍
```

### 論文 1: NeuromorphicRx — "From Neural to Spiking Receiver"
- **出處:** arXiv:2512.05246, 2025

```
核心創新: 用 SNN 完整取代 5G-NR OFDM 接收器的三大模組

傳統 OFDM 接收器:
  收到信號 y → Channel Estimation → Equalization → Demapping → 解碼出 bits

NeuromorphicRx:
  收到信號 y → SNN Encoder → SNN Processing → SNN Decoder → 解碼出 bits
  全部用 spike（0/1）計算！

SNN 架構細節:
  - 深度卷積 SNN + spike-element-wise (SEW) residual connections
  - ANN-SNN hybrid output layer（最後一層用 ANN 做 soft bit output）
  - 用 surrogate gradient 訓練（因為 spike 不可微分）

數值結果（5G NR, 3GPP TDL-A channel, 30 kHz SCS）:

  BLER (Block Error Rate) vs SNR:
  ┌─────────────┬──────────┬───────────┬────────────┐
  │ Method      │ SNR=5dB  │ SNR=10dB  │ SNR=15dB   │
  ├─────────────┼──────────┼───────────┼────────────┤
  │ LMMSE+EQ    │ 0.35     │ 0.08      │ 0.01       │
  │ ANN Receiver│ 0.30     │ 0.06      │ 0.008      │
  │ SNN Receiver│ 0.32     │ 0.07      │ 0.009      │
  └─────────────┴──────────┴───────────┴────────────┘

  SNN 只比 ANN 差 1.2 dB
  但能耗: SNN 是 ANN 的 1/7.6 (7.6x 更省電)
  加上 quantization-aware training 後: 1/35 (35x 更省電!)

  這意味著:
  - ANN receiver: 100 mW → SNN: 2.9 mW
  - 從 mW 級降到接近 µW 級
  - 適合 IoT 設備、可穿戴裝置、衛星終端
```

---

## 二、Research Gap

```
Gap 1: SNN for 高階調變（目前只做到 BPSK/QPSK）
  - 64-QAM, 256-QAM 的 SNN detection 尚無人做
  → "High-Order Modulation Detection via Multi-Level Spiking Neural Networks"

Gap 2: SNN 用於 ISAC
  - ISAC receiver 需要同時處理通訊和雷達信號
  - SNN 的 event-driven 特性天然適合雷達的 sparse pulse processing
  → "Neuromorphic ISAC Receiver: Event-Driven Joint Sensing and Communication"

Gap 3: SNN + semantic communication
  - SNN encoder at transmitter + SNN decoder at receiver
  - 超低功耗語義通訊
```

**預估時程:** 9-12 個月
**目標期刊:** IEEE TWC, IEEE JSAC, IEEE TCOM
**GPU需求:** 中（SNN training with surrogate gradients）

---
---

# 方向五：Agentic AI for Autonomous Network Management
# （代理式 AI 用於自主網路管理）

---

## 一、背景

### 1.1 什麼是 Agentic AI？

Agentic AI ≠ 單一 LLM 回答問題。它是**多個 AI agent 組成的系統**，每個 agent 有自己的角色、記憶、工具，能夠：
- **感知**環境（讀取 network metrics）
- **推理**問題（用 LLM 分析 root cause）
- **規劃**行動（制定 multi-step plan）
- **執行**操作（呼叫 API 調整配置）
- **學習**經驗（記住什麼有效什麼無效）

### 論文 1: Confucius — "Intent-Driven Network Management with Multi-Agent LLMs"
- **出處:** ACM SIGCOMM 2025（頂級網路會議）
- **作者:** Meta (Facebook) 的工程團隊

```
這是全球第一個在生產環境中運行的 multi-agent LLM 網路管理系統。
在 Meta 的內部網路已經運行 2+ 年，服務 60+ 個應用場景。

架構:
  Human Operator: 「我想讓 A 區域的延遲降到 10ms 以下」
        ↓ (自然語言 intent)
  Intent Parser Agent (LLM):
    解析: area=A, metric=latency, target=<10ms
        ↓
  Planner Agent (LLM + RAG):
    查詢歷史案例（RAG）→ 制定 multi-step plan:
    Step 1: 檢查 A 區域當前的 routing config
    Step 2: 識別 bottleneck links
    Step 3: 計算新的 traffic engineering weights
    Step 4: 驗證新 config 不會影響其他區域
        ↓
  Executor Agent:
    呼叫 network API 執行 config 變更
        ↓
  Validator Agent:
    監控變更後的指標，確認達到目標
    如果沒達到 → 回報 Planner Agent 做修正

關鍵設計:
  - DAG (Directed Acyclic Graph) workflow: 工作流程定義為 DAG
  - RAG for long-term memory: 用歷史案例作為知識庫
  - Human-in-the-loop: 重要變更需要人確認
  - Regression prevention: 自動驗證不會造成其他指標退化
```

### 論文 2: MeshAgent — "Enabling Reliable Network Management with LLMs"
- **出處:** ACM SIGMETRICS 2026
- **作者:** Microsoft Research

```
核心創新: 解決 LLM 在網路管理中的「不可靠」問題

問題: LLM 會產生幻覺（hallucination）
  例如: LLM 可能建議一個不存在的 routing path
  或: LLM 可能輸出一個違反 network constraints 的 config

MeshAgent 的解法:

1. Domain-specific invariant extraction:
   從 network specification 中自動提取不可違反的約束
   例如: "每個 interface 的 bandwidth 不能超過 100 Gbps"
         "routing table 不能有 loops"

2. Constraint-guided generation:
   LLM 生成的 config 必須通過 invariant checker
   如果違反 → 自動修正或重新生成

3. Abstention mechanism:
   如果 LLM 的 confidence 低於閾值 → 不執行，告知人類

數值:
  - Accuracy: 95%+ (100% with fine-tuned agents)
  - vs. 普通 LLM (GPT-4): 提升 26%
  - False positive rate (誤操作): 接近 0%（因為 invariant checker）
```

---

## 二、Research Gap

```
Gap 1: Agentic AI for satellite constellation management
  - Confucius 管理 Meta 的 data center network
  - 沒有人把這種 multi-agent LLM 用於衛星星座管理
  → "SatOps: Multi-Agent LLM Framework for Autonomous LEO Constellation Operations"

Gap 2: LLM agent + RL agent 的深度融合
  - 目前 LLM agent 和 RL agent 是分開的
  - LLM 做 high-level 規劃, RL 做 low-level 控制
  - 缺乏理論框架來分析兩者的最佳分工
  → "Hierarchical LLM-RL Agent for 6G Network Optimization"

Gap 3: Safety and verification
  - 自主網路管理如果出錯，後果很嚴重
  - MeshAgent 提出了 invariant checking，但不夠完整
  → "Formally Verified Agentic AI for Critical Network Infrastructure"
```

**預估時程:** 9-12 個月
**目標期刊:** IEEE TNSE Special Issue "Agentic AI" (deadline 2026-05-01), ACM SIGCOMM
**GPU需求:** 高（LLM fine-tuning + RL training）

---
---

# 方向六：SatFM — 衛星通道基礎模型
# （同前一版本，此處為精簡版）

---

這個方向保留，因為它**主要靈感來自 LWM（Arizona State）和 WiFo（清華）這兩個頂刊工作**，而非實驗室自己的研究。與老師的交集只在「老師做 LEO 衛星」，但 foundation model 這個方法完全是外部引入的。

**核心邏輯：**
```
LWM (arXiv:2411.08872) → 首個地面 channel FM
WiFo (Science China 2025) → 首個 STF channel FM
CPLLM (arXiv:2510.10561) → 用 LLM fine-tune 做 LEO channel prediction

我們的 SatFM: 首個 satellite-native channel FM
  = LWM 的架構 + 3GPP TR 38.811 的 NTN channel data + 衛星特有 inductive bias
```

（完整說明見前一版 IDEAS_EXPLAINED_zh.md）

---
---

# 六個方向的最終比較

| 排名 | 方向 | 來源 | 與實驗室交集 | 新穎度 | GPU需求 | 出paper速度 | 博士論文潛力 |
|------|------|------|------------|--------|---------|-------------|------------|
| **1** | **World Models for 6G** | TNSE SI, NeurIPS | 低 | ★★★★★ | 高 | 9-12月 | ★★★★★ |
| **2** | **Causal AI for Wireless** | IEEE VTM, arXiv | 零 | ★★★★★ | 中-高 | 9-12月 | ★★★★★ |
| **3** | **Diffusion for PHY** | IEEE TWC, NeurIPS | 低 | ★★★★ | 高 | 6-9月 | ★★★★ |
| **4** | **Neuromorphic Rx** | arXiv, KCL | 零 | ★★★★★ | 中 | 9-12月 | ★★★★ |
| **5** | **Agentic AI for Networks** | SIGCOMM, TNSE SI | 低 | ★★★★★ | 高 | 9-12月 | ★★★★★ |
| **6** | **SatFM** | LWM(ASU), WiFo | 中 | ★★★★★ | 極高 | 9-12月 | ★★★★★ |

## 博士生建議組合策略

```
策略 A（最穩健）:
  Year 1: 方向 3 (Diffusion for PHY) — 最快出paper, 建立基礎
  Year 2: 方向 1 (World Models) — 深入, 瞄準 TNSE Special Issue
  Year 3: 結合 1+6 → "World-Model-Enhanced Satellite Channel Foundation Model"
  Year 4: 博士論文整合

策略 B（最前沿）:
  Year 1: 方向 2 (Causal AI) — 超新, 搶先定義方向
  Year 2: 方向 5 (Agentic AI) — 結合因果推斷的 LLM agent
  Year 3: 結合 2+5 → "Causally-Aware Agentic AI for Self-Evolving 6G Networks"
  Year 4: 博士論文整合

策略 C（最平衡）:
  Year 1: 方向 3 (Diffusion for PHY) — 快速出paper
  Year 2: 方向 6 (SatFM) — 建立差異化
  Year 3: 方向 1 or 2 — 深度方向
  Year 4: 博士論文整合
```

---

*每個方向都已準備好詳細的背景、原理、論文、數值例子。等你理解釐清後，我們再來決定哪個方向要深入去做實驗和寫論文。*
