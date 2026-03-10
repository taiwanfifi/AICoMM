# 三大研究方向 — 完整繁體中文講解

> 專有名詞維持英文，其餘以繁體中文說明。
> 每個方向包含：背景、參考論文原理、方法細節、數值例子、流程說明。

---

# 方向一：Split-Inference Diffusion Semantic Communication
# （拆分推理擴散語義通訊）

---

## 一、背景：為什麼需要這個研究？

### 1.1 什麼是 Semantic Communication（語義通訊）？

傳統通訊的目標是「正確傳送每一個 bit」。Shannon 理論告訴我們，只要 data rate 低於 channel capacity，就能做到接近零錯誤的傳輸。但問題是：**很多時候我們不需要傳送每一個 bit**。

舉個例子：你用手機拍了一張 4K 照片（約 12 MB），要傳給另一端做「這是什麼物體？」的分類任務。傳統做法是把 12 MB 的 JPEG 壓縮後傳過去，對方解壓後再丟進分類器。但如果我們只需要知道「這是一隻貓」，為什麼要傳 12 MB？

**Semantic Communication 的核心想法：只傳送對接收端「有意義」的資訊。**

具體來說，transmitter 端用一個 neural network（稱為 semantic encoder）萃取出影像的「語義特徵」（比如物件輪廓、紋理、語義標籤），只傳送這些特徵，receiver 端再用另一個 neural network（semantic decoder）來還原或直接用於下游任務。

### 1.2 什麼是 Deep Joint Source-Channel Coding（DeepJSCC）？

傳統通訊系統是「分層」的：
```
Source Coding (JPEG/H.264) → Channel Coding (LDPC/Turbo) → Modulation → Channel → 反向
```
每一層各自優化。但 Shannon 在 1959 年就證明了：**joint source-channel coding（聯合源通道編碼）在理論上可以達到最佳效能**，只是找不到實際的 optimal joint code。

DeepJSCC 用 deep learning 來逼近這個 optimal joint code：
```
影像 → Neural Network Encoder → 通道符號 → Wireless Channel → Neural Network Decoder → 重建影像
```
整個系統 end-to-end 訓練，encoder 和 decoder 同時學會「壓縮」和「保護」（即同時做 source coding 和 channel coding）。

**關鍵優勢：** DeepJSCC 不會出現傳統系統的「cliff effect」。傳統系統在 SNR 低於某個門檻時會完全崩潰（因為 channel code 解不出來），但 DeepJSCC 是 graceful degradation — SNR 越低，影像品質越差但不會完全壞掉。

### 1.3 什麼是 Diffusion Model（擴散模型）？

Diffusion Model 是目前最強大的生成式 AI 模型之一（Stable Diffusion、DALL-E 3、Midjourney 都是基於此）。它的原理分兩步：

**Forward Process（正向過程）：** 把乾淨的圖片逐步加 Gaussian noise，經過 T 步之後變成純雜訊。
```
x₀ (原圖) → x₁ (微噪) → x₂ (更噪) → ... → xₜ (純雜訊)
```
每一步加的 noise 量由一個 schedule（β₁, β₂, ..., βₜ）控制。

**Reverse Process（逆向過程 / Denoising）：** 訓練一個 neural network（通常是 U-Net）學會「從 xₜ 預測 xₜ₋₁」，也就是逐步去噪。
```
xₜ (純雜訊) → xₜ₋₁ → ... → x₁ → x₀ (重建圖片)
```

**數值例子：**
- Stable Diffusion 通常用 T = 1000 步的 forward process
- 推理時用 DDIM sampling 可以縮減到 20-50 步
- 每一步 denoising 需要跑一次完整的 U-Net forward pass
- Stable Diffusion 的 U-Net 約有 860M parameters
- 在 NVIDIA A100 上，一步 denoising 約需 50-100 ms
- 所以完整 50 步推理需要 2.5-5 秒

### 1.4 Diffusion Model 用在 Semantic Communication 的好處

為什麼要把 Diffusion Model 放進 Semantic Communication？

**理由一：Diffusion 天生就在做「去噪」。** Wireless channel 也會加 noise。所以 diffusion 的 reverse process 可以同時學會去除 channel noise 和 source noise，達到「joint denoising」的效果。

**理由二：生成品質極高。** 在 SNR 很低（訊號很弱）的情況下，傳統 DeepJSCC 重建的影像會很模糊。但 Diffusion Model 可以利用它學到的「影像先驗知識（prior）」來「補腦」，生成看起來合理的細節。

**數值比較（來自 DiffJSCC 論文）：**
```
在 Kodak 資料集上，SNR = 1 dB，bandwidth ratio = 1/16：
- 傳統 JPEG + LDPC：完全失敗（cliff effect）
- DeepJSCC (CNN-based)：PSNR ≈ 20.5 dB
- DiffJSCC (Diffusion-based)：PSNR ≈ 24.3 dB，且主觀品質遠超
                              LPIPS ≈ 0.15（越低越好）
                              FID ≈ 45（越低越好）
```

### 1.5 問題在哪？→ Diffusion 太重了！

一個完整的 Diffusion Model（如 Stable Diffusion）需要：
- 860M parameters（U-Net 部分）
- 數 GB GPU memory
- 每張圖推理 2-5 秒

如果 receiver 是手機或 IoT 裝置，根本跑不動。

**但 server（edge server 或 cloud）可以跑。**

所以自然的想法是：**把 denoising 的一部分跑在 server，一部分跑在 device。**

---

## 二、參考論文與它們的原理

### 論文 1: DiffJSCC — "Diffusion-Aided Joint Source Channel Coding"
- **作者:** Mingyu Yang, Bowen Liu et al.
- **出處:** arXiv:2404.17736, 2025
- **開源:** [GitHub](https://github.com/mingyuyng/DiffJSCC)

**它做了什麼：**
DiffJSCC 把 pre-trained Stable Diffusion 接在 DeepJSCC 的 receiver 端。流程如下：

```
Transmitter 端：
  原圖 x → JSCC Encoder → 通道符號 z → 送入 wireless channel

Receiver 端：
  1. 收到有噪的 ẑ = z + channel noise
  2. JSCC Decoder 做初步重建 → x̂_coarse（粗略重建，可能很模糊）
  3. 同時從 ẑ 萃取 spatial features（空間特徵）和 text features（用 CLIP）
  4. 把 spatial + text features 作為 condition，輸入 pre-trained Stable Diffusion
  5. Stable Diffusion 做 conditional denoising（50步）→ x̂_fine（精細重建）
```

**關鍵數據：**
- 在 Kodak 資料集（768×512 影像）上
- 只需 3072 個 channel symbols（< 0.008 symbols/pixel）
- 在 SNR = 1 dB 的超低訊噪比下
- 仍能產生「看起來真實」的高品質重建
- LPIPS（感知相似度）從 DeepJSCC 的 0.35 降到 0.15

**局限：** Diffusion 的 50 步 denoising **全部在 receiver 端執行**。如果 receiver 是手機→跑不動。

---

### 論文 2: Digital-SC — "Digital Semantic Communication with Adaptive Network Split"
- **作者:** Mingyao Guo, Jia Chen et al.
- **出處:** IEEE JSAC, 2024 (arXiv:2305.13553)

**它做了什麼：**
Digital-SC 把一個 DNN-based semantic communication 系統的 encoder/decoder 拆分在 device 和 edge 之間。

```
                    Device                          Edge Server
               ┌──────────────┐               ┌──────────────────┐
  原始資料 x → │ Encoder 前 k 層│ → 中間特徵 f → │ Encoder 後 N-k 層  │
               └──────────────┘    (壓縮+量化    │      ↓           │
                                    後傳送)      │ Channel Symbols  │
                                                 │      ↓           │
                                                 │ Decoder 全部層   │
                                                 │      ↓           │
                                                 │ 重建結果 x̂       │
                                                 └──────────────────┘
```

一個 **policy network** 根據當前 channel condition（SNR）和 device 的計算能力，動態決定：
- k = split point（在第幾層切）
- 傳送的 feature dimension（壓縮多少）

**數值例子：**
- 使用 ResNet-based encoder（12 層）
- 如果 SNR 高（通道好）→ policy 選 k=3（device 只算前3層），傳較少特徵
- 如果 SNR 低（通道差）→ policy 選 k=8（device 算更多層），傳更壓縮的特徵
- 相比固定 split point，自適應 split 在各種 SNR 下平均提升 1-2 dB PSNR

**局限：** 它用的是傳統 CNN encoder/decoder，**不是 diffusion model**。所以生成品質不如 diffusion-based 方法。

---

### 論文 3: Yang et al. — "Efficient Multi-user Offloading of Personalized Diffusion Models"
- **出處:** arXiv:2411.15781, 2024

**它做了什麼：**
這篇是目前**最接近**我們想法的論文。它把 diffusion 的 denoising 拆分在 server 和 device 之間：

```
Server:  denoising steps T → T-1 → ... → nᵢ*+1  （前面的粗略去噪步驟）
            ↓ 傳送中間結果 x_{nᵢ*}
Device:  denoising steps nᵢ* → nᵢ*-1 → ... → 0    （後面的精細去噪步驟）
            ↓
         最終生成結果 x₀
```

用 **PER-DQN（Prioritized Experience Replay DQN）** 來決定：
- 每個 user 的 split point nᵢ*
- 是否要 offload 到 server
- 頻寬分配

**數值例子（多用戶場景）：**
- 4 個 users，每人要生成不同的圖（不同 text prompt）
- Server: NVIDIA A100, 每步 denoising ~80ms
- Device: smartphone GPU, 每步 denoising ~500ms
- 如果全部在 server 做（T=50 步）：server 排隊延遲大
- 如果全部在 device 做：每張圖 50×500ms = 25秒
- 用 split（server 做 30 步, device 做 20 步）：
  - Server: 30×80ms = 2.4s per user
  - 傳輸中間結果: ~200ms (假設 10 Mbps)
  - Device: 20×500ms = 10s
  - 總延遲: max(server排隊時間, 10s) ≈ 12s（比全device的25s快一倍）

**關鍵差異：這篇做的是 AIGC（AI Generated Content）的服務，不是 Semantic Communication。**

在 AIGC 場景中，wireless channel 只是一個「傳送 intermediate feature 的管道」。但在 Semantic Communication 中，wireless channel 是整個系統設計的一部分 — channel noise 會影響語義品質，split point 的選擇不只影響延遲，還影響語義保真度。

---

### 論文 4: 老師實驗室 — Jason Muliawan, WCNC 2026
**"Diffusion Model-Assisted Task-Oriented Semantic Communication with DRL-Based Resource Optimization"**

**它做了什麼：**
這是老師實驗室最新的論文，把 diffusion model 放進 task-oriented semantic communication，並用 DRL 做資源優化。

```
Transmitter: Semantic Encoder 萃取 task-relevant features
        ↓
Wireless Channel (加 noise)
        ↓
Receiver: Diffusion Model 做 feature enhancement（去除 channel noise）
        ↓
Task Head: 執行下游任務（例如分類）
        ↓
DRL Agent: 根據 channel state，決定 bandwidth 和 power allocation
```

**重點：** Diffusion 整個跑在 receiver 端，沒有 split。

---

### 論文 5: 老師實驗室 — 許欣芸, COMNETSAT 2025
**"Semantics-Preserving Dynamic Pruning for Split Inference in Log Anomaly Detection"**

**它做了什麼：**
把一個 DNN 的推理拆分在 device 和 server 之間（split inference），並在傳送中間 feature 之前做「dynamic pruning（動態剪枝）」來減少傳輸量，同時保證語義資訊不被破壞。

```
Device:  DNN 前 k 層 → intermediate features
         ↓ dynamic pruning（移除不重要的 feature channels）
         ↓ 只傳送重要的 channels
Edge Server: 收到 pruned features → DNN 後面的層 → 輸出結果
```

---

## 三、我們的創新想法：SplitDiffSem

### 3.1 核心概念

**把 diffusion model 的 reverse denoising process 拆分在 edge server 和 device 之間，專門用於 semantic communication，並用 DRL 自適應優化 split point 和 resource allocation。**

```
                 Transmitter                        Receiver (Edge + Device)
            ┌──────────────────┐            ┌─────────────────────────────────┐
            │                  │            │                                 │
  原始資料   │  Semantic Encoder │  encoded   │   Edge Server                   │
  x ───────→│  (lightweight)   │──features──→│   ┌─────────────────────┐      │
            │                  │  over      │   │ Denoising Steps     │      │
            └──────────────────┘  wireless  │   │ T → T-1 → ... → k+1│      │
                                  channel   │   └─────────┬───────────┘      │
                                            │             │ 中間特徵 x_k      │
                                            │             ↓ (local transfer) │
                                            │   Device                       │
                                            │   ┌─────────────────────┐      │
                                            │   │ Denoising Steps     │      │
                                            │   │ k → k-1 → ... → 0  │      │
                                            │   └─────────┬───────────┘      │
                                            │             ↓                  │
                                            │   重建結果 x̂ 或 Task Output     │
                                            │                                 │
                                            │   DRL Agent:                    │
                                            │   決定 k, bandwidth, power      │
                                            └─────────────────────────────────┘
```

### 3.2 為什麼這是新的？

我們做了徹底的新穎性驗證（搜索了 12 篇最接近的論文），發現：

```
Yang et al.  = split diffusion + DRL          但做 AIGC 不做 SemCom ✗
Tang et al.  = diffusion + SemCom + split     但拆 forward process 不拆 reverse ✗
Digital-SC   = split + SemCom + adaptive      但用 CNN 不用 diffusion ✗
Jason WCNC'26 = diffusion + SemCom + DRL      但不拆分 ✗

我們      = split reverse diffusion + SemCom + DRL  → 三者交集為空 ✓
```

### 3.3 具體方法設計

**Step 1: Semantic Encoder（在 transmitter 端）**

使用一個輕量級的 encoder（例如 MobileNet 或小型 ViT），把原始影像 x 編碼成一組 semantic features z：

```
z = Encoder(x)
z 的維度: (C, H', W') 例如 (256, 16, 16) = 65,536 個值

這些值直接映射到 channel symbols（OFDM subcarriers）
channel symbols 數量 = C × H' × W' / compression_ratio
```

**Step 2: 通過 Wireless Channel**

channel symbols 通過無線通道後受到衰落和噪聲影響：

```
ẑ = h × z + n

其中：
h = channel fading coefficient (Rayleigh: |h| ~ Rayleigh(σ))
n = AWGN noise (n ~ N(0, σ²_n))
SNR = |h|² × E[|z|²] / σ²_n
```

**數值例子：**
```
假設 SNR = 5 dB:
  信號功率 = 1
  噪聲功率 = 10^(-5/10) = 0.316
  所以收到的 ẑ 大約有 30% 的 noise corruption
```

**Step 3: Edge Server 做前半段 Denoising（步驟 T → k+1）**

Edge server 有完整的 Diffusion U-Net（860M parameters）。它收到 ẑ 後：

```
(a) 把 ẑ 當作 condition
(b) 從 Gaussian noise xₜ 開始做 conditional reverse denoising:
    xₜ → xₜ₋₁ → ... → x_{k+1} → x_k

這些步驟負責「粗略的全局結構」重建。
```

**為什麼前面的步驟更重要？**

Diffusion denoising 的特性：
- 前面的步驟（T → T/2）：決定全局結構（構圖、物體位置、大致形狀）
- 後面的步驟（T/2 → 0）：補充細節（紋理、邊緣、色彩微調）

所以 server 做前面的「重要步驟」，device 做後面的「輕量步驟」是合理的。

**數值例子：**
```
假設 T = 50 步 DDIM, split point k = 30:
  Server 做 20 步（步驟 50→31）: 20 × 80ms = 1.6 秒（A100 GPU）
  傳送 x_30 到 device: x_30 大小 ≈ 4×64×64 = 16,384 floats = 64 KB
    傳輸時間（假設 10 Mbps）≈ 50ms
  Device 做 30 步（步驟 30→1）:
    但 device 用的是 pruned/distilled 小模型（例如 100M params）
    每步 ~150ms → 30 × 150ms = 4.5 秒

  總延遲: 1.6 + 0.05 + 4.5 = 6.15 秒

  對比:
  - 全部在 server: 50 × 80ms = 4 秒，但 server 要排隊 + 傳完整影像回來
  - 全部在 device: 50 × 500ms = 25 秒
  - 不用 diffusion（純 DeepJSCC）: 0.1 秒，但品質差很多
```

**Step 4: Device 做後半段 Denoising（步驟 k → 0）**

Device 收到中間特徵 x_k 後，用一個**輕量版 diffusion model**（例如用 knowledge distillation 壓縮過的）完成剩餘的 denoising。

這裡有一個關鍵設計：device 端的模型可以用 **consistency distillation** 把多步 denoising 壓縮成更少步（甚至 1 步），大幅加速。

```
原本: x_k → x_{k-1} → ... → x_0    （k 步，每步跑一次 U-Net）
蒸餾後: x_k → x_0                    （1步，跑一次 distilled model）

延遲從 4.5 秒降到 0.15 秒！
```

**Step 5: DRL Agent 決定最佳 Split Point**

**State（狀態）：**
```
s = [SNR, bandwidth_available, server_queue_length, device_compute_capacity, task_type]
```

**Action（動作）：**
```
a = [k (split point, 0~T), bandwidth_allocation, power_level]

k = 0: 全部在 server 做
k = T: 全部在 device 做
k = 中間值: split
```

**Reward（獎勵）：**
```
r = α × Quality(x̂) - β × Latency - γ × Energy

其中:
- Quality 可以是 PSNR, SSIM, 或 task accuracy
- Latency = server_compute + transmission + device_compute
- Energy = device 的能耗
- α, β, γ 是權重（根據 task priority 調整）
```

**DRL 訓練例子：**
```
用 PPO (Proximal Policy Optimization):
- Episode: 隨機生成 SNR (0~20 dB), bandwidth, queue length
- Agent 選擇 (k, bw, power)
- 計算 quality + latency + energy
- 更新 policy network

訓練 10,000 episodes 後:
- SNR = 15 dB 時: agent 選 k=35（server少做，device多做，因為channel好）
- SNR = 0 dB 時: agent 選 k=10（server多做，device少做，因為需要更精確的去噪）
- Server 很忙時: agent 選 k=45（幾乎全在device做）
```

### 3.4 與其他工作的本質區別

| 特性 | AIGC Offloading (Yang et al.) | 我們的 SplitDiffSem |
|------|------------------------------|---------------------|
| 目標 | 生成新圖片 (text→image) | 通訊重建 (transmit→receive) |
| Channel 角色 | 只是傳輸管道 | 系統設計的核心（noise影響split策略）|
| Split 考量 | 只考慮延遲和計算量 | 考慮語義保真度 + channel condition + 延遲 |
| Diffusion 的角色 | 生成內容 | 去除 channel noise + 補充語義細節 |
| 優化目標 | 延遲最小化 | 語義品質-延遲-能耗聯合最佳化 |

---

## 四、預期實驗設計

```
基礎程式碼: DiffJSCC (GitHub 開源)
Channel model: Rayleigh fading + AWGN
Diffusion: Stable Diffusion v2.1 (server) + Distilled version (device)
DRL: PPO with 2-layer MLP policy network
Dataset: CIFAR-10 (快速實驗), Kodak (正式結果), ImageNet (大規模)
GPU: server 用 A100, device 模擬用限制 memory 的小 GPU

Baseline 比較:
1. DeepJSCC (no diffusion) — 品質下界
2. DiffJSCC (all on server) — 品質上界但延遲高
3. Digital-SC (split but no diffusion) — split 但品質不如
4. Yang et al. (split diffusion but no SemCom) — 不考慮 channel
5. 我們的 SplitDiffSem — 預期在 quality-latency tradeoff 上 Pareto optimal
```

---
---

# 方向二：SatFM — 衛星通道基礎模型
# （Satellite Channel Foundation Model）

---

## 一、背景：為什麼需要衛星 Channel Foundation Model？

### 1.1 什麼是 Foundation Model？

Foundation Model（基礎模型）是一種在**大規模數據**上**預訓練**的模型，可以透過 **fine-tuning** 或 **zero-shot transfer** 適應各種下游任務。

最知名的例子：
- **語言：** GPT-4, Claude → 在數兆 token 上預訓練 → fine-tune 後可做翻譯、摘要、coding
- **視覺：** ViT (Vision Transformer) → 在 ImageNet 上預訓練 → fine-tune 後可做物體偵測、分割

**無線通訊也開始有了自己的 Foundation Model。**

### 1.2 已有的 Wireless Foundation Models（全部是地面場景）

**LWM (Large Wireless Model)** — 2024年11月，全球首個 channel foundation model
- **作者:** Alkhateeb et al. (Arizona State University)
- **做法:**
  1. 用 [DeepMIMO](https://www.deepmimo.net/) 生成 1,000,000+ 組無線通道矩陣
     - 來自 15 個不同場景（城市街道、校園、波士頓等）
     - 每組通道: H ∈ ℂ^{32×32}（32根天線 × 32個 subcarrier）
  2. 把 H 轉成 token sequence（展平 + patching）
  3. 用 **Masked Channel Modeling (MCM)** 預訓練：
     - 隨機遮住 75% 的 channel patches
     - Transformer 學會預測被遮住的部分
     - 概念和 BERT 的 Masked Language Modeling 一樣
  4. Pre-trained model 產生 **universal channel embeddings**
  5. Fine-tune 做各種下游任務

**數值結果（LWM）：**
```
下游任務1: Beam Prediction
  - LWM fine-tune: Top-3 accuracy = 89%
  - 從零訓練的 CNN: Top-3 accuracy = 76%
  - 提升: +13%

下游任務2: LOS/NLOS Classification
  - LWM fine-tune: Accuracy = 95%
  - 從零訓練: Accuracy = 87%
  - 提升: +8%

關鍵: LWM 在從未見過的新場景上（zero-shot）也能達到 82% beam prediction accuracy
```

**WiFo (Wireless Foundation Model for Channel Prediction)** — 2025
- 用 Masked Autoencoder (MAE) 在 space-time-frequency domain 預訓練
- 160K CSI 樣本
- **Zero-shot**（完全不用 fine-tune）就能做 channel prediction
- 還壓縮出 **Tiny-WiFo**（5.5M parameters），推理只要 1.6ms

### 1.3 問題：這些都是「地面」的 Foundation Model

LWM 和 WiFo 的訓練數據全部來自**地面場景**（城市、郊區的基地台到手機）。

**LEO 衛星通道有什麼不同？**

```
                 地面通道                              LEO 衛星通道
    ──────────────────────────              ──────────────────────────────
    距離: 100m - 10km                       距離: 300-2000 km
    Doppler: 10-1000 Hz                     Doppler: 10-40 kHz（衛星高速移動）
    Path Loss: 60-120 dB                    Path Loss: 150-190 dB
    延遲: < 1 ms                            延遲: 2-20 ms
    通道變化: 慢（秒級）                      通道變化: 快（衛星 7.5 km/s）
    仰角: 固定（水平）                        仰角: 動態（10°-90°）
    衛星經過時間: N/A                         衛星可見時間: 5-15 分鐘
    Fading: Rayleigh/Rician                 Fading: Shadowed-Rician（受建物/樹遮蔽）
```

因為這些根本性差異，**地面 FM 無法直接用於衛星通道**。但目前沒有人做過 satellite-specific channel FM。

### 1.4 最接近的工作：CPLLM

**CPLLM** (arXiv:2510.10561, 2025) 用 pre-trained LLM（GPT-2 系列）做 LEO channel prediction。它設計了一個 CSI encoder 把通道矩陣映射到 LLM 的 text embedding space，然後用 LoRA fine-tune。

```
CSI history [H_{t-L}, ..., H_{t-1}] → CSI Encoder → Token Embeddings
    → Frozen LLM (GPT-2) + LoRA → Predicted H_t
```

**結果：** 比專門設計的 DNN baseline 提升 6 dB NMSE。

**但它不是 Foundation Model。** 它只是把通用 LLM fine-tune 來做一個特定任務。它沒有在大規模衛星通道數據上預訓練，無法 zero-shot transfer 到其他任務。

---

## 二、我們的想法：SatFM

### 2.1 核心概念

在 **大規模模擬 LEO 衛星通道數據**上，用 **Transformer + Masked Channel Modeling** 預訓練一個 foundation model，使其學會衛星通道的通用表徵（universal representation），再 fine-tune 到各種下游任務。

### 2.2 數據生成 Pipeline

```
Step 1: 定義場景參數空間
    - 軌道高度: [300, 600, 1200, 2000] km
    - 頻段: S-band (2 GHz), Ka-band (20/30 GHz), L-band (1.5 GHz)
    - 環境: Dense Urban, Urban, Suburban, Rural (3GPP TR 38.811)
    - 仰角: [10°, 20°, 30°, 45°, 60°, 75°, 90°]
    - Doppler: 根據軌道高度和仰角計算
    - 天線配置: [4, 8, 16, 32, 64] antenna elements

Step 2: 用 OpenNTN + Sionna 生成通道
    - OpenNTN 實作了 3GPP TR 38.811 的 NTN-TDL channel models
    - Sionna 是 NVIDIA 的 GPU-accelerated 通訊模擬器
    - 每個場景組合生成 10,000 個 channel realization
    - 總數據量: 4 × 3 × 4 × 7 × 5 × 10,000 = 16,800,000 通道矩陣

Step 3: 格式化
    每個通道矩陣 H ∈ ℂ^{N_rx × N_subcarrier × N_tap}
    轉成實數表示: [Re(H), Im(H)] → 維度 (2, N_rx, N_subcarrier, N_tap)
```

**數值例子：**
```
一個典型的 LEO channel sample:
  軌道: 600 km (Starlink-like)
  頻段: Ka-band, 20 GHz
  環境: Urban
  仰角: 45°

  NTN-TDL-D model (LOS dominant):
    Rician K-factor = 10 dB
    第一條 path (LOS): delay = 0 ns, power = 0 dB
    第二條 path: delay = 30 ns, power = -12 dB
    第三條 path: delay = 70 ns, power = -18 dB
    Doppler shift: ~24 kHz (衛星相對速度 ~7.5 km/s)

  Channel matrix H (32 antennas × 64 subcarriers):
    H[i,j] = complex value representing the channel between antenna i and subcarrier j
    |H[i,j]| 大約在 0.001 ~ 0.1 的範圍（考慮 path loss 後 normalize）
```

### 2.3 Model Architecture

```
┌─────────────────────────────────────────────────────┐
│                      SatFM                           │
│                                                      │
│  Input: H ∈ ℝ^{2 × N_ant × N_sub}                  │
│         + metadata [orbit, freq, elev, env]          │
│                                                      │
│  ┌─────────────────────────────────┐                 │
│  │ Satellite-Aware Embedding       │                 │
│  │  - Patch embedding (like ViT)   │                 │
│  │  - Orbit-phase position embed   │                 │
│  │  - Elevation angle embedding    │                 │
│  │  - Doppler-aware frequency embed│                 │
│  └──────────┬──────────────────────┘                 │
│             ↓                                        │
│  ┌─────────────────────────────────┐                 │
│  │ Transformer Encoder             │                 │
│  │  - 12 layers                    │                 │
│  │  - 768 hidden dim               │                 │
│  │  - 12 attention heads           │                 │
│  │  - ~85M parameters              │                 │
│  └──────────┬──────────────────────┘                 │
│             ↓                                        │
│  Universal Channel Embeddings                        │
│             ↓                                        │
│  Task-specific heads (fine-tune 時加上):              │
│  ├─ Channel Prediction Head                          │
│  ├─ Beam Prediction Head                             │
│  ├─ Handover Prediction Head                         │
│  └─ LOS/NLOS Classification Head                     │
└─────────────────────────────────────────────────────┘
```

### 2.4 Pre-training 方法: Masked Channel Modeling (MCM)

```
原始 channel matrix H (32 × 64):
┌──┬──┬──┬──┬──┬──┬──┬──┐
│p1│p2│p3│p4│p5│p6│p7│p8│     每個 patch = 4×8 的 channel block
├──┼──┼──┼──┼──┼──┼──┼──┤     共 8×8 = 64 patches
│p9│..│..│..│..│..│..│16│
├──┼──┼──┼──┼──┼──┼──┼──┤
│..│..│..│..│..│..│..│..│
├──┼──┼──┼──┼──┼──┼──┼──┤
│..│..│..│..│..│..│..│..│
├──┼──┼──┼──┼──┼──┼──┼──┤
│..│..│..│..│..│..│..│..│
├──┼──┼──┼──┼──┼──┼──┼──┤
│..│..│..│..│..│..│..│..│
├──┼──┼──┼──┼──┼──┼──┼──┤
│..│..│..│..│..│..│..│..│
├──┼──┼──┼──┼──┼──┼──┼──┤
│57│..│..│..│..│..│..│64│
└──┴──┴──┴──┴──┴──┴──┴──┘

Masking: 隨機遮住 75% 的 patches (48 out of 64)
保留 25% (16 patches) + satellite metadata 作為 input
Transformer 預測被遮住的 48 個 patches

Loss = MSE(predicted_patches, ground_truth_patches)
```

**訓練過程：**
```
Batch size: 256
Learning rate: 1.5e-4 with cosine annealing
Training epochs: 100
GPU: 4× NVIDIA A100 (80GB)
Training time estimate: ~2-3 days (1600萬個 channel samples)
```

### 2.5 下游任務與預期結果

**任務 1: Channel Prediction（通道預測）**
```
Input: H_t, H_{t-1}, H_{t-2} (過去 3 個 time slot 的通道)
Output: Ĥ_{t+1} (預測下一個 time slot 的通道)

預期結果:
  - SatFM (fine-tune): NMSE ≈ -20 dB
  - CPLLM (baseline): NMSE ≈ -14 dB
  - 從零訓練的 LSTM: NMSE ≈ -10 dB
  - LWM zero-shot (地面FM直接用): NMSE ≈ -5 dB (因為地面和衛星差太多)
```

**任務 2: Beam Prediction（波束預測）— 直接延伸 MetaBeam (ICC'26)**
```
Input: 過去的通道量測 + 衛星軌道資訊
Output: 最佳 beam index

預期: SatFM 可以替代 MetaBeam 中的 meta-learning module
    並且在 unseen orbital configuration 上有更好的 zero-shot generalization
```

**任務 3: Handover Prediction（換手預測）**
```
Input: 時序通道品質 + 衛星仰角變化
Output: 最佳換手時機和目標衛星

這個任務直接連結到老師多位學生的 LEO 衛星工作
```

### 2.6 為什麼適合博士生？

1. **系統性工作**：從數據生成、模型設計、預訓練到多個下游任務，足夠撐一整個博士論文
2. **論文產出多**：
   - Paper 1: SatFM 本身（IEEE JSAC / TWC）
   - Paper 2: SatFM for beam management（延伸 MetaBeam, ICC/GLOBECOM）
   - Paper 3: Tiny-SatFM for on-board inference（IoT Journal / TVT）
   - Paper 4: Transfer learning 地面→衛星 FM（IEEE WCL / TMLCN）
3. **全新領域**：沒有人做過，你可以定義這個方向

---
---

# 方向三：GNN-MADRL for Joint LEO Resource Management
# （圖神經網路 + 多代理強化學習 統一LEO資源管理）

---

## 一、背景

### 1.1 LEO 巨型星座的規模問題

SpaceX Starlink 目前有 **6,000+** 顆衛星在軌，最終計畫 **42,000** 顆。每顆衛星要同時處理：

- **路由（Routing）：** 資料要從哪條 inter-satellite link (ISL) 轉發？
- **快取（Caching）：** 哪些熱門內容要預先存在衛星上？
- **任務卸載（Offloading）：** 用戶的計算任務要在衛星上算還是轉到地面？

**問題：** 這三個問題傳統上分開解決。但它們是高度耦合的：

```
例子：用戶 A 在太平洋上空，要存取一個影片

路由問題: 從哪顆地面站經哪條ISL路徑傳到衛星 S1?
快取問題: 如果衛星 S1 已經有這個影片的 cache → 不需要路由到地面站
卸載問題: 如果 S1 正在幫另一個用戶做邊緣計算 → S1 很忙 → 應該轉到 S2

三者互相影響：
- 快取決策影響路由（有 cache 就不用跨衛星傳）
- 路由影響卸載（路由延遲大 → 不如在本地算）
- 卸載影響快取（計算用完的結果可以 cache 給下一個用戶）
```

### 1.2 什麼是 GNN？為什麼適合衛星網路？

**Graph Neural Network（圖神經網路）** 是專門處理**圖結構數據**的 neural network。

衛星星座天然就是一個 graph：
```
Node（節點）= 衛星
Edge（邊）= Inter-Satellite Link (ISL)

Starlink 的圖:
- ~6000 nodes
- 每個 node 有 4 個 ISL edges（2個 intra-plane, 2個 inter-plane）
- 圖的結構隨時間變化（因為衛星在移動）
```

**GNN 的核心操作：Message Passing（訊息傳遞）**

```
每個 node v 有一個 feature vector h_v（包含：位置、負載、快取狀態等）

一輪 message passing:
  1. 每個 node 收集鄰居的 features: m_v = AGG({h_u : u ∈ N(v)})
  2. 更新自己的 feature: h_v' = UPDATE(h_v, m_v)

經過 K 輪 message passing 後，每個 node 的 feature 包含了 K-hop 鄰域的資訊。
```

**數值例子：**
```
衛星 S1 的初始 feature:
  h_S1 = [position_x, position_y, position_z, traffic_load, cache_status, compute_load]
       = [6978.2, 0.0, 0.0, 0.73, [1,0,1,0,1], 0.45]
       （距離地心6978.2km, traffic load 73%, 快取了內容1,3,5, compute用了45%）

S1 的鄰居: S2, S3, S4, S5 (四個ISL連接的衛星)

1輪 message passing 後:
  m_S1 = MeanPool([h_S2, h_S3, h_S4, h_S5])
       = 鄰居的平均 traffic load, cache 狀態等
  h_S1' = MLP(concat(h_S1, m_S1))
       = S1 現在「知道」鄰居的狀況了

2輪後: S1 知道 2-hop 鄰居的狀況
3輪後: S1 知道 3-hop 鄰居的狀況

這讓每顆衛星能做出「考慮到周圍環境」的決策
```

### 1.3 什麼是 MADRL（Multi-Agent DRL）？

在單一 agent 的 DRL 中，一個 agent 觀察環境、採取行動、獲得獎勵。

在 MADRL 中，**每顆衛星都是一個 agent**：

```
Agent i (衛星 i):
  觀察: o_i = local observation (自己的狀態 + GNN 提供的鄰域資訊)
  行動: a_i = [路由決策, 快取決策, 卸載決策]
  獎勵: r_i = f(全局網路效能)

所有 agent 同時行動，環境的 next state 取決於所有 agent 的聯合行動
```

**CTDE（Centralized Training, Decentralized Execution）** 是主流範式：
```
訓練時: 有一個中央 critic 可以看到所有 agent 的狀態（centralized）
部署時: 每個 agent 只根據自己的 local observation 行動（decentralized）
```

### 1.4 老師實驗室的相關工作

| 論文 | 問題 | 方法 | 局限 |
|------|------|------|------|
| SpaceEdge (TWC'24) | 任務卸載 | DRL | 只做卸載，不考慮路由和快取 |
| ICMLCN'25 | 快取 | Multi-agent DRL | 只做快取，不考慮路由和卸載 |
| ToN'26 | 路由 | 演算法（非DL）| 不用學習，且只做路由 |
| IEEE Network'25 | Contact Graph模型 | 圖建模 | 只是模型，沒有學習 |
| WCNC'23 (莫初拓) | MEC卸載 | GCN + DRL | 地面場景，非衛星 |

**Gap: 沒有人把這三個問題統一在一個 GNN-MADRL framework 中。**

---

## 二、我們的想法：UniLEO

### 2.1 系統架構

```
                    LEO Mega-Constellation (e.g., Starlink, 4000+ sats)
                                    │
                                    ↓
                ┌───────────────────────────────────────┐
                │        Contact Graph                   │
                │   (from IEEE Network'25 paper)         │
                │                                        │
                │   每個時間段 t 有一張圖 G_t = (V, E_t)   │
                │   V = 所有衛星 + 地面站                  │
                │   E_t = 該時段可用的 ISL + 衛星-地面連結  │
                └───────────────────┬───────────────────┘
                                    │
                                    ↓
                ┌───────────────────────────────────────┐
                │     Temporal GNN Encoder               │
                │                                        │
                │  Input: node features + edge features   │
                │  + temporal features (軌道相位)          │
                │                                        │
                │  Process:                               │
                │  1. K-layer GraphSAGE/GAT               │
                │  2. Temporal attention across time       │
                │  3. Output: node embeddings z_i          │
                └───────────────────┬───────────────────┘
                                    │
                        ┌───────────┼───────────┐
                        ↓           ↓           ↓
                  ┌──────────┐┌──────────┐┌──────────┐
                  │ Routing  ││ Caching  ││Offloading│
                  │ Agent    ││ Agent    ││ Agent    │
                  │          ││          ││          │
                  │ π_route  ││ π_cache  ││ π_offload│
                  │ (z_i)    ││ (z_i)    ││ (z_i)    │
                  └────┬─────┘└────┬─────┘└────┬─────┘
                       ↓           ↓           ↓
                  Joint Action a_i = [route_decision, cache_decision, offload_decision]
```

### 2.2 具體數值例子

```
場景: Starlink Phase 1 (1584 衛星, 72 planes × 22 sats/plane)

每顆衛星 i 的 node feature (維度 = 32):
  h_i = [
    orbital_position (3),    # x, y, z in ECI frame
    velocity (3),            # vx, vy, vz
    traffic_load (1),        # 0~1, 當前負載
    compute_utilization (1), # 0~1, CPU/GPU 使用率
    cache_state (10),        # binary, 哪些內容已快取
    buffer_queue (1),        # 0~1, 排隊長度
    visible_ground_stations (3),  # 哪些地面站可見
    beam_config (4),         # 當前 beam 指向
    energy_level (1),        # 太陽能面板電量
    link_quality (4),        # 四條 ISL 的 SNR
    padding (1)
  ]

Edge feature (每條 ISL):
  e_{ij} = [distance (1), propagation_delay (1), available_bandwidth (1), SNR (1)]

GNN 設計:
  3-layer GraphSAGE with mean aggregation
  Hidden dim = 128
  每層後 ReLU + BatchNorm

  經過 3 層後，每顆衛星的 embedding z_i ∈ ℝ^128 包含了 3-hop 鄰域資訊
  對 Starlink 來說，3-hop 涵蓋了約 50-100 顆衛星的區域
```

### 2.3 三個 Agent 的具體決策

**Routing Agent:**
```
Input: z_i (衛星 i 的 GNN embedding) + packet info (destination, priority)
Output: next_hop ∈ {neighbor_1, neighbor_2, neighbor_3, neighbor_4}

例子:
  衛星 S42 收到一個 packet，目的地是地面站 GS-Tokyo
  GNN embedding 告訴 S42:
    - neighbor S43 (同軌道下一顆): 負載 30%, 距 Tokyo 更近
    - neighbor S41 (同軌道上一顆): 負載 90%, 距 Tokyo 更遠
    - neighbor S114 (相鄰軌道): 負載 50%, 但有 cross-seam ISL 問題
    - neighbor S108 (相鄰軌道): 負載 45%, 有通往 Tokyo 的好路線

  Agent 選: S43（低負載且方向正確）
```

**Caching Agent:**
```
Input: z_i + content popularity prediction
Output: cache_or_not ∈ {0, 1} for each content

例子:
  衛星 S42 飛越太平洋，即將進入日本上空
  GNN 告訴 S42:
    - 日本地區最近 1 小時最熱門內容: video_A (30%), video_B (25%), news_C (15%)
    - S42 的 cache 容量: 10 GB, 目前已用 7 GB
    - 鄰居 S43 已經有 video_A 的 cache

  Agent 決定:
    - 不 cache video_A（因為鄰居有了）
    - Cache video_B（自己做，避免所有用戶都要跨衛星取）
    - Cache news_C（小檔案，容量夠）
```

**Offloading Agent:**
```
Input: z_i + task info (computation_requirement, deadline)
Output: execute_locally 或 offload_to_neighbor_j 或 offload_to_ground

例子:
  用戶透過 S42 送來一個 AI inference 任務（20 GFLOPS, deadline 200ms）
  S42 的情況:
    - 本地計算: 150ms（來得及，但會占用 CPU）
    - Offload 到 S43: ISL delay 5ms + S43 計算 80ms + return 5ms = 90ms（更快）
    - Offload 到 ground: 衛星到地面 20ms + ground 計算 10ms + return 20ms = 50ms
      但目前沒有可見的地面站！

  Agent 決定: Offload 到 S43（快且可行）
```

### 2.4 MADRL 訓練

```
算法: QMIX 或 MAPPO (Multi-Agent PPO)

Global Reward:
  R = w₁ × avg_throughput + w₂ × (-avg_latency) + w₃ × cache_hit_rate + w₄ × (-energy)

其中 w₁=0.3, w₂=0.3, w₃=0.2, w₄=0.2（可調）

Training:
  - 環境: OpenSN 模擬 Starlink constellation
  - Traffic model: Poisson arrival, hot spots over populated areas
  - Episode length: 1 orbital period (~90 minutes simulated time)
  - 每個 episode 切成 5 分鐘的 decision intervals → 18 steps per episode
  - Training: 50,000 episodes
  - GPU: 4× A100 (GNN + MADRL 聯合訓練)
  - Estimated training time: 3-5 days

Baseline 比較:
  1. Shortest Path Routing + LRU Caching + Local Computing (傳統方法)
  2. Separate DRL for each (三個獨立 DRL, 不共享 GNN)
  3. SpaceEdge (只做 offloading)
  4. ICMLCN'25 (只做 caching)
  5. UniLEO (我們的聯合方法)

預期結果:
  - UniLEO vs. 傳統方法: throughput +25%, latency -30%
  - UniLEO vs. 獨立 DRL: throughput +10%, latency -15%（因為聯合優化的 synergy）
  - UniLEO vs. SpaceEdge (只 offloading): latency -20%（因為有 caching 減少不必要的傳輸）
```

### 2.5 為什麼適合博士生？

1. **直接站在老師的 legacy 上：** 統一實驗室 3-4 篇頂刊的工作，這是博士級的 contribution
2. **可擴展性強：**
   - Paper 1: GNN-MADRL for joint routing+caching (ICC/GLOBECOM)
   - Paper 2: 加入 offloading 的完整 framework (TWC/JSAC)
   - Paper 3: 加入 digital twin for online adaptation (IEEE Network/TMC)
   - Paper 4: 加入 foundation model for GNN pre-training (TMLCN/WCL)
3. **計算需求大：** 4000+ node 的 GNN + MADRL 需要大量 GPU，這是你的優勢
4. **和實驗室其他學生互補：** 你做 framework，他們做 individual problems，可以合作

---
---

# 三個方向的比較與建議

## 博士生策略建議

| 面向 | 方向一 SplitDiffSem | 方向二 SatFM | 方向三 UniLEO |
|------|---------------------|-------------|--------------|
| 第一篇paper速度 | ★★★★★ (6-9月) | ★★★ (9-12月) | ★★★★ (8-10月) |
| 論文總產出量 | ★★★ (2-3篇) | ★★★★★ (4-5篇) | ★★★★ (3-4篇) |
| 博士論文完整度 | ★★★ | ★★★★★ | ★★★★ |
| 與實驗室衝突風險 | ⚠️ 中 | ✅ 零 | ⚠️ 低 |
| GPU活用度 | ★★★★ | ★★★★★ | ★★★★★ |
| 開創新方向能力 | ★★★★ | ★★★★★ | ★★★ |
| 工業界價值 | ★★★★ | ★★★★★ | ★★★★★ |

## 我的建議排序（針對博士生）

**首推：方向二 SatFM**
- 理由：完全不與實驗室任何人衝突、論文產出最多、可以定義一個新方向、GPU活用度最高
- 但缺點是需要 9-12 個月才能出第一篇

**次推：方向三 UniLEO**
- 理由：與老師最核心的 LEO 方向高度契合、統一框架是博士級 contribution
- 與碩士生是互補關係（你做 framework，他們做 single problem）

**第三：方向一 SplitDiffSem**
- 理由：出 paper 最快、新穎度極高
- 但需要先確認不與許欣芸和 Jason 衝突

**最佳組合策略：**
```
Year 1: 方向二 SatFM（建立數據 pipeline + 預訓練，同時出一篇 conference paper）
Year 2: SatFM 的多個下游任務（2-3 篇 journal papers）
Year 3: 結合方向三（SatFM + GNN for LEO → "FM-enhanced Joint LEO Management"）
Year 4: 博士論文整合

或者:

Year 1: 方向一 SplitDiffSem（快速出一篇 conference paper 建立信心）
Year 1-2: 方向二 SatFM（主線）
Year 3-4: 結合方向一+二（"SatFM for Split Diffusion SemCom over LEO"）
```
