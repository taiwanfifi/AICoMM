# 相關論文分析：D2L & DualPath vs. Scout

## 1. Doc-to-LoRA (D2L) — Sakana AI
**arXiv:2602.15902 (2026/02)**

### 核心想法
傳統做法：LLM 要回答關於某份文件的問題，就把整份文件塞進 context window → 產生巨大 KV-cache。

D2L 換個思路：**訓練一個 hypernetwork，讀一次文件就產出一組 LoRA adapter**。之後 LLM 掛上這個 LoRA，就像「記住」了那份文件，不需要任何 context tokens。

### 技術細節

```
傳統方式：
  [文件 tokens] + [問題] → LLM → 答案
  代價：128K tokens = 12+ GB KV-cache

D2L 方式：
  [文件 tokens] → Hypernetwork → LoRA weights (~50 MB)
  LLM + LoRA → [問題] → 答案（context window 裡完全沒有文件）
```

- **Hypernetwork**：Perceiver 架構，~309M 參數，8 層 cross-attention
- **輸入**：base LLM 處理文件時每層的 token activations
- **輸出**：Rank-8 LoRA matrices（只改 MLP layers）
- **長文處理**：把文件切 chunks，每段各自產 LoRA，再沿 rank 維度拼接

### 結果

| 指標 | D2L | 傳統微調 | Full Context |
|------|-----|---------|-------------|
| 知識內化時間 | 0.55 秒 | 72+ 秒 | N/A |
| 推理記憶體 | <50 MB | ~1 GB | 12+ GB |
| SQuAD 準確率 | 83.5% of full-context | ~90% | 100% |
| 32K NIAH | 近乎完美 | — | 近乎完美 |

### 限制
- Meta-training hypernetwork 非常貴（多 GPU 跑數天到數週）
- 換目標模型就要重新訓練
- 品質只有 full-context 的 83.5%
- 只測了 Gemma-2B，沒有跨模型

---

## 2. DualPath — DeepSeek
**arXiv:2602.21548 (2026/02)**

### 核心問題
DeepSeek 在自家 production 發現：**Agent 場景的瓶頸不是 GPU，是 storage I/O**。

為什麼？Agent 跟環境互動數十到數百輪，每輪只新增少量 token，但累積 context 超長。98.7% 的 KV-cache 可以從之前的 cache 直接讀取（cache hit），不需要重算。所以 GPU 大部分時間不是在算，而是在等 KV-cache 從遠端 SSD 讀回來。

### PD 分離架構的問題

```
目前架構：
  Prefill Engine (PE): 負責處理 prompt + 讀取 KV-cache from SSD
  Decode Engine (DE): 負責 autoregressive generation

問題：
  PE 的 storage NIC (~50 GB/s) 被 KV-cache 讀取打滿
  DE 的 storage NIC 完全閒置
  → PE 成為瓶頸
```

### DualPath 解法

```
原本 (Single Path):
  SSD → PE storage NIC → PE GPU

DualPath:
  Path 1: SSD → PE storage NIC → PE GPU        (跟以前一樣)
  Path 2: SSD → DE storage NIC → DE DRAM → InfiniBand RDMA → PE GPU  (新增！)
```

把 DE 閒置的 storage NIC 利用起來，透過 InfiniBand 高速網路把讀到的 KV-cache 轉給 PE。

### 關鍵技術
- **Layerwise Prefill**：不一次載入整個 KV-cache（太大放不下 HBM），而是一層一層載入、運算、丟掉
- **Traffic Isolation**：用 InfiniBand Virtual Lanes 隔離 KV-cache 流量和模型推論流量，確保不互相干擾
- **Global Scheduler**：根據各節點的 disk queue length 分配讀取任務

### 結果

| 場景 | 加速 |
|------|------|
| 離線推論 | 最高 1.87× |
| 線上服務 | 平均 1.96× |
| 規模 | 1,152 GPU 近線性擴展 |

### 硬體需求
- NVIDIA Hopper GPUs
- InfiniBand (每節點 8 × 400 Gbps)
- 3FS 分散式 SSD 儲存
- **純資料中心方案，無法用在邊緣**

---

## 3. 跟 Scout 的比較

### 三種方法解決的問題層次不同

```
Scout:  「我根本不傳 KV-cache data，只傳哪些 position 重要」 → 28,800× 壓縮
D2L:    「我把文件知識壓進 weights，不需要 KV-cache」          → 消除 KV-cache
DualPath:「KV-cache 還是要傳，但我用兩條路一起搬，搬更快」    → 2× I/O 加速
```

### 詳細比較表

| 維度 | Scout (我們) | D2L (Sakana) | DualPath (DeepSeek) |
|------|-------------|-------------|---------------------|
| **核心方法** | 傳 position indices | 文件→LoRA weights | 雙路徑 I/O |
| **傳輸大小** | 336 bytes | ~50 MB adapter | 全量 KV-cache (GB) |
| **壓縮倍率** | 28,800× | ~240× (vs KV) | 0× (不壓縮) |
| **跨模型** | 有 (小→大) | 無 | 無 |
| **需要訓練** | 不需要 | 需要 (hypernetwork) | 不需要 |
| **品質** | 99.4% of server's own | 83.5% of full-context | 100% (不壓縮) |
| **場景** | 邊緣/WAN (10-200 Mbps) | 單機 long-context | 資料中心 (InfiniBand) |
| **解決的瓶頸** | 網路頻寬 | GPU 記憶體 | Storage NIC 頻寬 |
| **可組合性** | 可疊在 DualPath 上 | 獨立方案 | 可搭配 Scout |

### 關鍵差異

1. **Scout 是唯一做跨模型 attention transfer 的**
   - D2L：single model, hypernetwork 綁定特定 base model
   - DualPath：single model, 搬的是同一個模型的 KV-cache
   - Scout：小模型的 attention pattern 指導大模型的 position selection

2. **Scout 是唯一 training-free 的語意壓縮**
   - D2L 需要訓練 hypernetwork（數天 multi-GPU）
   - DualPath 不做壓縮
   - Scout 零訓練，直接用 attention scores

3. **DualPath 驗證了 KV-cache I/O 是 production bottleneck**
   - DeepSeek 在真實 agent workload 觀察到 98.7% cache hit rate
   - GPU utilization 很低，因為大部分時間在等 KV-cache 從 SSD 讀回來
   - 這強化了 Scout 的 motivation：如果連資料中心都被卡住，邊緣場景更嚴重

### 互補關係

```
資料中心場景：
  DualPath (搬更快) + Scout (傳更少) = 最佳組合
  → KV-cache 讀取量從 GB 降到 bytes，DualPath 的雙路徑甚至不需要啟動

邊緣場景：
  Scout 獨立運作，DualPath 不適用（沒有 InfiniBand）
  D2L 理論上可用但需要 per-model hypernetwork 訓練

Long-context 單機場景：
  D2L 最適合（直接消除 KV-cache）
  Scout 和 DualPath 不適用（沒有網路傳輸需求）
```

### 我們在 MLSys paper 怎麼引用

**DualPath** → Related Work "Disaggregated serving" 段落：
> DualPath identifies that KV-cache storage I/O saturates prefill-engine NICs in agentic workloads (98.7% cache hit rate) and introduces a second read path through idle decode-engine NICs via RDMA, achieving 1.9× throughput. DualPath's approach—moving bytes faster through dual I/O paths—is complementary to Scout's—transmitting fewer bytes via semantic indices; the two could be composed.

**D2L** → Related Work 新增 "Context distillation into weights" 段落：
> Doc-to-LoRA trains a hypernetwork to convert documents into LoRA adapters in a single forward pass, eliminating KV-cache at inference (<50 MB vs. 12 GB). This weight-space compression is an alternative to Scout's position-space transfer, but requires expensive meta-training, operates on a single model (no cross-model transfer), and achieves 83.5% of full-context quality versus Scout's 99.4%.
