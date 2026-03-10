# Research Directions — Status Index

> 19 個研究方向的狀態總覽。標記 ✅ 表示已驗證並寫入論文，💡 表示未來方向，❌ 表示已否定。

## Tier 1: 已驗證，已寫入論文

| # | 方向 | 狀態 | 寫入 | 核心發現 |
|---|------|------|------|---------|
| 01 | KV-Cache Compression Protocol | ✅ PAPER-READY | Paper A | Q2C >> SnapKV >> H2O; 96% quality @ 18.75% BW |
| 02 | Cross-Model KV Transfer | ✅ NEAR-READY | Paper B | 82-92% overlap; scout improves cloud quality |
| 06 | Quantization vs SVD | ✅ CONFIRMED | Paper A | INT8 lossless; SVD cliff at rank-32 |
| 11 | Layer-Heterogeneous Compression | ✅ CONFIRMED | Paper A | Layer 0 bottleneck (model-specific); mixed-precision recovers |
| 16 | Key-Value Asymmetry | ✅ DISCOVERED | Paper B | Keys cos=0.9997, Values cos=0.222 |
| 17 | Quantization is Free | ✅ CROSS-VALIDATED | Paper A | INT4 fragility is model-specific, not architectural |

## Tier 2: 部分驗證 / 已納入論文框架

| # | 方向 | 狀態 | 寫入 | 說明 |
|---|------|------|------|------|
| 07 | Task Attention Patterns | 部分 | JSAC | 不同任務需要不同壓縮策略 |
| 08 | KV-Cache as Semantic State | 理論 | JSAC §II | Information Bottleneck 框架 |
| 12 | Communication Cost Model | 部分 | JSAC §IV, §VI | 頻寬-品質 tradeoff 決策模型 |
| 19 | Related Work Landscape | 參考 | JSAC §VIII | 競爭對手文獻調查 |

## Tier 3: 未來方向（假說，待驗證）

| # | 方向 | 風險 | 潛在投稿 | 概述 |
|---|------|------|---------|------|
| 03 | Adaptive KV Streaming | 中 | INFOCOM | 即時頻寬條件下動態調整壓縮 |
| 04 | Importance-Aware Retransmission | 中 | ICC/Globecom | 用 attention 作為 ARQ 優先級 |
| 05 | Multi-Agent KV Sharing | 高 | ACL/NeurIPS | 多 agent 共享 base KV + task-specific 層 |
| 09 | Speculative KV Prefetch | 中 | MobiCom | 多輪對話中預取 KV-cache |
| 10 | Privacy-Preserving KV | 中 | S&P Workshop | 差分隱私 KV 壓縮 |
| 13 | VLM KV Compression | 高 | CVPR/ECCV | 視覺語言模型的 modality-aware 壓縮 |
| 14 | Knowledge Distillation via KV | 高 | NeurIPS/ICML | 大模型 KV 投射到小模型空間 |
| 15 | KV-Cache External Memory | 高 | AAAI | KV 片段作為 agent 外部記憶 |

## 已否定

| # | 方向 | 結論 |
|---|------|------|
| 18 | Zeroed Positions Improve Selection | ❌ DEBUNKED — mask_only == zero_mask; 是 generation path artifact |

## 也可以看

- `survey/` — 文獻調查
- `future-ideas/` — 未來論文規劃（中文）
- `PROGRESS_REPORT.md` — 詳細進度（較舊，2026-02-08）
