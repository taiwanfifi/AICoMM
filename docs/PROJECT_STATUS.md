# Project Status — KV-Cache Semantic Communication

**最後更新**: 2026-03-10
**作者**: 鄭維倫、廖婉君，台大電機系

---

## 一、整體狀態

| 項目 | 狀態 | 下一步 |
|------|------|--------|
| Paper A (壓縮) | ✅ 完成 (7頁) | 等投稿 INFOCOM 2027 (~Aug 2026) |
| Paper B (Scout 協議) | ✅ 完成 (7頁) | 等投稿 ICC / JSAC |
| JSAC 合併論文 | 🔄 審稿迭代中 (R4) | 需繼續到 10+ rounds |
| MLSys 版本 | ✅ 排版完成 | 同 JSAC 內容 |
| GPU 實驗 | ✅ 全部完成 (49 JSON) | GPU server 已銷毀 |
| 理論框架 | ✅ Information Bottleneck | — |
| 跨架構驗證 | ✅ Qwen + Llama + Gemma | — |

## 二、是否可以投稿？

### Paper A — 可以投
- 7 個模型、4 個任務的完整實驗
- Q2C 顯著優於 SnapKV 和 H2O
- 量化研究完整（INT8/INT4/Mixed）
- **建議投**: INFOCOM 2027 (deadline ~Aug 2026)

### Paper B — 可以投
- Scout 協議設計完整
- 跨架構驗證（3 個模型家族）
- 真實 5G trace 模擬
- **建議投**: ICC 2027 或 IEEE JSAC (letter)

### JSAC 合併 — 需要更多迭代
R4 審稿分數：Claude 72.6, Gemini 71, ChatGPT 60

**已修正的問題**:
1. ✅ Prompt-only baseline 加入 Table 2
2. ✅ 頻寬估計驗證實驗
3. ✅ Proposition 1 加入 bounded quality loss
4. ✅ Poisson arrival 多 agent 擴展
5. ✅ Fallback hierarchy 擴展到 7 modes

**仍需改進**:
1. ❌ CacheGen 直接品質對比（需 GPU）
2. ❌ Random-k baseline（正則化控制）
3. ❌ 7B→14B @4K+ tokens（需 GPU, OOM）
4. ❌ 論文分數仍在 60-72 範圍，需大幅改進

## 三、我們的創新在哪？

### 跟別人不一樣的地方

| 面向 | 現有工作 | 我們的貢獻 |
|------|---------|-----------|
| **傳輸單元** | 傳 bits/packets | 傳 KV-cache（語義狀態） |
| **壓縮決策** | 固定策略 | 用 attention 分數動態決定 (Q2C) |
| **跨模型** | 同模型複製 | 小→大模型 KV 共享，品質提升 10% |
| **協議設計** | 無 / 固定模式 | 7 種 fallback 自適應 |
| **理論基礎** | 經驗性 | Information Bottleneck + Rate-Distortion |

### Competitors（主要競爭者）

| 工作 | 發表 | 做了什麼 | 我們的區別 |
|------|------|---------|-----------|
| CacheGen | SIGCOMM 2024 | KV-cache 壓縮傳輸 | 我們加了 task-aware selection + 協議設計 |
| SnapKV | NeurIPS 2024 | Attention-based selection | 我們的 Q2C 在 25% retention 勝 29-47% |
| H2O | NeurIPS 2023 | Heavy-hitter selection | 我們在所有模型上都大幅勝出 |
| CacheBlend | EuroSys 2025 | 接收端品質修復 | 互補：他們修復，我們設計傳輸 |
| KVCOMM | NeurIPS 2025 | 跨 context KV 重用 | 我們是跨模型，不只跨 context |
| C2C (清華) | 2025 | KV-cache 用於 agent 通訊 | 我們有協議設計 + 頻寬自適應 |

### 學界目前做到哪？

- **KV-cache 壓縮**: SnapKV, H2O, CacheGen 都是單模型壓縮，沒有跨模型考量
- **Edge-Cloud LLM**: 大部分工作假設頻寬充足，專注 model partition (DistServe, Splitwise)
- **語義通訊**: 主流仍在做 image/video 重建，沒有針對 LLM 推理狀態
- **我們填的空白**: 第一個把 KV-cache 當作跨模型語義傳輸單元，並設計 task-aware 自適應協議

## 四、關鍵數據摘要

### Q2C Token Selection (SQuAD v2, 25% retention)

| 模型 | Q2C (ours) | SnapKV | H2O | 勝出幅度 |
|------|-----------|--------|-----|---------|
| Qwen-7B | **0.428** | 0.292 | 0.205 | +47% vs SnapKV |
| Qwen-14B | **0.360** | 0.279 | 0.160 | +29% vs SnapKV |
| Mistral-7B | **0.294** | 0.205 | 0.129 | +43% vs SnapKV |

### Cross-Model Attention Overlap (@75% retention)

| 配對 | 家族 | Overlap | 統計顯著性 |
|------|------|---------|-----------|
| 3B→7B | Qwen 2.5 | 82% | ✅ |
| 7B→14B | Qwen 2.5 | 83% | ✅ |
| 3B→8B | Llama 3 | 92% | marginal (p=0.06) |
| 2B→9B | Gemma 2 | 86% | NS (p=0.36) |

### 傳輸模式對比

| 模式 | 大小 | 品質 | 壓縮倍數 |
|------|------|------|---------|
| Full BF16 | 9.7 MB | 100% | 1x |
| INT8 | 4.7 MB | 99.6% | 2x |
| Mixed-INT4 | 2.6 MB | 93.6% | 3.7x |
| Scout | 336 B | 81-110% | **28,800x** |

## 五、投稿時間表

```
2026-03 ─── JSAC: 繼續審稿迭代 (R5, R6, ...)
2026-04 ─── JSAC: 達到投稿品質 (目標分數 > 85)
2026-06 ─── Paper A: 準備 INFOCOM 投稿
2026-08 ─── Paper A: INFOCOM 2027 deadline
2026-08 ─── Paper B: ICC / JSAC 投稿
```
