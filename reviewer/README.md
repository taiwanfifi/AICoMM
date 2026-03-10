# Review History — JSAC Merged Paper

本目錄紀錄 JSAC 合併論文的所有審稿迭代過程。我們使用 Claude、Gemini、ChatGPT 三個 AI reviewer 做交叉審稿。

## 審稿流程概述

```
Round 0 (初始)  → 內部自我檢查，列出潛在問題
Round 1 (Gemini) → 第一輪正式審稿，89/100
Round 2 (多源)   → Gemini 第二輪 97/100 + 早期 review
Round 3 (交叉)   → Claude 72.3 + Gemini 對 Paper A/B 分別審
Round 4 (三方)   → Claude 72.6 + ChatGPT 60 + Gemini 71
```

## 分數趨勢

| Round | Claude | Gemini | ChatGPT | 主要問題 |
|-------|--------|--------|---------|---------|
| R1 | — | 89 | — | 初版，基礎架構 |
| R2 | — | 97 | — | 大幅改進後 |
| R3 | 72.3 | — | — | 實驗方法論、baseline 不足 |
| R4 | 72.6 | 71 | 60 | 仍需 CacheGen 對比、Random-k baseline |

> 注意：R2 的 97 分是 Gemini 「fresh eyes」模式，可能偏樂觀。R3-R4 的 Claude/ChatGPT 較嚴格。

## 目錄結構

```
reviewer/
├── round0-initial/          # 內部自我審查（6 個問題清單）
├── round1/                  # Gemini 第一輪
├── round2/                  # 早期多源審稿
├── round3/                  # Claude R3 + Gemini 對各子論文
│   ├── REVIEW_*_claude_JSAC_round3.md
│   ├── REVIEW_*_gemini_*.md (多份)
│   └── REVIEW_*_給老師摘要.md (中文)
├── round4/                  # 三方交叉審稿
│   ├── REVIEW_*_chatgpt_JSAC_round4.md
│   ├── REVIEW_*_claude_JSAC_round4.md
│   └── REVIEW_*_gemini_JSAC_round4.md
├── prompts/                 # 審稿 prompt 模板
├── analysis/                # 跨輪分析、問題追蹤
│   ├── BLIND_SPOTS_*.md
│   ├── REVIEW_STATUS.md
│   └── REVIEW_REPORT.md
└── run_r4_reviews.py        # 自動化審稿腳本
```

## R4 後仍需改進的問題

1. **CacheGen 直接品質對比**（需 GPU 實驗）
2. **Random-k baseline**（正則化控制實驗）
3. **7B→14B @4K+ tokens**（需 GPU，OOM workaround）
4. 繼續迭代至 10+ rounds
