# Research Directions

本目錄包含 19 個研究方向的探索文件、文獻調查、和未來論文規劃。

## 結構

```
research/
├── INDEX.md                 # ⭐ 所有方向的狀態總覽（先看這個）
├── 01-19_*.md               # 各研究方向詳細文件
├── survey/                  # 文獻調查
│   ├── Survey_Communications_AI_Research.md
│   ├── HOT_RESEARCH_TOPICS_AI_COMMUNICATIONS_2025_2026.md
│   └── FINAL_RESEARCH_REPORT.md
├── future-ideas/            # 未來論文規劃
│   ├── SIX_IDEAS_FROM_TOP_VENUES_zh.md
│   └── IDEAS_EXPLAINED_zh.md
├── BATCH_EXPERIMENT_LOG.md  # 完整 batch 實驗紀錄（歷史參考）
└── PROGRESS_REPORT.md       # GPU server 設置紀錄（歷史參考）
```

## 快速導覽

**已驗證寫入論文** (6 個)：Topics 01, 02, 06, 11, 16, 17
**未來方向** (8 個)：Topics 03-05, 09-10, 13-15
**已否定** (1 個)：Topic 18 (zeroed positions — generation path artifact)
**參考/部分納入** (4 個)：Topics 07, 08, 12, 19

→ 詳細狀態見 [INDEX.md](INDEX.md)

## 研究方法論

```
假說 → 實驗設計 → GPU 執行 → 結果分析 → 驗證/否定 → 寫入論文或歸檔
```

所有實驗都遵循：固定種子 (seed=42)、逐樣本記錄 (JSON)、統計顯著性檢驗 (paired t-test)。
