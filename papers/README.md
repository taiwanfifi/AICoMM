# Papers

## 論文一覽

| 論文 | 路徑 | 頁數 | 格式 | 目標 | 狀態 |
|------|------|------|------|------|------|
| Paper A | `paper-A/main.tex` | 7 | IEEE Conference | INFOCOM 2027 | ✅ 完成 |
| Paper B | `paper-B/main.tex` | 7 | IEEE Conference | ICC / JSAC | ✅ 完成 |
| JSAC | `jsac/main.tex` | 15 | IEEE JSAC | IEEE JSAC | 🔄 審稿迭代 (R4) |
| MLSys | `mlsys/main.tex` | — | MLSys 2025 | MLSys | ✅ 排版完成 |

## Paper A: Task-Aware KV-Cache Compression

Q2C selection + mixed-precision quantization，7 個模型、4 個任務的系統性評估。

**核心賣點**：Q2C 比 SnapKV 好 29-47%；INT8 對所有模型無損；INT4 脆弱性是 model-specific。

## Paper B: Scout Protocol

跨模型 attention 對齊 + 自適應傳輸協議。

**核心賣點**：只傳位置索引 (336 bytes vs 9.7 MB)，反而讓大模型品質提升 10.2%。

## JSAC Merged Paper

合併 Paper A + B，加上跨架構實驗、完整協議設計、理論分析。

**審稿分數**: Gemini R1:89 → R2:97 → Claude R3:72.3 → Claude R4:72.6

## 編譯

```bash
# 任一論文
cd papers/<paper-dir> && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 圖表生成
python papers/paper-A/generate_figures.py
python papers/jsac/generate_figures.py
python experiments/figures/generate_paper_b_figures.py
```

## 共用圖表

JSAC 和 MLSys 共用同一組圖表（`jsac/figures/` 的內容複製到 `mlsys/figures/`）。更新 JSAC 圖表後需同步。
