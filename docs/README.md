# docs/ — 專案文件與內部筆記

## 結構

```
docs/
├── PROJECT_STATUS.md       # 專案整體狀態、投稿計畫、創新定位
├── advisor-notes/          # 給老師的信件草稿、Scout 方法說明
│   ├── mail.md
│   ├── email_draft.md
│   └── mail_thoughts.md
├── notes/                  # 研究筆記、想法、規劃
│   ├── Q.md               # 核心研究問題深入 Q&A
│   ├── new_concepts.md     # 新研究概念 (Doc-to-LoRA, DualPath)
│   ├── p8想法補充.md        # Diffusion in 6G PHY 方向分析
│   ├── paper-ideas-7x-zh.md # 7 個未來論文規劃
│   ├── progress_0301.md    # 3/1 進度追蹤
│   └── start.md            # 重構指示（本次整理的起點）
├── reference/              # 工具與操作參考手冊
│   ├── REF_reviewer.md     # AI reviewer 工作流程
│   ├── REF_vastai_cli.md   # vast.ai GPU server 操作
│   ├── REF_skills.md       # Claude Code skills
│   ├── REF_remote_ops.md   # 遠端操作
│   ├── REF_llm_api.md      # LLM API 參考
│   └── REF_explainer.md    # 論文寫作風格指南
└── export_project.py       # 專案匯出工具
```

## 最重要的文件

- **PROJECT_STATUS.md** — 一頁了解全部：狀態、創新、競爭者、投稿計畫
- **notes/Q.md** — 深入分析 Scout vs prompt-only baseline 的根本問題
