# Core Framework: Semantic State Communication

本目錄包含本研究的核心理論框架。

## 核心概念
Semantic State Communication (SSC): 通訊的目的是同步 semantic state，而非傳輸 data。

## 文件清單

| 文件 | 內容 | 狀態 |
|------|------|------|
| `semantic-state-sync.md` | SSC 框架：Drift Theorem, Reset Policy, 設計準則 | ✅ 完成（含理論修復） |
| `semantic-token-definition.md` | Semantic Token 定義、Latent/Structured 雙模式 | ✅ 完成 |
| `architecture-overview.md` | STL 統一架構、Control/Data/Management Plane | ✅ 完成 |
| `t3-original-reference.md` | 原始 t3.md（僅供歷史參考，不再修改） | 📁 Archive |

## 未創建但已被其他文件涵蓋的內容

- `communication-paradigm.md` → 內容已整合至 `semantic-state-sync.md` 和 `../01-problem-formulation/contributions.md`
- `protocol-layers.md` → 內容已整合至 `architecture-overview.md` Section 5

## 建議閱讀順序
1. `semantic-token-definition.md`（先懂 token 是什麼）
2. `semantic-state-sync.md`（再懂怎麼同步）
3. `architecture-overview.md`（最後看整體架構）
