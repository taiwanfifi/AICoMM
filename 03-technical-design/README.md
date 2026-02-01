# Technical Design

本目錄包含技術實現的詳細設計。

## 核心技術
Attention-Based Filtering: 使用 Attention 機制決定哪些 semantic token 值得傳輸。

## 文件清單

| 文件 | 內容 | 狀態 |
|------|------|------|
| `attention-filtering.md` | Semantic Indexer（基於 DSA Lightning）、雙通道架構 | ✅ 完成 |
| `token-encoding.md` | Protobuf schema、量化、壓縮（Structured Mode） | ✅ 完成 |
| `state-integration.md` | Receiver 端整合：Anchor 對齊 + Neural Projector | ✅ 完成 |
| `kv-cache-alignment.md` | 異質 KV-Cache 維度對齊（512→4096） | ✅ 完成（含理論修復） |
| `t6-original-reference.md` | 原始 t6.md（僅供歷史參考，不再修改） | 📁 Archive |

## 未創建但已被其他文件涵蓋的內容

- `token-representation.md` → 內容已整合至 `token-encoding.md`
- `control-plane.md` → 內容已整合至 `../02-core-framework/architecture-overview.md` Section 3.1
- `data-plane.md` → 內容已整合至 `../02-core-framework/architecture-overview.md` Section 3.2
- `implementation-notes.md` → 內容已整合至 `../06-implementation/ssc-pipeline-spec.md`

## 建議閱讀順序
1. `attention-filtering.md`（Source 端：怎麼選 token）
2. `token-encoding.md`（中間：怎麼編碼傳輸）
3. `state-integration.md`（Receiver 端：怎麼整合）
4. `kv-cache-alignment.md`（異質模型：怎麼做維度對齊）
