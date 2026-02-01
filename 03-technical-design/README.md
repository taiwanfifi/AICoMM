# Technical Design

本目錄包含技術實現的詳細設計。

## 核心技術
Attention-Based Filtering: 使用 Attention 機制決定哪些 semantic token 值得傳輸。

## 待創建的文件
- [ ] `attention-filtering.md`: 基於 DeepSeek DSA 的設計（從 t6.md 重構）
- [ ] `token-representation.md`: Token 的編碼與解碼
- [ ] `control-plane.md`: Control Plane 協定
- [ ] `data-plane.md`: Data Plane 協定
- [ ] `implementation-notes.md`: 實現細節

## 參考資料
- `t6-original-reference.md`: 原始的 t6.md（需要重構）

## 下一步
1. 重構 t6.md，強調通訊角度而非 AI 角度
2. 設計具體的協定流程
3. 準備 pseudocode
