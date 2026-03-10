# 給廖老師的信 — 草稿（挑選版本後合併）

---

## 信件主旨（建議）

AI × 通訊近期頂刊趨勢整理 — 六大方向摘要

---

## 正文

老師好，

我最近花了一些時間整理 AI 與通訊交叉領域在近兩年（2024–2026）頂刊/頂會的研究動態。附件是整理後的摘要，以下先把我看到的重點跟老師報告。

---

### ===== POINT 1：告訴老師 insights（二選一）=====

#### 【版本 A：直接講結論型 — 簡潔有力，適合老師很忙的情境】

我掃了 IEEE JSAC、TWC、TCOM、COMST、ACM SIGCOMM 2025、NeurIPS AI4NextG Workshop 等場域約 300 篇論文後，觀察到一個收斂中的趨勢：**通訊領域的 AI 正在從「用 DNN 取代某個 block」（例如 CNN 做 channel estimation）轉向系統級的 paradigm shift。** 具體來說有六個方向特別值得注意：

1. **Semantic Communication** — COMST 2024 連刊三篇 survey，DeepJSCC 已演化到 MIMO + 數位化版本（JSAC 2025）
2. **Diffusion Model for Physical Layer** — TWC 2025 用 score-based diffusion 做 channel estimation，超越完美 CSI 下的 MMSE
3. **World Model for 6G** — IEEE TNSE 開了 special issue（deadline 2026-03-15），NeurIPS 2025 有 Dual-Mind World Model
4. **Causal AI for Wireless** — 從 correlation learning 升級到 intervention 和 counterfactual reasoning，IEEE VTM 2024 有 vision paper
5. **Agentic AI** — Meta 的 Confucius 系統已在生產網路跑兩年（ACM SIGCOMM 2025），IEEE TNSE 2026 開了至少四個相關 special issue
6. **Foundation Model for Wireless** — Arizona State 的 Large Wireless Model 做到 zero-shot beam prediction 82%；但 LEO 衛星目前無人做過

這六個方向不是獨立的，它們指向同一件事：通訊系統正在從「被動傳輸」走向「端到端可學習、可推理、可自主決策的智慧系統」。附件中有每個方向的技術原理和代表論文，供老師參考。

---

#### 【版本 B：連結實驗室型 — 把 insights 跟老師/學生的研究做對照】

我掃了近兩年（2024–2026）IEEE JSAC、TWC、COMST、ACM SIGCOMM 等場域約 300 篇 AI×通訊論文，想跟老師分享幾個觀察：

**大趨勢：AI 在通訊中的角色正在從「輔助工具」變成「原生組件」。**

具體來說，我看到六個特別活躍的方向，其中幾個跟我們實驗室正在做的事有交集：

| 方向 | 頂刊訊號 | 與實驗室的關係 |
|------|---------|-------------|
| Semantic Comm + DeepJSCC | COMST 2024 三篇 survey、JSAC 2025 | Jason 的 Diffusion SemCom (WCNC'26) |
| Diffusion for PHY | TWC 2025 超越 MMSE | Jason 的 diffusion 方向可擴展 |
| World Model for 6G | TNSE special issue 2026 | 與 digital twin 相關但方法全新 |
| Causal AI | VTM 2024 vision paper | 全新方向，但可接 beam management |
| Agentic AI | SIGCOMM 2025 (Meta)、TNSE 4 個 SI | 佳宏的 multi-agent 可擴展 |
| Foundation Model for Wireless | LWM (Arizona State) | LEO 版本尚未有人做 — 子恒/紹寶可考慮 |

也就是說，我們實驗室已經有人在做其中 2-3 個方向的子問題，但有幾個全新的方向（Causal AI、World Model、Agentic AI）可能值得留意，因為 IEEE editorial board 正在積極開 special issue。

附件有每個方向的從零講解和代表論文，老師有空可以翻翻。

---

### ===== POINT 2：說明用 AI 做 survey（三選一）=====

#### 【版本 2A：坦率直說型 — 強調工具效率】

補充說明一下方法：這份整理是我用 AI 工具（主要是 Claude 和 Gemini）系統性搜尋和彙整的。具體做法是：給定頂刊清單和關鍵字，讓 AI 搜尋近兩年的論文、提取重點、交叉比對不同來源。我再逐一驗證論文的真實性（確認 DOI、期刊、年份）。現在的 AI 工具在做這類結構化文獻搜尋時已經相當可靠，搭配人工驗證可以在幾天內完成過去可能需要幾週的 survey 工作。

---

#### 【版本 2B：低調帶過型 — 重點放在結果而非工具】

這份整理涵蓋了約 300 篇頂刊論文，我用了一些自動化工具來加速文獻搜尋和分類，再人工篩選和驗證每一篇的出處。附件中列出的論文都確認過 DOI 和發表資訊。

---

#### 【版本 2C：展示可能性型 — 暗示 AI 可以成為研究加速器】

附帶一提，這份 survey 的初步搜尋和分類是借助 AI 工具完成的——我發現現在的 LLM（如 Claude、Gemini）在做結構化文獻調查時已經不太會「編造」論文了，只要搭配 web search 和人工驗證 DOI，效率比傳統的手動搜尋快很多。我自己在這個過程中也學到不少，或許這個工作方式本身對實驗室其他同學做文獻調查也有參考價值。

---

### ===== POINT 3：科普式摘要（直接放在信中）=====

最後簡單分享幾個我讀完之後印象比較深的技術觀察：

- **Shannon 理論的邊界正在被重新定義：** 傳統通訊只管 bit 對不對，不管有沒有用。Semantic communication 挑戰的就是這件事——只傳「有意義的東西」。DeepJSCC 用端到端神經網路取代分層的 source coding + channel coding，最大好處是沒有 cliff effect（SNR 降低時品質是漸進退化，不會突然崩潰）。

- **Diffusion model 在通訊的應用比想像中自然：** Channel estimation 的本質是從有噪的 y = Hx + n 恢復乾淨的 H，這和 diffusion 的去噪在數學上是同構的。TWC 2025 的結果顯示 diffusion 做 channel estimation 可以超越完美 CSI 的 MMSE estimator，因為它學到了 channel 的先驗分佈。

- **Agentic AI 已經不是概念了：** Meta 的 Confucius 是多個 AI agent 組成的系統（Intent Parser → Planner → Executor → Validator），已在 Meta 的生產網路上跑了兩年以上，管理 60+ 個應用場景。這是 ACM SIGCOMM 2025 的論文。

- **IEEE editorial board 的風向很明確：** TNSE 2026 同時開了 World Model、Agentic AI、Causal AI 的 special issue。JSAC 持續在 semantic comm 和 ISAC 開題。這些不是邊緣方向，而是 community 正在 converge 的方向。

---

### 結尾

以上是我目前的整理，附件中有更完整的技術說明和論文清單供老師參考。如果老師覺得哪個方向值得進一步討論，我可以再深入調查。

William

---

## 附件清單
1. **六大頂刊級研究方向 — 完整講解**（SIX_IDEAS_FROM_TOP_VENUES_zh.pdf）— 主要附件，6個方向的從零講解
2. **AI×通訊頂刊論文總覽**（Survey_Communications_AI_Research.pdf）— 參考附件，54篇論文清單與分類
