
⏺ 好的，我看完了全部五份文件。以下是它們的關係和差異：

  ---
  五份文件的演化關係

  Survey (基底調查)
      │
      ├──→ HOT_RESEARCH_TOPICS (擴展)
      │
      └──→ FINAL_RESEARCH_REPORT (彙整+鎖定實驗室)
                │
                ├──→ IDEAS_EXPLAINED_zh (教學版 — 3個方向)
                │
                └──→ SIX_IDEAS_FROM_TOP_VENUES_zh (修正版 — 6個方向)

  逐份說明

  #: 1
  文件: Survey_Communications_AI_Research.md
  角色: 基底調查
  內容: 頂刊清單 + 18個方向 + 每方向2-3篇代表論文
  論文數: 54
  語言: EN
  ────────────────────────────────────────
  #: 2
  文件: HOT_RESEARCH_TOPICS_...md
  角色: 廣度擴展
  內容: 在Survey基礎上再挖出35個更新穎的方向（Tier 1/2/3），含Special
    Issue deadlines
  論文數: 35+
  語言: EN
  ────────────────────────────────────────
  #: 3
  文件: FINAL_RESEARCH_REPORT.md
  角色: 核心報告
  內容: 針對廖婉君教授實驗室，做10輪深度調查：教授背景→18方向→Top
    10排名→3個具體研究提案（SplitDiffSem, SatFM, UniLEO）→313篇論文清單
  論文數: ~313
  語言: 中
  ────────────────────────────────────────
  #: 4
  文件: IDEAS_EXPLAINED_zh.md
  角色: 教學講解（版本A）
  內容: 把 FINAL_RESEARCH_REPORT
    的前3名方向用「從零講起」的方式詳細解釋，含數值例子、公式、系統圖
  論文數: —
  語言: 中
  ────────────────────────────────────────
  #: 5
  文件: SIX_IDEAS_FROM_TOP_VENUES_zh.md
  角色: 教學講解（版本B）
  內容: 6個方向，刻意降低實驗室比重（3/6與lab零交集），加入 World
    Models、Causal AI、Neuromorphic Rx、Agentic AI 等來自頂會的全新方向
  論文數: —
  語言: 中

  誰是誰的迭代？

  - 4 (IDEAS_EXPLAINED) 是 3 (FINAL_REPORT) 的教學展開 —
  只講前3個方向，偏實驗室導向
  - 5 (SIX_IDEAS) 是 4 的修正版 — 你可能覺得太偏實驗室，所以重做了一版，
  6個方向中只有1個（SatFM）與lab有中度交集，其餘都是頂會新方向

  你應該怎麼讀？

  ┌─────────────────────────────────┬────────────────────┬──────────┐
  │            你的需求             │       看哪份       │  不用看  │
  ├─────────────────────────────────┼────────────────────┼──────────┤
  │ 想快速了解「AI+通訊有哪些大方向 │ 1 (Survey) 即可    │ 2        │
  │ 」                              │                    │          │
  ├─────────────────────────────────┼────────────────────┼──────────┤
  │ 想知道最新最前沿的方向+投稿dead │ 2 (HOT_TOPICS)     │ —        │
  │ line                            │                    │          │
  ├─────────────────────────────────┼────────────────────┼──────────┤
  │ 想完整了解實驗室適合做什麼+論文 │ 3 (FINAL_REPORT) — │ 1, 2     │
  │ 庫                              │  最重要的一份      │ (因為 3  │
  │                                 │                    │ 已包含)  │
  ├─────────────────────────────────┼────────────────────┼──────────┤
  │ 想從頭理解某個方向的原理/數值/  │ 5 (SIX_IDEAS) ← 最 │ 4 (被5取 │
  │ 公式                            │ 新版，涵蓋範圍最廣 │ 代)      │
  └─────────────────────────────────┴────────────────────┴──────────┘

  建議

  - 1 和 2 是 3 的前置材料，資訊已被 3 吸收，可歸檔不再看
  - 4 被 5 取代（5 = 4 的擴充修正版，方向從3個變6個，且降低lab bias）
  - 你真正需要讀的只有兩份：3 (FINAL_REPORT) + 5 (SIX_IDEAS)

✻ Cooked for 43s


⏺ 好，我直接寫出來給你。

  ---
  AI × 通訊：從 300+ 篇頂刊論文整理出的全貌筆記

  這份筆記是什麼、憑什麼可信

  這份筆記不是我自己憑空想的。它來自對過去兩年（2024–2026）IEEE
  JSAC、TWC、TCOM、COMST（IF 46.7，通訊領域第一名的 survey 期刊）、ACM
  SIGCOMM 2025、IEEE INFOCOM 2025 Best Paper、NeurIPS AI4NextG Workshop
  等頂級場域，超過 300
  篇論文的系統性掃描。我想做的事情很簡單：身為通訊人，AI
  這些工具到底在幹嘛？每個方向背後的道理是什麼？哪些是真的有人在頂刊發、
  有人在生產環境跑的？以下是我讀完之後看到的東西。

  ---
  一、通訊正在從「傳每一個 bit」變成「只傳有意義的東西」

  Shannon 在 1948 年告訴我們：只要 data rate 低於 channel capacity，就能
  做到幾乎零錯誤的傳輸。但他自己也說過，他的理論不管「意義」——他只管 bit
   對不對，不管 bit 有沒有用。

  Semantic Communication（語義通訊）就是挑戰這件事。想像你用手機拍了一張
   12 MB
  的照片要傳給對方做物體辨識，傳統做法是壓縮、通道編碼、傳過去、解碼、解
  壓縮、再丟進分類器。但如果對方只需要知道「這是一隻貓」，為什麼要傳 12
  MB？語義通訊的做法是：transmitter 端用一個神經網路把影像壓縮成「語義特
  徵」（比如物體的輪廓、紋理的統計量），只傳這些特徵，receiver
  端再用另一個神經網路還原或直接拿去做任務。IEEE COMST 2024
  年連續刊了三篇 semantic communication survey（Chaccour et al., Lu et
  al., Guo et al.），可見這個方向的熱度。

  在這個框架底下有一個關鍵技術叫 Deep Joint Source-Channel
  Coding（DeepJSCC）。傳統通訊系統是分層的：source coding（像
  JPEG）負責壓縮，channel coding（像 LDPC）負責抗噪，兩邊各自最佳化。但
  Shannon 1959 年就證明了 joint source-channel coding
  理論上更好，只是找不到實際的最佳解。DeepJSCC
  用深度學習來逼近這個解——整個 encoder-channel-decoder
  當成一個端到端的神經網路一起訓練。它最大的好處是不會出現傳統系統的
  cliff effect：傳統系統在 SNR 掉到某個門檻以下時影像直接崩壞（因為
  channel code 解不出來），但 DeepJSCC 是漸進式退化，SNR
  越低品質越差但不會突然斷崖。這個概念最早由 Bourtsoulatze et al. 在
  IEEE TCCN 2019 提出，到現在已經演化出 MIMO 版本（DeepJSCC-MIMO, IEEE
  TWC 2024）和數位化版本（D2-JSCC, IEEE JSAC 2025）。

  ---
  二、Diffusion Model 不只會畫圖，它正在改變 physical layer

  大家都知道 Stable Diffusion 和 DALL-E
  會生成圖片。它的原理是這樣的：先把一張乾淨的圖逐步加 Gaussian
  noise，加 T 步之後變成純雜訊（forward
  process）；然後訓練一個神經網路（通常是 U-Net）學會「從第 t
  步的雜訊預測第 t-1 步的稍乾淨版本」（reverse
  process）。推理的時候就從純雜訊開始，一步步去噪，最後生成一張圖。

  現在把這個想法放到通訊裡。channel estimation 的本質是什麼？你送
  pilot，經過通道加了 noise，收到 y = Hx + n，你要從有噪的 y 恢復乾淨的
  H。這和 diffusion 的去噪在數學上是同構的。Zhou et al. 在 IEEE TWC
  2025（vol. 24, no. 7）直接把 score-based diffusion model 用在 massive
  MIMO channel estimation 上，結果超越了完美統計資訊下的 MMSE
  estimator。為什麼？因為 diffusion model 在訓練過程中學到了 channel
  的先驗分佈（prior），相當於它「知道」真實的通道長什麼樣子，所以去噪的
  時候不只是做線性濾波，而是在整個通道分佈上做最佳的 posterior
  sampling。

  Diffusion 在語義通訊中也被用上了。DiffJSCC（arXiv:2404.17736，有開源
  code）把 pre-trained Stable Diffusion 接在 DeepJSCC 的 receiver
  端：先用傳統 DeepJSCC 做粗略重建，再把粗略結果當作 condition 餵進
  Stable Diffusion 做精細重建。在 Kodak 資料集上，SNR = 1 dB
  的極端條件下，LPIPS（感知相似度）從純 DeepJSCC 的 0.35 降到
  0.15。代價是什麼？Stable Diffusion 的 U-Net 有 860M 參數，在 A100 上跑
   50 步 denoising 要 2.5–5 秒。如果 receiver 是手機——跑不動。

  這就帶出了下一個關鍵概念。

  ---
  三、模型太大塞不進手機，所以要「拆開來算」

  Split Inference（拆分推理）的想法很直覺：一個大模型，前面幾層在 device
   跑，中間的 intermediate feature 傳到 edge server，後面幾層在 server
  跑。這不是新概念，但它和通訊的結合是近兩年才爆發的。Digital-SC（Guo et
   al., IEEE JSAC 2024）做了一個自適應版本：根據當前的 SNR 和 device
  的算力，用一個 policy network
  動態決定要在第幾層切、中間特徵要壓縮多少。SNR 好的時候 device
  多算一點（少傳一點），SNR 差的時候 server 多扛一點。

  把 split inference 和 diffusion 結合起來就更有意思了。Yang et
  al.（arXiv:2411.15781）把 diffusion 的 50 步 reverse denoising
  拆成兩段：server 做前 30 步（負責粗略的全局結構），傳一個 64 KB
  的中間特徵給 device，device 做後 20 步（負責細節）。用 DRL
  來決定最佳的 split point。但這篇做的是 AIGC 服務的 offloading，不是
  semantic communication——wireless channel
  在他們的設定裡只是傳輸管道，不是系統設計的核心。這個差異很關鍵，因為在
   semantic communication 中，channel noise 會直接影響語義品質，split
  的決策不只關乎延遲，還關乎重建的保真度。這個交集目前是空白的。

  ---
  四、不想每個問題從頭訓練，所以要 Foundation Model

  GPT 和 BERT 改變 NLP 的方式是：先在巨量文字上 pre-train
  一個通用模型，再針對特定任務 fine-tune。這個 paradigm
  現在進入了無線通訊。

  Large Wireless Model（LWM, Alkhateeb et al., Arizona State,
  arXiv:2411.08872）是全球第一個 channel foundation model。做法是用
  DeepMIMO 生成 100 萬組以上的通道矩陣（來自 15
  個不同場景），把每個通道矩陣切成 patches，隨機遮住 75%，訓練一個
  Transformer 去預測被遮住的部分——和 BERT 遮住字詞讓模型猜是完全一樣的道
  理，只是把「字」換成了「通道的一小塊」。預訓練完的模型產生的 embedding
   可以 fine-tune 到各種下游任務：beam prediction（Top-3 accuracy 從 76%
   提升到 89%）、LOS/NLOS classification（從 87% 到 95%）。更厲害的是
  zero-shot：在從未見過的新場景也能達到 82% beam prediction
  accuracy。WiFo（清華，Science China 2025）做了類似的事，還壓縮出只有
  5.5M 參數的 Tiny-WiFo，推理只要 1.6ms。

  但——這些全部是地面場景。LEO 衛星通道和地面差太多了：Doppler
  從地面的幾百 Hz 跳到衛星的 10–40 kHz，path loss 從 60–120 dB 跳到
  150–190 dB，通道從秒級變化變成毫秒級（衛星以 7.5 km/s
  的速度飛過），仰角從固定變成 10°到 90° 動態變化。目前沒有人為衛星做過
  channel foundation model。CPLLM（arXiv:2510.10561）用 GPT-2 做 LEO
  channel prediction，但它只是 fine-tune 一個通用 LLM，不是在衛星數據上
  pre-train 的 foundation model。

  ---
  五、網路天然就是圖，所以 GNN 是對的工具

  Graph Neural Network（GNN）在通訊中被大量使用的原因很簡單：無線網路本
  來就是一張圖。node 是基地台或用戶，edge 是干擾連結或通訊連結。GNN
  的核心操作叫 message passing：每個 node
  收集鄰居的特徵做聚合，更新自己的表示。經過 K 層之後，每個 node
  就「知道」K-hop 鄰域的狀態了。

  GNN 在通訊中最大的賣點是 size generalization——你在 10
  個基地台的小網路上訓練，可以直接部署到 100 個基地台的大網路上，因為
  GNN 的參數和圖的大小無關，只和鄰域結構有關。IEEE TWC 2025 的
  model-based GNN beamforming 那篇把通訊領域的 domain knowledge（WMMSE
  算法的迭代結構）和 GNN 結合起來，既保留了可解釋性，又有 data-driven
  的靈活性。ACM MobiCom 2024 也接了一篇 heterogeneous GNN for cell-free
  ISAC。

  把 GNN 和 multi-agent RL
  結合（GNN-MADRL）是處理大規模分散式網路的主流做法。每顆 LEO 衛星是一個
   agent，用 GNN 提取鄰域拓撲資訊作為
  observation，然後各自做決策。訓練用 centralized training decentralized
   execution（CTDE）：訓練時有一個 global critic 看全局，部署時每個
  agent 只看自己的 local observation。這套框架在 LEO routing（Computer
  Networks 2025）、beam hopping（Computer Comm 2025）、handover（IEEE
  TWC 2025）都有頂刊論文了。

  ---
  六、World Model：讓 AI 在「想像」中學會管理網路

  傳統 DRL 最大的問題是 sample inefficiency——agent
  要和真實環境互動上百萬次才能學好一個 policy，在真實網路中這意味著要不
  斷改配置然後觀察後果，太慢也太危險。World Model
  的做法來自認知科學：人不是每次都要真的試，大腦裡有一個「世界的內部模型
  」，可以在腦中模擬「如果我這樣做會怎樣」。DreamerV3 在 Atari
  遊戲上證明了這件事——先學一個 world model
  預測環境轉移，然後在「想像」中模擬上萬條
  trajectory，找到最佳策略，真實環境互動次數減少 10 到 100 倍。

  這個概念進入通訊是 2025 年的事。IEEE TNSE 專門開了一期 special issue
  "Fusing Digital Twins and World Models for Proactive 6G
  Networks"（deadline 2026-03-15），代表 IEEE editorial board
  認可這是方向。Dual-Mind World Model（arXiv:2510.24546，被 NeurIPS 2025
   AI4NextG Workshop 接受）更進一步，靈感來自 Kahneman
  的《快思慢想》：System 1 是輕量的 RNN 做即時預測，System 2 是重量級
  model 做因果分析和長期規劃。平常用 System 1，遇到異常啟動 System 2。在
   mmWave V2X 場景中，比純 model-free RL 的數據效率好 26%，收斂快 2 倍。

  World model 和前面的 digital twin 不一樣。Digital twin
  是「模擬器」——你需要有一個夠準的環境模型。World model
  是「學出來的模擬器」——它從數據中自己學會環境怎麼轉移，所以就算你沒有完
  美的通道模型，它也能用。

  ---
  七、因果推斷：不只知道「什麼和什麼相關」，還要知道「為什麼」

  這是我讀到最讓我覺得方向對的東西。目前通訊裡幾乎所有的 AI 都在做
  correlation learning：觀察到 RSRP 下降時選 beam #3
  效果好，就學到「RSRP 低→beam #3」。但為什麼 RSRP
  下降？可能是遮擋物（應該換 beam）、可能是 handover 邊界（應該換
  cell）、可能是功率衰減（應該調功率）。純相關性分不出來，所以一旦環境改
  變（比如路上多了一棟大樓），學到的 policy 就壞掉了。

  Causal AI 要學的是因果圖（structural causal model）：User Location →
  Path Loss → RSRP，Blockage → NLOS → Beam Selection →
  Throughput。知道因果關係之後可以做三件傳統 AI 做不到的事：第一是
  intervention——「如果我主動把功率調高，結果會怎樣」（不是觀察到功率高時
  結果怎樣，而是我介入改變功率）；第二是 counterfactual——「我選了 beam
  #3 結果不好，如果當時選 beam #5 會更好嗎」；第三是
  robustness——因果機制不隨環境表面特徵改變，所以泛化性更好。

  arXiv:2511.10291 把因果模型融入 multi-agent RL for IoT channel
  access，環境互動次數減少 58%，因為 agent 只在因果圖說「有影響」的
  channel 上探索，不浪費時間在因果上不相干的 channel。IEEE Vehicular
  Technology Magazine 2024 的 vision paper（Thomas et
  al.）把這件事講得很清楚：6G 的 AI 必須從 Pearl 因果階梯的 Level
  1（觀察相關性）升級到 Level 2（介入）和 Level 3（反事實推理）。

  ---
  八、Neuromorphic Computing：用脈衝代替浮點數，能耗降 30 倍

  傳統 DNN 的每個 neuron 每次都要做浮點數乘加運算。Spiking Neural
  Network（SNN）模仿生物神經元：累積輸入電位，超過閾值就發一個
  spike（二值的 0 或
  1），沒超過就什麼都不做——event-driven，不做事的時候不耗電。用
  LIF（Leaky Integrate-and-Fire）neuron 建模：V[t] = β·V[t-1] +
  I[t]，超過 threshold 就 fire 然後 reset。

  arXiv:2512.05246 用 SNN 完整取代了 5G NR OFDM 接收器的 channel
  estimation + equalization + demapping。在 3GPP TDL-A 通道上，SNN
  receiver 的 BLER 只比傳統 ANN receiver 差 1.2 dB，但能耗是 ANN 的
  1/7.6。加上 quantization-aware training 之後達到 1/35——從 100 mW 降到
  2.9 mW。這對
  IoT、穿戴裝置、衛星終端這種電池敏感的場景是根本性的改變。King's
  College London 的 KCLIP Lab 也在做 neuromorphic wireless split
  computing（2025 年 3 月）。

  ---
  九、Agentic AI：不是 LLM 回答問題，是多個 AI agent 自己管網路

  最後一個我覺得很重要的方向。Agentic AI 不等於 ChatGPT。它是多個 AI
  agent 組成的系統，每個 agent
  有角色、有記憶、有工具，能感知環境、推理問題、規劃行動、執行操作。

  最有說服力的案例是 Confucius（ACM SIGCOMM
  2025）——Meta（Facebook）的工程團隊做的，在 Meta
  的生產網路上已經跑了兩年以上，服務 60+ 個應用場景。架構是：人說「我要
  A 區域延遲低於 10ms」→ Intent Parser Agent 解析意圖 → Planner Agent 用
   RAG 查歷史案例、制定 multi-step plan → Executor Agent 呼叫 network
  API 改配置 → Validator Agent 監控結果、確認達標。這不是實驗室
  demo，是在管 Meta 真實的 data center network。Microsoft Research 的
  MeshAgent（ACM SIGMETRICS 2026）則是解決 LLM 的 hallucination 問題：從
   network specification 中自動提取不可違反的約束（如「每個 interface
  頻寬不超過 100 Gbps」、「routing table 不能有 loop」），LLM
  生成的配置必須通過 invariant checker。

  IEEE TNSE 在 2026 年開了至少四個 agentic AI 相關的 special
  issue（Secure Agentic Edge, Integrating Agentic AI in Wireless,
  Knowledge-driven Autonomous Agents, 加上 IEEE OJCOMS 的 Orchestrating
  Agentic AI），editorial board 的訊號很清楚：這是下一波。

  ---
  十、幾個你可能聽過但不確定原理的東西

  ISAC（Integrated Sensing and Communications）
  是讓同一套硬體、同一段波形同時做通訊和雷達感知。不是「通訊系統順便做
  sensing」，是從 waveform design 開始就把兩個目標放在一起最佳化。IEEE
  JSAC 和 TSP 是主力場域。進階版叫 ISCC（加上
  Computation），用一個信號同時做三件事。

  RIS（Reconfigurable Intelligent Surface）
  是一面可以程式控制的「智慧牆」，每個元素可以獨立調整反射的相位，等於你
  可以用軟體改變電磁波的傳播路徑。和傳統的中繼器不同，RIS
  不需要射頻鏈路和功率放大器，所以成本和功耗極低。AI 用在 RIS
  上主要是因為 phase shift 的最佳化是高維非凸問題，DRL
  是目前最有效的方法之一。

  Over-the-Air Computation（空中計算） 利用無線通道的疊加特性：多個
  device 同時傳，基地台收到的是所有信號的疊加。如果每個 device
  傳的是自己的 model gradient，基地台收到的直接就是 aggregated
  gradient——不需要一個一個收再加起來。這讓 Federated Learning
  的通訊效率提升了一個數量級。IEEE Signal Processing Magazine 2025 年 3
  月刊了一篇 waveform design for AirComp，標誌著這個方向進入成熟期。

  O-RAN（Open RAN） 是把基地台的軟硬體拆開、標準化介面，讓不同廠商的設備
  可以互通。其中最關鍵的是 Near-RT RIC（RAN Intelligent
  Controller），它提供了一個標準化的平台讓你部署 AI 算法作為
  xApp——這意味著你的研究不再是純模擬，可以直接跑在真實的 O-RAN testbed
  上。

  ---
  我看到的大圖

  讀完這些之後，我看到的格局是這樣的：通訊領域的 AI 正在從「用 DNN
  取代某個 block」（比如用 CNN 做 channel
  estimation）這種點狀的應用，轉向系統級的 paradigm shift。Semantic
  communication 改變了通訊的目標函數，foundation model
  改變了訓練的方式，world model 改變了 RL 和環境互動的方式，causal AI
  改變了學習的層次，agentic AI
  改變了網路管理的範式。這些不是彼此獨立的方向，而是一個正在收斂的大趨勢
  ：通訊系統正在變成一個端到端可學習、可推理、可自主決策的智慧系統。

  我們實驗室的每個人切入的角度不同——有人做 split inference +
  diffusion（P7），有人做 satellite foundation model（P3），有人做
  causal AI for beam management（P5），有人做 world model for BS
  sleep（P4），有人做 neuromorphic
  receiver（P6）——但底層都是同一件事：怎麼讓 AI 從「輔助工具」變成通訊系
  統的「原生組件」。理解彼此在做什麼，其實就是理解這個大趨勢的不同切面。





Sincerely,
程煒倫 William
Research Assistant, Institute of Information Science, Academia Sinica
+886 9080706 02 
