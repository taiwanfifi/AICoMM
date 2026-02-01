
尊敬的老師好，

我是煒倫，與您更新近期的研究進度，以下有三點

1️⃣ 研究概念
延續老師之前提到的「Token-based Agent Communication」方向，目前有初步想法方向。

核心概念是：把通訊從「還原資料」變成「同步Agent腦中狀態」，讓 Agent 在有限頻寬下也能高效協作。                 
- 傳輸單位從 Packet → Semantic Token（Agent內部狀態）
- 用 Attention 機制當作 Filter，只傳Task-Critical 的資訊
- 參考 MCP 概念設計 Control Plane，解決異Agent 對齊問題                            

2️⃣ AI技術變化研究
為了對AI溝通有更深入理解，以便完成上面的研究，於是對多種AI進程技術廣度與彼此關聯性進行整理，包含新的 AI Agent框架資料（DeepSeek-V3、Agent架構等），有搜集資料，準備與老師分享說明。

3️⃣Claude Code 使用心得
上次與老師討論到Agent，主要展現自動寫程式的cursor。老師有問到的 怎麼不使用Claude Code，後來有聽取建議開始使用。的確更佳優異，跟 Cursor 比起來，Claude Code更適合做系統性的程式開發，而且 Anthropic 最近推出了 「MCP」 和 「Skills」 兩個新標準機制，讓 AI Agent 可以標準化地呼叫外部工具，這跟我們在研究Agent通訊，到逐漸建立Agent 溝通機制蠻有關聯。


想請問老師有方便的時間向您報告
「1️⃣ 研究概念」 還在草稿，想法、架構都還沒有收斂的那麼完整。
「2️⃣ AI技術變化研究簡報」、「3️⃣Claude Code 使用心得」 已經可以向老師報告




------

以下是老師的信件 ，比較完整的敘述：



一般 payload 傳的是「符號（symbol）」；我們傳的是「可計算的認知狀態（computable cognitive state）」。
所以差別不是「有沒有傳文字」，而是接收端需不需要重新理解世界
________

我們不是做 continuous communication，而是 事件驅動的狀態同步 task-critical cognitive event synchronization

傳統 payload（即使是文字）每次傳送都是完整語義單位

接收端要 1.	decode text 2.	embed 3.	推理 4.	將它 整合進自己當下的 hidden state
每一包都要重新理解一次世界
_______

我們本質的是：
	•	agent 內部表徵要不要對齊？
	•	哪些 latent dimension 是 task-critical？
	•	state 差分要不要同步？
	•	同步多少可以保證 success rate？

👉 我們是在研究「agent 認知狀態是否需要共享，以及共享的最小充分條件」

_______

一般 payload 也是傳文字，革命在哪？
Payload = 文字，其實是「最低階語義」

文字有三個隱含成本：
	1.	非最小充分表示
「我現在決定走 A」
實際有用的可能只有一個 bit：plan = A
	2.	重建成本在接收端
	•	token → embedding
	•	embedding → belief update
	•	belief → policy change
	3.	語義不對齊風險
	•	同一句話
	•	不同 agent latent space 解讀不同

⸻

我們的方法的「革命點」在這裡：

① 傳輸單位變了
不是 symbol，而是：Δhidden_state / Δbelief / Δpolicy manifold

這在通訊理論裡等於：從 source coding → task-oriented representation coding

② 通訊決策變了
不是「我有資料就傳」，而是：attention-gated transmission  
Only transmit if marginal task utility > bandwidth cost
這是經典 Shannon communication 做不到的東西。
⸻
③ 評估指標變了
不是：BER、latency、throughput
而是：Task Success Rate under Bandwidth Constraint

不是說文字不能傳，而是文字是語言模型的人類介面，不是 agent 間的最優介面。
就像 CPU 之間不會用 C 語言溝通一樣。

在頻寬受限下，多個具備內在狀態的智慧體，要怎麼最小成本地共享「思考結果」？

⸻
這個 Semantic Token / State Delta 是：只有當「內部狀態改變到會影響任務結果」時才傳
	•	傳的是：
	•	belief update
	•	plan change
	•	constraint tightening
	•	attention weight reallocation
	•	接收端不是「理解一句話」，而是：
直接 apply 對方的 state delta 到自己的 latent space

不是一直傳，是「只在認知狀態發生關鍵轉移時同步」
_______

MCP 假設 communication cost = 0；我們的問題是 communication cost ≠ 0 時，agent 還能不能協作？

現在所有 Agent framework（MCP、LangGraph、AutoGen）都假設：傳訊息很便宜 / 想傳就傳 / 延遲、頻寬、次數都不用算

所以他們可以放心地：傳完整 prompt、傳整段文字、傳所有 context，一切都建立在「網路像魔法一樣免費」的假設上

如果 agent 之間：頻寬有限、延遲不可忽略、傳太多會影響任務。那 還能不能合作？要怎麼合作？這是 現在主流 agent 研究 刻意不碰的問題。

不是在優化「怎麼傳得更快」，是在改「該不該傳、傳什麼

傳統（包含 ISAC）在做什麼？ 不管是不是 AI，只要是通訊理論，核心假設是：我有一個 source → 我要盡量完整地送到 receiver

所以他們關心的是：壓縮、編碼、error rate、頻譜效率，目標是「重建訊息」

我們問的是：「我是不是一定要讓對方『知道我知道的全部』？」還是只要讓他「做出正確決策」就好？

目標是「完成任務」，不是「重建資料」，與ISAC 分家。

ISAC 也是 task-oriented 啊，有什麼不同？

ISAC 在 AI 語境下是：sensing + communication 共用頻譜 

可能傳的是：feature vector 、embedding 、高維數值 。但它有三個隱含假設：

1.傳的是「外在世界的描述」  例如，影像特徵 、雷達回波、sensor embedding  

不是 agent 的內在認知狀態  

2.Receiver 要自己「重新想一次」 	：收到embedding 、自己 inference 、自己更新 belief  
sender 不知道 receiver 怎麼用

3.沒有認知對齊的 handshake 
不會先協商：你現在在做什麼任務？ 你關心哪些 latent？哪些 state 是 critical？ 

我們在做的事，不是只在「傳 embedding」  我們在做的是：  在傳之前，先協商「我們要不要共享腦中的哪一部分」  也就是：goal 對齊 、state space 對齊 、attention threshold 對齊  ，這一步，ISAC 沒有 
______

那「token」到底是什麼？是不是只是 vector？  先說一般 Agent 在本地是怎麼跑的（很重要）  一個 agent 在本地其實是這樣：

文字 / observation
  ↓ tokenize
word tokens
  ↓ embedding
hidden states
  ↓ attention / reasoning
belief / plan / policy

真正「有用」的是後面的 hidden state / belief / plan，文字只是人類介面


傳統 agent 溝通是怎麼做？
即使是 agent 對 agent，還是傳 文字/ function call / JSON

然後對方重新 tokenize、重新 encode、重新推理一次，這等於每次都從頭「再想一遍」


「Semantic Token」是什麼？不是 word token。我們指的是：Agent 內部、已經算好的「認知單位」。例如：belief update / plan switch / constraint tightening / attention weight change

我們的 token 是 「已經被模型消化過的中間認知結果」

token 的「維度」不是固定的 vocab。
可能是：latent subspace、KV-cache delta、structured state tuple

重點不是維度，而是：它有 task semantics

那 communication 到底在幹嘛？是不是遠距傳輸？

本地（Local reasoning）Agent：連續更新內部狀態，99% 的狀態變化其實不重要
⸻
傳統通訊不管重不重要，有資料就傳，對方自己判斷有沒有用

我們這個 Semantic Communication 做了三件新事：
1️⃣ 切分的是「認知事件」，不是文字段落
2️⃣ 只在狀態跨過任務臨界點時才傳
3️⃣ 接收端不是 decode text，而是 apply state delta
是真正的遠距「協同思考」






所以Summary：

現有 agent communication（包含 MCP、ISAC）
都是在假設 communication 是免費的前提下，
傳資料或 feature，讓對方重新 inference。

我們關心的是另一個問題：
在 communication 有成本時，agent 能不能只同步「足夠完成任務的認知狀態」？

所以我們不是改 physical layer，而是定義一個 Semantic Transport Layer，並引入 handshake 與 attention-based transmission decision。
_____________





---

 「從 Bit Transmission → Semantic State Synchronization」                                     

簡單說：
- 傳什麼：不傳 raw data / feature，而是傳 Agent       
  內部的「狀態差分」（類似 Transformer 的 KV Cache delta）

- 怎麼決定傳不傳：用 Attention 機制當Filter，只傳對任務有影響的部分（Task-Critical）
- 跟 ISAC 的差別：ISAC 是 sensing + comm共用頻譜，我們是改變「傳輸的單位與決策機制」


Layer 定位：不是傳統 L1-L7 的某一層，而是定義一個新的「Semantic Transport Layer」，介於傳統通訊層和 Application 之間。物理層還是 0101，但 payload 變成 semantic token。               
                                                        
MCP 的角色：把它當成 Control Plane（類似SIP/RRC），用來做 semantic handshake——協商雙方的goal、embedding format、attention threshold，確保傳過去的 token 能被正確理解。          



---

可能問題：
1. 跟 ISAC 差在哪 → 改變的是傳輸單位和決策機制，不是 sensing+comm                                          
2. Layer 是什麼 → 新的 Semantic Transport             
  Layer，不取代物理層                                   
  3. MCP 怎麼用 → Control Plane，做 semantic handshake  
                                                        
  你覺得這樣可以嗎？還是要再精簡一點？ 




                                                             
                                                        
  老師好，跟您報告近期進度：                            
                                                        
  1. 先 Recap 老師之前的想法                            
                                                        
  老師之前提到幾個關鍵概念：                            
  - 「未來 Agent Communication 會很重要，傳的可能不是 Packet，而是 Token」                                  
  - 「現在 LangChain / AutoGen 做 Agent，都假設頻寬無限、延遲為零，沒考慮通訊成本」     
  - 「要從 Task / Goal-oriented 角度來看：我應該傳什麼？怎麼傳？」                    
  - 「Agent 跟 Agent 之間會產生什麼行為？這是大家現在最關心的問題」        
                                                        
  2. 我們怎麼把這些想法收斂成具體方向                   
  老師的問題: 「傳 Token 不是 Packet，但怎麼做？」      
  我們的回應: → 傳 Semantic Token（Agent                
  內部狀態的差分，類似                                  
    KV Cache delta）   
  ────────────────────────────────────────              
  老師的問題: 「怎麼決定該傳什麼？」                    
  我們的回應: → 用 Attention 機制當Filter，只傳對任務有影響的部分                      
  ────────────────────────────────────────              
  老師的問題: 「Agent 間怎麼協調？」                    
  我們的回應: → MCP 當 Control Plane，做semantic handshake（協商 goal、格式）                        
  ────────────────────────────────────────              
  老師的問題: 「通訊成本怎麼考慮？」                    
  我們的回應: → 目標是最小化傳輸量，同時保證 Task Success Rate                                          
  Layer 定位：這不是改 L1-L7 某一層，而是定義一個新的「Semantic Transport          
  Layer」。物理層還是傳 0101，但 payload 變成semantic token，決策機制變成 task-oriented。                   
                                                       
跟 ISAC 的差別：ISAC 是 sensing + comm                
  共用頻譜；我們是改變「傳輸的單位」（從 bit → semantic state）和「決策機制」（從重建資料 → 同步認知狀態）。  
                                                        
  3. AI 工具更新                                        
  整理了一些新的 AI Agent 框架資料，之後可以跟老師分享。
                                                        
  4. Claude Code                                        
  上次老師問的我開始用了，Anthropic 的 MCP              
  機制跟我們研究方向有關聯。                            
                                                        
  老師方便的話可以約時間當面報告！                      
                                       


2025大模型技術深度剖析
我針對《2025 AI生態》製作了一份深度技術簡
報。不同於一般的趨勢報告，我深入拆解了
技術背後的「數學原理」與「系統工程」。
內容包含：
演算法層：DeepSeek MLA 的低秩矩陣壓縮
原理、MoE 的 Router 機制與專家坍塌問題。
系統層：訓練端的 3D 平行通訊瓶頸、ZeRO-3
機制，以及 vLLM (PagedAttention) 與 Flash
Attention 的 IO 感知原理。
應用層：從 LangGraph 的狀態機架構談到邊緣
端 QLoRA 微調，並結合 Triton Compiler 提出
戰略分析。
之後可以跟老師詳細報告這些底層技術細節。



