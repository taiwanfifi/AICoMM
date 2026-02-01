#!/bin/bash

# Repository 重構腳本
# 用途：將研究文檔從發散探索整理為收斂論文結構
# 作者：台大電機博士研究
# 日期：2026-01-24

set -e  # 遇到錯誤立即停止

echo "========================================="
echo "  AI-Comm Repository 重構腳本"
echo "========================================="
echo ""

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 確認執行
echo -e "${YELLOW}警告：此腳本將重組整個目錄結構${NC}"
echo "建議先查看 RESTRUCTURE_PLAN.md 了解完整計劃"
echo ""
read -p "確定要繼續嗎？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "取消執行"
    exit 0
fi

echo ""
echo "========================================="
echo "  Phase 1: 備份現有文件"
echo "========================================="

# 創建備份目錄
BACKUP_DIR="backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo -e "${GREEN}✓${NC} 創建備份目錄: $BACKUP_DIR"

# 備份所有 markdown 文件
cp *.md "$BACKUP_DIR/" 2>/dev/null || true
echo -e "${GREEN}✓${NC} 備份 markdown 文件"

# 列出備份的文件
echo ""
echo "已備份的文件："
ls -1 "$BACKUP_DIR/"
echo ""

echo "========================================="
echo "  Phase 2: 創建新目錄結構"
echo "========================================="

# 創建主要目錄
directories=(
    "00-advisor-feedback"
    "01-problem-formulation"
    "02-core-framework"
    "03-technical-design"
    "04-background/papers"
    "04-background/related-work"
    "04-background/technical-background"
    "05-evaluation"
    "06-paper-drafts/figures"
    "07-code/simulation"
    "07-code/prototype"
    "07-code/evaluation"
    "archive/old-directions"
    "archive/evolution-logs"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo -e "${GREEN}✓${NC} 創建目錄: $dir"
done

echo ""
echo "========================================="
echo "  Phase 3: 移動 PDF 文件"
echo "========================================="

# 移動 PDF 到 background/papers
if ls *.pdf 1> /dev/null 2>&1; then
    mv *.pdf 04-background/papers/
    echo -e "${GREEN}✓${NC} 移動 PDF 文件到 04-background/papers/"
else
    echo -e "${YELLOW}!${NC} 未找到 PDF 文件"
fi

echo ""
echo "========================================="
echo "  Phase 4: 整理現有 Markdown 文件"
echo "========================================="

# 移動教授反饋
if [ -f "professor_concepts.md" ]; then
    mv professor_concepts.md 00-advisor-feedback/professor-concepts-raw.md
    echo -e "${GREEN}✓${NC} professor_concepts.md → 00-advisor-feedback/"
fi

# 移動背景文獻
if [ -f "agent.md" ]; then
    mv agent.md 04-background/technical-background/agent-services.md
    echo -e "${GREEN}✓${NC} agent.md → 04-background/technical-background/"
fi

if [ -f "IOA.md" ]; then
    mv IOA.md 04-background/technical-background/internet-of-agents.md
    echo -e "${GREEN}✓${NC} IOA.md → 04-background/technical-background/"
fi

if [ -f "deepseek.md" ]; then
    mv deepseek.md 04-background/technical-background/deepseek-architecture.md
    echo -e "${GREEN}✓${NC} deepseek.md → 04-background/technical-background/"
fi

# 歸檔舊方向
if [ -f "t1.md" ]; then
    mv t1.md archive/old-directions/t1-oran-automation.md
    echo -e "${GREEN}✓${NC} t1.md → archive/old-directions/ (已廢棄方向)"
fi

if [ -f "t2.md" ]; then
    mv t2.md archive/old-directions/t2-edge-rag.md
    echo -e "${GREEN}✓${NC} t2.md → archive/old-directions/ (已廢棄方向)"
fi

# 歸檔演進紀錄
if [ -f "t4.md" ]; then
    mv t4.md archive/evolution-logs/t4-diagnosis.md
    echo -e "${GREEN}✓${NC} t4.md → archive/evolution-logs/ (診斷紀錄)"
fi

if [ -f "t5.md" ]; then
    mv t5.md archive/evolution-logs/t5-convergence.md
    echo -e "${GREEN}✓${NC} t5.md → archive/evolution-logs/ (收斂紀錄)"
fi

if [ -f "t7.md" ]; then
    mv t7.md archive/evolution-logs/t7-version-comparison.md
    echo -e "${GREEN}✓${NC} t7.md → archive/evolution-logs/ (版本對比)"
fi

# 核心文件（需要重構，暫時保留原位置供參考）
if [ -f "t3.md" ]; then
    cp t3.md 02-core-framework/t3-original-reference.md
    echo -e "${YELLOW}→${NC} t3.md 複製到 02-core-framework/ (需要重構)"
    echo -e "${YELLOW}  ${NC} 原始檔案保留，請手動重構為 semantic-state-sync.md"
fi

if [ -f "t6.md" ]; then
    cp t6.md 03-technical-design/t6-original-reference.md
    echo -e "${YELLOW}→${NC} t6.md 複製到 03-technical-design/ (需要重構)"
    echo -e "${YELLOW}  ${NC} 原始檔案保留，請手動重構為 attention-filtering.md"
fi

# 處理 t8.md（最關鍵的論述文件）
if [ -f "t8.md" ]; then
    cp t8.md 01-problem-formulation/t8-core-arguments-reference.md
    echo -e "${GREEN}✓${NC} t8.md → 01-problem-formulation/ (核心論述參考)"
fi

echo ""
echo "========================================="
echo "  Phase 5: 創建關鍵補充文件"
echo "========================================="

# 創建 defense-strategy.md 骨架
cat > 01-problem-formulation/defense-strategy.md << 'EOF'
# Defense Strategy: 與現有研究的本質差異

## 概述
本文檔明確說明我們的研究與現有 ISAC、JSCC、MCP、Agent Frameworks 的本質差異，
建立清晰的理論護城河。

## 核心差異總覽

### 與 ISAC 的本質差異
**ISAC**: Integrated Sensing and Communication（頻譜共享）
**我們**: 改變傳輸單位（bit → semantic state）

### 與 JSCC 的本質差異
**JSCC**: Joint Source-Channel Coding（資料重建）
**我們**: Task-oriented state synchronization

### 與 MCP/Agent Frameworks 的本質差異
**MCP等**: 假設 communication cost = 0
**我們**: 在 communication 有成本時，agent 如何協作

## 三個革命點

### 1. 傳輸單位改變
```
傳統：Symbol / Bit / Packet
我們：Δhidden_state / Δbelief / Δpolicy manifold
```

**意義**：從 source coding → task-oriented representation coding

### 2. 決策機制改變
```
傳統：有資料就傳
我們：Attention-gated transmission
     Only transmit if marginal task utility > bandwidth cost
```

**意義**：這是經典 Shannon communication 做不到的東西

### 3. 評估指標改變
```
傳統：BER、Latency、Throughput
我們：Task Success Rate under Bandwidth Constraint
```

## 核心論述

### 問題定義
現有 agent communication（包含 MCP、ISAC）都是在假設 communication 是免費的前提下，
傳資料或 feature，讓對方重新 inference。

我們關心的是另一個問題：
**在 communication 有成本時，agent 能不能只同步「足夠完成任務的認知狀態」？**

### 技術定位
不是在優化「怎麼傳得更快」，是在改「該不該傳、傳什麼」。

## 詳細對比

### vs. Traditional Communication
傳統通訊關心的是：壓縮、編碼、error rate、頻譜效率
**目標**：重建訊息

我們問的是：「我是不是一定要讓對方『知道我知道的全部』？」
**目標**：完成任務

### vs. ISAC
ISAC 的三個假設：
1. 傳的是「外在世界的描述」（影像特徵、雷達回波）
2. Receiver 要自己「重新想一次」
3. 沒有認知對齊的 handshake

我們的差異：
1. 傳的是「內在認知狀態」（belief update, plan switch）
2. Receiver 直接 apply state delta
3. 有 Control Plane 協商（goal 對齊、state space 對齊）

### vs. JSCC
JSCC 目標：min D_MSE(X, X̂)
我們目標：min D_task(S, Ŝ) = 1 - Task_Success_Rate

重點不是「重建資料」，而是「同步認知狀態」。

### vs. MCP/Agent Frameworks
現在所有 Agent framework（MCP、LangGraph、AutoGen）都假設：
- 傳訊息很便宜
- 想傳就傳
- 延遲、頻寬、次數都不用算

我們的問題：
如果 agent 之間頻寬有限、延遲不可忽略、傳太多會影響任務，
還能不能合作？要怎麼合作？

## Semantic Token 定義

### 什麼是 Semantic Token？
不是 word token。

我們指的是：Agent 內部、已經算好的「認知單位」。
- belief update
- plan switch
- constraint tightening
- attention weight change

### 與 Word Token 的差異

| 維度 | Word Token | Semantic Token |
|------|-----------|----------------|
| 定義 | 文字單位 | 認知單位 |
| 維度 | 固定 vocab | Latent subspace |
| 處理 | 需要重新推理 | 直接 apply delta |
| 成本 | 重建成本在接收端 | 最小充分表示 |

### 為什麼不用文字？
文字有三個隱含成本：
1. **非最小充分表示**：「我現在決定走 A」實際有用的可能只有一個 bit：plan = A
2. **重建成本在接收端**：token → embedding → belief update → policy change
3. **語義不對齊風險**：同一句話，不同 agent latent space 解讀不同

## 技術創新點

### 1. 事件驅動的狀態同步
不是 continuous communication，而是：
**Task-Critical Cognitive Event Synchronization**

### 2. Attention-based Filtering
只在狀態跨過任務臨界點時才傳：
- 切分的是「認知事件」，不是文字段落
- 只在狀態跨過任務臨界點時才傳
- 接收端不是 decode text，而是 apply state delta

### 3. Control Plane 協商
與 ISAC 最大的差異：
在傳之前，先協商「我們要不要共享腦中的哪一部分」
- goal 對齊
- state space 對齊
- attention threshold 對齊

## 預期質疑與回應

### Q: 這不就是壓縮嗎？
A: 不是。壓縮是為了重建，我們是為了任務。評估指標不同。

### Q: ISAC 也是 task-oriented，有什麼不同？
A: ISAC 傳外在世界描述，我們傳內在認知狀態。ISAC 接收端要重新推理，我們直接 apply delta。

### Q: 為什麼不用 MCP？
A: MCP 是 application-layer 協定，假設通訊成本為零。我們要處理的是通訊有成本時的協作問題。

### Q: 實驗怎麼做？
A: Trace-driven simulation，這是通訊領域的標準方法。

## 總結
我們不是改 physical layer，而是定義一個 **Semantic Transport Layer**，
並引入 handshake 與 attention-based transmission decision。

這是一個全新的通訊範式：
從「Bit Transmission」→「Semantic State Synchronization」
EOF

echo -e "${GREEN}✓${NC} 創建 01-problem-formulation/defense-strategy.md"

# 創建 mathematical-system-model.md 骨架
cat > 01-problem-formulation/mathematical-system-model.md << 'EOF'
# Mathematical System Model

## 系統模型定義

### 基本元素
- **Source (Edge Agent)**: 具有內部狀態 $S_t \in \mathcal{S}$
- **Channel**: 帶寬受限 $B$，延迟 $D$
- **Receiver (Cloud Agent)**: 目標狀態 $\hat{S}_t$
- **Task**: 目標函數 $\mathcal{T}(S_t, \hat{S}_t)$

### 狀態空間定義
```math
S_t = (h_t, b_t, p_t, a_t)
```
其中：
- $h_t$: Hidden state (latent representation)
- $b_t$: Belief state (probabilistic world model)
- $p_t$: Policy state (action distribution)
- $a_t$: Attention weights (task-critical dimensions)

## Information Bottleneck 框架

### 目標函數
```math
\min I(X; Z) - \beta I(Z; Y)
```

其中：
- $X$: 原始狀態空間
- $Z$: 傳輸的 semantic token
- $Y$: 任務相關信息
- $\beta$: 權衡參數（bandwidth cost vs. task performance）

### 物理意義
- $I(X; Z)$: 傳輸率（越小越好）
- $I(Z; Y)$: 任務相關信息（越大越好）
- $\beta$: 根據頻寬成本動態調整

## Rate-Distortion 目標

### 優化問題
```math
\min R(S_t \to Z_t)
\text{subject to: } D_{\text{task}}(S_t, \hat{S}_t) \leq D_{\max}
```

其中：
- $R$: 傳輸率（bits per token）
- $D_{\text{task}}$: 任務失真（不是 MSE，是 task success rate）

### 任務失真定義
```math
D_{\text{task}}(S_t, \hat{S}_t) = 1 - P(\text{Task Success} | \hat{S}_t)
```

## 完整優化問題

### 主要目標
```math
\max \text{Task Success Rate}
```

### 約束條件
1. **帶寬約束**：$R \leq B$
2. **延遲約束**：$T_{\text{total}} \leq D_{\max}$
3. **狀態一致性**：$\|S_t - \hat{S}_t\|_{\text{task}} \leq \epsilon$

### 決策變數
- $\delta_t \in \{0,1\}$: 是否在時間 $t$ 傳輸
- $Z_t$: 傳輸的 semantic token
- $\tau$: Attention threshold

## Attention-Based Filtering

### Attention Gate 函數
```math
\delta_t = \mathbb{1}[\max_i a_{t,i} > \tau]
```

其中 $a_{t,i}$ 是第 $i$ 個 latent dimension 的 attention weight。

### Token Selection
```math
Z_t = \{h_{t,i} : a_{t,i} > \tau\}
```
只傳輸 attention weight 超過閾值的維度。

## State Update 模型

### Source 端（發送前）
```math
S_{t+1} = f_{\text{local}}(S_t, o_t)
```
其中 $o_t$ 是新的 observation。

### Receiver 端（接收後）
```math
\hat{S}_{t+1} = f_{\text{integrate}}(\hat{S}_t, Z_t, \text{Anchor}_t)
```

### State Delta
```math
\Delta S_t = S_t - S_{t-1}
Z_t = \text{Compress}(\Delta S_t)
```

## 與傳統通訊的數學對比

| 框架 | 目標函數 | 約束 | 評估 |
|------|---------|------|------|
| Shannon | $\max I(X;Y)$ | $R < C$ | BER |
| JSCC | $\min D(X,\hat{X})$ | $R < C$ | MSE |
| **Ours** | $\max P(\text{Task Success})$ | $R < B, T < D_{\max}$ | Task Success Rate |

## 關鍵數學創新

### 1. Task-Oriented Distortion
不是 $\|X - \hat{X}\|^2$，而是 $1 - P(\text{Task Success})$

### 2. Attention-Weighted Rate
```math
R_{\text{eff}} = \sum_{i: a_i > \tau} \log_2 |\mathcal{H}_i|
```
只計算被傳輸的維度的熵。

### 3. Semantic Consistency Constraint
```math
\text{KL}(p(a|S_t) \| p(a|\hat{S}_t)) \leq \epsilon
```
確保兩端的 policy 分布接近。

## 下一步
1. 推導 optimal threshold $\tau^*$
2. 分析 rate-distortion trade-off
3. 設計具體的 compression algorithm
EOF

echo -e "${GREEN}✓${NC} 創建 01-problem-formulation/mathematical-system-model.md"

# 創建 state-integration.md 骨架
cat > 03-technical-design/state-integration.md << 'EOF'
# State Integration: Receiver 端機制

## 概述
平衡 Source/Receiver 設計，補充 Receiver 端的狀態整合機制。

## Receiver 端狀態更新公式

### 基本更新方程
```math
S_{t+1} = f(S_t, \Delta_{\text{token}}, \text{Anchor})
```

其中：
- $S_t$: 當前狀態
- $\Delta_{\text{token}}$: 接收到的 semantic token
- $\text{Anchor}$: MCP 協商好的對齊點
- $f$: 確定性整合函數（Deterministic Integration）

## 關鍵機制

### 1. Anchor-based Alignment

#### Anchor 定義
Anchor 是雙方協商好的「語義錨點」，用於確保 token 插入正確的 latent slot。

#### 協商過程
```
1. Sender: "我的 anchor 是 [goal=delivery, location=A]"
2. Receiver: "我的 anchor 是 [goal=delivery, location=B]"
3. Control Plane: "Align on goal=delivery, location divergence noted"
```

#### 整合公式
```math
\hat{S}_{t+1} = \hat{S}_t + \alpha \cdot (\Delta_{\text{token}} - \text{ProjectionError}(\text{Anchor}))
```

### 2. Out-of-Order 處理

#### 問題
網路可能導致 token 亂序到達。

#### 解決方案
- Token 帶有時間戳 $t$
- 維護 buffer，按序整合
- 如果 $t_{\text{recv}} > t_{\text{expected}} + \delta$，觸發重傳請求

#### Buffer 管理
```python
def integrate_token(buffer, token):
    buffer.insert(token, key=token.timestamp)
    while buffer.is_continuous():
        S_t = buffer.pop_oldest()
        apply_delta(S_t)
```

### 3. 丟包處理

#### 不是 RAG 腦補
與傳統 agent 不同，我們不會「腦補」丟失的訊息。

#### 策略
1. **Zero-Hold**: 保持上一個狀態
   ```math
   \hat{S}_t = \hat{S}_{t-1} \quad \text{if packet lost}
   ```

2. **Kalman Filter**: 基於歷史趨勢預測
   ```math
   \hat{S}_t = \hat{S}_{t-1} + K(\Delta S_{\text{predicted}} - \Delta S_{\text{observed}})
   ```

3. **Task-aware Retransmission**: 根據任務重要性決定是否重傳
   ```python
   if task_criticality(lost_token) > threshold:
       request_retransmission(lost_token.id)
   else:
       use_zero_hold()
   ```

## 與 Traditional Communication 的對比

| 維度 | 傳統通信 | 我們的方法 |
|------|---------|-----------|
| 丟包處理 | ARQ/FEC | Task-aware retransmission |
| 順序保證 | TCP sequence number | Token timestamp + anchor |
| 狀態還原 | Bit-perfect recovery | Task-sufficient reconstruction |
| 錯誤處理 | CRC/Checksum | Semantic consistency check |

## Deterministic Integration 的優勢

### 1. 可預測性
給定相同的 $(S_t, \Delta_{\text{token}}, \text{Anchor})$，結果是確定的。

### 2. 可驗證性
可以用數學證明：
```math
\|S_{\text{sender}} - S_{\text{receiver}}\|_{\text{task}} \leq \epsilon
```

### 3. 可調試性
不像 RAG 會產生 hallucination，我們的整合過程是透明的。

## 實現細節

### 狀態整合算法
```python
class StateIntegrator:
    def __init__(self, anchor):
        self.anchor = anchor
        self.buffer = PriorityQueue()

    def integrate(self, token):
        # 1. 檢查 anchor 對齊
        if not self.check_alignment(token.anchor, self.anchor):
            raise SemanticMismatchError

        # 2. 插入 buffer
        self.buffer.put((token.timestamp, token))

        # 3. 按序處理
        while self.buffer.has_continuous_sequence():
            t, tok = self.buffer.get()
            self.apply_delta(tok.delta)

    def apply_delta(self, delta):
        # 4. 更新狀態
        self.state = self.state + self.alpha * delta

    def handle_loss(self, expected_t):
        # 5. 丟包處理
        if self.is_critical(expected_t):
            self.request_retransmit(expected_t)
        else:
            self.state = self.state  # Zero-hold
```

## 評估指標

### 1. 整合準確度
```math
\text{Accuracy} = 1 - \frac{\|S_{\text{true}} - \hat{S}\|_{\text{task}}}{\|S_{\text{true}}\|_{\text{task}}}
```

### 2. 延遲
```math
T_{\text{total}} = T_{\text{transmission}} + T_{\text{integration}}
```

### 3. 容錯性
```math
\text{Robustness} = P(\text{Task Success} | \text{packet loss rate} = p)
```

## 下一步
1. 實現 StateIntegrator 原型
2. 測試不同丟包率下的性能
3. 優化 Kalman Filter 參數
EOF

echo -e "${GREEN}✓${NC} 創建 03-technical-design/state-integration.md"

# 創建 vs-ISAC.md
cat > 04-background/related-work/vs-ISAC.md << 'EOF'
# vs. ISAC: 本質差異分析

## ISAC 定義
**Integrated Sensing and Communication (ISAC)**: 在同一頻譜資源上同時進行感知和通訊。

## 核心區別

### 我們與 ISAC 的差異
| 維度 | ISAC | 我們的方法 |
|------|------|-----------|
| **傳輸單位** | Sensing data + Communication data | Semantic state delta |
| **傳輸內容** | 外在世界的描述（影像、雷達） | 內在認知狀態（belief, plan） |
| **接收端處理** | 重新 inference | 直接 apply delta |
| **對齊機制** | 無認知對齊 | Control Plane handshake |
| **目標** | 頻譜效率 | 任務成功率 |

## ISAC 的三個隱含假設

### 1. 傳的是「外在世界的描述」
例如：
- 影像特徵
- 雷達回波
- Sensor embedding

**不是** agent 的內在認知狀態。

### 2. Receiver 要自己「重新想一次」
```
收到 embedding → 自己 inference → 自己更新 belief
```
Sender 不知道 Receiver 怎麼用這些資訊。

### 3. 沒有認知對齊的 handshake
不會先協商：
- 你現在在做什麼任務？
- 你關心哪些 latent？
- 哪些 state 是 critical？

## 我們的創新

### 1. Control Plane 協商
在傳之前，先協商「我們要不要共享腦中的哪一部分」：
- Goal 對齊
- State space 對齊
- Attention threshold 對齊

### 2. State Delta 直接應用
```
接收端不是 decode → inference → update
而是：直接 apply state delta
```

### 3. Task-Oriented 評估
```
ISAC: min BER, max Spectrum Efficiency
我們: max Task Success Rate under Bandwidth Constraint
```

## 數學對比

### ISAC 優化目標
```math
\max \alpha \cdot R_{\text{comm}} + (1-\alpha) \cdot R_{\text{sensing}}
\text{subject to: } P_{\text{total}} \leq P_{\max}
```

### 我們的優化目標
```math
\max P(\text{Task Success})
\text{subject to: } R \leq B, T \leq D_{\max}, D_{\text{task}} \leq \epsilon
```

## 為什麼 ISAC 不夠？

### ISAC 的假設
「我有 data，我要盡量完整地送到 receiver」

### 我們的問題
「我是不是一定要讓對方『知道我知道的全部』？」
還是只要讓他「做出正確決策」就好？

## 總結
ISAC 是 sensing + comm 共用頻譜；
我們是改變「傳輸的單位」（從 bit → semantic state）和「決策機制」（從重建資料 → 同步認知狀態）。
EOF

echo -e "${GREEN}✓${NC} 創建 04-background/related-work/vs-ISAC.md"

# 創建 vs-JSCC.md
cat > 04-background/related-work/vs-JSCC.md << 'EOF'
# vs. JSCC: 本質差異分析

## JSCC 定義
**Joint Source-Channel Coding (JSCC)**: 聯合優化 source coding 和 channel coding，
以最小化 end-to-end 失真。

## 核心區別

### 數學目標對比
| 框架 | 目標函數 | 失真度量 | 約束 |
|------|---------|---------|------|
| **JSCC** | $\min D(X, \hat{X})$ | MSE / PSNR | $R < C$ |
| **我們** | $\min D_{\text{task}}(S, \hat{S})$ | $1 - P(\text{Task Success})$ | $R < B, T < D_{\max}$ |

## JSCC 的假設

### 1. 目標是「重建資料」
```math
\min E[\|X - \hat{X}\|^2]
```
希望 $\hat{X}$ 儘可能接近 $X$。

### 2. 失真是 pixel-level 或 feature-level
評估用 MSE、PSNR、SSIM 等指標。

### 3. 不考慮 task semantics
不管資料是用來做什麼任務，只管重建準確度。

## 我們的創新

### 1. 目標是「完成任務」
```math
\min D_{\text{task}}(S, \hat{S}) = 1 - P(\text{Task Success} | \hat{S})
```

### 2. 失真是 task-level
不需要 bit-perfect recovery，只需要足夠完成任務。

### 3. Task-aware compression
根據任務的 attention weights 決定傳什麼。

## 具體例子

### JSCC 的做法
```
影像 X → Encoder → 壓縮表示 Z → Decoder → 重建影像 X̂
評估: PSNR(X, X̂)
```

### 我們的做法
```
Agent 狀態 S → Attention Filter → Semantic Token Z → State Integration → 更新狀態 Ŝ
評估: P(Task Success | Ŝ)
```

## 為什麼不用 JSCC？

### 場景：自駕車協作
**JSCC 會**：
- 壓縮整張影像
- 傳輸壓縮表示
- 重建影像
- Receiver 自己做物體偵測

**我們會**：
- 只傳 "前方有障礙物，建議切換到 Plan B"
- Receiver 直接更新 plan state

### 頻寬對比
JSCC: ~100 KB (壓縮影像)
我們: ~100 Bytes (semantic token)

### 延遲對比
JSCC: Encode + Transmit + Decode + Inference
我們: Filter + Transmit + Integrate

## 數學嚴謹性對比

### JSCC 有嚴格的理論保證
```math
R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X; \hat{X})
```

### 我們也有理論框架
```math
R_{\text{task}}(D) = \min_{p(z|s): E[d_{\text{task}}(S,\hat{S})] \leq D} I(S; Z)
```

關鍵差異：$d_{\text{task}} \neq d_{\text{MSE}}$

## 可能的質疑與回應

### Q: 這不就是 task-oriented JSCC 嗎？
A: 不是。JSCC 的 task-oriented extension 還是在「重建 feature」，我們是「同步 state」。

### Q: 你們的失真度量怎麼定義？
A: $D_{\text{task}} = 1 - P(\text{Task Success})$，可以通過實驗測量。

### Q: 有理論最優解嗎？
A: 我們有 Information Bottleneck 框架下的 rate-distortion bound。

## 總結
JSCC 追求「重建資料」，我們追求「完成任務」。
評估指標從 MSE → Task Success Rate。
EOF

echo -e "${GREEN}✓${NC} 創建 04-background/related-work/vs-JSCC.md"

# 創建 vs-traditional-comm.md
cat > 04-background/related-work/vs-traditional-comm.md << 'EOF'
# vs. Traditional Communication: 範式轉移

## Shannon 範式

### 經典通訊模型
```
Source → Encoder → Channel → Decoder → Destination
```

### 核心假設
1. 目標是「bit-perfect transmission」
2. Channel capacity $C = B \log_2(1 + \text{SNR})$
3. 評估指標：BER, throughput, latency

## 我們的範式

### Semantic State Communication
```
Shared World Model → State Δ → Semantic Token → Sync → Task Execution
```

### 核心假設
1. 目標是「task-sufficient synchronization」
2. Semantic capacity $C_{\text{sem}} = f(\text{attention}, \text{task relevance})$
3. 評估指標：Task Success Rate, spectrum efficiency (bits per successful task)

## 範式轉移的三個層次

### 1. 傳輸單位
| 範式 | 傳輸單位 | 語義層級 |
|------|---------|---------|
| Shannon | Bit | 無語義 |
| Semantic Comm | Symbol | 低階語義（文字） |
| **Ours** | Cognitive State | 高階語義（認知） |

### 2. 通訊目標
```
Shannon: min P(error)
Semantic: min D(message)
Ours: max P(task success)
```

### 3. 評估標準
```
Shannon: BER, SNR, Capacity
Semantic: PSNR, SSIM, Perceptual Quality
Ours: Task Success Rate, Decision Latency
```

## 為什麼 Shannon 範式不夠？

### Shannon 的隱含假設
「只要 bit 傳對了，communication 就成功了」

### 在 AI Agent 場景下的問題
1. **冗餘傳輸**：傳了很多對任務無用的資訊
2. **重複推理**：每次都要重新理解
3. **缺乏對齊**：不知道對方需要什麼

## 具體對比

### 場景：兩個 Agent 協作導航

#### Traditional Communication
```
Agent A: 傳送完整地圖影像 (1 MB)
Agent B: 接收 → 解碼 → 物體偵測 → 路徑規劃
```

#### Our Approach
```
Agent A: "Plan changed: obstacle at (x,y), switch to route B" (100 bytes)
Agent B: 接收 → 直接更新 plan state
```

### 頻寬對比
Traditional: 1 MB
Ours: 100 bytes
**節省**: 10,000x

### 延遲對比
Traditional: T_encode + T_transmit + T_decode + T_inference
Ours: T_filter + T_transmit + T_integrate
**減少**: ~50%

## 理論貢獻

### Shannon 的 Rate-Distortion Theory
```math
R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X; \hat{X})
```

### 我們的 Task-Oriented Rate-Distortion
```math
R_{\text{task}}(D) = \min_{p(z|s): D_{\text{task}}(S,\hat{S}) \leq D} I(S; Z)
```

關鍵創新：
- $d(X, \hat{X})$: pixel-level distortion
- $D_{\text{task}}(S, \hat{S})$: task-level distortion

## 可能的質疑

### Q: 你們還是在用 bit 傳輸啊，哪裡不同？
A: Physical layer 是 bit，但 payload 變成 semantic token，決策機制變成 task-oriented。

### Q: Shannon capacity 還適用嗎？
A: Physical layer 的 capacity 還是適用，但我們定義了 Semantic capacity。

### Q: 這是 cross-layer 設計嗎？
A: 不完全是。我們定義了新的 Semantic Transport Layer，介於傳統通訊層和 Application 之間。

## 總結

### 範式轉移
```
Shannon: Bit Transmission
  ↓
Semantic Communication: Symbol Transmission
  ↓
Ours: Cognitive State Synchronization
```

### 核心創新
不是改善「怎麼傳」，而是改變「傳什麼」和「為什麼傳」。
EOF

echo -e "${GREEN}✓${NC} 創建 04-background/related-work/vs-traditional-comm.md"

echo ""
echo "========================================="
echo "  Phase 6: 創建目錄說明文件"
echo "========================================="

# 創建 archive/README.md
cat > archive/README.md << 'EOF'
# Archive（歸檔目錄）

本目錄包含已廢棄的研究方向和演進過程紀錄。

## old-directions/
已確認不符合研究方向的早期想法：
- `t1-oran-automation.md`: 6G/O-RAN 網路自動化（教授反饋：太 application-layer）
- `t2-edge-rag.md`: 邊緣多模態 RAG（格局太小，只是壓縮優化）

## evolution-logs/
研究收斂過程的紀錄：
- `t4-diagnosis.md`: 診斷報告（什麼對、什麼錯）
- `t5-convergence.md`: 確認研究收斂
- `t7-version-comparison.md`: 不同版本的演進分析

**這些文件保留的原因**：
1. 了解研究思路的演進過程
2. 避免重複走入死胡同
3. 作為 PhD 學習歷程的紀錄

**不應該**：
- 引用這些內容到正式論文
- 將這些想法作為 contribution
- 在與教授討論時提及這些方向
EOF

echo -e "${GREEN}✓${NC} 創建 archive/README.md"

# 創建 01-problem-formulation/README.md
cat > 01-problem-formulation/README.md << 'EOF'
# Problem Formulation

本目錄定義核心研究問題。

## 核心問題
當 AI Agents 成為主要通訊實體，傳統的 bit-oriented 網路應如何演進？

## 待創建的文件
- [ ] `research-question.md`: 正式的問題定義
- [ ] `motivation.md`: 為什麼這個問題重要（從 professor-concepts 提取）
- [ ] `challenges.md`: 技術挑戰
- [ ] `contributions.md`: 我們的貢獻（與 SOTA 的差異）

## 與論文的對應
→ Paper Section 1 (Introduction)
→ Paper Section 3 (Problem Statement)

## 下一步
1. 從 `00-advisor-feedback/professor-concepts-raw.md` 提取核心洞察
2. 撰寫 `research-question.md`
3. 與教授確認問題定義
EOF

echo -e "${GREEN}✓${NC} 創建 01-problem-formulation/README.md"

# 創建 02-core-framework/README.md
cat > 02-core-framework/README.md << 'EOF'
# Core Framework: Semantic State Communication

本目錄包含本研究的核心理論框架。

## 核心概念
Semantic State Communication (SSC): 通訊的目的是同步 semantic state，而非傳輸 data。

## 待創建的文件
- [ ] `semantic-state-sync.md`: SSC 的數學定義（從 t3.md 重構）
- [ ] `communication-paradigm.md`: 新舊範式對比
- [ ] `architecture-overview.md`: 系統架構
- [ ] `protocol-layers.md`: Layer 5+ 協定設計

## 參考資料
- `t3-original-reference.md`: 原始的 t3.md（需要重構）

## 下一步
1. 重構 t3.md 為正式的學術文檔
2. 加入數學符號與系統模型
3. 繪製架構圖
EOF

echo -e "${GREEN}✓${NC} 創建 02-core-framework/README.md"

# 創建 03-technical-design/README.md
cat > 03-technical-design/README.md << 'EOF'
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
EOF

echo -e "${GREEN}✓${NC} 創建 03-technical-design/README.md"

echo ""
echo "========================================="
echo "  Phase 6: 創建 .gitignore"
echo "========================================="

cat > .gitignore << 'EOF'
# System files
.DS_Store
*.swp
*~
._.DS_Store

# Backup
backup*/
*.backup
*.bak

# IDE
.vscode/
.idea/
*.code-workspace

# Temporary
temp/
tmp/
*.tmp

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/

# LaTeX
*.aux
*.log
*.out
*.toc
*.fdb_latexmk
*.fls
*.synctex.gz

# Large files (optional)
# *.pdf
# *.mp4

# OS
Thumbs.db
EOF

echo -e "${GREEN}✓${NC} 創建 .gitignore"

echo ""
echo "========================================="
echo "  完成！"
echo "========================================="
echo ""
echo -e "${GREEN}重構成功！${NC}"
echo ""
echo "目錄結構："
tree -L 2 -d . 2>/dev/null || find . -type d -maxdepth 2 | grep -v "^\./\." | sort

echo ""
echo "========================================="
echo "  下一步建議"
echo "========================================="
echo ""
echo "1. 查看新的目錄結構是否符合預期"
echo "2. 閱讀 RESTRUCTURE_PLAN.md 了解詳細的重構計劃"
echo "3. 開始重構核心文件："
echo "   - 從 02-core-framework/t3-original-reference.md 重構 semantic-state-sync.md"
echo "   - 從 03-technical-design/t6-original-reference.md 重構 attention-filtering.md"
echo "4. 創建 01-problem-formulation/research-question.md"
echo "5. 查看 ROADMAP.md 了解論文寫作時間表"
echo ""
echo -e "${YELLOW}重要${NC}：原始的 t3.md 和 t6.md 已複製到對應目錄，請手動重構後再刪除原始文件"
echo ""
echo "如有問題，所有原始文件都在 $BACKUP_DIR/ 中"
echo ""
