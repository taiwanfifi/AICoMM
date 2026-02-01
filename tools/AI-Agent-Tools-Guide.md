# AI Agent 工具使用指南

本指南涵蓋三個 AI coding agent 工具，搭配本地 Ollama 模型使用。

---

## 環境狀態

| 工具 | 版本 | 安裝方式 | 狀態 |
|------|------|---------|------|
| **aider** | v0.86.1 | pip (conda env: llama.cpp) | 已測試通過 |
| **SWE-agent** | v1.1.0 | git clone + pip editable (`/Users/william/SWE-agent/`) | 已安裝 |
| **OpenHands** | v1.2 | Docker images | 已安裝 |
| **Ollama** | - | brew | 已安裝，23 個模型 |

### 推薦模型

| 模型 | 大小 | 用途 |
|------|------|------|
| `qwen2.5-coder:32b` | 19GB | 最強 coding 模型（推薦） |
| `qwen2.5-coder:14b` | 9GB | 中等，速度較快 |
| `qwen3:32b` | 20GB | 通用推理 |
| `deepseek-r1:14b` | 9GB | 推理鏈模型 |

---

## 1. aider（已測試通過）

### 是什麼

aider 是 terminal-based 的 AI pair programming 工具。你給它檔案，用自然語言下指令，它直接幫你改 code。

### 基本用法

```bash
# 進入你要改的專案目錄
cd /Users/william/Downloads/AI-Comm

# 啟動 aider，指定模型和要編輯的檔案
aider --model ollama/qwen2.5-coder:32b 01-problem-formulation/research-question.md
```

啟動後會進入互動模式，直接打字就能下指令。

### 常用指令（在 aider 互動模式內）

```
/add <file>          # 加入更多檔案讓 aider 看到
/drop <file>         # 移除檔案
/ask <question>      # 只問問題，不改檔案
/diff                # 顯示目前改動
/undo                # 復原上次改動
/quit                # 離開
```

### 實際範例

**範例 1：改善一篇研究文件**
```bash
cd /Users/william/Downloads/AI-Comm
aider --model ollama/qwen2.5-coder:32b 01-problem-formulation/motivation.md
```
進入後輸入：
```
請用更精確的學術語言重寫第一段，強調 semantic state synchronization 而非 semantic communication
```

**範例 2：同時編輯多個相關檔案**
```bash
aider --model ollama/qwen2.5-coder:32b \
  02-core-framework/semantic-state-sync.md \
  02-core-framework/semantic-token-definition.md
```
進入後輸入：
```
確保兩個檔案中 semantic token 的定義完全一致，如果有矛盾以 semantic-state-sync.md 為準
```

**範例 3：讓 AI 只讀不改（純問問題）**
```bash
aider --model ollama/qwen2.5-coder:32b 03-technical-design/attention-filtering.md
```
進入後輸入：
```
/ask 這個 attention filtering 機制跟原始 DeepSeek DSA 的主要差異是什麼？
```

### 重要參數

```bash
# 不自動 git commit（推薦）
aider --model ollama/qwen2.5-coder:32b --no-auto-commits <files>

# 不需要 git repo
aider --model ollama/qwen2.5-coder:32b --no-git <files>

# 用較小模型（跑更快）
aider --model ollama/qwen2.5-coder:14b <files>

# 唯讀模式（只問不改）
aider --model ollama/qwen2.5-coder:32b --read <files>
```

---

## 2. SWE-agent

### 是什麼

SWE-agent 是自主型 coding agent。你描述一個問題（bug、feature），它會自己瀏覽程式碼、思考、做修改、測試。比 aider 更自主，適合給一個任務然後讓它自己跑。

### 前置條件

- Docker 必須在跑（OrbStack 要開著）
- 安裝位置：`/Users/william/SWE-agent/`

### 基本用法

```bash
# 對一個 GitHub issue 自動修 bug
sweagent run \
  --model ollama/qwen2.5-coder:32b \
  --problem_statement "Add type hints to all functions in calc.py" \
  --repo /path/to/your/repo
```

### 實際範例

**範例 1：讓 agent 自主分析並改善檔案**
```bash
sweagent run \
  --model ollama/qwen2.5-coder:32b \
  --problem_statement "Review semantic-state-sync.md and fix any inconsistencies in mathematical notation. All semantic states should use S_t notation and deltas should use the Δ symbol." \
  --repo /Users/william/Downloads/AI-Comm
```

**範例 2：對 GitHub repo 的 issue 自動修復**
```bash
sweagent run \
  --model ollama/qwen2.5-coder:32b \
  --problem_statement.github_url https://github.com/YOUR/REPO/issues/1
```

**範例 3：互動模式（像 shell 一樣一步步操作）**
```bash
sweagent run-replay \
  --traj_path <trajectory_file>
```

### SWE-agent vs aider 的差異

| | aider | SWE-agent |
|---|---|---|
| 互動方式 | 對話式，你指揮 | 自主式，給任務就跑 |
| 適合場景 | 即時編輯、pair programming | Bug 修復、自動化任務 |
| 需要 Docker | 不需要 | 需要 |
| 一次改多少 | 你控制 | 它自己決定 |

---

## 3. OpenHands

### 是什麼

OpenHands（前身 OpenDevin）是 Web GUI 的自主 coding agent。有完整的瀏覽器介面，agent 可以寫 code、跑 terminal、甚至操作瀏覽器。功能最完整但也最重。

### 前置條件

- Docker 必須在跑（OrbStack 要開著）
- 兩個 Docker images 已下載完成

### 啟動方式

```bash
docker run -it --rm \
  -e AGENT_SERVER_IMAGE_REPOSITORY=docker.openhands.dev/openhands/runtime \
  -e AGENT_SERVER_IMAGE_TAG=1.2-nikolaik \
  -e LOG_ALL_EVENTS=true \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.openhands:/.openhands \
  -p 3000:3000 \
  --add-host host.docker.internal:host-gateway \
  --name openhands-app \
  docker.openhands.dev/openhands/openhands:1.2
```

啟動後打開瀏覽器：**http://localhost:3000**

### 設定 Ollama 連線

1. 開啟 http://localhost:3000
2. 點右上角 **Settings**（齒輪圖示）
3. 到 **LLM** tab
4. 打開 **Advanced** 開關
5. 設定：
   - **Model**: `ollama/qwen2.5-coder:32b`
   - **Base URL**: `http://host.docker.internal:11434`
   - **API Key**: 隨便打（例如 `local-key`）— 本地模型不需要真的 key 但欄位不能空
6. 儲存

### 實際範例

**範例 1：在 Web GUI 中下指令**

啟動後在聊天框輸入：
```
請閱讀 /workspace 裡的所有 markdown 檔案，然後幫我整理出一個研究摘要
```

**範例 2：掛載本地專案目錄**

```bash
docker run -it --rm \
  -e AGENT_SERVER_IMAGE_REPOSITORY=docker.openhands.dev/openhands/runtime \
  -e AGENT_SERVER_IMAGE_TAG=1.2-nikolaik \
  -e LOG_ALL_EVENTS=true \
  -e SANDBOX_USER_ID=$(id -u) \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.openhands:/.openhands \
  -v /Users/william/Downloads/AI-Comm:/opt/workspace_base \
  -p 3000:3000 \
  --add-host host.docker.internal:host-gateway \
  --name openhands-app \
  docker.openhands.dev/openhands/openhands:1.2
```

這樣 OpenHands 就能讀寫你的研究檔案。

### 停止 OpenHands

```bash
docker stop openhands-app
```

### 三個工具的比較

| | aider | SWE-agent | OpenHands |
|---|---|---|---|
| **介面** | Terminal 對話 | Terminal 自主跑 | Web GUI |
| **自主程度** | 低（你指揮） | 高（給任務跑） | 高（給任務跑） |
| **需要 Docker** | 不需要 | 需要 | 需要 |
| **適合場景** | 日常編輯、快速改檔 | Bug 修復、code review | 複雜任務、需要瀏覽器的場景 |
| **學習成本** | 最低 | 中等 | 最低（有 GUI） |
| **資源消耗** | 最少 | 中等 | 最多（Docker container） |

---

## 快速開始 Checklist

### 用 aider（最簡單，建議先從這開始）
```bash
# 1. 確認 Ollama 在跑
ollama list

# 2. 進專案目錄
cd /Users/william/Downloads/AI-Comm

# 3. 啟動
aider --model ollama/qwen2.5-coder:32b --no-auto-commits 02-core-framework/semantic-state-sync.md
```

### 用 SWE-agent
```bash
# 1. 確認 Docker 在跑（OrbStack 要開）
docker ps

# 2. 跑任務
sweagent run \
  --model ollama/qwen2.5-coder:32b \
  --problem_statement "你的任務描述" \
  --repo /Users/william/Downloads/AI-Comm
```

### 用 OpenHands
```bash
# 1. 確認 Docker 在跑
docker ps

# 2. 啟動（含掛載專案目錄）
docker run -it --rm \
  -e AGENT_SERVER_IMAGE_REPOSITORY=docker.openhands.dev/openhands/runtime \
  -e AGENT_SERVER_IMAGE_TAG=1.2-nikolaik \
  -e LOG_ALL_EVENTS=true \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.openhands:/.openhands \
  -v /Users/william/Downloads/AI-Comm:/opt/workspace_base \
  -p 3000:3000 \
  --add-host host.docker.internal:host-gateway \
  --name openhands-app \
  docker.openhands.dev/openhands/openhands:1.2

# 3. 打開瀏覽器 http://localhost:3000
# 4. Settings → LLM → 設定 Ollama
```

---

## 疑難排解

### Ollama 沒回應
```bash
# 確認 Ollama 在跑
curl http://localhost:11434/api/tags

# 如果沒回應，重啟
ollama serve
```

### aider 連不到 Ollama
```bash
# 設定環境變數
export OLLAMA_API_BASE=http://localhost:11434

# 然後重新啟動 aider
aider --model ollama/qwen2.5-coder:32b <files>
```

### OpenHands Docker 連不到 Ollama
確保用了 `--add-host host.docker.internal:host-gateway`，
然後 Base URL 用 `http://host.docker.internal:11434`（不是 localhost）。

### SWE-agent Docker 問題
```bash
# 確認 Docker daemon 在跑
docker info

# 如果 OrbStack 沒開，手動開啟
open -a OrbStack
```

### 模型跑太慢
換用較小的模型：
```bash
# 14b 版本（約快 2 倍）
aider --model ollama/qwen2.5-coder:14b <files>

# 或用 qwen3:4b（最快但品質較差）
aider --model ollama/qwen3:4b <files>
```
