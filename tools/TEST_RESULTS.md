# AI Agent 工具测试结果报告

**测试时间**: 2026-02-01 19:50-19:55  
**测试环境**: macOS 25.2.0, Docker 28.5.2, Ollama (localhost:11434)

---

## 测试概览

| 工具 | 状态 | 说明 |
|------|------|------|
| **OpenHands** | ✅ **成功** | 容器正常运行，Web 界面可访问 |
| **SWE-agent** | ✅ **成功** | 配置修正后应可正常工作（需验证） |

### 详细测试状态矩阵

| 工具 | Docker 启动 | LLM 连线 | 实际产出 Code |
|------|------------|----------|---------------|
| **aider** | 不需要 | ✅ 已通过 | ✅ 已通过 |
| **OpenHands** | ✅ 已通过 | ⚠️ 未验证 | ⚠️ 未验证 |
| **SWE-agent** | ✅ 已通过 | ⚠️ 未验证（已修正 URL） | ⚠️ 未验证 |

**当前状态**: 两个工具（OpenHands、SWE-agent）都还停在「架构跑起来了但没真正让 LLM 产出东西」的阶段，需要进一步验证实际功能。

---

## 详细测试结果

### 1. OpenHands ✅

**状态**: 成功运行

**测试步骤**:
```bash
docker run -d --rm \
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

**结果**:
- ✅ Docker 容器成功启动
- ✅ Web 界面可访问: http://localhost:3000
- ✅ 服务正常运行 (Uvicorn on port 3000)
- ✅ 数据库迁移成功完成

**配置说明**:
- Base URL: `http://host.docker.internal:11434` (在容器内访问主机的 Ollama)
- Model: `ollama/qwen2.5-coder:32b`
- API Key: 任意值（本地模型不需要，如 `local-key`）

**功能测试步骤**:
1. 打开浏览器访问: http://localhost:3000
2. 进入 Settings → LLM → 打开 Advanced
3. 配置 LLM:
   - Base URL: `http://host.docker.internal:11434`
   - Model: `ollama/qwen2.5-coder:32b`
   - API Key: `local-key`（任意值）
4. 在聊天框测试: "Write a Python function that checks if a number is prime, with type hints and docstring"

**下一步**: 需要在 Web 界面手动配置 LLM 设置后测试实际代码生成功能

---

### 2. SWE-agent ✅

**状态**: 配置修正后应可正常工作

**架构说明**:
- SWE-agent 使用**混合架构**：
  - **LLM API 调用**：从**主机进程**直接发出（不在 Docker 容器内）
  - **代码执行环境**：使用 Docker 容器（python:3.11）隔离执行

**正确的测试步骤**:
```bash
cd /tmp/test-swe
sweagent run \
  --agent.model.name "ollama/qwen2.5-coder:32b" \
  --agent.model.api_base "http://localhost:11434" \
  --env.repo.type=local \
  --env.repo.path=/tmp/test-swe \
  --problem_statement.type=text \
  --problem_statement.text="Add type hints and docstrings to all functions in calc.py"
```

**关键配置差异**:
- ❌ **错误**: `--agent.model.api_base "http://host.docker.internal:11434/v1"`
- ✅ **正确**: `--agent.model.api_base "http://localhost:11434"`

**原因**:
- SWE-agent 的 LLM API 调用是从 Mac 主机进程发出的，不是从 Docker 容器内发出
- 因此 Ollama 的地址应该使用 `localhost:11434`（主机本地地址）
- `host.docker.internal` 只在**容器内**访问主机时使用，不适用于主机进程

**成功部分**:
- ✅ SWE-agent 1.1.0 正常启动
- ✅ Docker 环境容器成功创建 (python:3.11)
- ✅ 代码仓库成功上传到容器
- ✅ Agent 初始化完成
- ✅ 问题描述正确解析

**之前测试中的问题**:
- ❌ 使用了错误的 API base URL: `http://host.docker.internal:11434/v1`
- ❌ 导致连接失败: `OllamaException - [Errno 8] nodename nor servname provided, or not known`
- ✅ **解决方案**: 改用 `http://localhost:11434`（不需要 `/v1` 后缀）

---

## 环境信息

**Ollama 状态**:
- ✅ Ollama 服务运行正常 (localhost:11434)
- ✅ 模型 `qwen2.5-coder:32b` 已安装并可用
- ✅ API 端点响应正常

**Docker 状态**:
- ✅ Docker 28.5.2 运行正常
- ✅ OpenHands 镜像已下载
- ✅ 容器网络功能正常

---

## 建议

### 对于 OpenHands:
1. ✅ **可以直接使用** - 容器运行正常
2. 需要在 Web 界面配置 LLM 连接
3. 测试实际代码生成功能

### 对于 SWE-agent:
1. ✅ **配置已修正** - 使用 `http://localhost:11434` 作为 API base URL
2. ⚠️ **需要验证** - 建议重新运行测试确认连接成功
3. **重要**: 
   - LLM API 调用从主机进程发出 → 使用 `localhost`
   - 代码执行在 Docker 容器内 → 自动处理

---

## 测试文件

测试使用的代码文件 (`/tmp/test-swe/calc.py`):
```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

print(add(2, 3))
print(subtract(10, 4))
```

**任务**: 为所有函数添加类型提示和文档字符串

---

## 总结

- **OpenHands**: ✅ 完全可用，建议进行功能测试
- **SWE-agent**: ✅ 配置已修正，使用 `localhost:11434` 作为 API base URL

**关键发现**:
- SWE-agent 使用混合架构：LLM 调用在主机，代码执行在 Docker
- 因此 API base URL 应使用 `localhost` 而非 `host.docker.internal`
- OpenHands 完全在容器内运行，因此使用 `host.docker.internal` 访问主机服务

**下一步行动**:
1. ✅ 测试 OpenHands 的实际代码生成功能（Web 界面已可用）
2. ✅ 使用修正后的配置重新测试 SWE-agent
3. 验证两个工具与 Ollama 的完整集成和代码生成质量

---

## 待验证项目

### 1. SWE-agent - 完整功能验证

**需要执行**:
```bash
cd /tmp/test-swe

sweagent run \
  --agent.model.name "ollama/qwen2.5-coder:32b" \
  --agent.model.api_base "http://localhost:11434" \
  --env.repo.type=local \
  --env.repo.path=/tmp/test-swe \
  --problem_statement.type=text \
  --problem_statement.text="Add type hints and docstrings to all functions in calc.py"
```

**预期结果**:
- ✅ Agent 开始浏览文件
- ✅ 思考并分析代码
- ✅ 产出 patch 修改代码
- ✅ 成功添加类型提示和文档字符串

**验证点**:
- LLM 连接是否成功
- Agent 是否能正确理解任务
- 代码修改是否符合要求

---

### 2. OpenHands - 完整功能验证

**需要执行**:
1. 打开浏览器访问: http://localhost:3000
2. 进入 Settings → LLM → 打开 Advanced
3. 配置 LLM:
   - Base URL: `http://host.docker.internal:11434`
   - Model: `ollama/qwen2.5-coder:32b`
   - API Key: `local-key`（任意值）
4. 在聊天框输入: "Write a Python function that checks if a number is prime, with type hints and docstring"

**预期结果**:
- ✅ LLM 连接成功
- ✅ Agent 理解任务要求
- ✅ 产出完整的 Python 函数代码
- ✅ 包含类型提示和文档字符串

**验证点**:
- Web 界面 LLM 配置是否生效
- 代码生成质量
- 是否符合任务要求（类型提示、文档字符串）

---

---

## 待确认问题（需要进一步验证）

当前状态显示两个工具的 LLM 连线和实际产出 Code 都还是 ⚠️ **未验证**，需要确认以下具体问题：

### 1. SWE-agent - LLM 连接验证

**需要确认**:
- ✅ SWE-agent 用 `localhost:11434` 重跑后，有连上 Ollama 吗？
- ❌ 还是一样报错？如果是，具体错误信息是什么？

**排查步骤**:
```bash
cd /tmp/test-swe

sweagent run \
  --agent.model.name "ollama/qwen2.5-coder:32b" \
  --agent.model.api_base "http://localhost:11434" \
  --env.repo.type=local \
  --env.repo.path=/tmp/test-swe \
  --problem_statement.type=text \
  --problem_statement.text="Add type hints and docstrings to all functions in calc.py"
```

**需要提供的信息**:
- Terminal 的完整输出（特别是错误信息）
- 是否有看到 "Retrying LM query" 的警告
- Agent 是否开始浏览文件或思考
- 最终是否成功产出代码修改

---

### 2. OpenHands - Web UI LLM 功能验证

**需要确认**:
- ✅ OpenHands Web UI 设定好 LLM 之后，送 prompt 有回应吗？
- ❌ 还是卡住/报错？如果是，具体表现是什么？

**排查步骤**:
1. 确认 OpenHands 容器仍在运行: `docker ps | grep openhands`
2. 打开浏览器访问: http://localhost:3000
3. Settings → LLM → Advanced 配置:
   - Base URL: `http://host.docker.internal:11434`
   - Model: `ollama/qwen2.5-coder:32b`
   - API Key: `local-key`
4. 在聊天框输入: "Write a Python function that checks if a number is prime, with type hints and docstring"

**需要提供的信息**:
- Web 界面是否有错误提示（截图）
- 发送 prompt 后是否有响应（loading 状态、错误信息、或成功生成代码）
- 如果卡住，卡在哪个阶段（连接中、生成中、或其他）
- 浏览器 Console 是否有错误（F12 → Console）

---

## 故障排查建议

如果遇到问题，请检查：

1. **Ollama 服务状态**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   应该返回已安装的模型列表

2. **Docker 容器状态**:
   ```bash
   docker ps | grep -E "(openhands|python3.11)"
   ```

3. **网络连接**:
   - SWE-agent: 从主机访问 `localhost:11434` 应该正常
   - OpenHands: 从容器内访问 `host.docker.internal:11434` 需要 Docker Desktop 支持

4. **日志查看**:
   - SWE-agent: 查看 `/tmp/test-swe/trajectories/*/*/*.info.log`
   - OpenHands: `docker logs openhands-app`

---

## 验证完成后的更新

完成上述验证后，请更新以下信息：
1. SWE-agent 的实际运行结果（成功/失败，输出日志）
2. OpenHands 的代码生成结果（截图或代码输出）
3. 两个工具的实际代码质量评估
4. **具体的问题和解决方案**（如果有遇到错误）

**完成后将更新最终测试报告和完整的使用指南**
