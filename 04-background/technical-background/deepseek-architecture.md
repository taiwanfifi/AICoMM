這是一份針對 LLM 閱讀優化的 `Markdown` 版本。

**處理策略說明：**
1.  **結構化標記**：使用標準 Markdown 標題層級（#）重建論文結構，使模型能理解上下文層級。
2.  **圖表轉譯**：將所有圖片（架構圖、趨勢圖）轉換為 `[Figure Description]` 區塊，詳細描述圖中數據趨勢、軸線與關鍵組件，確保模型能「看見」圖表內容。
3.  **表格重建**：將截圖中的表格轉換為 Markdown Table，這是 LLM 理解結構化數據最省 Token 且最精準的方式。
4.  **數學公式**：將 OCR 的亂碼公式修復為標準 LaTeX 格式（如 `$\mathcal{L}$`），這是 LLM 理解數學的母語。
5.  **去噪**：移除了頁眉、頁腳、重複的標題與參考文獻（References），保留所有技術細節、Prompt 範本與實驗數據。

---

# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

**DeepSeek-AI**
**Abstract**
We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs of DeepSeek-V3.2 are as follows:
1.  **DeepSeek Sparse Attention (DSA):** We introduce DSA, an efficient attention mechanism that substantially reduces computational complexity while preserving model performance in long-context scenarios.
2.  **Scalable Reinforcement Learning Framework:** By implementing a robust reinforcement learning protocol and scaling post-training compute, DeepSeek-V3.2 performs comparably to GPT-5. Notably, our high-compute variant, **DeepSeek-V3.2-Speciale**, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving gold-medal performance in both the 2025 International Mathematical Olympiad (IMO) and the International Olympiad in Informatics (IOI).
3.  **Large-Scale Agentic Task Synthesis Pipeline:** To integrate reasoning into tool-use scenarios, we developed a novel synthesis pipeline that systematically generates training data at scale. This methodology facilitates scalable agentic post-training, yielding substantial improvements in generalization and instruction-following robustness within complex, interactive environments.

> **[Figure 1 Description]**
> A bar chart comparing DeepSeek-V3.2 variants against GPT-5-High, Claude-4.5-Sonnet, and Gemini-3.0-Pro across various benchmarks.
> *   **Benchmarks:** AIME 2025, HMMT 2025, HLE (text-only), Codeforces, SWE Verified, Terminal Bench 2.0, $\tau^2$ Bench, Tool Decathlon.
> *   **Key Insight:** DeepSeek-V3.2-Speciale (light blue) consistently scores highest or near-highest, matching or exceeding Gemini-3.0-Pro (grey) on math/code tasks. DeepSeek-V3.2-Thinking (dark blue) shows competitive performance against GPT-5-High.
> *   **Highlighted Score:** DeepSeek-V3.2-Speciale achieves Gold Medal level.

## 1. Introduction

The release of reasoning models (DeepSeek-AI, 2025; OpenAI, 2024a) marked a pivotal moment in the evolution of Large Language Models (LLMs). While the open-source community continues to make strides, the performance gap between closed-source and open-source models appears to be widening.

We identify three critical deficiencies limiting open-source models:
1.  **Architectural Efficiency:** Predominant reliance on vanilla attention constrains efficiency for long sequences.
2.  **Resource Allocation:** Insufficient computational investment during post-training (RL) for hard tasks.
3.  **Agentic Capabilities:** Lag in generalization and instruction-following compared to proprietary counterparts.

To address these:
1.  We introduce **DSA (DeepSeek Sparse Attention)**, reducing computational complexity while preserving long-context performance.
2.  We develop a scalable RL protocol, allocating a post-training budget exceeding **10% of pre-training cost**.
3.  We propose a pipeline for **agentic task synthesis**, generating over 1,800 environments and 85,000 prompts to drive RL.

**DeepSeek-V3.2** achieves parity with GPT-5. **DeepSeek-V3.2-Speciale**, a high-compute variant with relaxed length constraints, achieves parity with Gemini-3.0-Pro and gold-medal performance in IOI 2025, ICPC World Final 2025, IMO 2025, and CMO 2025.

## 2. DeepSeek-V3.2 Architecture

### 2.1. DeepSeek Sparse Attention (DSA)
DeepSeek-V3.2 uses the same architecture as DeepSeek-V3.1-Terminus, with the introduction of DSA through continued training.

**Prototype of DSA:**
Consists of a **lightning indexer** and a **fine-grained token selection mechanism**.

The **lightning indexer** computes the index score $I_{t,s}$ between query token $\mathbf{h}_t \in \mathbb{R}^d$ and preceding token $\mathbf{h}_s \in \mathbb{R}^d$:

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left( \mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I \right) \quad (1)
$$

Where $H^I$ is the number of indexer heads. $\mathbf{q}^I$ and $w^I$ are derived from $\mathbf{h}_t$, and $\mathbf{k}^I$ from $\mathbf{h}_s$.
The **fine-grained token selection mechanism** retrieves key-value entries $\{\mathbf{c}_s\}$ corresponding to the top-k index scores. The attention output $\mathbf{u}_t$ is:

$$
\mathbf{u}_t = \text{Attn}(\mathbf{h}_t, \{ \mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:}) \}) \quad (2)
$$

**Instantiate DSA Under MLA:**
DSA is instantiated based on the MQA (Multi-Query Attention) mode of MLA (Multi-Head Latent Attention), where each latent vector (key-value entry) is shared across all query heads.

> **[Figure 2 Description]**
> Architecture diagram of DSA under MLA.
> *   **Input:** Hidden states $\mathbf{h}_t$.
> *   **Path 1 (Core Attention):** Processed via RoPE and concatenated to form Queries.
> *   **Path 2 (Lightning Indexer):** $\mathbf{h}_t$ feeds into a "Lightning Indexer" module (green block).
> *   **Selection:** The Indexer outputs scores to a "Top-k Selector".
> *   **Retrieval:** The Selector picks specific KV pairs from the cache ($\mathbf{c}_t^{KV}$).
> *   **Output:** The selected KV pairs are fed into the Multi-Query Attention block to produce Output Hidden $\mathbf{u}_t$.

**Dense Warm-up Stage:**
Freeze all parameters except the indexer. Align indexer outputs with main attention distribution $p_{t,:}$ using KL-divergence:

$$
\mathcal{L}^I = \sum_t D_{\text{KL}}(p_{t,:} \parallel \text{Softmax}(I_{t,:})) \quad (3)
$$

**Sparse Training Stage:**
Optimize all parameters. Align indexer outputs considering only the selected token set $\mathcal{S}_t$:

$$
\mathcal{L}^I = \sum_t D_{\text{KL}}(p_{t,\mathcal{S}_t} \parallel \text{Softmax}(I_{t,\mathcal{S}_t})) \quad (4)
$$
Indexer input is detached from the computational graph; its signal comes only from $\mathcal{L}^I$.

### 2.2. Parity Evaluation
DeepSeek-V3.2 performs similarly to V3.1-Terminus on standard benchmarks and human preference (ChatbotArena proxy), proving no regression despite sparse attention. On long-context benchmarks (AA-LCR, Fiction.live), V3.2-Exp outperforms or matches V3.1.

### 2.3. Inference Costs
DSA reduces core attention complexity from $O(l^2)$ to $O(lk)$, where $k \ll l$.

> **[Figure 3 Description]**
> Two line charts comparing "Cost Per Million Tokens" (Y-axis) vs "Token Position" (X-axis, up to 128k).
> *   **(a) Prefilling:** V3.1-Terminus cost increases linearly/steeply. V3.2 cost is nearly flat and significantly lower.
> *   **(b) Decoding:** Similar trend; V3.2 maintains low cost while V3.1 scales up with context length.

## 3. Post-Training

Includes specialist distillation and mixed RL training.

**Specialist Distillation:**
We develop specialized models for 6 domains: mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search. Supports both "thinking" (long CoT) and "non-thinking" modes.

**Mixed RL Training:**
Adopt **GRPO (Group Relative Policy Optimization)**. Merges reasoning, agent, and human alignment into one RL stage to prevent catastrophic forgetting.
*   **Rewards:** Rule-based outcome, length penalty, language consistency, and generative reward models.

**DeepSeek-V3.2-Speciale:**
Experimental variant trained exclusively on reasoning data with reduced length penalty and incorporated DeepSeekMath-V2 dataset/rewards.

### 3.1. Scaling GRPO

**GRPO Objective:**

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{old}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right] \quad (5)
$$

**Stabilization Strategies:**

1.  **Unbiased KL Estimate:**
    Corrects the K3 estimator using importance sampling to eliminate systematic estimation errors when $\pi_\theta \ll \pi_{ref}$.
    $$
    D_{KL} = \frac{\pi_\theta}{\pi_{old}} \left( \frac{\pi_{ref}}{\pi_{old}} - \log \frac{\pi_{ref}}{\pi_{old}} - 1 \right) \quad (7)
    $$

2.  **Off-Policy Sequence Masking:**
    Masks negative sequences with significant policy divergence ($M_{i,t}$) to prevent learning from highly off-policy mistakes.
    $$
    M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0, \frac{1}{|o_i|} \sum \log \frac{\pi_{old}}{\pi_\theta} > \delta \\ 1 & \text{otherwise} \end{cases} \quad (9)
    $$

3.  **Keep Routing:** Enforces consistent MoE expert routing between training and inference.
4.  **Keep Sampling Mask:** Preserves top-p/top-k truncation masks during training to match inference action spaces.

### 3.2. Thinking in Tool-Use

**3.2.1. Thinking Context Management**
Replicating DeepSeek-R1's strategy (discarding reasoning) is inefficient for multi-turn tools.
*   **Strategy:** Retain reasoning content throughout tool interactions. Discard historical reasoning *only* when a new **user message** arrives.
*   **Benefit:** Avoids redundant re-reasoning for each tool call.

> **[Figure 4 Description]**
> Diagram of Thinking Retention:
> *   Turn 1.1: User msg -> Thinking 1.1 -> Tool call 1.1.
> *   Turn 1.2: Tool result 1.1 (Thinking 1.1 is KEPT) -> Thinking 1.2 -> Tool call 1.2.
> *   Turn 1.3: Tool result 1.2 (Thinking 1.1 & 1.2 KEPT) -> Thinking 1.3 -> Answer 1.
> *   Turn 2.1: **New User message 2** -> **DISCARD** previous thinking -> Thinking 2.1.

**3.2.2. Cold-Start**
Integrate reasoning and tool-use via prompting.
*   System prompt explicitly asks to reason within `<think>` tags before calling tools.
*   See Appendix B for templates.

**3.2.3. Large-Scale Agentic Tasks**
We employ real-world tools (search APIs, Jupyter) with synthesized prompts.

**Table 1: Agent Tasks Description**
| Task Type | Count | Environment | Prompt Source |
| :--- | :--- | :--- | :--- |
| Code Agent | 24,667 | Real | Extracted |
| Search Agent | 50,275 | Real | Synthesized |
| General Agent | 4,417 | Synthesized | Synthesized |
| Code Interpreter | 5,908 | Real | Extracted |

*   **Search Agent:** Multi-agent pipeline (Question Construction -> Answer Generation -> Verification) to create verifiable QA pairs from web corpora.
*   **Code Agent:** Mined GitHub Issue-PR pairs. Environment setup agent builds executable environments (Docker/Sandbox) to verify patches (Pass-to-Fail/False-to-Positive checks).
*   **General Agent:** Synthetic environment synthesis. An agent builds a sandbox database, tools, tasks, solutions, and verification functions iteratively.

> **[Example Box: Synthesized Task - Trip Planning]**
> A user request for a 3-day trip to Hangzhou with specific constraints (budget, ratings, location).
> **Tool Set:** `get_all_attractions_by_city`, `get_all_hotels`, `get_weather`, etc.

## 4. Evaluation

### 4.1. Main Results

**Table 2: Comparison between DeepSeek-V3.2 and closed/open models**
*Numbers in bold are best in class. Note: $\tau^2$-Bench is average of categories.*

| Benchmark (Metric) | Claude-4.5-Sonnet | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking | MiniMax M2 | DeepSeek-V3.2 Thinking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **English** | | | | | | |
| MMLU-Pro (EM) | 88.2 | 87.5 | **90.1** | 84.6 | 82.0 | **85.0** |
| GPQA Diamond (Pass@1) | 83.4 | 85.7 | **91.9** | **84.5** | 77.7 | 82.4 |
| HLE (Pass@1) | 13.7 | 26.3 | **37.7** | 23.9 | 12.5 | **25.1** |
| **Code** | | | | | | |
| LiveCodeBench (Pass@1-COT) | 64.0 | 84.5 | **90.7** | 82.6 | 83.0 | **83.3** |
| Codeforces (Rating) | 1480 | 2537 | **2708** | - | - | 2386 |
| **Math** | | | | | | |
| AIME 2025 (Pass@1) | 87.0 | 94.6 | **95.0** | **94.5** | 78.3 | 93.1 |
| HMMT Feb 2025 | 79.2 | 88.3 | **97.5** | 89.4 | - | **92.5** |
| IMOAnswerBench | - | 76.0 | **83.3** | **78.6** | - | 78.3 |
| **Code Agent** | | | | | | |
| Terminal Bench 2.0 | **42.8** | 35.2 | **54.2** | 35.7 | 30.0 | **46.4** |
| SWE Verified | **77.2** | 74.9 | 76.2 | 71.3 | 69.4 | **73.1** |
| **Search Agent** | | | | | | |
| BrowseComp | 24.1 | **54.9** | - | -/60.2* | 44.0 | **51.4/67.6\*** |
| **ToolUse** | | | | | | |
| $\tau^2$-Bench | 84.7 | 80.2 | **85.4** | 74.3 | 76.9 | **80.3** |
| MCP-Universe | 46.5 | 47.9 | **50.7** | 35.6 | 29.4 | **45.9** |

*Note: BrowseComp scores with context management are marked with *.*

DeepSeek-V3.2 significantly narrows the gap with frontier models. In Code Agents, it outperforms open-source models on SWE-bench Verified and Terminal Bench.

### 4.2. Results of DeepSeek-V3.2-Speciale

DeepSeek-V3.2-Speciale surpasses Gemini-3.0-Pro on multiple benchmarks by leveraging increased reasoning tokens.

**Table 3: Benchmark performance and efficiency (Accuracy & Output Tokens)**
*Tokens in thousands (k).*

| Benchmark | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking | DeepSeek-V3.2 Thinking | DeepSeek-V3.2 Speciale |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AIME 2025 | 94.6 (13k) | 95.0 (15k) | 94.5 (24k) | 93.1 (16k) | **96.0 (23k)** |
| HMMT Feb 2025 | 88.3 (16k) | 97.5 (16k) | 89.4 (31k) | 92.5 (19k) | **99.2 (27k)** |
| HMMT Nov 2025 | 89.2 (20k) | 93.3 (15k) | 89.2 (29k) | 90.2 (18k) | **94.4 (25k)** |
| IMOAnswerBench | 76.0 (31k) | 83.3 (18k) | 78.6 (37k) | 78.3 (27k) | **84.5 (45k)** |
| LiveCodeBench | 84.5 (13k) | **90.7 (13k)** | 82.6 (29k) | 83.3 (16k) | 88.7 (27k) |
| GPQA Diamond | 85.7 (8k) | **91.9 (8k)** | 84.5 (12k) | 82.4 (7k) | 85.7 (16k) |
| HLE | 26.3 (15k) | **37.7 (15k)** | 23.9 (24k) | 25.1 (21k) | 30.6 (35k) |

**Table 4: Competition Performance (Speciale)**

| Competition | Result | Medal |
| :--- | :--- | :--- |
| **IMO 2025** | Score 35/42 | **Gold** |
| **CMO 2025** | Score 102/126 | **Gold** |
| **IOI 2025** | Score 492/600 | **Gold** |
| **ICPC WF 2025** | 10/12 Solved (Rank 2nd) | **Gold** |

### 4.3. Synthesis Agentic Tasks

Ablation study on synthetic data.
*   **Table 5:** Accuracy on general synthesized tasks: DeepSeek-V3.2-Exp (12%) vs GPT-5-Thinking (62%). Synthetic tasks are challenging.
*   **Generalization:** RL on synthetic data (V3.2-SFT + Synthetic RL) improves performance on real-world benchmarks ($\tau^2$-Bench, MCP-Universe) compared to models trained only on code/search RL.

> **[Figure 5 Description]**
> RL training curves. The model trained with synthetic data (Blue line) shows steady improvement in score over steps, significantly outperforming V3.2-Exp (Green dotted) and V3.2-SFT (Grey dotted).

### 4.4. Context Management of Search Agent

Strategies to handle context overflow (128k limit):
1.  **Summary:** Summarize overflowed trajectory.
2.  **Discard-75%:** Drop first 75% of tool history.
3.  **Discard-all:** Reset context.
4.  **Parallel-fewest-step:** Baseline.

> **[Figure 6 Description]**
> Chart showing Accuracy of BrowseComp (Y-axis 52.5-70.0) vs Real Steps (X-axis).
> *   **Summary:** High accuracy but high cost (steps).
> *   **Discard-all (Green line):** Efficient and scalable, reaching ~67.6 accuracy with fewer steps. Matches Parallel scaling performance.

## 5. Conclusion

DeepSeek-V3.2 bridges the gap between efficiency and advanced reasoning via DSA and scaled RL. **DeepSeek-V3.2-Speciale** achieves milestones in open LLMs with Gold Medals in IMO and IOI. Future work includes scaling pre-training compute to close the knowledge gap and optimizing token efficiency.

---

## Appendices

### A. MHA and MQA Modes of MLA

> **[Figure 7 Description]**
> Illustration of Multi-Head Latent Attention (MLA) modes:
> *   **(a) MHA Mode:** Used for training and prefilling. Queries $\{q_{t,i}\}$ attend to Keys $\{k_{t,i}\}$.
> *   **(b) MQA Mode:** Used for decoding. Key-Value entries are shared across query heads (compressed representation), enabling DSA instantiation.

### B. Cold Start Template

**Table 6: Reasoning System Prompt**
> "You are an expert Python programmer... Please first reason before giving the final answer. The reasoning process enclosed within `<think> </think>`. The final answer is output after the `</think>` tag."

**Table 7: Agent System Prompt**
> "Use Python interpreter tool to execute Python code... Important: ALWAYS adhere to this exact format for tool use: `{TOOLCALL-FORMAT}`."

**Table 8: Thinking-in-Tool-Use Prompt**
> "You may use the Python tool **multiple times** during your reasoning, a.k.a in `<think></think>`... Call the Python tool early... Do NOT invoke any tools in your presented final solution steps."

### C. Non-thinking DeepSeek-V3.2 Agentic Evaluation

**Table 9: Non-thinking vs Thinking Mode**
| Benchmark | Non-thinking | Thinking |
| :--- | :---: | :---: |
| Terminal Bench 2.0 | 37.1 | 46.4 |
| SWE Verified | 72.1 | 73.1 |
| $\tau^2$-bench | 77.2 | 80.3 |
| MCP-Universe | 38.6 | 45.9 |

Thinking mode consistently outperforms non-thinking mode.

### D. Evaluation Method of IOI, ICPC, IMO, and CMO

*   **IOI/ICPC:** Max 128k context. Sample 500 (IOI) / 32 (ICPC) candidates. Filter invalid/refusals. Select submission based on longest thinking trace.
*   **IMO/CMO:** Generate-verify-refine loop. Model improves solution until perfect self-eval or max revision cap.