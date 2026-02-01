這是一份針對 LLM（Large Language Model）閱讀優化的 **Markdown** 版本。我已經進行了以下處理以確保模型能最高效地吸收資訊：

1.  **格式重整**：使用標準 Markdown 標題層級（# H1, ## H2, ### H3）建立清晰的知識樹結構。
2.  **OCR 修正**：修復了斷詞（如 "op- timization" → "optimization"）、刪除了頁眉/頁尾/版權聲明/頁碼等無意義的噪聲 Token。
3.  **圖表轉譯**：將 PDF 中的視覺圖表（Figures）與表格（Tables）轉換為 `[Visual Description]` 或標準 Markdown 表格。這能讓 LLM 理解圖表中的邏輯流、架構層級和數據對比，而不會遺失視覺資訊。
4.  **去冗餘**：移除了 Reference 列表與作者簡介，僅保留核心學術內容。

---

# Deploying Foundation Model Powered Agent Services: A Survey

**Authors:** Wenchao Xu, Jinyu Chen, Peirong Zheng, Xiaoquan Yi, Tianyi Tian, Wenhui Zhu, Quan Wan, Haozhao Wang, Yunfeng Fan, Qinliang Su, Xuemin Shen (Fellow, IEEE)

**Abstract:**
Foundation model (FM) powered agent services are regarded as a promising solution to develop intelligent and personalized applications for advancing toward Artificial General Intelligence (AGI). To achieve high reliability and scalability in deploying these agent services, it is essential to collaboratively optimize computational and communication resources, thereby ensuring effective resource allocation and seamless service delivery. In pursuit of this vision, this paper proposes a unified framework aimed at providing a comprehensive survey on deploying FM-based agent services across heterogeneous devices, with the emphasis on the integration of model and resource optimization to establish a robust infrastructure for these services. Particularly, this paper begins with exploring various low-level optimization strategies during inference and studies approaches that enhance system scalability, such as parallelism techniques and resource scaling methods. The paper then discusses several prominent FMs and investigates research efforts focused on inference acceleration, including techniques such as model compression and token reduction. Moreover, the paper also investigates critical components for constructing agent services and highlights notable intelligent applications. Finally, the paper presents potential research directions for developing real-time agent services with high Quality of Service (QoS).

**Keywords:** Foundation Model, AI Agent, Cloud/Edge Computing, Serving System, Distributed System, AGI.

---

## I. INTRODUCTION

The rapid advancement of artificial intelligence (AI) has positioned foundation models (FMs) as a cornerstone of innovation, driving progress in various fields such as natural language processing, computer vision, and autonomous systems. These models incubate numerous applications from automated text generation to advanced multi-modal question answering and autonomous robot services. Popular FMs (GPT, Llama, ViT, CLIP) push boundaries by processing large volumes of data across formats.

However, traditional FMs are often confined to Q&A services based on pre-existing knowledge. FM-powered agents enhance this by incorporating dynamic memory management, long-term task planning, advanced tools, and external environment interactions. For example, agents can call APIs for real-time data to improve reliability and personalization.

Developing a serving system with low latency, high reliability, elasticity, and minimal resource consumption is crucial. Challenges include:
1.  **Fluctuating query loads:** Requires elastic scaling.
2.  **Huge parameter space:** Challenges storage and inference overhead, necessitating model compression and parallelism.
3.  **Diverse service requirements:** Trade-offs between latency and accuracy require dynamic resource allocation.
4.  **Complex agent tasks:** Requires management of large-scale memory and multi-agent collaboration.

To address these, this survey proposes a unified framework (see Figure 1).

> **[Visual Description: Figure 1 - The framework of FM-powered agent services]**
> The framework is a layered architecture composed of five distinct layers, from top to bottom:
> 1.  **Application Layer (§ 2):** Showcases end-user applications like ChatGPT, Copilot, AI Pin, Hippocratic AI, and DeepSeek.
> 2.  **Agent Layer (§ 3):** Contains core agent capabilities: Multi-agent coordination, Planning, Memory, and Tool Use (RAG/Use).
> 3.  **Model Layer (§ 4):** Focuses on the models themselves (GPT, Llama, Vicuna) and optimization techniques:
>     *   **Model Compression:** Pruning, Quantization, Distillation.
>     *   **Token Reduction:** Pruning, Merging, Summary.
> 4.  **Resource Layer (§ 5):** Manages infrastructure via Parallelism and Resource Scaling.
> 5.  **Execution Layer (§ 6):** The physical hardware layer (Edge vs. Cloud) utilizing Computation Optimization, Input/Output Optimization, and Communication Optimization.

> **[Visual Description: Figure 2 - Survey Structure]**
> A flow chart mapping the paper's sections to the layers in Figure 1:
> *   **Section II:** Application (FM & AI Agent Applications).
> *   **Section III:** AI Agent (Multi-agent Framework, Planning, Memory, Tool Use).
> *   **Section IV:** Foundation Model & Compression (Current Models, Compression, Adaptation, Token Adaptation).
> *   **Section V:** Resource Allocation & Parallelism (Edge/Cloud Scaling, Parallelism types).
> *   **Section VI:** Execution Optimization (Computation: FPGA/ASIC/IMC/CPU/GPU, Memory, Communication, Integrated Frameworks).
> *   **Section VII & VIII:** Lessons Learned & Conclusion.
> The diagram indicates a span from "Software" (top sections) to "Hardware" (bottom sections).

---

## II. APPLICATION LAYER

### A. Foundation Model Applications
Commercial generative AI is categorized into:
1.  **Text-generating/Multimodal:** ChatGPT (OpenAI), ERNIE Bot (Baidu).
2.  **Code-generating:** Github Copilot (based on Codex/GPT-3).
3.  **Image-generating:** Midjourney.
4.  **Video-generating:** Sora (OpenAI), Runway Gen-2.

### B. AI Agent Applications
AI agents are considered a principal avenue toward AGI.
*   **AutoGPT:** Automatically optimizes GPT's hyperparameters and executes tasks autonomously.
*   **Generative Agents (Smallville):** Simulated environment with 25 agents exhibiting social interactions.
*   **VOYAGER:** An embodied agent in Minecraft that continuously explores and learns skills.
*   **ChatDev:** A virtual software company operated by multiple intelligent agents using a waterfall model.
*   **Recent Commercial Agents:** OpenAI's Operator, Deep Research (based on o3), Google's Project Astra/Mariner/Jules, GLM-PC (ZhiPu), Microsoft Dynamics 365 agents, and Manus (Monica).

### C. Lessons Learned
*   **Multimodal capabilities:** Text/code/image are mature; video requires more research in temporal modeling.
*   **Operational cost:** Token-level inference costs limit adoption. Hybrid architectures (combining large/small models) and infrastructure optimization are key mitigation strategies.

---

## III. AI AGENT

> **[Visual Description: Figure 3 - LLM Agent framework]**
> The LLM acts as the "Brain" or Central Nervous System. It connects to three types of agent specializations:
> 1.  **Conversational Agent**
> 2.  **Assistant Agent**
> 3.  **UserProxAgent / Tool Agent**
> The diagram emphasizes that the LLM processes information, makes decisions, and generates actions.

### A. Multi-agent Framework
Leverages multiple LLMs working collaboratively.
*   **Coordination:** Can be centralized (supervisor allocates tasks) or decentralized (autonomous collaboration).
*   **Examples:** AWS Bedrock (hierarchical), LLM-Co, CLIP (embodied settings), Semantic In-Context Learning (ICL).

### B. Planning
Agents break down large tasks into sub-goals.
*   **Characteristics:** Hierarchical approach (strategic to operational), Parallel processing, Dynamic adjustment based on feedback.
*   **Key Works:**
    *   **MindAgent:** Gaming interactions.
    *   **DEPS:** Integration of planning with LLMs for open-world environments (Minecraft).
    *   **LLM-MCTS:** Combines LLM with Monte Carlo Tree Search.
    *   **ReCon & MPC:** Improvement of planning without fine-tuning.

> **[Visual Description: Figure 4 - LLM for planning]**
> Shows a cycle: User Query -> PlanAgent (Outer Loop) -> Split into N sub-tasks -> Inner Loop (Iterative Refinement) -> Agent Dispatch & Tool Retrieval -> ToolAgent -> Feedback/Reflection -> Modification/Deletion/Addition of tasks -> Final Result.

### C. Long/Short Term Memory
Memory management is crucial for context retention and resource efficiency.
*   **Short-term:** Processes current context (limited window).
*   **Long-term:** Retains info over time (user preferences, facts) using external storage or vector databases (RAG).
*   **Key Works:**
    *   **REMEMBERER:** Integrates long-term experience memory updated via RL.
    *   **MoT (Memory-of-Thought):** Self-improvement via external memory.
    *   **RAISE:** Scratchpad and examples for continuity.

> **[Visual Description: Figure 5 - LLM Agent of Memory]**
> A flowchart showing the memory process:
> User message -> NL understanding -> Status tracking (In-memory/Vector database) -> Conversation strategy (using Adapters/Policies like Slot filling/Intent detection) -> Action execution -> Agent reply. It highlights dynamic maintenance of historical sequences.

### D. Tool Use
Enables agents to call external APIs (calculators, search, proprietary databases).
*   **Benefits:** Access to current info, enhanced functions, interactivity with proprietary systems, dynamic adaptation.
*   **Key Works:**
    *   **Toolformer:** Self-supervised learning to use tools.
    *   **Gpt4tools:** Low-Rank Adaptation for tool usage.
    *   **Gorilla & ToolLLM:** Fine-tuning for API calls.
    *   **NESTFUL & Reverse Chain:** Evaluation and implementation of multi-step API planning.

> **[Visual Description: Figure 6 - LLM agent of using tools]**
> Shows the interaction between the "Environment" (Perception/Inputs) and the "Brain" (LLM).
> The Brain contains: Storage (Memory/Knowledge) and Decision Making (Planning/Reasoning).
> The Agent takes Action (Text, Tools/API Calls, Embodiment) which affects the Environment.

### E. Lesson Learned
*   **Synergistic architecture:** Effective agents require the integration of Planning, Memory, and Tool-use modules around the LLM core.

---

## IV. FOUNDATION MODEL & COMPRESSION

### A. Current Foundation Model
*   **Timeline (Figure 7 Summary):**
    *   ~2020: T5, GPT-3, mT5, Bert.
    *   2021: PanGu-α, ERNIE 3.0, Gopher.
    *   2022: LaMDA, PaLM, OPT, BLOOM, ChatGPT.
    *   2023: LLaMA series, Qwen, Falcon, Mistral, Gemini.
    *   2024: Llama-3, Phi-3, Yi.

*   **Large Language Models (LLMs):**
    *   **Encoder-Decoder:** T5, mT5.
    *   **Decoder-only:** GPT-3 (few-shot), PanGu-α, Gopher, LaMDA (dialogue), PaLM (scaling), LLaMA (open-source efficiency), Llama 2/3 (grouped-query attention), Qwen (strong open-source performance), DeepSeek-r1 (Reasoning/RL).
    *   **Comparison (Table II Data Summary):**
        *   Models range from 60M to 500B+ parameters.
        *   Architectures mostly use Decoder-only (GPT-style) or Encoder-Decoder (T5).
        *   Attention types: Multi-head, Multi-query, Grouped-query.
        *   Activations: ReLU, GELU, SwiGLU.
        *   Position Embeddings: Relative, Sinusoidal, RoPE, ALiBi.

*   **Multimodal Models (MLLMs):**
    *   **Alignment:** Connecting Vision Models (VM) to LLMs.
    *   **Methods:** Flamingo (Perceiver Resampler), miniGPT-4/PandaGPT (Linear layer), CogVLM (Trainable visual experts), LLaVA.
    *   **Universal:** NExT-GPT (Any-to-any), OneLLM, Gemini 1.5 (MoE, long context).

> **[Visual Description: Figure 8 - Radar Chart Benchmark]**
> A comparison of models (DeepSeek-R1, OpenAI-o1, Qwen-2.5, Llama-3.1, etc.) across tasks:
> *   **General:** MMLU Pro.
> *   **Math/Science:** MATH-500, GPQA Diamond, AIME 2024.
> *   **Coding:** LiveCodeBench, SWE-bench.
> *   *Result:* DeepSeek-r1 and OpenAI-o1 show exceptional performance, covering the largest area on the chart.

### B. Model Compression

> **[Visual Description: Figure 9 - Model Compression Methods]**
> Illustrates three techniques:
> 1.  **Pruning:** Removing redundant connections/parameters (Scissors cutting connections).
> 2.  **Quantization:** Converting high-precision numbers (Floating Point, e.g., -1.4, 0.4) to lower precision (Binary/Fixed Point, e.g., -1, 0, 1) to save space.
> 3.  **Knowledge Distillation:** A large "Teacher" model transfers knowledge (logits/soft targets) to a smaller "Student" model.

1.  **Pruning:** Removing weights or neurons.
    *   **Structured vs. Unstructured:** Wanda, LLM-Pruner, FLAP, Sheared Llama.
    *   **Table III (Pruning Methods):** Lists challenges (e.g., "Retraining required") and solutions (e.g., Wanda uses weight magnitude * input activation).
2.  **Quantization:** Reducing bit-width (e.g., W8A8, W4A16).
    *   **Techniques:** SmoothQuant (outlier smoothing), AWQ (activation-aware), GPTQ, QLoRA.
    *   **Table IV (Quantization):** Categorizes into W8A8 (Weights & Activations 8-bit) and Low-bit weight-only. Mentions SmoothQuant, AWQ, Omniquant.
3.  **Knowledge Distillation:** Teacher-student training.
    *   **Methods:** Distilling step-by-step (Chain-of-Thought), Zephyr (Direct Preference Optimization), MiniLLM.

### C. Model Adaptation
Dynamically selecting or adjusting models based on resources/context.
*   **Model Selection:** INFaaS, Edgeadaptor (routing queries to appropriate model sizes).
*   **Model Iteration:** Tabi (early exit or cascading).
*   **Speculative Decoding:** Leviathan et al. (Draft model generates tokens, target model verifies).
*   **Table V:** Summarizes adaptation methods (INFaaS, JellyBean, Tabi) by Target (Latency, Cost) and Scenario (Cloud/Edge).

> **[Visual Description: Figure 10 - Model Adaptation Methods]**
> (a) **Model Selection:** A selector routes inputs to Small, Medium, or Large models based on Latency/Accuracy needs.
> (b) **Model Iteration:** The model has "Exits" at different layers. If confidence is high, it exits early; otherwise, it continues to deeper layers.
> (c) **Speculative Decoding:** A small model generates a sequence (dots), and a verifier (large model) checks them in parallel.

### D. Token Adaptation
Reducing input sequence length to save compute (Attention is quadratic $O(N^2)$).
*   **Token Pruning:** Removing uninformative tokens (H2O, dynamicViT).
*   **Token Merging:** Combining similar tokens (ToMe).
*   **Token Summary:** Compressing context into summary vectors (AutoCompressor, ICAE).

> **[Visual Description: Figure 11 - Token Reduction Methods]**
> (a) **Pruning:** Tokens are evaluated; low-score tokens are dropped.
> (b) **Merging:** Similar tokens (Set A and Set B) are identified and averaged into a single token.
> (c) **Summary:** Long Instruction/Memory tokens are compressed into specific "Memory Tokens" before entering the LLM.

### E. Lessons Learned
*   Decoder-only architectures dominate.
*   Training data quality is critical (Llama/Qwen success).
*   MoE (Mixture of Experts) balances efficiency and capability (DeepSeek-V3).

---

## V. RESOURCE ALLOCATION AND PARALLELISM

### A. Resource Allocation and Scaling
Dynamic allocation between Edge and Cloud (Figure 12).
*   **Cloud:** Model containers (Clipper), Squishy bin packing (Nexus), Autoscaling (INFaaS).
*   **Edge:** Offloading algorithms, DRL-based optimization.
*   **Table VI:** Summary of methods (Clipper, SpotServe, HexGen) categorized by Cloud vs. Edge.

> **[Visual Description: Figure 12 - Resource Allocation]**
> A loop showing: Condition (Query Load) -> Resource Allocator/Autoscaler -> Target (Latency/Throughput) -> Adjustment of KV cache and Parameters across GPUs.

### B. Parallelism
Crucial for models that don't fit on one GPU.
1.  **Data Parallelism (DP):** Replicating model, splitting data batch.
2.  **Model Parallelism (MP):** Splitting model layers (Pipeline Parallelism - PP) or splitting tensors (Tensor Parallelism - TP).
3.  **Sequence Parallelism (SP):** Splitting long sequences.
4.  **Auto-parallelism:** Frameworks like Alpa, Megatron-LM, DeepSpeed.

> **[Visual Description: Figure 13 - Parallelism Methods]**
> 1.  **Data Parallelism:** Same model on GPU 1 & 2, different inputs.
> 2.  **Model Parallelism:** Model partitioned. Layers 1-2 on GPU 1, Layers 3-4 on GPU 2.
> 3.  **Tensor Parallelism:** Matrix multiplication $X \times W$ is split column/row-wise across GPUs.

### C. Lessons Learned
*   **Adaptive resource management** is required for fluctuating agent workloads.
*   **Hybrid parallelism** (combining DP, TP, PP) is standard for large scale.

---

## VI. EXECUTION OPTIMIZATION

> **[Visual Description: Figure 14 - Multi-layer optimization framework]**
> *   **Top (Communication):** Semantic Communication between Edge Server and Devices (Router/Switch).
> *   **Middle (Integrated Frameworks):** Software like llama.cpp, MLC-LLM acting as middleware.
> *   **Bottom (Hardware):** Specific backends: FPGA, ASIC, IMC (In-Memory Compute), CPU, GPU.
> *   **Optimizations:** Computation Optimization (Kernels), Memory Optimization (KV cache offloading).

### A. Computation Optimization
Tailoring algorithms to hardware (Table VII features):
1.  **FPGA:** Reconfigurable, low power. Specialized accelerators for Attention (FlightLLM).
2.  **ASIC:** Custom chips (TPU, NPU). High efficiency (A3 accelerator, SpAtten).
3.  **In-Memory Compute (IMC):** processing inside memory to reduce I/O (ReRAM).
4.  **CPU & GPU:** Collaborative inference (PowerInfer), kernel optimizations (FlashAttention).

### B. Memory Optimization
*   **KV Cache Management:** PagedAttention (vLLM) treats KV cache like virtual memory pages to reduce fragmentation.
*   **Offloading:** Moving parameters/cache to CPU RAM or SSD (DeepSpeed-ZeRO-Offload, FlexGen) when GPU memory is full.
*   **MoE Specific:** Expert offloading/prefetching.

### C. Communication Optimization
*   **Semantic Communication:** Transmitting only semantic-relevant info (activations) rather than raw data.
*   **Split Computing:** Distributing layers between Edge and Cloud.

### D. Integrated Frameworks
*   **Table VIII:** Lists frameworks like **LLaMA.cpp** (CPU optimized), **MLC-LLM** (TVM compiler), **DeepSpeed-Inference**, **TensorRT-LLM** (NVIDIA), **vLLM** (High throughput).

### F. Lessons Learned
*   **Hardware-specific optimization** is paramount (e.g., customized dataflow for FPGA).
*   **Phase-aware coordination:** Prefill phase is compute-bound; Decoding phase is memory-bound. Different strategies are needed for each.

---

## VII. LESSON LEARNED AND FUTURE WORK

1.  **Elastic Agent Serving System:** Agents currently lack internal elasticity. Future agents should dynamically decide when to use external tools vs. internal reasoning.
2.  **Workflow-Aware Resource Scheduling:** Scheduling should understand the specific workflow (RAG vs. Reasoning) to pre-load specific experts or databases.
3.  **Development of Multi-modal Agents:** Moving from text-based agents with visual encoders to natively multi-modal processing (audio/video/text interleaved).
4.  **Heterogeneous Edge Computing:** Better utilization of diverse edge accelerators (NPU, DSP) beyond just standard GPUs.

## VIII. CONCLUSION
The survey presented a unified framework for FM-powered agent services. It covered the full stack from Application to Execution. Key technologies discussed include model compression, token reduction, parallelism, and resource allocation. Future directions emphasize real-time, multi-modal, and elastic agent services on heterogeneous edge-cloud environments to advance AGI.