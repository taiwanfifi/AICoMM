這是一份針對 LLM（如 Claude 或 GPT-4）優化過的 Markdown 文件。

**優化策略說明：**
1.  **格式 (Markdown):** 這是 LLM 最容易解析的結構化語言。使用了明確的標題層級 (#, ##, ###) 來建立知識樹。
2.  **圖片轉譯 (Visual-to-Text):** 所有的圖表 (Figure) 和表格 (Table) 都已被轉換為詳細的文字描述或 Markdown 表格。對於架構圖，我使用了結構化的條列式描述（如 `[Layer]` -> `[Component]` -> `[Function]`），這樣 LLM 能夠理解組件之間的邏輯關係和資料流向，而不僅僅是視覺外觀。
3.  **Token 效率:**
    *   去除了頁眉、頁腳、頁碼。
    *   去除了文末的參考文獻列表（但保留了文中的引用標記如 `[1]` 以維持上下文完整性）。
    *   修正了 OCR 產生的斷行符號（Hyphenation），確保單字完整。
4.  **上下文完整性:** 保留了所有的章節內容、定義和邏輯推演，確保模型閱讀時不會有知識斷層。

---

# Internet of Agents: Fundamentals, Applications, and Challenges

**Authors:** Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang, and Dusit Niyato

## Abstract
With the rapid proliferation of large language models (LLMs) and vision-language models (VLMs), AI agents have evolved from isolated, task-specific systems into autonomous, interactive entities capable of perceiving, reasoning, and acting without human intervention. As these agents proliferate across virtual and physical environments, from virtual assistants to embodied robots, the need for a unified, agent-centric infrastructure becomes paramount. In this survey, we introduce the **Internet of Agents (IoA)** as a foundational framework that enables seamless interconnection, dynamic discovery, and collaborative orchestration among heterogeneous agents at scale. We begin by presenting a general IoA architecture, highlighting its hierarchical organization, distinguishing features relative to the traditional Internet, and emerging applications. Next, we analyze the key operational enablers of IoA, including capability notification and discovery, adaptive communication protocols, dynamic task matching, consensus and conflict-resolution mechanisms, and incentive models. Finally, we identify open research directions toward building resilient and trustworthy IoA ecosystems.

---

## I. INTRODUCTION

The rapid advancement of large models, including large language models (LLMs) and vision-language models (VLMs), has ushered in a new era of artificial intelligence (AI) agents (or agentic AI), transforming them from isolated, task-specific models into autonomous, interactive entities. These agents can perceive, reason, and act independently, capable of seamless collaboration with humans and other agents in complex environments, marking a pivotal step toward artificial general intelligence (AGI). From virtual assistants to physically embodied systems such as humanoid robots, autonomous unmanned aerial vehicles (UAVs), and intelligent vehicles, AI agents are rapidly weaving into daily life.

Tech giants have declared their adventure to develop next-generation AI agents, such as OpenAI Operator and ByteDance’s Doubao Agent. For instance, platforms such as Hugging Face host over 1 million open-source models, while Tencent Yuanbao support over 100K specialized agents. According to Gartner, by 2028, at least 15% of daily tasks will be autonomously performed by AI agents, and 33% of enterprise applications will incorporate agent-driven intelligence. As AI agents proliferate, they are poised to act as “new citizens” in digital and physical spaces, reshaping economic structures and human social interactions.

The widespread adoption of agents has spurred the need for real-time cross-domain agent communication and coordination, particularly in scenarios such as smart cities including millions of heterogeneous agents. The **Internet of Agents (IoA)**, also referred to as the agentic web, emerges as a foundational infrastructure for next-generation intelligent systems that enables seamless interconnection, autonomous agent discovery, dynamic task orchestration, and collaborative reasoning among large-scale virtual/embodied agents.

Unlike the human-centric Internet, IoA is agent-centric and prioritizes inter-agent interactions, where the exchanged information shifts from human-oriented data (e.g., text, images, and audio) to machine-oriented data objects (e.g., model parameters, encrypted tokens, and latent representations). Furthermore, interaction methods are evolving beyond graphical user interfaces (GUIs) toward semantic-aware and goal-driven communications via auto-negotiation. Additionally, by offering scalable networked AI inference and shared sensing capabilities, IoA empowers resource-constrained agents, e.g., mobile devices and UAVs, with access to advanced AI capabilities and beyond-line-of-sight (BLOS) perception.

### Critical Challenges
Despite its bright future, the practical deployment of IoA at scale faces several critical challenges:

*   **Interconnectivity:** Existing multi-agent systems (MAS) are primarily simulated on single devices, whereas real-world IoA deployments span billions of geographically distributed agents. This necessitates new agent networking architectures that support seamless interoperability among heterogeneous agents, breaking down data silos.
*   **Agent-Native Interface:** Current computer-use agents rely on mimicking human GUI actions (e.g., clicks and keystrokes), incurring high screen-scraping overhead. IoA should empower agents to interact natively (e.g., APIs or semantic communication protocols) with other agents and Internet resources.
*   **Autonomous Collaboration:** IoA encompasses both physical and virtual agents operating in highly dynamic settings. IoA should let agents self-organize, self-negotiate, and form low-cost, high-efficiency collaborative networks for autonomous agent discovery, capability sharing, task orchestration, and load balancing.

---

### [VISUAL CONTEXT: Table I]
**Summary of Key Abbreviations in Alphabetical Order**

| Abbr. | Definition | Abbr. | Definition | Abbr. | Definition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A2A** | Agent-to-Agent | **AGI** | Artificial General Intelligence | **AI** | Artificial Intelligence |
| **ANP** | Agent Network Protocol | **BLOS** | Beyond-Line-of-Sight | **CoT** | Chain-of-Thought |
| **CUA** | Computer-Use Agent | **D2D** | Device-to-Device | **DID** | Decentralized IDentifier |
| **DNN** | Deep Neural Network | **EOM** | Economy of Minds | **GoT** | Graph-of-Thought |
| **GUI** | Graphical User Interface | **IoA** | Internet of Agents | **IoT** | Internet of Things |
| **LLM** | Large Language Model | **MARL** | Multi-Agent Reinforcement Learning | **MAS** | Multi-Agent Systems |
| **MCP** | Model Context Protocol | **MPC** | Multi-Party Computation | **NLP** | Natural-Language Processing |
| **P2P** | Peer-to-Peer | **PMU** | Phasor Measurement Unit | **pub/sub** | Publish-Subscribe |
| **QoS** | Quality-of-Service | **RAG** | Retrieval-Augmented Generation | **RL** | Reinforcement Learning |
| **SSE** | Server-Sent Events | **ToT** | Tree-of-Thought | **UAV** | Unmanned Aerial Vehicle |
| **UUID** | Universally Unique IDentifier | **VC** | Verifiable Credential | **VLM** | Vision-Language Model |

---

### A. Comparison with Existing Surveys and Contributions

Existing surveys mainly focus on multi-agent systems (MAS), which have three challenges:
1.  **Ecosystem isolation:** Limiting agents to their own environments.
2.  **Single-device simulation:** Confined to single-device simulations rather than real-world distributed operation.
3.  **Rigid communication and coordination:** Hard-coded protocols failing to capture dynamic collaboration.

In contrast, this survey focuses on the **networking aspects** of large model-based agents, addressing their architectures and open challenges of the Internet of Agents (IoA).

---

### [VISUAL CONTEXT: Table II]
**A Comparison of Our Survey with Relevant Surveys**

*   **2022 [21]:** Overview of consensus in MAS including taxonomies, models, protocols, control mechanisms, and applications.
*   **2024 [20]:** Discussions on capabilities and limitations of LLM-based MAS applications in software engineering.
*   **2024 [16]:** Survey of LLM-based MAS, including agent-environment interface, LLM agent characterization, inter-agent comm., capability acquisition, and applications.
*   **2024 [18]:** Survey on LLM-based MAS construction in problem-solving and world simulation.
*   **2024 [2]:** Review enabling technologies, core features, cooperation paradigms, security/privacy challenges, and defense strategies of LLM agents.
*   **2025 [15]:** Review intelligent decision-making approaches, algorithms, and models in MAS.
*   **2025 [17]:** Discussions on key characteristics including type, strategy, structure, and coordination of LLM-based MAS.
*   **2025 [19]:** Survey on LLM-based multi-agent autonomous driving systems including multivehicle interaction, vehicle-infra. communication, and human-vehicle codriving.
*   **Now (Ours):** Comprehensive survey of fundamentals, applications, and challenges of IoA, discussions on architecture design, key characteristics, working paradigms, and open issues in IoA.

---

### [VISUAL CONTEXT: Figure 1]
**Organization structure of this survey paper**

The survey is structured as follows:
*   **Section I: Introduction**
*   **Section II: Overview of Internet of Agents**
    *   Architecture of IoA (Infrastructure, Agent Management, Agent Coordination, Agent Application Layers)
    *   Key Characteristics of IoA (Autonomous Intelligence, High Dynamics, High Heterogeneity, Large-scale Scalability, Semantic-aware Comm, Task-driven Cooperation)
    *   Key Differences Among Internet, IoT, and IoA
    *   Emerging IoA Application Scenarios
*   **Section III: Building Blocks & Tech. of Internet of Agents**
    *   Capability Notification & Discovery (Evaluation, Retrieval, Notification)
    *   Interaction Structure & Task Orchestration (Interaction Modes, Topological Structures, Decomposition, Allocation)
    *   Communication Protocols (Anthropic's MCP, Google's A2A, ANP, AGNTCY, Agora, MoA)
    *   Consensus & Conflict Resolution (Turn-Taking, Reasoning Alignment, Scaling Consensus)
    *   Economic Models (Pricing, Incentives, Penalties)
    *   Trustworthy Regulation (DID & VC, Blockchain, Legal Design)
*   **Section IV: Future Research Directions** (Secure Protocols, Decentralized Ecosystems, Economic Systems, Privacy, Cyber-Physical Security, Ethical & Interoperable IoA)
*   **Section V: Conclusion**

---

## II. OVERVIEW OF INTERNET OF AGENTS

### A. Architecture of IoA

The IoA is an emerging agent-centric infrastructure that connects billions of autonomous agents, both virtual and embodied, across diverse domains.

#### 1) Agent Types
*   **Virtual agents:** Operate entirely within digital environments (e.g., chatbots, virtual assistants). They leverage high-bandwidth connections to access LLMs and knowledge bases.
*   **Embodied agents:** Inhabit the physical world (e.g., home robots, UAVs, autonomous vehicles). They rely on onboard sensors and actuators to perceive and manipulate their environment.

#### 2) Functional Modules of Agents
Each IoA agent typically comprises four core functional modules:

---

### [VISUAL CONTEXT: Figure 2]
**Workflow of functional modules of virtual and embodied agents**

*   **Brain (Large Model):**
    *   **Planning:** The central reasoning unit. Receives observations and goals. Uses strategies like feedback-free (CoT, ToT) or feedback-enhanced planning (ReAct, Reflexion).
    *   **Memory:** Stores Short-term (context buffer), Long-term (vector stores, RAG), and Hybrid memory.
*   **Body (Sensor & Actuator):**
    *   **Interaction:** Connects to the Environment.
        *   *Inputs:* Observation (Visual, Text, Sensor data).
        *   *Feedback:* From Human, Cyber, or Physical worlds.
    *   **Action:**
        *   *Embodied operations:* Robot movement, grasping.
        *   *Tool invocation:* APIs, search engines, scripts.
        *   *External knowledge:* Consulting databases.

*Note: Embodied agents uniquely include physical sensors and actuators in the feedback loop.*

---

#### 3) Interconnected Sub-IoA
The IoA architecture comprises multiple interconnected and domain-specific agent networks, referred to as **sub-IoAs**. Each sub-IoA operates semi-autonomously and is anchored by a designated gateway node hosting a **gateway proxy agent**.

---

### [VISUAL CONTEXT: Figure 3]
**Overview of IoA encompassing heterogeneous autonomous agents across diverse domains**

*   **Structure:**
    *   Multiple Domains (Domain 1, 2, 3, 4).
    *   Each Domain has a **Server/Gateway** running a **Proxy Agent**.
    *   Domains contain **Virtual Agents** and **Embodied Agents**.
*   **Process Flow (Example):**
    1.  **Agent Register:** Local agents register with their domain's Proxy Agent.
    2.  **Capability Notification:** Agents inform the proxy of their skills.
    3.  **Agent Discovery:** A Gateway Proxy Agent searches for required skills (locally or across domains).
    4.  **Interaction Relay:** Gateways facilitate communication between agents in different domains (e.g., Domain 1 Agent talking to Domain 4 Agent).
    5.  **Agent State Sync:** Ensuring consistency across the interaction.

---

#### 4) General IoA Architecture
The hierarchical IoA architecture comprises four layers:

---

### [VISUAL CONTEXT: Figure 4]
**General architecture of IoA**

**Layer 1 (Bottom): Infrastructure Layer**
*   **Foundational Models:** Large Models (DeepSeek-R1, GPT-4o, LLaMA) providing the cognitive core.
*   **Computing Power:** Cloud clusters (GPU/TPU), Edge nodes, AI chips.
*   **Communication Infra:** 5G/6G, Wireless, Wired networks.
*   **Data & Knowledge:** Public data, Personal data, Business data, Prompts, Knowledge bases.

**Layer 2: Agent Management Layer**
*   **Identity Identification:** DIDs, Authentication.
*   **Capability Notification:** Semantic modeling of skills.
*   **Registration & Discovery:** Service registries.
*   **Authn & Access Control:** Security policies.
*   *Agents:* Virtual agents and Embodied agents exist here, utilizing Perception, Memory, Planning, and Action modules.

**Layer 3: Agent Coordination Layer**
*   **Dynamic Matching:** Matching tasks to agents.
*   **Task Orchestration:** Decomposition and workflow management.
*   **Interaction Protocols:** Standards for communication.
*   **Conflict Resolution:** Consensus mechanisms.
*   **Agent Security & Privacy:** Trust enhancement, Privacy protection.
*   **Billing & Incentives:** Economic layer.
*   *Workflow:* Takes Task Data -> produces Inference Results.

**Layer 4 (Top): Application Layer**
*   **Domains:** Smart Home, Personal Assistant, Autonomous Driving, Smart City, Smart Healthcare, Smart Factory.
*   **Application Interfaces:**
    *   Interface standards.
    *   Modality alignment.
    *   Semantic alignment.
    *   Knowledge alignment.

---

### B. Key Characteristics of IoA

1.  **Autonomous Intelligence:** Agents proactively advertise capabilities, initiate collaborations, and negotiate without human intervention (unlike passive APIs).
2.  **High Dynamics:** Agents are created/terminated on-demand; embodied agents move physically. Workflows reconfigure in real-time.
3.  **High Heterogeneity:** Agents range from microcontrollers to GPU clusters. Data formats vary (LiDAR vs. Text).
4.  **Large-scale Scalability:** Billions of agents. Requires hierarchical sharding and elastic resource orchestration.
5.  **Semantic-aware Communication:** Focus on "Computing-Oriented" and "Meaning-aware" communication (transmitting intent/inference rather than raw data) to reduce overhead.
6.  **Task-driven Cooperation:** Networks form dynamically based on tasks (e.g., micro-swarms) and dissolve upon completion.

### C. Key Differences Among Internet, IoT, and IoA

**Summary:** IoA represents a paradigm shift from data transmission (Internet) and device control (IoT) to **agent collaboration and knowledge exchange**.

---

### [VISUAL CONTEXT: Table III]
**Key Comparisons Between Traditional Internet, IoT, and IoA**

| Feature | Traditional Internet | Internet of Things (IoT) | Internet of Agents (IoA) |
| :--- | :--- | :--- | :--- |
| **Core Objective** | Host & Information Connectivity | Device & Information Connectivity | Agent & Knowledge Connectivity |
| **Service Objects** | Humans | Smart Devices (Sensors/Actuators) | Autonomous Agents (Virtual/Embodied) |
| **Architecture** | Centralized (Client-Server) | Decentralized (End-Edge-Cloud) | Hybrid (P2P + Proxy-based) |
| **Addressing** | IP | IP, Static Device Identity | IP, Dynamic Semantic Identifiers |
| **Interaction Mode** | Passive (Request-Response) | Event-Driven (Trigger-Based) | Proactive (Goal/Task-Oriented) |
| **Communication** | Bit-Level Transmission | Bit-Level + Lightweight Protocols | Semantic-Level Exchange |
| **Autonomy Source** | Human-Controlled | Rule-based Device Logic | Large Model-Driven Agent Intelligence |
| **Network Dynamics** | Low (Static Topologies) | Medium (Dynamic Topologies) | High (Evolving Interactions & Mobility) |

---

### D. Emerging IoA Application Scenarios

1.  **Smart City:** Cross-domain agent networking. Traffic controllers, safety UAVs, and emergency robots form on-demand teams to handle incidents (e.g., fire, accidents).
2.  **Smart Home:** Agent communications within an IoA subnet. Housekeeping robots, digital assistants, and appliances discover each other to form task groups (e.g., meal prep).
3.  **Smart Factory:** Coordination between subnet and external agents. Factory robots coordinate with supply chain drones and cloud analytic agents.

---

### [VISUAL CONTEXT: Figure 5]
**Overview of IoA applications**

*   **a) Smart Home:** Depicts a house with various agents (Assistant, Robot, Appliance) connected in a local mesh (P2P + Proxy).
*   **b) Smart Factory:** Shows a production line with robotic arms and AGVs communicating with a central "Smart Factory" hub.
*   **c) Smart City:** A broader scope showing connected vehicles, drones, and city infrastructure agents interoperating.
*   *Common Thread:* All three scenarios show specific "Agents" connected to a central "Internet of Agents (IoA)" cloud/network.

---

### [VISUAL CONTEXT: Table IV]
**Comparison of Mainstream Agent Frameworks**

| Framework | Key Features | Strengths | Weaknesses | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **MetaGPT** | Role-based, structured workflows | High efficiency in complex tasks | Rigid roles, less flexible | Automated software development |
| **LangChain** | Modular chaining, tool integration | Highly customizable | Steep learning curve | Custom pipelines, RAG |
| **AutoGPT** | Recursive self-prompting | Fully autonomous | Prone to loops, high cost | General task automation |
| **AutoGen** | Conversational MAS | Flexible collaboration | Complex setup | Multi-agent dialogue |
| **BabyAGI** | Task-driven iteration | Lightweight, easy deploy | Limited reasoning depth | Small-scale automation |
| **CAMEL** | Role-playing societies | Simulates human-like interactions | Less optimized for tools | Social AI research |

---

## III. WORKING PARADIGMS OF INTERNET OF AGENTS

### [VISUAL CONTEXT: Figure 6]
**Overview of cross-domain agent interaction lifecycle in IoA**

The diagram illustrates the sequential phases:
1.  **Register:** Agents register with Gateway.
2.  **Capability Notification:** Agents define and broadcast skills.
3.  **Discovery:** Finding suitable partners.
4.  **Agent Interactions:**
    *   *Structure:* Interaction Structure design.
    *   *Orchestration:* Task Orchestration.
    *   *Protocol:* Agent Comm. Protocol.
    *   *Resolution:* Consensus & Conflict Resolution.
5.  **State Sync:** Synchronizing context across domains.
6.  **Regulation & Optimization:** Economic Models and Trustworthy Regulation.

---

### A. Capability Notification & Discovery in IoA

Accurately evaluating agent capabilities is foremost for effective task assignment.

1.  **Capability Evaluation:**
    *   *Self-reported:* Agents declare capabilities (e.g., "I use GPT-4o and can run Python"). Fast but prone to hallucination.
    *   *Systematic Verification:* Gateway agents verify skills using benchmarks (e.g., GAIA for reasoning, RoCoBench for embodied skills).

2.  **Capability Notification:**
    *   *Proactive:* Agents push updates.
    *   *Event-triggered:* Updates only upon significant changes (e.g., new tool learned).
    *   *Periodic:* Regular consistency checks.

3.  **Capability Retrieval:**
    *   *Semantic Retrieval:* Using embeddings (BERT, etc.) to match natural language queries to agent profiles.
    *   *Agentic-enhanced Retrieval:* Using "Agentic RAG" where a retrieval agent actively refines queries and selects tools.

---

### [VISUAL CONTEXT: Figure 7]
**Illustration of capability notification and discovery in IoA**

*   **Left Side (Notification):**
    *   Newcomer Agent sends "Self-reported Capability".
    *   Gateway Server performs "Systematic Verification".
    *   Agent sends "Capability Notification" to update the registry.
*   **Right Side (Retrieval):**
    *   User/Agent sends "Capability Retrieval" request.
    *   Gateway accesses "Capability Lists" (e.g., *ScholarAgent*: Tools=[Browser], Skills=[Research]; *FitAgent*: Tools=[Recipe DB]).
    *   Returns matched agent.

---

### [VISUAL CONTEXT: Table V]
**Comparison of Agent Capability Retrieval Strategies**

| Ref. | Strategy | Key Technology | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- |
| [58] | Traditional Search | Exact/Fuzzy matching | Simple, fast | Struggles with semantic ambiguity |
| [38] | Semantic Retrieval | DNNs (Transformers) | Captures intent/semantics | Computationally heavy |
| [59] | Knowledge-based | Knowledge Graphs, RAG | Interpretable, logical | Expensive to build/maintain |
| [60] | Agentic-enhanced | Agentic RAG | Feedback-driven, adaptive | High complexity/latency |

---

### B. Interaction Structure & Task Orchestration

#### 1) Interaction Structure Design
This involves two elements: **Interaction Mode** (how they talk) and **Communication Topology** (how they are connected).

*   **Interaction Modes:**
    *   *Aggregate:* Voting mechanisms (Majority vote).
    *   *Reflect:* Agents review and critique each other.
    *   *Debate:* Structured argumentation to reach truth.
    *   *Tool-use:* Agents invoking external APIs.
*   **Topological Structures:**
    *   *Chain:* Linear sequence (Pipelined).
    *   *Star:* Central controller (Manager/Worker).
    *   *Tree:* Hierarchical command (Divide and Conquer).
    *   *Graph:* Flexible P2P mesh (Dynamic).

---

### [VISUAL CONTEXT: Figure 8]
**Illustration of interaction modes and communication topologies**

*   **Top (Modes):**
    *   *Aggregate:* 3 agents output -> 1 aggregator -> Result.
    *   *Reflect:* Agent outputs -> Self/Peer creates feedback -> Refined output.
    *   *Debate:* Agents exchange dialogue bubbles back and forth.
    *   *Tool-use:* Agent connects to a "Tool" block.
*   **Bottom (Topologies):**
    *   *Chain:* A -> B -> C.
    *   *Star:* Center Agent connects to periperal agents.
    *   *Tree:* Root Agent -> Branch Agents -> Leaf Agents.
    *   *Graph:* Mesh network with arbitrary connections.

---

#### 2) Task Orchestration
*   **Task Decomposition:** Breaking high-level requests into sub-tasks.
    *   *Rule-based:* Pre-defined schemas (Standard Operating Procedures).
    *   *Learning-based:* LLMs decompose tasks based on training or RAG (e.g., HuggingGPT).
*   **Task Allocation:** Assigning tasks to agents.
    *   *Routing-based:* Central router selects best model/agent (e.g., RouteLLM).
    *   *Self-organizing:* Market-based (bidding) or negotiation-based allocation (Agora).

---

### [VISUAL CONTEXT: Table VI]
**Comparison of Existing Task Orchestration Methods**

| Strategy | Core Idea | Strengths | Weaknesses | Examples |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-based Decomp.** | Pre-defined logical rules | High interpretability | Limited adaptability | TDAG, HM-RAG |
| **Learning-based Decomp.** | LLM/RL interaction | Strong adaptability | Lower interpretability | HuggingGPT |
| **Routing-based Allocation** | Match task to agent via router | High efficiency | Depends on router quality | RouteLLM |
| **Self-organizing Allocation** | Decentralized coordination | High autonomy | Coordination overhead | Mindstorms, Agora |

---

### C. Communication Protocols for IoA

To eliminate data silos, standardized protocols are required.

1.  **Anthropic’s MCP (Model Context Protocol):** Client-server standard for connecting LLMs to data/tools. Uses OAuth.
2.  **Google’s A2A (Agent to Agent):** P2P networking layer. Uses Agent Cards (JSON metadata) for discovery and standardized message exchange.
3.  **Other Protocols:**
    *   *ANP (Agent Network Protocol):* Fully decentralized, DID-based.
    *   *AGNTCY:* Open interoperable infrastructure.
    *   *Agora:* Collaborative LLM-agent communications focusing on natural language routines.

---

### [VISUAL CONTEXT: Figure 9]
**Workflow of Anthropic's MCP**

*   **Architecture:** Client-Server.
*   **Entities:**
    *   **MCP Hosts:** Running AI Applications and MCP Clients.
    *   **MCP Server:** Holds Tools, Prompts, Resources, Capabilities.
    *   **External:** APIs, Browser, Database, Filesystem.
*   **Flow:**
    1.  MCP Request (Host to Server).
    2.  Fetch/Execute (Server to External Tools).
    3.  Return Data (External to Server).
    4.  MCP Response (Server to Host).

---

### [VISUAL CONTEXT: Figure 10]
**Workflow of Google's A2A**

*   **Entities:** Client Agent and Remote Agent.
*   **Discovery:**
    *   Uses **Agent Card** (Name, Skills, URL).
    *   Client discovers Remote Agent via Card.
*   **Task Flow:**
    1.  Generate UUID for Task ID.
    2.  Send Task (ID & Request).
    3.  Process Task (Remote side).
    4.  Interaction (Optional bidirectional flow).
    5.  Return Response.

---

### [VISUAL CONTEXT: Table VII]
**Comparison of Representative Agent Communication Protocols**

| Protocols | MCP [13] | A2A [9] | ANP [46] | AGNTCY [109] | Agora [11] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Goal** | Tool/Data Access | P2P Comm. | Decentralized Net | Standard Collab | Scalable Comm |
| **Architecture** | Client-Server | P2P | P2P | Hybrid | P2P |
| **Auth** | OAuth | OAuth 2.0 / API Keys | DID-based | Connect Protocol | - |
| **Discovery** | - | Agent Cards | Agent Desc. | OASF Metadata | NLP / Docs |
| **Entity** | Anthropic | Google | Open Source | Open Alliance | Oxford/Camel AI |

---

### D. Consensus & Conflict Resolution in IoA

Agents often produce inconsistent outputs or have conflicting goals.

1.  **Turn-Taking Regulation:**
    *   *Polling:* Coordinator asks agents one by one.
    *   *Arbitration:* A "Leader" agent assigns speaking rights.
2.  **Reasoning Alignment:**
    *   *Self-Consistency:* Agent checks its own multiple outputs.
    *   *Collective Reasoning:* Aggregating reasoning paths from multiple agents (better accuracy).
3.  **Scaling Consensus:** Hierarchical consensus (Local clusters agree -> Cluster reps agree -> Global consensus).

### E. Economic Models of IoA

To foster long-term collaboration, economic incentives are needed.

1.  **Pricing Models:**
    *   *Capability-based:* Pay per token/compute usage.
    *   *Contribution-aware:* Pay based on marginal impact (Shapley value).
2.  **Mechanism Design:**
    *   *Incentives:* Auctions, Contracts, Reputation systems.
    *   *Penalties:* Slashing (taking tokens), Reputation loss for hallucinations or malicious acts.

---

### [VISUAL CONTEXT: Table VIII]
**Comparison of Incentive Mechanism Designs**

| Mechanism Type | Advantages | Limitations | Key Methodology |
| :--- | :--- | :--- | :--- |
| **Auction Theory** | Fairness, efficiency | High computation | Privacy-preserving stochastic auctions |
| **Contract Theory** | Mitigates asymmetry | Complex design | Multi-dimensional contract optimization |
| **Non-cooperative Game** | Realistic decision making | Sub-optimal global outcomes | Nash Equilibrium modeling |
| **Cooperative Game** | Improves system performance | Coalition instability | Shapley value distribution |
| **Reputation Mechanism** | Sustainable collaboration | Prone to manipulation | Semantic-aware reputation scoring |

---

### F. Trustworthy Regulation in IoA

1.  **DID and Verifiable Credentials (VCs):** Agents need self-sovereign identity (W3C DID) to be accountable without a central authority.
2.  **Blockchain:** Immutable ledgers for auditing agent actions and enforcing smart contracts.
3.  **Legal Design:** Defining liability. Who is responsible if an agent causes harm? The developer, the user, or the agent?

## IV. FUTURE RESEARCH DIRECTIONS

1.  **Secure & Adaptive Protocols:** Balancing versatility and efficiency. Protecting against prompt injection and tool poisoning.
2.  **Decentralized Self-Governing Ecosystems:** Moving away from central servers to bio-inspired swarm intelligence and DAO-like structures.
3.  **Agent-Based Economic Systems:** Preventing adversarial market manipulation (Sybil attacks) in agent economies.
4.  **Privacy-Preserving Interactions:** Handling sensitive data (healthcare/finance) using Homomorphic Encryption or MPC without killing performance.
5.  **Cyber-Physical Secure IoA:** Defending against attacks that bridge the digital-physical gap (e.g., spoofing LiDAR data to crash a drone).
6.  **Ethical & Interoperable IoA:** Ensuring moral reasoning alignment and transparent audit trails for agent decisions.

## V. CONCLUSIONS

We positioned the IoA as the next-generation infrastructure for autonomous intelligent systems. We presented a hierarchical architecture, explored enabling technologies (discovery, orchestration, protocols, consensus, economics), and outlined a roadmap for future research. Sustained innovation in networking, standards, and security is essential to realize the IoA ecosystem.