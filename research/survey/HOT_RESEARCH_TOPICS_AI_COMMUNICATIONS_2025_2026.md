# HOT & NOVEL Research Topics in AI + Communications (2025-2026)
## Comprehensive Survey from Top Journals and Conferences

**Date compiled:** February 24, 2026
**Sources:** IEEE JSAC, IEEE COMST, IEEE TWC, IEEE TCOM, ACM SIGCOMM 2025, IEEE INFOCOM 2025, IEEE TNSE, IEEE IoTJ, and others

---

## EXECUTIVE SUMMARY

This report identifies **35+ genuinely novel and high-value research directions** from top-tier AI + Communications venues. Each topic is distinct from: semantic communication, LEO satellite management, split inference, and foundation models for wireless (already identified). Topics are organized by novelty tier and include paper references, GPU/resource needs, and open research gaps.

---

## TIER 1: HOTTEST & MOST NOVEL DIRECTIONS (Highest Impact Potential)

---

### 1. Metasurface-Driven Physical Neural Networks for Over-the-Air AI

**Key Paper:** "Enabling Over-the-Air AI for Edge Computing via Metasurface-Driven Physical Neural Networks"
**Venue:** ACM SIGCOMM 2025 (top-tier acceptance!)
**Authors:** Researchers from Northwest University and University at Buffalo

**What makes it exciting:** This paper was accepted at SIGCOMM (extremely competitive, ~12% acceptance rate). It proposes using programmable multi-layer metasurfaces and MIMO channels to realize computational layers in the wave propagation domain -- turning the wireless channel itself into a neural network. The system (called MINNs - Metasurfaces-Integrated Neural Networks) offloads computation into the physical layer, achieving performance comparable to fully digital DNNs while dramatically reducing power consumption. This is a fundamentally new computing paradigm.

**GPU/Resources:** Primarily simulation-based + some physical experiment components. Can be done as dry lab with electromagnetic simulation tools (CST, HFSS) + PyTorch/TensorFlow.

**Research Gaps:**
- Multi-user MINNs with interference management
- Training algorithms for physical neural network layers
- Robustness to hardware imperfections and environmental changes
- Integration with existing 5G/6G protocol stacks
- Scaling to larger networks and more complex tasks

**JSAC Special Issue Alignment:** "Multi-functional Programmable Metasurfaces for 6G and Beyond" (deadline: Nov 1, 2025, pub Q3 2026)

---

### 2. Diffusion Models for Physical Layer Problems

**Key Papers:**
- "Generative Diffusion Models for High Dimensional Channel Estimation" -- **IEEE TWC**, vol. 24, no. 7, pp. 5840-5854, July 2025
- "CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications" -- **IEEE TWC**, 2024
- "Generative Diffusion Model-Based Variational Inference for MIMO Channel Estimation" -- **IEEE TCOM**, 2025
- "Near-Field Channel Estimation for XL-MIMO via Deep Generative Model" -- **IEEE Trans. Cognitive Comm. and Networking**, May 2025

**What makes it exciting:** Diffusion models are the latest generative AI breakthrough (behind DALL-E, Stable Diffusion). Their application to wireless is just beginning. They capture the structure of MIMO channels via deep generative priors and achieve state-of-the-art channel estimation, signal denoising, and RF fingerprinting. This is a rapidly growing area with very few papers so far -- a perfect window of opportunity.

**GPU/Resources:** 100% simulation-based / dry lab. Requires GPU training for diffusion model architectures. Standard deep learning framework (PyTorch).

**Research Gaps:**
- Diffusion models for real-time beam tracking (latency challenge)
- Conditional diffusion for joint channel estimation + signal detection
- Score-based models for OFDM signal recovery
- Diffusion-based data augmentation for wireless datasets
- Lightweight diffusion for edge deployment
- Application to near-field / THz channel estimation

---

### 3. Intent-Driven Network Management with Multi-Agent LLMs

**Key Papers:**
- "Intent-Driven Network Management with Multi-Agent LLMs: The Confucius Framework" -- **ACM SIGCOMM 2025** (Meta production system!)
- "Intent-Based Management of Next-Generation Networks: an LLM-Centric Approach" -- **IEEE Network**, 2024
- "INTA: Intent-Based Translation for Network Configuration with LLM Agents" -- arXiv, Jan 2025

**What makes it exciting:** This is **deployed at Meta production scale** (Confucius has been operational for 2 years with 60+ applications). It uses multi-agent LLM systems for intent translation, network configuration, and autonomous management. The SIGCOMM acceptance validates this as a major new direction. It bridges the gap between natural language intent specification and low-level network operations.

**GPU/Resources:** Primarily software/simulation. Requires access to LLM APIs or local LLM deployment. Can leverage open-source LLMs (Llama, Mistral).

**Research Gaps:**
- Hallucination detection and safety for network-critical decisions
- Formal verification of LLM-generated network configurations
- Domain adaptation of LLMs for specific network environments (O-RAN, 5G core)
- Multi-agent coordination protocols for distributed network management
- Latency and reliability guarantees for LLM-based control loops

**IEEE Special Issue Alignment:** "Integrating Agentic AI in Intelligent Wireless Networks" (IEEE TNSE, deadline: May 1, 2026); "Knowledge-driven Autonomous Agent Systems" (IEEE TNSE, deadline: June 15, 2026)

---

### 4. Neuromorphic Computing for Green Wireless Communications

**Key Papers:**
- "Enabling Green Wireless Communications with Neuromorphic Continual Learning" (SpikACom) -- arXiv 2502.17168, Feb 2025
- "Neuromorphic Wireless Split Computing with Multi-Level Spikes" -- King's College London KCLIP Lab, March 2025

**What makes it exciting:** Spiking Neural Networks (SNNs) on neuromorphic hardware achieve **order-of-magnitude improvement in energy efficiency** compared to conventional deep learning, while matching performance on critical wireless tasks (MIMO beamforming, channel estimation, semantic communication). This is the intersection of brain-inspired computing with wireless -- a truly novel angle that addresses the sustainability crisis of AI in communications.

**GPU/Resources:** Simulation-based with neuromorphic simulator frameworks (Norse, snnTorch, Lava). Can also use Intel Loihi or IBM TrueNorth hardware if available. Mostly dry lab.

**Research Gaps:**
- SNN architectures optimized for specific wireless tasks
- Neuromorphic hardware-aware protocol design
- Continual/online learning with SNNs for dynamic channels
- Comparison with quantized conventional DNNs (fair benchmarking)
- Energy-latency tradeoffs in neuromorphic edge inference
- Integration with ISAC systems

---

### 5. World Models and Digital Twins for Proactive 6G Networks

**Key Papers:**
- "Dual Mind World Model Inspired Network Digital Twin for Access Scheduling" -- arXiv 2602.04566, Feb 2026
- "Generative AI Empowered Network Digital Twins" -- **ACM Computing Surveys**, 2025
- "Digital Twin-Accelerated Online Deep Reinforcement Learning for Admission Control in Sliced Communication Networks" -- **IEEE TNSM**, 2024

**What makes it exciting:** World models (from robotics/gaming) are being applied to build predictive digital twins of wireless networks that can "imagine" future network states and proactively optimize. The "Dual Mind" approach combines fast intuitive inference with deliberate planning -- inspired by human cognitive psychology. This converts digital twins from passive monitoring tools into proactive, self-improving cognitive systems.

**GPU/Resources:** Simulation-based / dry lab. Requires GPU for world model training. Can use network simulators (ns-3, OpenAI Gym-based envs).

**Research Gaps:**
- Sim-to-real transfer for world model accuracy
- Scalability of world models to city-scale networks
- Integration with real O-RAN testbeds
- Online adaptation and continuous learning
- Generalization across different network topologies

**IEEE Special Issue Alignment:** "Fusing Digital Twins and World Models for Proactive 6G Networks" (IEEE TNSE, deadline: March 15, 2026); "Digital Twins for Wireless Networks" (IEEE JSAC, deadline: May 1, 2026)

---

### 6. Distributed Inference in Network User Plane (DUNE)

**Key Paper:** "DUNE: Distributed Inference in the User Plane" -- **IEEE INFOCOM 2025 Best Paper Award**
**Authors:** IMDEA Networks

**What makes it exciting:** Won the Best Paper Award at INFOCOM 2025. It decomposes AI inference tasks across network devices (programmable switches, DPUs, smartNICs) directly in the data plane, achieving high scalability and improved accuracy while preserving ultra-low-latency and line-rate throughput. This is fundamentally different from cloud/edge inference -- it's AI embedded IN the network fabric itself.

**GPU/Resources:** Requires network simulation tools (P4, Tofino switch emulators, DPU development kits). Primarily software-based but benefits from hardware emulation.

**Research Gaps:**
- Model partitioning strategies for heterogeneous network devices
- Fault tolerance and graceful degradation
- Training pipelines for distributed user-plane models
- Application to specific wireless functions (scheduling, handover)
- Security implications of in-network inference

---

## TIER 2: RAPIDLY EMERGING DIRECTIONS (Strong Publication Momentum)

---

### 7. AI for O-RAN / Open RAN Optimization

**Key Papers:**
- "Meta Reinforcement Learning Approach for Adaptive Resource Optimization in O-RAN" -- IEEE Xplore, 2025
- "PPO-EPO: Energy and Performance Optimization for O-RAN Using Reinforcement Learning" -- **IEEE ICC 2025**
- "xSlice: Near-Real-Time Resource Slicing for QoS Optimization in 5G O-RAN using Deep RL" -- arXiv, Sept 2025

**What makes it exciting:** O-RAN standardization is happening NOW. The Near-RT RIC (RAN Intelligent Controller) provides a standardized platform for deploying AI/ML algorithms as xApps. This creates immediate practical impact -- your research can be deployed on real O-RAN testbeds.

**GPU/Resources:** Can use O-RAN simulation platforms (OpenAirInterface, srsRAN, O-RAN Software Community). Dry lab feasible with some computational requirements.

**Research Gaps:**
- Multi-objective optimization across conflicting KPIs (energy, latency, throughput)
- Safe RL that avoids catastrophic network failures
- Transfer learning across different O-RAN deployments
- Explainable xApps for operator trust
- Conflict resolution between multiple xApps running simultaneously

---

### 8. Causal Inference for AI-Native Wireless Networks

**Key Papers:**
- "Causal Discovery in Dynamic Fading Wireless Networks" -- arXiv, May 2025
- "Causal Model-Based Reinforcement Learning for Sample-Efficient IoT Channel Access" -- arXiv, Nov 2025
- "Causal Reasoning: Charting a Revolutionary Course for Next-Generation AI-Native Wireless Networks" -- arXiv, Sept 2023

**What makes it exciting:** Most AI-for-wireless work uses correlational learning. Causal inference can discover WHY interference occurs, enabling more robust and generalizable solutions. The causal MARL framework reduces environment interactions by 58% -- hugely important for real-world deployment where exploration is expensive.

**GPU/Resources:** 100% simulation-based / dry lab. Standard ML frameworks + causal discovery libraries (DoWhy, CausalNex).

**Research Gaps:**
- Causal discovery from limited/noisy wireless measurements
- Integration with reinforcement learning for causal-aware resource allocation
- Causal explanations for network anomalies (XAI connection)
- Counterfactual reasoning for network planning ("what if we add a base station here?")
- Scalable causal graph learning for large networks

---

### 9. Self-Supervised / Contrastive Learning for Wireless Foundation Models

**Key Papers:**
- "ContraWiMAE: A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning" -- arXiv, May 2025
- "FLA-CL: Feature-Level Augmented Contrastive Learning for Wireless Signal Recognition" -- Electronics, 2025
- "Multi-Representation Domain Attentive Contrastive Learning for Unsupervised AMR" -- **Nature Communications**, 2025

**What makes it exciting:** The Nature Communications acceptance is a major signal. Self-supervised learning solves the labeled data scarcity problem in wireless (you can't easily label millions of channel measurements). These methods achieve strong performance with only 10% labeled data. Foundation model approaches that pre-train on massive unlabeled wireless data and fine-tune for specific tasks are the next frontier.

**GPU/Resources:** GPU-intensive for pre-training. Dry lab / simulation-based. Can leverage existing wireless datasets (DeepMIMO, etc.).

**Research Gaps:**
- Wireless-specific augmentation strategies (beyond image augmentations)
- Cross-domain transfer (indoor to outdoor, sub-6GHz to mmWave)
- Zero-shot adaptation to new environments
- Efficient fine-tuning for resource-constrained edge devices
- Benchmark datasets for wireless foundation model evaluation

---

### 10. Graph Neural Networks for Scalable Wireless Resource Management

**Key Papers:**
- "Model-Based GNN Enabled Energy-Efficient Beamforming for Ultra-Dense Wireless Networks" -- **IEEE TWC**, April 2025
- "Graph Neural Networks for Wireless Networks: Graph Representation, Architecture and Evaluation" -- **IEEE Wireless Communications**, Oct 2024
- "Graph Neural Networks for Distributed Power Allocation: Aggregation Over-the-Air" -- **IEEE TWC**, 2023

**What makes it exciting:** GNNs naturally model wireless networks as graphs (nodes = users/BSs, edges = interference links). They provide SIZE GENERALIZATION -- train on small networks, deploy on large ones. The newest work combines model-based domain knowledge with GNN architectures for better performance and interpretability.

**GPU/Resources:** 100% simulation-based / dry lab. Standard PyTorch Geometric or DGL frameworks.

**Research Gaps:**
- Temporal GNNs for time-varying wireless graphs
- GNN + RL for joint scheduling and resource allocation
- Heterogeneous GNNs for multi-tier networks (macro + small cells + RIS)
- GNN-based distributed algorithms with convergence guarantees
- Application to cell-free massive MIMO coordination

---

### 11. Over-the-Air Computation and Federated Edge Learning

**Key Papers:**
- "Waveforms for Computing Over the Air" -- **IEEE Signal Processing Magazine**, March 2025
- "Communication Efficient Ciphertext-Field Aggregation via Over-the-Air Computation" -- **IEEE TIFS**, Jan 2025
- "Task-Oriented Over-the-Air Computation for Multi-Device Edge AI" -- **IEEE TWC**, 2023

**What makes it exciting:** Over-the-air computation (AirComp) exploits the superposition property of wireless channels to aggregate data from multiple devices simultaneously, enabling orders-of-magnitude speedup for federated learning. The IEEE Signal Processing Magazine publication (2025) signals mainstream maturity. New directions include AirComp + homomorphic encryption for privacy.

**GPU/Resources:** Simulation-based. Can combine communication simulation (MATLAB/Python) with ML frameworks.

**Research Gaps:**
- Waveform design for heterogeneous AirComp tasks
- Robustness to imperfect CSI and channel estimation errors
- Multi-function AirComp (simultaneous aggregation + sensing)
- Integration with ISAC systems
- Practical testbed implementations over 5G NR

---

### 12. AI for THz Communications and Ultra-Massive MIMO

**Key Papers:**
- "AI and Deep Learning for Terahertz Ultra-Massive MIMO: From Model-Driven to Foundation Models" -- ScienceDirect, 2025
- "Near-Field Beamforming for THz Communications with NLG and Massive MIMO" -- Wiley, 2025

**What makes it exciting:** THz communications (0.1-10 THz) offer massive bandwidth for 6G but face extreme propagation challenges. AI is essential for beam management (RL reduces training by 80%), molecular absorption prediction, and near-field channel estimation. The move toward "Native-AI" where AI is embedded intrinsically in the THz communication fabric is a paradigm shift.

**GPU/Resources:** Simulation-based / dry lab. THz channel modeling tools + deep learning. Some experimental validation may need THz testbeds.

**Research Gaps:**
- Foundation models for THz channel prediction
- Joint beam tracking + sensing in THz bands
- AI for THz device impairment compensation
- Near-field-specific neural network architectures
- Energy-efficient AI for THz massive antenna arrays

---

### 13. Explainable AI (XAI) for 6G Network Slicing and Management

**Key Papers:**
- "Advancing 6G: Survey for Explainable AI on Communications and Network Slicing" -- **IEEE Open Journal of the Communications Society**, 2025
- "A Survey on XAI for 5G and Beyond Security" -- arXiv, updated 2025
- "Towards Transparent 6G AI-RAN: Explainable DRL for Intelligent Network Slicing" -- ScienceDirect, 2025

**What makes it exciting:** Telecom operators will NEVER deploy AI they can't understand. XAI for network management is becoming a regulatory requirement (EU AI Act). SHAP, LIME, and attention-based explanations are being integrated into AIOps pipelines. This is high-impact because it's the bridge between AI research and real-world network deployment.

**GPU/Resources:** Simulation-based / dry lab. Standard ML + XAI libraries (SHAP, LIME, Captum).

**Research Gaps:**
- XAI-aware training that optimizes for both performance AND interpretability
- Real-time explanation generation for network control decisions
- Causal explanations vs. correlational (SHAP/LIME limitations)
- Operator-friendly visualization of AI decisions
- Formal verification of XAI-guided network policies

---

### 14. Embodied AI for Low-Altitude Economy Networking

**Key Papers:**
- "EIoT: Embodied Intelligence of Things" -- Science China Press, 2025
- "Embodied Edge Intelligence Meets Near Field Communication" -- arXiv, Aug 2025
- IEEE TNSE Special Issue: "Embodied AI for Low-Altitude Economy Networking" (deadline: March 31, 2026)

**What makes it exciting:** This addresses the communication needs of robots, drones, and autonomous systems in the emerging "low-altitude economy." Embodied AI agents need to simultaneously perceive, communicate, and act -- creating new joint optimization problems. The IEEE TNSE special issue signals this as a recognized emerging direction.

**GPU/Resources:** Simulation-based (drone simulators like AirSim + network simulators). Can also involve physical robot experiments.

**Research Gaps:**
- Communication-aware path planning for embodied agents
- Multi-agent communication protocols for robot swarms
- Edge computing resource allocation for embodied AI workloads
- Semantic communication tailored for robot-to-robot interaction
- Safety-critical communication guarantees for autonomous drones

---

### 15. Agentic AI for Autonomous Network Operations

**Key Papers:**
- SIGCOMM 2025: "Confucius" multi-agent LLM framework (Meta)
- IEEE TNSE Special Issue: "Secure, Trustworthy, and Autonomous Intelligent Edge Networking with Agentic AI" (deadline: March 15, 2026)
- IEEE TNSE Special Issue: "Integrating Agentic AI in Intelligent Wireless Networks" (deadline: May 1, 2026)
- IEEE OJCOMS Special Issue: "Orchestrating Computing, Communication, and Agentic AI for Human-Centric Wireless Systems" (deadline: May 15, 2026)

**What makes it exciting:** "Agentic AI" -- AI systems that can autonomously plan, reason, and execute multi-step tasks -- is THE hottest topic in AI right now. Its application to networking is just beginning. FOUR separate IEEE special issues on agentic AI for networks signals massive editorial interest. This is the next evolution beyond simple ML-for-networking.

**GPU/Resources:** Dry lab. Requires LLM APIs and network simulation environments.

**Research Gaps:**
- Safety and guardrails for autonomous network agents
- Multi-agent negotiation for cross-domain network management
- Agentic AI for cross-layer optimization
- Benchmarks for evaluating autonomous network agents
- Human-in-the-loop override mechanisms

---

## TIER 3: STRONG EMERGING DIRECTIONS (Active Research Communities)

---

### 16. AI-Native WiFi / WiFi Sensing

**Key Papers:**
- "Machine Learning & Wi-Fi: Path Towards AI/ML-Native IEEE 802.11 Networks" -- arXiv, 2024
- "Next-Generation Wi-Fi Networks with Generative AI" -- arXiv, 2024
- "From Wi-Fi 7 to Wi-Fi 8: Survey of Technological Evolution" -- Computer Networks, 2025

**What makes it exciting:** WiFi 7 (802.11be) and WiFi 8 (802.11bn) are being designed with AI/ML as native components. WiFi sensing (using WiFi signals for presence detection, gesture recognition, health monitoring) is a NEW application domain. ML can reduce MIMO overheads by learning compressed channel representations.

**GPU/Resources:** Simulation + potential for real WiFi hardware experiments. Dry lab feasible.

**Research Gaps:**
- AI-driven multi-AP coordination for WiFi 8
- Generative AI for WiFi traffic prediction and resource allocation
- Cross-device WiFi sensing generalization
- Privacy-preserving WiFi sensing
- Joint WiFi communication + sensing optimization

---

### 17. Quantum Machine Learning for Communications

**Key Venues:**
- IEEE GLOBECOM 2025 Workshop: "Quantum ML for Reliable Communications in UIoT"
- IEEE WCNC 2025 Workshop: "Quantum Computing for Communications and Learning"
- QCNC 2025 Conference (March-April 2025, Nara, Japan)
- IEEE JSAC Quantum Series: "Quantum Communications and Networking" (continuous submissions, pub from July 2026)

**What makes it exciting:** Quantum ML promises exponential speedups for specific optimization problems in communications. Major companies (Google, IBM) have demonstrated quantum advantage. The IEEE JSAC now has a CONTINUOUS quantum series -- signaling permanent editorial commitment to this topic.

**GPU/Resources:** Quantum simulators (Qiskit, PennyLane, Cirq) on classical hardware. Can use IBM Quantum Experience for small experiments.

**Research Gaps:**
- Practical quantum advantage for specific wireless problems (spectrum allocation, beamforming)
- Hybrid quantum-classical algorithms for near-term quantum devices
- Quantum federated learning with communication constraints
- Error mitigation for noisy quantum circuits applied to networking
- Quantum-secure communication protocols with ML enhancement

---

### 18. Fluid Antenna Systems (FAS) for Next-Gen Wireless

**Key Reference:** IEEE OJCOMS Special Issue: "Fluid Antenna Systems for Next-Generation Wireless Communications" (deadline: March 15, 2026)

**What makes it exciting:** Fluid antennas can dynamically change their physical position/shape to optimize signal reception, unlike fixed antennas. This introduces a NEW degree of freedom for wireless optimization. Combined with AI for real-time position/configuration optimization, this is a genuinely novel hardware-software co-design direction.

**GPU/Resources:** Primarily simulation-based with electromagnetic modeling.

**Research Gaps:**
- AI-driven fluid antenna position optimization
- Joint fluid antenna + RIS optimization
- Channel estimation for fluid antenna systems
- Multi-user scheduling with fluid antennas
- Practical hardware implementation challenges

---

### 19. Integrated Sensing, Communication, AND Computation (ISCC)

**Key Papers:**
- "A Survey on Integrated Sensing, Communication, and Computation" -- arXiv, Aug 2024
- "ISCC with Adaptive DNN Splitting in Multi-UAV Networks" -- **IEEE TWC**, 2024
- "ISCC for Over-the-Air Federated Edge Learning" -- arXiv, Aug 2025
- IEEE IoTJ Special Issue: "Integrated Sensing, Memory, Communication, and Computation for Large-Scale AI Based IoT" (deadline: May 15, 2026)

**What makes it exciting:** Goes beyond ISAC by adding computation as a third dimension. Triple-functional signal design uses one signal to perform sensing, communication, AND computation simultaneously. The addition of "Memory" in the IEEE IoTJ special issue (ISMCC) adds a fourth dimension.

**GPU/Resources:** Simulation-based. Mathematical optimization + deep learning.

**Research Gaps:**
- Triple-functional waveform design optimization
- Resource allocation for three competing objectives
- ISCC for autonomous driving scenarios
- Information-theoretic limits of ISCC
- Practical ISCC protocol design for 6G

---

### 20. Knowledge-Driven Deep Learning for Wireless Optimization

**Key Paper:** "A Comprehensive Survey of Knowledge-Driven Deep Learning for Intelligent Wireless Network Optimization in 6G" -- **IEEE COMST**, 2025

**What makes it exciting:** Published in the #1 ranked journal (impact factor 46.7). This approach integrates domain knowledge (physics, communication theory) into neural network design, combining the strengths of model-based and data-driven approaches. It addresses the key limitations of pure deep learning: insufficient datasets and poor interpretability.

**GPU/Resources:** 100% simulation-based / dry lab.

**Research Gaps:**
- Systematic methods for encoding wireless domain knowledge
- Knowledge-driven architecture search
- Physics-informed neural networks for specific wireless problems
- Benchmarking knowledge-driven vs. purely data-driven approaches
- Transfer of knowledge across different wireless systems

---

### 21. LLM-Driven Network Traffic Engineering

**Key Papers:**
- "NetLLM: Adapting Large Language Models for Networking" -- **ACM SIGCOMM 2024**
- "Hattrick: Solving Multi-Class TE using Neural Models" -- ACM SIGCOMM 2025
- "Towards LLM-Based Failure Localization in Production-Scale Networks" -- ACM SIGCOMM 2025

**What makes it exciting:** NetLLM (SIGCOMM 2024) showed LLMs can be adapted for networking tasks. SIGCOMM 2025 continued this trend with LLM-based failure localization and neural traffic engineering. This represents a new paradigm where general-purpose AI models are repurposed for network optimization.

**GPU/Resources:** Requires GPU for LLM fine-tuning. Can use open-source LLMs. Dry lab.

**Research Gaps:**
- Efficient LLM adaptation with minimal networking data
- Real-time LLM inference for network control
- Multi-modal LLMs that process both text and network telemetry
- LLM-based "copilot" for network operators
- Benchmarks for LLM networking capabilities

---

### 22. Networking Infrastructure for LLM Training and Inference

**Key Papers (all ACM SIGCOMM 2025):**
- "Alibaba Stellar: New Generation RDMA Network for Cloud AI"
- "InfiniteHBD: Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching"
- "MegaScale-Infer: Efficient MoE Model Serving with Disaggregated Expert Parallelism"
- "ByteScale: Communication-Efficient LLM Training with 2048K Context on 16384 GPUs"
- "Astral: Datacenter Infrastructure for Large Language Model Training at Scale"
- "SGLB: Scalable Global Load Balancing in Commodity AI Clusters"

**What makes it exciting:** 6 out of ~75 SIGCOMM 2025 papers (8%) are about networking for AI/LLM workloads. This is the HOTTEST infrastructure topic. The scale is staggering (16,384 GPUs, 2M context length). These represent real production systems from ByteDance, Alibaba, and others.

**GPU/Resources:** Requires large-scale simulation or access to cluster environments. Some aspects can be studied in simulation.

**Research Gaps:**
- Co-design of network topology and LLM parallelism strategy
- Optical switching for dynamic MoE routing
- Network-aware LLM serving optimization
- Wireless backhaul for distributed AI training
- Energy-efficient networking for AI workloads

---

### 23. Multi-Agent Reinforcement Learning for Distributed Wireless Networks

**Key Papers:**
- "Multi-Agent RL in Wireless Distributed Networks for 6G" -- arXiv, Feb 2025
- "Multi-Agent RL for Distributed Resource Allocation in Cell-Free Massive MIMO-Enabled MEC" -- IEEE Xplore, 2023
- "Multi-Agent RL for Wireless Networks Against Adversarial Communications" -- IEEE Xplore, 2024

**What makes it exciting:** Centralized optimization doesn't scale for future distributed networks. MARL enables agents to learn cooperative/competitive strategies with decentralized execution. Applications span spectrum sharing, power control, and MAC protocol learning.

**GPU/Resources:** Simulation-based. Standard RL libraries (Stable-Baselines3, RLlib).

**Research Gaps:**
- Scalable MARL beyond 10-20 agents
- Communication-efficient MARL (agents need to share info over wireless)
- Safe MARL with guaranteed minimum performance
- Multi-task MARL for joint optimization
- MARL for heterogeneous network architectures

---

### 24. Joint Source-Channel Coding (DeepJSCC) -- New Frontiers

**Key Papers:**
- "DD-JSCC: Dynamic Deep Joint Source-Channel Coding" -- **IEEE ICC 2025**
- "D2-JSCC: Digital Deep Joint Source-Channel Coding" -- **IEEE Xplore**, 2025
- "DeepJSCC-f: Deep JSCC of Images with Feedback" -- IEEE JSTSP

**What makes it exciting:** DeepJSCC is maturing with new variants: dynamic architectures that adapt to device capabilities (2dB PSNR improvement + 40% training cost reduction), digital implementations compatible with existing systems, and feedback-aware designs. This is moving from theoretical novelty toward practical deployment.

**GPU/Resources:** 100% simulation-based / dry lab. GPU for training encoder-decoder networks.

**Research Gaps:**
- DeepJSCC for video (temporal correlation exploitation)
- Multi-user DeepJSCC with interference awareness
- Standardization-compatible DeepJSCC designs
- DeepJSCC for multi-modal data (text + image + sensor)
- Graceful degradation under extreme channel conditions

---

### 25. Advanced Waveform Design with AI

**Key Reference:** IEEE JSAC Special Issue: "Advanced Waveforms Embracing Channel Dynamics for Future Wireless Systems" (deadline: March 15, 2026)

**What makes it exciting:** This JSAC special issue targets the co-design of waveforms with channel dynamics -- moving beyond static waveform design. AI can learn optimal waveforms that adapt to channel conditions in real-time, potentially replacing handcrafted OFDM-based designs.

**GPU/Resources:** Simulation-based. Signal processing tools + deep learning.

**Research Gaps:**
- End-to-end learned waveforms for specific channel types
- Waveform design for joint communication and sensing
- AI-designed waveforms that satisfy spectral mask constraints
- Waveform optimization for energy harvesting systems
- Compatibility with existing standards

---

### 26. Bio-Inspired Communication and Molecular Communications

**Key Reference:** IEEE TMBMC Special Issue: "Bio-Inspired Communication and Information Processing for Self-Organized Living Systems" (deadline: Feb 28, 2026)

**What makes it exciting:** Molecular communication for in-body nanoscale networks (drug delivery, health monitoring) is a genuinely different domain. AI can optimize molecule release patterns, detect signals in high-noise biological environments, and enable self-organizing nanonetworks.

**GPU/Resources:** Simulation-based with molecular dynamics and stochastic channel models.

**Research Gaps:**
- Deep learning for molecular channel estimation
- RL-based drug delivery optimization
- Information theory for molecular MIMO
- Integration with macro-scale communication networks
- Biocompatible communication protocols

---

### 27. Federated Unlearning for Privacy-Preserving Networks

**Key Reference:** IEEE TNSE Special Issue: "Federated Machine Learning and Unlearning for Privacy-Preserving Networked Intelligence" (deadline: June 1, 2026)

**What makes it exciting:** Federated UNLEARNING is brand new -- the ability to remove a client's contribution from a trained model (required by GDPR "right to be forgotten"). This creates novel challenges at the intersection of machine unlearning, communication efficiency, and privacy guarantees.

**GPU/Resources:** Simulation-based / dry lab.

**Research Gaps:**
- Efficient unlearning protocols that minimize communication
- Verification that unlearning actually occurred
- Unlearning in heterogeneous wireless networks
- Impact of unlearning on model utility
- Adversarial attacks on unlearning mechanisms

---

### 28. AI-on-RAN for Backhaul-free Edge Inference (AoRA)

**Key Paper:** "AoRA: AI-on-RAN for Backhaul-free Edge Inference" -- ACM SIGCOMM 2025 (short paper)

**What makes it exciting:** Performs AI inference directly on RAN infrastructure without requiring backhaul to a cloud/edge server. This dramatically reduces latency and eliminates backhaul as a bottleneck. Published at SIGCOMM 2025.

**GPU/Resources:** Requires RAN simulation environment. Primarily software-based.

**Research Gaps:**
- Which AI tasks are suitable for RAN-native inference?
- Resource sharing between communication and inference functions
- Multi-cell coordination for distributed RAN inference
- Model update mechanisms without backhaul
- Integration with O-RAN architecture

---

### 29. Physical Layer Security with Deep Learning

**Key Papers:**
- "Enhancing Security in 5G NR with Channel-Robust RF Fingerprinting" -- **IEEE TIFS**, 2025
- "Physical Layer-Based Device Fingerprinting: From Theory to Practice" -- arXiv, June 2025

**What makes it exciting:** Deep learning enables RF fingerprinting that works across different channel conditions. New approaches use denoise diffusion models for robust fingerprinting, and adversarial learning to test authentication resilience. With the explosion of IoT devices, physical-layer authentication is critical.

**GPU/Resources:** Simulation + potential real hardware experiments. Can use software-defined radios.

**Research Gaps:**
- Generalization across different environments and hardware
- Adversarial robustness of deep learning-based authentication
- Lightweight fingerprinting for IoT devices
- Integration with upper-layer security protocols
- Large-scale deployment challenges

---

### 30. Multimodal and Multi-Agent Systems in 6G

**Key Reference:** IEEE OJCOMS Special Issue: "Multimodal and Multi-Agent Systems in 6G Networks" (deadline: May 31, 2026)

**What makes it exciting:** 6G networks will need to handle diverse data types (images, text, sensor data, point clouds) and coordinate autonomous agents. This special issue signals the convergence of multimodal AI with next-generation networking -- a truly interdisciplinary frontier.

**GPU/Resources:** Simulation-based / dry lab.

**Research Gaps:**
- Multi-modal data fusion for network optimization
- Communication protocols for multi-agent AI coordination
- Semantic communication for multi-modal data
- Resource allocation for multimodal AI workloads
- Cross-modal generalization in wireless systems

---

### 31. Concept-Based Explainability for Learning-Enabled Network Systems

**Key Paper:** "Agua: A Concept-Based Explainer for Learning-Enabled Systems" -- ACM SIGCOMM 2025

**What makes it exciting:** Goes beyond standard XAI (SHAP/LIME) to concept-based explanations that are more meaningful to network operators. Published at SIGCOMM 2025, indicating high novelty and impact.

**GPU/Resources:** Dry lab. Standard ML + XAI frameworks.

**Research Gaps:**
- Domain-specific concepts for wireless network explanation
- Real-time concept extraction for online network control
- Concept-based safety verification
- Integration with intent-based networking
- User studies with network operators

---

### 32. Age of Information Optimization with Deep Learning

**Key Papers:**
- Multiple IEEE papers on AoI-aware scheduling with DRL (2024-2025)
- Applications in IoT, vehicular networks, smart warehouses

**What makes it exciting:** AoI is a fundamentally different metric from throughput or latency -- it measures information FRESHNESS. DRL-based scheduling that minimizes AoI while managing energy is an active and growing area, especially for real-time IoT and digital twin applications.

**GPU/Resources:** 100% simulation-based / dry lab.

**Research Gaps:**
- AoI optimization for multi-modal sensing
- AoI-aware semantic communication
- Joint AoI and privacy optimization
- AoI in massive IoT deployments (scalability)
- Theoretical AoI bounds with learning-based policies

---

### 33. Foundation Model-Empowered Intelligent Edge Networks

**Key Reference:** IEEE TNSE Special Issue: "Foundation Model Empowered Intelligent Networks at the AI-Native Edge" (deadline: Feb 28, 2026)

**What makes it exciting:** Deploying foundation models AT the network edge (not in the cloud) creates new challenges in model compression, distributed inference, and communication-computation tradeoffs. This is different from "foundation models for wireless" -- it's about making edge networks intelligent enough to run foundation models.

**GPU/Resources:** Simulation + edge computing testbed experiments possible.

**Research Gaps:**
- Edge-optimized foundation model architectures
- Collaborative inference across edge nodes
- Dynamic model selection based on task and resources
- Foundation model fine-tuning at the edge
- Privacy-preserving foundation model serving

---

### 34. Holographic MIMO Communications

**Key Papers:**
- "Holographic MIMO Communications: Theoretical Foundations, Enabling Technologies, and Future Directions" -- **IEEE COMST**, 2023 (highly cited)
- "Dual-Channel Near-Field Holographic MIMO Communications" -- **Nature Communications**, 2025

**What makes it exciting:** Holographic MIMO uses spatially continuous apertures with massive elements, enabling holographic radio that can precisely control electromagnetic fields. The Nature Communications publication (2025) validates this as a cutting-edge direction. Combined with AI for beam/channel management, this is a key 6G technology.

**GPU/Resources:** Electromagnetic simulation + ML frameworks. Primarily simulation-based.

**Research Gaps:**
- AI-driven holographic beamforming optimization
- Near-field channel estimation with deep learning
- Holographic MIMO + RIS joint optimization
- Hardware-efficient holographic MIMO implementations
- Holographic MIMO for ISAC applications

---

### 35. Blockchain and Theoretical Intelligent Networks

**Key Reference:** IEEE TNSE Special Issue: "Theoretical Intelligent Blockchain Networks" (deadline: June 15, 2026)

**What makes it exciting:** Using blockchain for decentralized, trustworthy AI model sharing and incentive mechanisms in wireless networks. The "theoretical" focus suggests formal analysis of convergence, security, and efficiency.

**GPU/Resources:** Simulation-based.

**Research Gaps:**
- Lightweight blockchain for resource-constrained wireless devices
- Smart contracts for automated spectrum trading
- Blockchain-enabled federated learning with verifiable contributions
- Cross-chain interoperability for multi-operator networks
- Formal security analysis of blockchain-based wireless systems

---

## IEEE JSAC SPECIAL ISSUES STILL ACCEPTING PAPERS (as of Feb 2026)

| Special Issue | Deadline | Publication |
|---|---|---|
| Digital Twins for Wireless Networks | May 1, 2026 | Q4 2026 |
| Advanced Waveforms Embracing Channel Dynamics | March 15, 2026 | Q1 2027 |
| Quantum Communications and Networking (continuous) | Ongoing | Mar/Jul/Nov |

## IEEE TNSE SPECIAL ISSUES ACCEPTING PAPERS

| Special Issue | Deadline |
|---|---|
| Foundation Model Empowered Intelligent Networks at the AI-Native Edge | Feb 28, 2026 |
| Secure, Trustworthy, and Autonomous Intelligent Edge with Agentic AI | March 15, 2026 |
| Fusing Digital Twins and World Models for Proactive 6G Networks | March 15, 2026 |
| Embodied AI for Low-Altitude Economy Networking | March 31, 2026 |
| Integrating Agentic AI in Intelligent Wireless Networks | May 1, 2026 |
| Federated ML and Unlearning for Privacy-Preserving Networks | June 1, 2026 |
| Knowledge-driven Autonomous Agent Systems | June 15, 2026 |
| Theoretical Intelligent Blockchain Networks | June 15, 2026 |
| AI-Driven Low-Altitude Intelligent Networks | Aug 1, 2026 |

## IEEE IoTJ SPECIAL ISSUES ACCEPTING PAPERS

| Special Issue | Deadline |
|---|---|
| Integrated Sensing, Memory, Communication, Computation for Large-Scale AI IoT | May 15, 2026 |
| AI at the Edge for Vehicular and Low-Altitude IoT Networks | May 31, 2026 |
| Large Model-Driven Intelligent Computing Optimization in AIoT | June 15, 2026 |
| IoT Empowered AI4Science | June 30, 2026 |

## IEEE OJCOMS SPECIAL ISSUES ACCEPTING PAPERS

| Special Issue | Deadline |
|---|---|
| Fluid Antenna Systems for Next-Gen Wireless | March 15, 2026 |
| Orchestrating Computing, Communication, and Agentic AI | May 15, 2026 |
| Resilient and Trustworthy Communications for 6G | May 30, 2026 |
| Multimodal and Multi-Agent Systems in 6G Networks | May 31, 2026 |
| Advanced Wireless for Satellite-Terrestrial Integrated Networks | June 30, 2026 |

---

## TOP RECOMMENDED RESEARCH DIRECTIONS (for dry lab / GPU-based research)

Based on novelty, publication momentum, and feasibility without physical hardware:

### A-Tier (Start immediately -- highest impact window)
1. **Diffusion models for physical layer** -- Very few papers, huge opportunity, 100% dry lab
2. **Agentic AI for autonomous networks** -- 4+ special issues, massive editorial demand
3. **World models + digital twins for proactive 6G** -- Dedicated TNSE special issue
4. **Causal inference for wireless** -- Almost unexplored, very novel
5. **Self-supervised/contrastive wireless foundation models** -- Nature Comms validation

### B-Tier (Strong directions with more competition)
6. **Intent-driven management with LLMs** -- Production-proven (Meta), SIGCOMM validated
7. **Neuromorphic computing for green wireless** -- Unique angle, sustainability narrative
8. **AI for O-RAN xApps** -- Practical impact, industry pull
9. **GNN for scalable resource management** -- Mature theory, new model-based variants
10. **Explainable AI for network slicing** -- Regulatory driver (EU AI Act)

---

## SOURCES

- [IEEE JSAC Call for Papers](https://www.comsoc.org/publications/journals/ieee-jsac/cfp)
- [IEEE JSAC 2025 Highly Cited Papers Blog](https://www.comsoc.org/publications/blogs/selected-ideas-communications/new-impact-factor-ieee-jsac-2025-and-highly-cited-papers-2021-2025)
- [ACM SIGCOMM 2025 Accepted Papers](https://conferences.sigcomm.org/sigcomm/2025/accepted-papers/)
- [IEEE INFOCOM 2025 Awards](https://infocom2025.ieee-infocom.org/awards)
- [IEEE Special Issue Tracker](https://klb2.github.io/ieee-special-issue-tracker/)
- [IEEE COMST Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9739)
- [IEEE TWC Journal](https://www.comsoc.org/publications/journals/ieee-twc)
- [IEEE GLOBECOM 2025 Workshops](https://globecom2025.ieee-globecom.org/)
- [IEEE WCNC 2025 Workshops](https://wcnc2025.ieee-wcnc.org/program/workshops)
- [IEEE ICMLCN 2025](https://icmlcn2025.ieee-icmlcn.org/)
- [IMDEA Networks - INFOCOM 2025 Best Paper](https://networks.imdea.org/imdea-networks-work-on-distributed-inference-in-mobile-networks-wins-best-paper-award-at-ieee-infocom-2025/)
