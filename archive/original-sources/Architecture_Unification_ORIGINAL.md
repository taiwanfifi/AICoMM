# Architecture Unification: Unified Meta-Framework for Semantic Transport Layer

## Executive Summary

This document provides a **unified architectural view** that reconciles the three previously independent layer frameworks:
- **FM-Agent 5-Layer** (from agent.md): Application → Agent → Model → Resource → Execution
- **IoA 4-Layer** (from IOA.md): Application → Coordination → Management → Infrastructure
- **SASL 5-Layer** (from t3.md): L0-L4 Semantic-Aware Sparse Layer
- **Professor's Semantic Plane** (from professor_concepts.md): New layer spanning traditional OSI

**Core Innovation**: Definition of the **Semantic Transport Layer (STL)** as a first-class communication paradigm for AI-native 6G networks.

---

## 1. Unified Layer Mapping Matrix

The following matrix establishes the **canonical mapping** across all architectural views:

| **FM-Agent Layer** | **IoA Layer** | **SASL Layer** | **OSI Analogy** | **Professor's Concept** | **Primary Function** | **Key Components** |
|-------------------|--------------|---------------|----------------|------------------------|---------------------|-------------------|
| **Application** | L4: Application Scenarios | - | L7 | Agent Logic / Task Goals | User-facing services | ChatGPT, Autonomous Vehicles, Smart Factory |
| **Agent** | L3: Agent Coordination | - | L6 | Control Plane | Planning, Memory, Tool Use | Multi-agent orchestration, Task decomposition |
| **Model (Compression)** | - | L2: Packetization | L5 | **Semantic Transport** | Token reduction, DSA | Model compression, Token pruning/merging |
| **Resource (Parallelism)** | L2: Agent Management | L1: Indexer | L4 | Resource Allocation | KV-Cache allocation, Scheduling | Parallelism (DP/TP/PP), Autoscaling |
| **Execution (Hardware)** | L1: Infrastructure | L0: Perception | L1-L3 | Bit Transport | Physical transmission | Edge devices, 5G/6G PHY, GPU/NPU |
| **[NEW] STL Data Plane** | [Embedded in L3] | L3: Sync Channel | **NEW** | **Semantic Data Plane** | KV-Cache Delta Streaming | Token encoding, Compression, Delta transmission |
| **[NEW] STL Management** | [Embedded in L2] | L4: Reconstruction | **NEW** | **Semantic Management** | Heterogeneity handling | Neural projector, RAG-based recovery |

### Key Insights from Mapping

1. **STL is NOT a replacement** for OSI layers - it's an **overlay protocol** that bridges L4-L7
2. **MCP's correct role**: Semantic Control Plane (handshake/negotiation), NOT application-layer API
3. **KV-Cache is the communication unit**: Analogous to how IP uses packets, STL uses semantic state deltas
4. **Three sub-layers of STL** (detailed in Section 3):
   - Control Plane (MCP, goal negotiation)
   - Data Plane (Token streaming, KV-Cache delta)
   - Management Plane (Resource allocation, model selection)

---

## 2. Meta-Architecture: The Complete System View

```
┌─────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                              │
│  (ChatGPT, Autonomous Vehicles, Smart Factory, UAV Swarms)      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FM-Agent: Application Services                           │   │
│  │ IoA: Application Scenarios                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  AGENT LAYER (Control Logic)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FM-Agent: Agent Layer (Planning, Memory, Tool Use)       │   │
│  │ IoA: Agent Coordination (Task Orchestration)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌═════════════════════════════════════════════════════════════════┐
║  🔥 SEMANTIC TRANSPORT LAYER (STL) - Core Contribution         ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ Control Plane (MCP)                                      │   ║
║  │  - Schema Negotiation (embedding dim, quantization)      │   ║
║  │  - Goal/Task Declaration (fire detection, object track)  │   ║
║  │  - Error Recovery Policy (RAG fallback, retransmit)      │   ║
║  ├─────────────────────────────────────────────────────────┤   ║
║  │ Data Plane (Token Streaming)                             │   ║
║  │  - Semantic Indexer (DSA-based selection)                │   ║
║  │    ↳ SASL L1: Lightning Indexer for attention-driven    │   ║
║  │  - Token Packetization (Encoding/Compression)            │   ║
║  │    ↳ SASL L2: Quantization (FP8/FP16), Serialization     │   ║
║  │  - KV-Cache Delta Transmission                           │   ║
║  │    ↳ SASL L3: Differential streaming, Sync protocol      │   ║
║  ├─────────────────────────────────────────────────────────┤   ║
║  │ Management Plane (Resource Optimization)                 │   ║
║  │  - Bandwidth Allocation (adaptive compression ratio)     │   ║
║  │  - Model Selection (Edge MobileVLM vs Cloud GPT-4V)      │   ║
║  │  - Heterogeneity Handling (Neural Projector alignment)   │   ║
║  │    ↳ SASL L4: Projector training, KV-Cache reconstruction│   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                 ║
║  Mapping:                                                       ║
║  - FM-Agent Model Layer → STL Data Plane                        ║
║  - IoA Agent Coordination → STL Control Plane                   ║
║  - SASL L0-L4 → STL complete implementation                     ║
╚═════════════════════════════════════════════════════════════════╝
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  RESOURCE LAYER                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FM-Agent: Resource Allocation (Parallelism, Scaling)     │   │
│  │ IoA: Agent Management (Discovery, Registration)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTION/INFRASTRUCTURE LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FM-Agent: Execution (Hardware Optimization)              │   │
│  │ IoA: Infrastructure (5G/6G, Computing Power)             │   │
│  │ SASL L0: Perception (Sensors, Multimodal Input)          │   │
│  │                                                           │   │
│  │ Physical Components:                                      │   │
│  │  - Edge: MobileVLM, LiDAR, Camera (SASL L0 Perception)   │   │
│  │  - Cloud: GPT-4V, LLaMA, DeepSeek (Model inference)      │   │
│  │  - Network: 5G NR, 6G, WiFi (Bit-level transmission)     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed STL Specification

### 3.1 Control Plane (MCP's Correct Role)

**NOT**: Application-layer API for tool calling
**IS**: Semantic handshake protocol for agent-to-agent communication

#### Functions:
1. **Schema Negotiation**
   - Embedding dimension alignment (e.g., Edge 512-dim ↔ Cloud 4096-dim)
   - Quantization format agreement (FP8, FP16, INT4)
   - Compression codec selection (Arithmetic coding, Huffman)

2. **Goal Declaration**
   - Task-oriented context: `{ task: "fire_detection", priority: "high", deadline: "500ms" }`
   - Semantic importance threshold: `{ attention_threshold: 0.8 }`

3. **Error Recovery**
   - Packet loss policy: Semantic redundancy (重複高attention token)
   - RAG fallback: Knowledge base補全 (when delta > threshold)

**Protocol Example**:
```json
{
  "mcp_version": "2.0",
  "agent_id": "edge_uav_01",
  "capabilities": {
    "model": "MobileVLM",
    "kv_dim": 512,
    "modalities": ["vision", "lidar"],
    "compression": ["fp8", "zstd"]
  },
  "task": {
    "goal": "fire_source_detection",
    "context": "forest_patrol",
    "attention_threshold": 0.85
  }
}
```

---

### 3.2 Data Plane (Token Streaming)

**Core Innovation**: Transmission unit is NOT packet, but **Semantic Token**

#### Pipeline:
```
Raw Sensor Data (SASL L0)
    ↓
Perception & Embedding (Edge Model: MobileVLM)
    ↓
Semantic Indexer (SASL L1: Lightning Indexer)
    ├─ Compute importance = f(embedding, task_context, attention_map)
    ├─ Select top-k tokens (k determined by bandwidth)
    └─ Discard background/redundant features
    ↓
Token Packetization (SASL L2)
    ├─ Quantization: FP32 → FP8 (3.5-4.3x compression, per CacheGen)
    ├─ Serialization: Protobuf/JSON schema
    └─ Compression: Arithmetic coding
    ↓
KV-Cache Delta Transmission (SASL L3)
    ├─ Initial: Full KV-Cache baseline (compressed <1KB for scene summary)
    ├─ Incremental: Only Δ(KV_t - KV_{t-1}) when semantic drift > ε
    └─ Silence: Zero transmission when no attention shift
    ↓
Receiver Reconstruction (Cloud)
```

**Semantic Token Format**:
```protobuf
message SemanticToken {
  uint32 token_id;
  enum Modality { VISION, LIDAR, AUDIO, TEXT };
  Modality modality;
  uint64 timestamp_us;

  message Payload {
    enum SemanticType { FIRE, HUMAN, VEHICLE, ANOMALY };
    SemanticType semantic_type;

    oneof spatial_scope {
      BoundingBox bbox;
      PointCloud pointcloud;
      GPS gps;
    }

    float confidence;  // FP16 quantized
    map<string, Value> attributes;  // Extensible
  }
  Payload payload;

  enum Compression { NONE, ZSTD, ARITHMETIC };
  Compression compression;
}
```

---

### 3.3 Management Plane (Heterogeneity & Resource)

#### Challenge:
Edge model (MobileVLM, 512-dim KV) ≠ Cloud model (GPT-4V, 4096-dim KV)

#### Solution: Neural Projector + Adaptive Resource Allocation

**Projector Design**:
```python
class KVCacheProjector(nn.Module):
    """
    Projects KV-Cache from edge model to cloud model dimension space.
    Based on C2C (Cache-to-Cache) neural fuser approach.
    """
    def __init__(self, d_source=512, d_target=4096):
        super().__init__()
        self.linear = nn.Linear(d_source, d_target)
        self.residual = nn.Parameter(torch.zeros(d_target))

    def forward(self, kv_cache_edge):
        # Project + residual connection
        kv_projected = self.linear(kv_cache_edge) + self.residual
        return kv_projected

    def inverse_project(self, kv_cache_cloud):
        # For cloud-to-edge feedback (optional)
        return F.linear(kv_cache_cloud - self.residual,
                       self.linear.weight.T)
```

**Training Method** (see Section 5 of `KV_Cache_Alignment_Design.md` for full details):
1. Distillation: Train projector using paired (edge_output, cloud_output) on same input
2. Fine-tuning: End-to-end on downstream task (fire detection accuracy)

**Resource Allocation**:
- Low bandwidth → Increase compression ratio (FP8 → INT4), reduce k in top-k
- High semantic drift → Trigger full re-sync (reset KV-Cache baseline)
- Model selection: Route simple queries to edge, complex to cloud

---

## 4. Cross-Document Consistency Checklist

### ✅ Terminology Standardization

| **Term** | **Canonical Definition** | **Usage Context** |
|---------|-------------------------|------------------|
| **Semantic Token** | Minimal self-contained semantic unit with (concept, confidence, scope, time) | STL Data Plane transmission unit |
| **KV-Cache Streaming** | Differential transmission of Transformer KV-Cache states | STL core mechanism (NOT KV-Cache management in model layer) |
| **MCP** | Model Context Protocol: Semantic Control Plane for agent handshake | STL Control Plane (NOT application API) |
| **Semantic Compression** | Task-aware reduction of semantic state, preserving task-relevant information | Communication layer optimization (NOT model-level compression like pruning) |
| **Lightning Indexer** | Attention-driven importance estimator for semantic token selection | SASL L1 component, analogous to DeepSeek DSA |
| **Semantic State Synchronization** | Core paradigm: Align receiver's world model state to sender's state | Replaces traditional "bit-perfect data transmission" |

### ✅ Layer Function Clarity

**Each document MUST reference this unified view**:

- **agent.md**:
  - Model Layer → Maps to STL Data Plane (compression/token reduction)
  - Execution Layer → SASL L0 (perception) + Infrastructure
  - Communication Optimization (Section VI.C) → Points to STL specification

- **IOA.md**:
  - Agent Coordination → STL Control Plane (MCP handshake)
  - Infrastructure → SASL L0 + traditional network stack
  - Communication Protocols (Section III.C) → Extends to include STL

- **t3.md**:
  - SASL L0-L4 → Complete STL implementation
  - Professor's feedback → Validated by STL design
  - MCP positioning → Clarified as Control Plane

- **t4-t7.md**:
  - All references to "semantic communication" → Link to STL framework
  - KV-Cache heterogeneity → Cite Management Plane projector

---

## 5. Relationship to Traditional OSI Model

### Why STL is NOT "Replacing OSI L2/L3"

**Common Misconception** (from initial research drafts):
> "Define new L2/L3 where packets are tensors"

**Reality**:
- Physical Layer (L1-L3) STILL transmits bits/symbols
- Channel coding, modulation, error correction remain NECESSARY
- Tensors/KV-Cache are **payload content**, NOT packet replacement

### Correct Positioning: STL as Overlay

```
┌─────────────────────────┐
│   Application (L7)      │
├─────────────────────────┤
│ 🔥 STL (NEW LAYER)      │  ← Semantic Transport Layer
│   Sits between L4-L7    │
├─────────────────────────┤
│   Transport (L4: TCP)   │
│   Network (L3: IP)      │
│   Data Link (L2)        │
│   Physical (L1)         │
└─────────────────────────┘
```

**STL's Relationship to OSI**:
1. **Uses L1-L4** for bit transmission (over 5G/6G PHY)
2. **Operates above L4** as semantic overlay protocol
3. **Influences cross-layer design**:
   - PHY: Semantic-aware channel coding (protect high-importance tokens)
   - MAC: Priority queuing for attention-weighted tokens
   - Network: Semantic routing (direct KV-Cache to co-located agents)

**Example Flow**:
```
Edge Agent:
  Camera Frame → MobileVLM → KV-Cache (512-dim)
    ↓ STL Data Plane
  Semantic Indexer → Top-k tokens
    ↓ STL Packetization
  Serialize to bytes → Compress
    ↓ Traditional Stack
  TCP/IP → 5G NR → Wireless Channel
    ↓
Cloud Agent:
  Receive bytes → Decompress
    ↓ STL Reconstruction
  Projector → KV-Cache (4096-dim)
    ↓ STL Management
  Inject into GPT-4V → Continue inference
```

---

## 6. Practical Implications

### 6.1 For System Designers
- **Interface Design**: Agents communicate via STL API (not raw HTTP/gRPC)
- **Resource Planning**: Budget for neural projector training overhead
- **Fault Tolerance**: Implement semantic redundancy + RAG fallback

### 6.2 For Researchers
- **Evaluation Metrics**:
  - Semantic distortion (task success rate vs. bandwidth)
  - Alignment error (KL divergence between edge/cloud world models)
  - Energy efficiency (Joules per semantic state update)
- **Baselines**:
  - H.264 video streaming
  - CLIP embedding transmission
  - Text prompt exchange (LangChain baseline)

### 6.3 For Future Work
- **Standards Development**: Propose STL as 3GPP Rel-20 study item
- **Hardware Acceleration**: Custom ASICs for semantic token encoding
- **Security**: Prevent KV-Cache side-channel attacks (per "I Know What You Asked" paper)

---

## 7. References to Related Documents

This unified architecture should be THE canonical reference. All other documents must align:

- **Core Theory**: `Theoretical_Foundations.md` (to be created in Phase 2)
- **Cost Model**: `Communication_Cost_Model.md` (to be created in Phase 1)
- **Implementation**: `KV_Cache_Alignment_Design.md` (to be created in Phase 1)
- **Evaluation**: Updates to `t5.md` (Phase 3)
- **Terminology**: `TERMINOLOGY.md` (to be created in Phase 4)

---

## 8. Validation Against Professor's Feedback

### ✅ Addresses Professor's Core Concerns (from professor_concepts.md):

1. **"未來傳Token不傳Packet"**
   → STL defines Semantic Token as transmission unit

2. **"Agent間會產生什麼行為?"**
   → Control Plane negotiation, emergent protocol adaptation

3. **"怎麼傳會好?"**
   → Attention-driven + Context-aware + Delta streaming

4. **"MCP不是應用層"**
   → Positioned as STL Control Plane (handshake/schema negotiation)

5. **"要有research flavor"**
   → Defines new communication paradigm + theoretical foundations (IB/Rate-Distortion)

---

## Conclusion

This document establishes the **single source of truth** for architectural discussions.

**Key Takeaways**:
1. STL is the **core PhD contribution** - a new communication paradigm
2. All three frameworks (FM-Agent, IoA, SASL) are **complementary views** of the same system
3. MCP's role is **Control Plane**, not application API
4. KV-Cache is the **communication unit**, analogous to how IP uses packets
5. This is **NOT** a replacement of OSI - it's an AI-native overlay that influences cross-layer design

**Next Steps**:
- Create `Communication_Cost_Model.md` (Phase 1 Task 1.3)
- Update `t3.md` to reference this unified view (Phase 1 Task 1.2)
- Develop theoretical foundations (Phase 2)
