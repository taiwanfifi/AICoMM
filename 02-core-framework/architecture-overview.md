# System Architecture Overview: Semantic Transport Layer (STL)

> **基于**: Architecture_Unification.md（Phase 1产出）
> **整合日期**: 2026-01-24
> **用途**: 统一所有架构视图，定义STL作为核心PhD贡献

---

## Executive Summary

This document provides the **unified architectural view** reconciling multiple framework perspectives:
- **FM-Agent 5-Layer** (from agent.md)
- **IoA 4-Layer** (from IOA.md)
- **SASL 5-Layer** (Semantic-Aware Sparse Layer)
- **Professor's Semantic Plane** (from advisor feedback)

**Core Innovation**: Definition of **Semantic Transport Layer (STL)** as a first-class communication paradigm for AI-native 6G networks.

---

## 1. Unified Layer Mapping Matrix

The following matrix establishes the **canonical mapping** across all architectural views:

| **FM-Agent** | **IoA** | **SASL** | **OSI** | **Professor's View** | **Function** | **Key Tech** |
|-------------|---------|----------|---------|---------------------|-------------|-------------|
| Application | L4: Scenarios | - | L7 | Agent Logic | User services | ChatGPT, Autonomous vehicles |
| Agent | L3: Coordination | - | L6 | Control Plane | Planning, Memory | Multi-agent orchestration |
| **Model** | - | **L2: Packetization** | **L5** | **🔥 Semantic Transport** | Token reduction, DSA | Compression, Token selection |
| Resource | L2: Management | L1: Indexer | L4 | Resource Allocation | KV-Cache scheduling | Parallelism, Autoscaling |
| Execution | L1: Infrastructure | L0: Perception | L1-L3 | Bit Transport | Physical transmission | Edge devices, 5G/6G PHY |

### Key Insights

1. **STL is an overlay** (NOT OSI replacement)
2. **MCP's correct role**: Semantic Control Plane (handshake), NOT application API
3. **KV-Cache as communication unit**: Analogous to IP packets
4. **Three STL sub-layers**:
   - Control Plane (MCP, goal negotiation)
   - Data Plane (Token streaming, KV-Cache delta)
   - Management Plane (Resource allocation, heterogeneity handling)

---

## 2. Meta-Architecture: Complete System View

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  (ChatGPT, Autonomous Vehicles, Smart Factory)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  AGENT LAYER (Control Logic)                                │
│  - Planning, Memory, Tool Use (FM-Agent)                    │
│  - Task Orchestration (IoA Coordination)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
╔═════════════════════════════════════════════════════════════╗
║  🔥 SEMANTIC TRANSPORT LAYER (STL) - PhD Contribution       ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ Control Plane (MCP)                                    │ ║
║  │  - Schema Negotiation (embedding dim, quantization)   │ ║
║  │  - Goal/Task Declaration (fire detection, tracking)   │ ║
║  │  - Error Recovery Policy (RAG fallback, retransmit)   │ ║
║  ├────────────────────────────────────────────────────────┤ ║
║  │ Data Plane (Token Streaming)                           │ ║
║  │  - Semantic Indexer (DSA-based, SASL L1)              │ ║
║  │  - Token Packetization (Encoding, Compression, L2)    │ ║
║  │  - KV-Cache Delta Transmission (SASL L3)              │ ║
║  ├────────────────────────────────────────────────────────┤ ║
║  │ Management Plane (Resource Optimization)               │ ║
║  │  - Bandwidth Allocation (adaptive compression)        │ ║
║  │  - Model Selection (Edge MobileVLM ↔ Cloud GPT-4V)    │ ║
║  │  - Heterogeneity: Neural Projector (SASL L4)          │ ║
║  └────────────────────────────────────────────────────────┘ ║
╚═════════════════════════════════════════════════════════════╝
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  RESOURCE LAYER                                             │
│  - Parallelism, Scaling (FM-Agent)                          │
│  - Agent Discovery, Registration (IoA Management)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  EXECUTION/INFRASTRUCTURE LAYER                             │
│  - Hardware Optimization (FM-Agent Execution)               │
│  - 5G/6G, Computing Power (IoA Infrastructure)              │
│  - Sensors, Multimodal Input (SASL L0 Perception)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed STL Specification

### 3.1 Control Plane: MCP's Correct Role

**NOT**: Application-layer API for tool calling
**IS**: Semantic handshake protocol for agent-to-agent communication

#### Functions

1. **Schema Negotiation**
   - Embedding dimension alignment (Edge 512-dim ↔ Cloud 4096-dim)
   - Quantization format (FP8, FP16, INT4)
   - Compression codec (Arithmetic, ZSTD)

2. **Goal Declaration**
   - Task context: `{task: "fire_detection", priority: "high", deadline: "500ms"}`
   - Semantic threshold: `{attention_threshold: 0.8}`

3. **Error Recovery**
   - Packet loss: Semantic redundancy (duplicate high-attention tokens)
   - RAG fallback: Knowledge base retrieval when delta > threshold

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

### 3.2 Data Plane: Token Streaming

**Core Innovation**: Transmission unit is **Semantic Token**, NOT packet.

#### Pipeline

```
Raw Sensor Data (SASL L0: Perception)
    ↓
Edge Model Inference (MobileVLM)
    ↓
Semantic Indexer (SASL L1: Lightning Indexer)
  - Compute importance = f(embedding, task, attention_map)
  - Select top-k tokens (k = bandwidth-dependent)
  - Discard background/redundant features
    ↓
Token Packetization (SASL L2)
  - Quantization: FP32 → FP8 (3.5-4.3x compression)
  - Serialization: Protobuf schema
  - Compression: Arithmetic coding
    ↓
KV-Cache Delta Transmission (SASL L3)
  - Initial: Full baseline (<1KB for scene summary)
  - Incremental: Only Δ(KV_t - KV_{t-1}) when drift > ε
  - Silence: Zero transmission when no attention shift
    ↓
Cloud Reception & Reconstruction
```

**Semantic Token Format** (详见 `03-technical-design/token-encoding.md`):
```protobuf
message SemanticToken {
  uint32 token_id;
  Modality modality;  // VISION, AUDIO, LIDAR, TEXT
  uint64 timestamp_us;

  SemanticPayload payload {
    SemanticType type;  // FIRE, HUMAN, VEHICLE
    SpatialScope scope;  // BBox, PointCloud, GPS
    float confidence;  // FP16 quantized
    map<string, Value> attributes;
  }

  CompressionType compression;  // ZSTD, ARITHMETIC
}
```

---

### 3.3 Management Plane: Heterogeneity & Resource

#### Challenge

Edge model (MobileVLM, 512-dim KV) ≠ Cloud model (GPT-4V, 4096-dim KV)

#### Solution: Neural Projector + Adaptive Allocation

**Projector Design**:
```python
class KVCacheProjector(nn.Module):
    """
    Projects KV-Cache from edge to cloud dimension space.
    Based on C2C (Cache-to-Cache) neural fuser.
    """
    def __init__(self, d_source=512, d_target=4096):
        super().__init__()
        self.linear = nn.Linear(d_source, d_target)
        self.residual = nn.Parameter(torch.zeros(d_target))

    def forward(self, kv_edge):
        return self.linear(kv_edge) + self.residual
```

**Training**: Distillation on paired (edge_output, cloud_output) samples.

**Resource Allocation**:
- Low bandwidth → Increase compression (FP8 → INT4), reduce top-k
- High drift → Trigger full re-sync
- Model selection: Route simple queries to edge, complex to cloud

---

## 4. Cross-Document Consistency

### Terminology Standardization

| Term | Definition | Usage Context |
|------|-----------|---------------|
| **Semantic Token** | Minimal semantic unit: (concept, confidence, scope, time) | STL Data Plane transmission unit |
| **KV-Cache Streaming** | Differential transmission of Transformer KV-Cache states | STL core mechanism |
| **MCP** | Model Context Protocol: Semantic Control Plane for handshake | STL Control Plane |
| **Semantic Compression** | Task-aware reduction preserving task-relevant info | Communication layer optimization |
| **Lightning Indexer** | Attention-driven importance estimator (analogous to DeepSeek DSA) | SASL L1 component |
| **Semantic State Sync** | Core paradigm: Align receiver's world model to sender's state | Replaces "bit-perfect transmission" |

### Document Cross-References

- **Core Theory**: `semantic-state-sync.md` (SSC framework)
- **Cost Model**: `../05-evaluation/cost-model.md` (Evaluation metrics)
- **Token Implementation**: `../03-technical-design/token-encoding.md` (Protobuf schema, quantization)
- **Attention Mechanism**: `../03-technical-design/attention-filtering.md` (DSA integration)

---

## 5. Relationship to Traditional OSI Model

### Why STL is NOT "Replacing OSI L2/L3"

**Misconception**: "Define new L2/L3 where packets are tensors"

**Reality**:
- Physical Layer (L1-L3) STILL transmits bits/symbols
- Channel coding, modulation, error correction remain NECESSARY
- Tensors/KV-Cache are **payload content**, NOT packet replacement

### Correct Positioning: STL as Overlay

```
┌──────────────────────┐
│  Application (L7)    │
├──────────────────────┤
│  🔥 STL (NEW)        │  ← Semantic Transport Layer
│  Between L4-L7       │
├──────────────────────┤
│  Transport (L4: TCP) │
│  Network (L3: IP)    │
│  Data Link (L2)      │
│  Physical (L1)       │
└──────────────────────┘
```

**STL's Relationship**:
1. **Uses L1-L4** for bit transmission (5G/6G PHY)
2. **Operates above L4** as semantic overlay
3. **Influences cross-layer**:
   - PHY: Semantic-aware channel coding (protect high-importance tokens)
   - MAC: Priority queuing for attention-weighted tokens
   - Network: Semantic routing (direct KV-Cache to co-located agents)

**Example Flow**:
```
Edge: Camera → MobileVLM → KV-Cache (512-dim)
  ↓ STL Data Plane
  Semantic Indexer → Top-k tokens
  ↓ STL Packetization
  Serialize → Compress
  ↓ Traditional Stack
  TCP/IP → 5G NR → Wireless Channel
  ↓
Cloud: Receive → Decompress
  ↓ STL Reconstruction
  Projector → KV-Cache (4096-dim)
  ↓
  Inject into GPT-4V → Continue inference
```

---

## 6. Validation Against Professor's Feedback

### ✅ Addresses Core Concerns (from `00-advisor-feedback/professor-concepts-raw.md`)

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

## 7. Practical Implications

### For System Designers
- **Interface**: Agents communicate via STL API (not raw HTTP/gRPC)
- **Resource Planning**: Budget for neural projector training overhead
- **Fault Tolerance**: Semantic redundancy + RAG fallback

### For Researchers
- **Evaluation Metrics**:
  - Semantic distortion (task success rate vs. bandwidth)
  - Alignment error (KL divergence between edge/cloud)
  - Energy efficiency (Joules per semantic state update)
- **Baselines**:
  - H.264 video streaming
  - CLIP embedding transmission
  - Text prompt exchange (LangChain)

### For Future Work
- **Standards**: Propose STL as 3GPP Rel-20 study item
- **Hardware**: Custom ASICs for semantic token encoding
- **Security**: Prevent KV-Cache side-channel attacks

---

## 8. Key Takeaways

1. **STL is the PhD contribution** - A new communication paradigm
2. **All frameworks are complementary** - FM-Agent, IoA, SASL unified
3. **MCP is Control Plane** - NOT application API
4. **KV-Cache is the unit** - Analogous to IP packets
5. **NOT OSI replacement** - AI-native overlay influencing cross-layer design

---

## Next Steps

- **Theory**: Formalize in `../01-problem-formulation/mathematical-system-model.md`
- **Implementation**: Detail in `../03-technical-design/token-encoding.md`
- **Evaluation**: Quantify in `../05-evaluation/cost-model.md`
