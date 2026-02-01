# Token Encoding Specification: From Concept to Binary

> **基于**: t3.md L2.1章节（Phase 1产出）
> **整合日期**: 2026-01-24
> **用途**: 详细的工程实现规范，解决"如何序列化、量化、压缩Semantic Token"

---

## Overview

前面文档定义了Semantic Token的**概念**（参见 `../02-core-framework/semantic-token-definition.md`），但缺少从concept到binary的**完整pipeline**。本文档补充工程实现细节，解决：

> **注意**：本文档描述的是 **Structured Mode**（结构化编码），适用于 Edge 已完成推理、只需传输高层语义结论的场景。
> 另一种 **Latent Mode**（KV-Cache delta 传输）的编码方式见 `06-implementation/ssc-pipeline-spec.md`。
> 两种模式的选择原则见 `../02-core-framework/semantic-token-definition.md` 的「兩種具體表示」章节。

1. **如何序列化**？（Protobuf schema）
2. **如何量化**？（FP32 → FP16/FP8/INT4）
3. **如何压缩**？（ZSTD, Arithmetic coding）
4. **如何跨模态统一**？（Vision/Audio/LiDAR）
5. **如何处理丢包**？（Redundancy + RAG）

---

## Token Encoding Pipeline（完整流程）

```
Semantic Concept (抽象)
    ↓
Structured Representation (Protobuf/JSON)
    ↓
Quantization (FP32 → FP16/FP8/INT4)
    ↓
Serialization (Binary format)
    ↓
Compression (Arithmetic Coding / ZSTD)
    ↓
Transmission (Over 5G/6G PHY)
```

---

## 1. Structured Schema Definition

使用**Protobuf**定义Semantic Token的标准格式（比JSON省50-70%带宽，向后兼容，跨平台）。

### Core Message Definition

```protobuf
// semantic_token.proto
syntax = "proto3";

message SemanticToken {
  // Header: Metadata
  uint32 token_id = 1;          // Unique identifier
  Modality modality = 2;         // Vision/Audio/LiDAR/Text
  uint64 timestamp_us = 3;       // Microsecond precision

  enum Modality {
    VISION = 0;
    AUDIO = 1;
    LIDAR = 2;
    TEXT = 3;
    MULTIMODAL = 4;
  }

  // Payload: Semantic content
  SemanticPayload payload = 4;

  // Compression metadata
  CompressionType compression = 5;

  enum CompressionType {
    NONE = 0;
    ZSTD = 1;
    ARITHMETIC = 2;
  }
}

message SemanticPayload {
  SemanticType semantic_type = 1;

  enum SemanticType {
    FIRE = 0;
    HUMAN = 1;
    VEHICLE = 2;
    ANOMALY = 3;
    OBJECT_GENERIC = 4;
  }

  // Spatial scope (variant type)
  oneof spatial_scope {
    BoundingBox bbox = 2;
    PointCloud pointcloud = 3;
    GPSCoordinate gps = 4;
  }

  // Confidence (quantized to FP16)
  float confidence = 5;  // Will be quantized before transmission

  // Extensible attributes
  map<string, AttributeValue> attributes = 6;
}
```

### Supporting Messages

```protobuf
message BoundingBox {
  float x_min = 1;
  float y_min = 2;
  float x_max = 3;
  float y_max = 4;
}

message PointCloud {
  repeated Point3D points = 1;  // Sparse representation
}

message Point3D {
  float x = 1;
  float y = 2;
  float z = 3;
}

message GPSCoordinate {
  double latitude = 1;
  double longitude = 2;
  float altitude = 3;
}

message AttributeValue {
  oneof value {
    float float_val = 1;
    int32 int_val = 2;
    string string_val = 3;
    bytes bytes_val = 4;
  }
}
```

### Why Protobuf?

- **Binary encoding** → 比JSON省50-70%带宽
- **Schema evolution** → 向后兼容（新增field不影响旧版本）
- **Cross-platform** → Edge (C++) ↔ Cloud (Python) 无缝对接
- **Built-in compression** → 结合varint编码自动优化整数

---

## 2. Quantization Policy（精度 vs. 带宽取舍）

### Confidence值的量化策略

| Precision | Bits | Range | Bandwidth Saving | Use Case |
|-----------|------|-------|------------------|----------|
| **FP32** (baseline) | 32 | Full | 0% | Debug only |
| **FP16** | 16 | ±65504 | 50% | **Default** (高精度场景) |
| **FP8** | 8 | ±240 | 75% | Edge-to-Cloud (带宽受限) |
| **INT4** | 4 | 0-15 (discrete) | 87.5% | Ultra-low bandwidth |

### Decision Rules

```python
def select_quantization(bandwidth_mbps, task_criticality):
    """
    Adaptive quantization based on network conditions and task requirements.

    Args:
        bandwidth_mbps: Available uplink bandwidth (Mbps)
        task_criticality: "safety_critical", "high", "medium", "low"

    Returns:
        Quantization format: "FP16", "FP8", or "INT4"
    """
    if task_criticality == "safety_critical":
        return "FP16"  # Fire detection, medical, autonomous driving
    elif bandwidth_mbps > 10:
        return "FP16"  # Sufficient bandwidth
    elif bandwidth_mbps > 5:
        return "FP8"   # Limited bandwidth
    else:
        return "INT4"  # Emergency fallback (may sacrifice accuracy)
```

### 坐标值的量化（Bounding Box）

**问题**: BBox坐标是浮点数，但精度过高浪费带宽。

**解决**: 根据image resolution量化到足够精度。

```python
def quantize_bbox(bbox, img_width=1920, img_height=1080):
    """
    Quantize bounding box coordinates to 11-bit integers.

    For 1920x1080 image:
    - x_min ∈ [0, 1920] → 11 bits (2^11 = 2048)
    - y_min ∈ [0, 1080] → 11 bits
    - Total: 44 bits for BBox (vs. 128 bits for FP32*4)

    Saving: 128 bits → 44 bits = 65.6% reduction
    """
    x_min_q = int(bbox.x_min / img_width * 2047)  # 11-bit
    y_min_q = int(bbox.y_min / img_height * 2047)
    x_max_q = int(bbox.x_max / img_width * 2047)
    y_max_q = int(bbox.y_max / img_height * 2047)

    return (x_min_q, y_min_q, x_max_q, y_max_q)

def dequantize_bbox(bbox_q, img_width=1920, img_height=1080):
    """Reverse quantization at receiver."""
    x_min = bbox_q[0] / 2047 * img_width
    y_min = bbox_q[1] / 2047 * img_height
    x_max = bbox_q[2] / 2047 * img_width
    y_max = bbox_q[3] / 2047 * img_height

    return BoundingBox(x_min, y_min, x_max, y_max)
```

### Quantization Error Analysis

| Component | Original (FP32) | Quantized (FP8) | Max Error | Acceptable? |
|-----------|----------------|----------------|-----------|-------------|
| **Confidence** | 0.923456 | 0.92 | ±0.004 | ✅ Yes (0.4%) |
| **BBox (x_min)** | 512.7 | 512 | ±1 pixel | ✅ Yes (<0.1%) |
| **Intensity** | 0.8731 | 0.87 | ±0.004 | ✅ Yes |

**Conclusion**: FP8/FP16 quantization introduces <0.5% error, acceptable for most tasks.

---

## 3. Compression Algorithm Selection

### Arithmetic Coding for Attributes

**Why?** Semantic tokens有高度结构性（例如fire_location的坐标分布集中在热区），适合统计压缩。

```python
# Pseudo-code: Arithmetic coding for token attributes
def compress_attributes(attributes, context_model):
    """
    Compress token attributes using adaptive arithmetic coding.

    Args:
        attributes: Dict of attribute key-value pairs
        context_model: Historical distribution of attribute values

    Returns:
        Compressed bitstream
    """
    encoder = ArithmeticEncoder(context_model)

    for key, value in attributes.items():
        # Encode using adaptive probability model
        encoder.encode(key, value)

    return encoder.get_binary()  # Compressed bitstream
```

**Compression Ratio** (based on CacheGen paper):
- Typical: **3.5-4.3x** for structured data
- Best case: **6-8x** (highly repetitive tokens, e.g., background)

### ZSTD for Point Cloud

对于PointCloud（稀疏3D点），使用**ZSTD**（快速通用压缩）。

```python
import zstandard as zstd

def compress_pointcloud(points):
    """
    Compress LiDAR point cloud using ZSTD.

    Args:
        points: Numpy array of shape (N, 3) containing (x, y, z) coordinates

    Returns:
        Compressed bytes
    """
    # Serialize to bytes
    points_bytes = points.tobytes()

    # Compress with ZSTD (level 3 for balance of speed/ratio)
    compressor = zstd.ZstdCompressor(level=3)
    compressed = compressor.compress(points_bytes)

    return compressed

def decompress_pointcloud(compressed_bytes):
    """Decompress at receiver."""
    decompressor = zstd.ZstdDecompressor()
    points_bytes = decompressor.decompress(compressed_bytes)
    points = np.frombuffer(points_bytes, dtype=np.float32).reshape(-1, 3)
    return points
```

**Why ZSTD?**
- **Fast**: Edge devices can compress in real-time (< 5ms for 1000 points)
- **Ratio**: 2-4x for geometric data
- **Streaming friendly**: Can compress incrementally

**Compression Comparison**:

| Method | Ratio | Speed (edge) | Use Case |
|--------|-------|--------------|----------|
| **Arithmetic** | 4-8x | Medium (20ms) | Structured attributes |
| **ZSTD Level 1** | 2x | Fast (5ms) | Real-time streaming |
| **ZSTD Level 3** | 3-4x | Medium (10ms) | **Default** |
| **ZSTD Level 9** | 5-6x | Slow (50ms) | Offline processing |

---

## 4. Modality-Agnostic Representation

### Challenge: 如何统一Vision/Audio/LiDAR?

**错误做法**: 每种modality定义不同的message type → 破坏interoperability

**正确做法**: 使用**抽象语义表示** + **modality-specific payload**

### Multi-Modal Token Schema

```protobuf
message MultiModalToken {
  // Unified semantic core (modality-agnostic)
  SemanticConcept concept = 1;  // e.g., "fire_source"
  float confidence = 2;

  // Modality-specific evidence (optional)
  oneof evidence {
    VisionEvidence vision = 3;
    AudioEvidence audio = 4;
    LiDAREvidence lidar = 5;
    FusedEvidence fused = 6;  // Multi-sensor fusion
  }
}

message SemanticConcept {
  string concept_id = 1;  // "fire_source", "human_presence"
  SpatialLocation location = 2;  // Unified spatial representation
  TemporalSpan timespan = 3;
}

message VisionEvidence {
  BoundingBox bbox = 1;
  bytes feature_vector = 2;  // Optional: CLIP embedding (512-dim)
}

message AudioEvidence {
  float frequency_hz = 1;
  float decibel = 2;
  bytes spectrogram = 3;  // Compressed spectrogram
}

message LiDAREvidence {
  PointCloud sparse_points = 1;
  float intensity = 2;
}

message FusedEvidence {
  repeated ModalityWeight weights = 1;  // Vision: 0.8, Audio: 0.2
}

message ModalityWeight {
  Modality modality = 1;
  float weight = 2;  // Confidence weight from this modality
}
```

### Key Insight

接收端不需要知道"这是从相机还是LiDAR来的"，只需要知道：
- **"fire_source 在 (x,y)"**
- **"confidence = 0.92"**
- **"来自vision (0.8) + audio (0.2) fusion"**

这种modality-agnostic设计实现了**sensor abstraction**，receiver可以处理任意传感器组合。

---

## 5. Complete Serialization + Transmission Example

### Fire Detection Scenario (Edge UAV → Cloud)

```python
# ============================================
# EDGE AGENT (UAV with camera)
# ============================================
def edge_transmit_fire_token():
    """
    Complete edge-side encoding and transmission pipeline.
    """
    # Step 1: Perception (SASL L0)
    frame = camera.capture()  # 1920x1080 RGB frame
    fire_detected, bbox, conf = fire_detector(frame)  # MobileVLM inference

    if not fire_detected:
        return  # Silence (no transmission) - key to bandwidth savings

    # Step 2: Create Semantic Token
    token = SemanticToken(
        token_id=generate_uuid(),
        modality=Modality.VISION,
        timestamp_us=get_timestamp_us(),
        payload=SemanticPayload(
            semantic_type=SemanticType.FIRE,
            bbox=quantize_bbox(bbox),  # 65.6% size reduction
            confidence=quantize_fp16(conf),  # 50% size reduction
            attributes={
                "intensity": quantize_fp8(estimate_intensity(frame, bbox)),  # 75% reduction
                "smoke_present": True,
                "flame_color": "orange-red"
            }
        ),
        compression=CompressionType.ZSTD
    )

    # Step 3: Serialize to binary (Protobuf)
    token_bytes = token.SerializeToString()
    print(f"Serialized size: {len(token_bytes)} bytes")
    # Typical: 300-400 bytes

    # Step 4: Compress (ZSTD level 3)
    compressed = zstd.compress(token_bytes, level=3)
    print(f"Compressed size: {len(compressed)} bytes")
    # Typical: 200-300 bytes (3.5x compression)

    # Step 5: Transmit over 5G NR
    transmit_packet(compressed)

    # Bandwidth comparison:
    # H.264 frame: ~50,000 bytes
    # Semantic token: ~250 bytes
    # Saving: 200x reduction

    return compressed

# ============================================
# CLOUD AGENT (Inference server)
# ============================================
def cloud_receive_fire_token(compressed_packet):
    """
    Complete cloud-side reception and decoding pipeline.
    """
    # Step 1: Decompress (ZSTD)
    token_bytes = zstd.decompress(compressed_packet)

    # Step 2: Deserialize (Protobuf)
    token = SemanticToken()
    token.ParseFromString(token_bytes)

    # Step 3: Dequantize
    fire_location = dequantize_bbox(token.payload.bbox)
    confidence = dequantize_fp16(token.payload.confidence)
    intensity = dequantize_fp8(token.payload.attributes["intensity"])

    # Step 4: Reconstruct semantic context (SASL L4)
    # Option A: Direct decision (if high confidence)
    if confidence > 0.9:
        decision = make_immediate_decision(fire_location, intensity)

    # Option B: Inject into LLM for complex reasoning
    else:
        context = f"Fire detected at {fire_location} with {confidence:.2f} confidence. " \
                  f"Intensity: {intensity:.2f}. Smoke present."
        decision = llm_agent.decide(
            context=context,
            action_space=["dispatch_drone", "alert_fire_dept", "monitor", "false_alarm"]
        )

    return decision
```

### Bandwidth Savings Analysis

| Method | Data Rate | Transmission Time (5G, 5 Mbps) | Notes |
|--------|-----------|--------------------------------|-------|
| **H.264 video** (30fps) | 5 Mbps = 625 KB/s | 80 ms per frame | Continuous streaming |
| **CLIP embedding** (10fps) | 0.4 Mbps = 50 KB/s | 8 ms per embedding | Feature-based |
| **Semantic Token** (event-driven) | 0.004 Mbps = 0.5 KB/s | **0.32 ms per token** | **Silence when no fire** |

**Key Insight**: Event-driven transmission (silence during normal operation) achieves **~1250x reduction** vs. H.264 in typical scenarios (fire occurs <10% of time). 各阶段详细分解见 `../05-evaluation/cost-model.md`「Bandwidth Savings Breakdown」。

---

## 6. Error Handling & Packet Loss

### Problem: Token丢包怎么办？

**传统方式**: TCP retransmission → High latency (200-500ms)
**Semantic方式**: **Redundancy** + **RAG Fallback**

### Sender: Semantic Redundancy

```python
def send_with_redundancy(token, importance):
    """
    Send critical tokens with redundancy to handle packet loss.

    Args:
        token: Semantic token to send
        importance: Importance score [0, 1] (from attention map)

    Returns:
        Number of copies sent
    """
    if importance > 0.9:  # Critical token (e.g., fire, obstacle)
        # Send 3 copies with different network paths
        send_packet(token, path=0, priority="high")
        send_packet(token, path=1, priority="high")  # Diversity path
        send_packet(token, path=2, priority="high")  # Diversity path
        return 3
    elif importance > 0.7:  # Important token
        # Send 2 copies
        send_packet(token, path=0, priority="medium")
        send_packet(token, path=1, priority="medium")
        return 2
    else:  # Normal token
        send_packet(token, path=0, priority="low")
        return 1
```

### Receiver: RAG Fallback

```python
def receive_with_fallback(token_id, timeout_ms=100):
    """
    Receive token with RAG-based fallback for missing data.

    Args:
        token_id: Expected token ID
        timeout_ms: Wait time before fallback

    Returns:
        Semantic token (received or reconstructed)
    """
    token = wait_for_packet(token_id, timeout=timeout_ms)

    if token is not None:
        return token  # Successfully received

    # Packet lost - use RAG to reconstruct
    print(f"Token {token_id} lost, using RAG fallback")

    # Step 1: Retrieve similar historical events from knowledge base
    similar_events = rag_search(
        query="fire detection in forest area",
        filters={"location": current_location, "time_window": "last_30min"},
        top_k=3
    )

    # Step 2: Reconstruct token from memory
    reconstructed_token = semantic_interpolation(
        historical_events=similar_events,
        current_context=get_current_context(),
        confidence_penalty=0.2  # Lower confidence for reconstructed
    )

    return reconstructed_token
```

### Trade-offs

| Method | Bandwidth Overhead | Latency | Accuracy | Use Case |
|--------|-------------------|---------|----------|----------|
| **Redundancy (3x)** | +200% | Low (no wait) | High (99.9%) | Safety-critical |
| **RAG Fallback** | 0% | Medium (query KB) | Medium (80-90%) | Non-critical |
| **Hybrid** | +0% to +200% | Low-Medium | High | **Recommended** |

**Hybrid Strategy**:
- Critical tokens (importance > 0.9): Send 3 copies
- Important tokens (0.7-0.9): Send 2 copies
- Normal tokens (<0.7): Send 1 copy + RAG fallback

---

## 7. Implementation Checklist

### Required Components

- [ ] **Protobuf Compiler**: Install `protoc` and generate Python/C++ bindings
- [ ] **Quantization Library**: Implement FP16/FP8/INT4 converters
- [ ] **ZSTD Library**: Install `zstandard` package
- [ ] **Arithmetic Coder**: Implement or use `pyarithmeticcoding`
- [ ] **RAG System**: Setup vector database (e.g., FAISS, Pinecone)

### Code Generation

```bash
# Generate Python bindings from .proto file
protoc --python_out=. semantic_token.proto

# Generate C++ bindings (for edge devices)
protoc --cpp_out=. semantic_token.proto
```

### Testing Protocol

1. **Unit Tests**: Test each quantization/compression function independently
2. **Integration Tests**: Test complete encode-decode pipeline
3. **Robustness Tests**: Simulate packet loss (10%, 20%, 30%)
4. **Performance Tests**: Measure latency on target hardware (Jetson Nano, RaspberryPi)

---

## 8. Summary Table

| Component | Technology | Bandwidth Impact | Latency Impact | Complexity |
|-----------|-----------|------------------|----------------|------------|
| **Schema** | Protobuf | -50% vs. JSON | +1ms (serialization) | Low |
| **Quantization** | FP16/FP8 | -50% to -75% | +0.5ms | Medium |
| **Compression** | ZSTD/Arithmetic | -3.5x to -6x | +5ms (edge) | Medium |
| **Modality Fusion** | Unified Concept | N/A | N/A | High |
| **Error Handling** | Redundancy+RAG | +0% to +200% | +10ms (RAG) | High |

### Total Savings (典型场景)

```
Raw Frame (50KB)
  ↓ Semantic extraction
Semantic features (2KB)
  ↓ Protobuf serialization
Serialized token (400 bytes) (-80%)
  ↓ Quantization
Quantized token (200 bytes) (-50%)
  ↓ ZSTD compression
Compressed token (60 bytes) (-70%)

Total: 50KB → 0.06KB = 833x reduction (单一 pipeline 极端值)
(加上 event-driven silence: 可达数千倍)
注：各阶段节省倍率的完整分解见 ../05-evaluation/cost-model.md「Bandwidth Savings Breakdown」
```

---

## Related Documents

- **Architecture**: `../02-core-framework/architecture-overview.md` (STL Data Plane详细说明)
- **Cost Model**: `../05-evaluation/cost-model.md` (Bandwidth/Energy评估)
- **Attention Filtering**: `attention-filtering.md` (如何选择top-k tokens)
- **State Integration**: `state-integration.md` (Receiver端如何重建)

---

## Future Enhancements

1. **Hardware Acceleration**: Custom ASIC for quantization/compression (target: <1ms)
2. **Adaptive Compression**: RL-based policy to select compression level dynamically
3. **Multi-Path Coding**: Network coding for loss resilience
4. **Security**: Encrypted token transmission (防止KV-Cache side-channel攻击)
