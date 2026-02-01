# SSC Pipeline Implementation Specification

**Version**: 1.0
**Status**: Implementation Ready
**Target**: Complete end-to-end SSC system for experimental validation

---

## 1. System Architecture

### 1.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SSC Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Edge Device (Jetson Nano)                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  1. Perception Module                                   │    │
│  │     - Camera/LiDAR input                                │    │
│  │     - Pre-processing (resize, normalize)                │    │
│  │                                                          │    │
│  │  2. Foundation Model (MobileVLM-3B)                     │    │
│  │     - Vision encoder                                    │    │
│  │     - Generate KV-Cache (512-dim × 24 layers)           │    │
│  │                                                          │    │
│  │  3. Semantic Indexer (DSA Lightning)                    │    │
│  │     - Attention score computation                       │    │
│  │     - Top-k token selection (k=32)                      │    │
│  │     - Context-aware thresholding                        │    │
│  │                                                          │    │
│  │  4. Token Encoder                                       │    │
│  │     - Protobuf serialization                            │    │
│  │     - FP8 quantization                                  │    │
│  │     - ZSTD compression                                  │    │
│  │                                                          │    │
│  │  5. Transmitter                                         │    │
│  │     - 5G NR / WiFi-6                                    │    │
│  │     - Packet framing with error correction              │    │
│  └────────────────────────────────────────────────────────┘    │
│                           ↓ (Wireless Channel)                  │
│  Cloud Server (A100 GPU)                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  6. Receiver                                            │    │
│  │     - Packet reassembly                                 │    │
│  │     - Error detection & recovery                        │    │
│  │                                                          │    │
│  │  7. Token Decoder                                       │    │
│  │     - ZSTD decompression                                │    │
│  │     - Protobuf deserialization                          │    │
│  │     - FP8 → FP32 dequantization                         │    │
│  │                                                          │    │
│  │  8. KV-Cache Projector                                  │    │
│  │     - Neural alignment: 512-dim → 4096-dim              │    │
│  │     - Layer normalization                               │    │
│  │                                                          │    │
│  │  9. Foundation Model (GPT-4V)                           │    │
│  │     - Inject aligned KV-Cache                           │    │
│  │     - Task execution (reasoning, decision)              │    │
│  │                                                          │    │
│  │ 10. Semantic Drift Monitor                              │    │
│  │     - Compute KL(P_edge || P_cloud)                     │    │
│  │     - Trigger reset if drift > threshold                │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Specifications

### 2.1 Perception Module

**Input**: Raw sensor data
**Output**: Preprocessed tensors

```python
class PerceptionModule:
    """
    Handles sensor input and preprocessing.
    """

    def __init__(self, input_size=(1920, 1080), target_size=(384, 384)):
        self.input_size = input_size
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process_frame(self, frame):
        """
        Process a single camera frame.

        Args:
          frame: numpy.ndarray (H, W, C), uint8, RGB

        Returns:
          tensor: torch.Tensor (1, C, H, W), float32
        """
        img = Image.fromarray(frame)
        tensor = self.transform(img).unsqueeze(0)
        return tensor

    def process_lidar(self, point_cloud):
        """
        Process LiDAR point cloud.

        Args:
          point_cloud: numpy.ndarray (N, 3), float32, (x, y, z)

        Returns:
          tensor: torch.Tensor (1, N, 3)
        """
        # Normalize coordinates to [0, 1]
        pc_normalized = (point_cloud - point_cloud.min()) / (
            point_cloud.max() - point_cloud.min()
        )
        return torch.from_numpy(pc_normalized).unsqueeze(0)
```

### 2.2 Foundation Model Wrapper

**Input**: Preprocessed tensors
**Output**: KV-Cache (512-dim × 24 layers)

```python
class EdgeFoundationModel:
    """
    Wrapper for MobileVLM-3B on edge device.
    """

    def __init__(self, model_path="MobileVLM-3B", device="cuda:0"):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        """Load pre-trained MobileVLM."""
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(model_path)
        model.to(self.device)
        return model

    def forward_with_cache(self, image_tensor):
        """
        Forward pass and extract KV-Cache.

        Args:
          image_tensor: torch.Tensor (1, 3, H, W)

        Returns:
          kv_cache: List of tuples (K, V) for each layer
            K, V: torch.Tensor (1, num_heads, seq_len, head_dim)
        """
        with torch.no_grad():
            outputs = self.model(
                pixel_values=image_tensor,
                output_attentions=True,
                use_cache=True
            )

        # Extract KV-Cache from model outputs
        kv_cache = []
        for layer_output in outputs.past_key_values:
            key, value = layer_output  # (1, num_heads, seq_len, head_dim)
            kv_cache.append((key, value))

        return kv_cache

    def get_cache_dim(self):
        """
        Return KV-Cache dimensions.

        Returns:
          dict: {'num_layers', 'num_heads', 'head_dim', 'total_dim'}
        """
        return {
            'num_layers': 24,
            'num_heads': 8,
            'head_dim': 64,
            'total_dim': 512  # num_heads * head_dim
        }
```

### 2.3 Semantic Indexer (DSA Lightning)

**Input**: KV-Cache
**Output**: Selected top-k tokens per layer

```python
class SemanticIndexer:
    """
    Implements DeepSeek DSA Lightning for token selection.
    """

    def __init__(self, top_k=32, context_aware=True):
        self.top_k = top_k
        self.context_aware = context_aware

    def compute_attention_scores(self, query, key):
        """
        Compute attention scores: softmax(Q @ K^T / sqrt(d_k))

        Args:
          query: torch.Tensor (1, num_heads, seq_len_q, head_dim)
          key: torch.Tensor (1, num_heads, seq_len_k, head_dim)

        Returns:
          scores: torch.Tensor (1, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights

    def select_top_k_tokens(self, kv_cache, top_k=None):
        """
        Select top-k important tokens based on attention scores.

        Args:
          kv_cache: List of (K, V) tuples
          top_k: Override default top_k

        Returns:
          selected_indices: List of torch.Tensor (top_k,) per layer
          selected_kv: List of (K_topk, V_topk) tuples
        """
        if top_k is None:
            top_k = self.top_k

        selected_indices = []
        selected_kv = []

        for layer_idx, (key, value) in enumerate(kv_cache):
            # key, value: (1, num_heads, seq_len, head_dim)

            # Aggregate attention across heads
            # Use last query position (most recent)
            query = key[:, :, -1:, :]  # (1, num_heads, 1, head_dim)

            attn_weights = self.compute_attention_scores(query, key)
            # attn_weights: (1, num_heads, 1, seq_len)

            # Average across heads
            avg_attn = attn_weights.mean(dim=1).squeeze(0).squeeze(0)
            # avg_attn: (seq_len,)

            # Select top-k tokens
            topk_values, topk_indices = torch.topk(avg_attn, k=top_k)
            selected_indices.append(topk_indices)

            # Extract selected KV
            key_selected = key[:, :, topk_indices, :]
            value_selected = value[:, :, topk_indices, :]
            selected_kv.append((key_selected, value_selected))

        return selected_indices, selected_kv

    def adaptive_top_k(self, scene_features):
        """
        Context-aware top-k adjustment.

        Args:
          scene_features: dict with keys {'complexity', 'fire_detected', 'motion'}

        Returns:
          top_k: int
        """
        if not self.context_aware:
            return self.top_k

        # Fire detected: need more tokens
        if scene_features.get('fire_detected', False):
            return min(64, self.top_k * 2)

        # High motion: moderate increase
        if scene_features.get('motion', 0) > 0.5:
            return min(48, int(self.top_k * 1.5))

        # Normal operation: use default or reduce
        if scene_features.get('complexity', 'medium') == 'low':
            return max(8, self.top_k // 2)

        return self.top_k
```

### 2.4 Token Encoder

**Input**: Selected KV-Cache
**Output**: Compressed binary packet

```python
class TokenEncoder:
    """
    Encodes semantic tokens to binary format.
    """

    def __init__(self, quantization='fp8', compression='zstd'):
        self.quantization = quantization
        self.compression = compression

    def quantize_tensor(self, tensor, dtype='fp8'):
        """
        Quantize tensor to lower precision.

        Args:
          tensor: torch.Tensor, float32
          dtype: str, one of ['fp32', 'fp16', 'fp8', 'int4']

        Returns:
          quantized: torch.Tensor, quantized dtype
        """
        if dtype == 'fp32':
            return tensor  # No quantization

        elif dtype == 'fp16':
            return tensor.half()

        elif dtype == 'fp8':
            # PyTorch doesn't natively support FP8, simulate with scale + int8
            scale = tensor.abs().max() / 127.0
            quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return quantized, scale  # Return scale for dequantization

        elif dtype == 'int4':
            # 4-bit quantization: map to [-8, 7]
            scale = tensor.abs().max() / 7.0
            quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
            return quantized, scale

        else:
            raise ValueError(f"Unsupported quantization: {dtype}")

    def serialize_token(self, token_data):
        """
        Serialize semantic token using Protobuf.

        Args:
          token_data: dict with keys:
            {
              'token_id': str,
              'modality': str,
              'timestamp': int,
              'kv_cache': List of (K, V) tensors,
              'metadata': dict
            }

        Returns:
          serialized: bytes
        """
        from semantic_token_pb2 import SemanticToken, Modality, SemanticPayload

        # Create Protobuf message
        token_msg = SemanticToken()
        token_msg.token_id = token_data['token_id']
        token_msg.modality = Modality.VISION  # or from token_data
        token_msg.timestamp_us = token_data['timestamp']

        # Encode KV-Cache as bytes
        kv_bytes = self._encode_kv_cache(token_data['kv_cache'])
        token_msg.payload.kv_cache_data = kv_bytes

        # Metadata
        for key, value in token_data.get('metadata', {}).items():
            token_msg.payload.attributes[key] = str(value)

        # Serialize to binary
        serialized = token_msg.SerializeToString()
        return serialized

    def _encode_kv_cache(self, kv_cache):
        """
        Encode KV-Cache tensors to bytes.

        Args:
          kv_cache: List of (K, V) tuples

        Returns:
          encoded: bytes
        """
        # Quantize
        quantized_kv = []
        scales = []

        for key, value in kv_cache:
            if self.quantization == 'fp8':
                key_q, key_scale = self.quantize_tensor(key, 'fp8')
                value_q, value_scale = self.quantize_tensor(value, 'fp8')
                quantized_kv.append((key_q, value_q))
                scales.append((key_scale, value_scale))
            else:
                quantized_kv.append((key, value))

        # Flatten to numpy array
        kv_array = self._flatten_kv_list(quantized_kv)

        # Serialize with pickle (or custom format)
        import pickle
        encoded = pickle.dumps({
            'kv_data': kv_array,
            'scales': scales if self.quantization == 'fp8' else None
        })

        return encoded

    def _flatten_kv_list(self, kv_list):
        """Flatten list of (K, V) tensors to numpy array."""
        flattened = []
        for key, value in kv_list:
            flattened.append(key.cpu().numpy())
            flattened.append(value.cpu().numpy())
        return np.concatenate([x.flatten() for x in flattened])

    def compress(self, data):
        """
        Compress binary data.

        Args:
          data: bytes

        Returns:
          compressed: bytes
        """
        if self.compression == 'zstd':
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=3)
            compressed = cctx.compress(data)
            return compressed

        elif self.compression == 'none':
            return data

        else:
            raise ValueError(f"Unsupported compression: {self.compression}")

    def encode(self, token_data):
        """
        Full encoding pipeline.

        Args:
          token_data: dict

        Returns:
          packet: bytes (ready for transmission)
        """
        # Step 1: Serialize
        serialized = self.serialize_token(token_data)

        # Step 2: Compress
        compressed = self.compress(serialized)

        return compressed
```

### 2.5 Token Decoder

**Input**: Compressed binary packet
**Output**: Reconstructed KV-Cache

```python
class TokenDecoder:
    """
    Decodes semantic tokens from binary format.
    """

    def __init__(self, quantization='fp8', compression='zstd'):
        self.quantization = quantization
        self.compression = compression

    def decompress(self, data):
        """Decompress binary data."""
        if self.compression == 'zstd':
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(data)
            return decompressed
        elif self.compression == 'none':
            return data
        else:
            raise ValueError(f"Unsupported compression: {self.compression}")

    def deserialize_token(self, data):
        """Deserialize Protobuf message."""
        from semantic_token_pb2 import SemanticToken

        token_msg = SemanticToken()
        token_msg.ParseFromString(data)

        token_data = {
            'token_id': token_msg.token_id,
            'modality': token_msg.modality,
            'timestamp': token_msg.timestamp_us,
            'kv_cache_bytes': token_msg.payload.kv_cache_data,
            'metadata': dict(token_msg.payload.attributes)
        }

        return token_data

    def dequantize_tensor(self, tensor, scale, dtype='fp8'):
        """Reverse quantization."""
        if dtype == 'fp8':
            return tensor.float() * scale
        elif dtype == 'fp16':
            return tensor.float()
        else:
            return tensor

    def decode_kv_cache(self, kv_bytes):
        """
        Decode KV-Cache from bytes.

        Args:
          kv_bytes: bytes

        Returns:
          kv_cache: List of (K, V) tensors
        """
        import pickle
        data = pickle.loads(kv_bytes)

        kv_array = data['kv_data']
        scales = data['scales']

        # Reconstruct KV-Cache tensors
        # (Implementation depends on storage format)
        # For simplicity, assume we stored shapes separately

        kv_cache = []
        # ... (reverse of _flatten_kv_list)

        return kv_cache

    def decode(self, packet):
        """
        Full decoding pipeline.

        Args:
          packet: bytes

        Returns:
          token_data: dict
        """
        # Step 1: Decompress
        decompressed = self.decompress(packet)

        # Step 2: Deserialize
        token_data = self.deserialize_token(decompressed)

        # Step 3: Decode KV-Cache
        kv_cache = self.decode_kv_cache(token_data['kv_cache_bytes'])
        token_data['kv_cache'] = kv_cache

        return token_data
```

### 2.6 KV-Cache Projector

**Input**: 512-dim KV-Cache (edge)
**Output**: 4096-dim KV-Cache (cloud)

```python
class KVCacheProjector(nn.Module):
    """
    Neural projector for heterogeneous KV-Cache alignment.
    """

    def __init__(self, d_source=512, d_target=4096, use_residual=True):
        super().__init__()
        self.d_source = d_source
        self.d_target = d_target
        self.use_residual = use_residual

        # Linear projection
        self.linear = nn.Linear(d_source, d_target, bias=False)

        # Residual connection (learnable)
        if use_residual:
            self.residual = nn.Parameter(torch.zeros(d_target))
        else:
            self.residual = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_target)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, kv_source):
        """
        Project source KV-Cache to target dimension.

        Args:
          kv_source: torch.Tensor (batch, seq_len, d_source)

        Returns:
          kv_target: torch.Tensor (batch, seq_len, d_target)
        """
        # Linear projection
        kv_proj = self.linear(kv_source)

        # Add residual
        if self.use_residual:
            kv_proj = kv_proj + self.residual

        # Normalize
        kv_target = self.layer_norm(kv_proj)

        return kv_target

    def project_kv_cache(self, kv_cache_source):
        """
        Project entire KV-Cache (all layers).

        Args:
          kv_cache_source: List of (K, V) tuples
            K, V: (batch, num_heads, seq_len, head_dim)

        Returns:
          kv_cache_target: List of (K_proj, V_proj) tuples
        """
        kv_cache_target = []

        for key, value in kv_cache_source:
            # Reshape: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads * head_dim)
            batch, num_heads, seq_len, head_dim = key.shape
            key_flat = key.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
            value_flat = value.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

            # Project
            key_proj = self.forward(key_flat)
            value_proj = self.forward(value_flat)

            # Reshape back: (batch, seq_len, d_target) → (batch, num_heads_target, seq_len, head_dim_target)
            num_heads_target = 32  # GPT-4V num_heads
            head_dim_target = self.d_target // num_heads_target

            key_proj = key_proj.reshape(batch, seq_len, num_heads_target, head_dim_target).permute(0, 2, 1, 3)
            value_proj = value_proj.reshape(batch, seq_len, num_heads_target, head_dim_target).permute(0, 2, 1, 3)

            kv_cache_target.append((key_proj, value_proj))

        return kv_cache_target
```

### 2.7 Semantic Drift Monitor

**Input**: Edge KV-Cache, Cloud KV-Cache
**Output**: Drift score, Reset signal

```python
class SemanticDriftMonitor:
    """
    Monitors semantic drift and triggers reset when needed.
    """

    def __init__(self, reset_threshold=0.10, contraction_param=0.98):
        """
        Args:
          reset_threshold: KL divergence threshold to trigger reset.
            理论推导值见 semantic-state-sync.md:
            - 0.05 (strict), 0.10 (moderate), 0.15 (lenient)
            - 稳态漂移 = ε_max / (1 - ρ)
          contraction_param: Error contraction parameter ρ ∈ [0,1).
            越大表示 error 消退越慢，需要更频繁 reset。
        """
        self.reset_threshold = reset_threshold
        self.rho = contraction_param
        self.drift_history = []

    def compute_kl_divergence(self, p, q):
        """
        Compute KL(P || Q).

        Args:
          p, q: torch.Tensor (same shape), probability distributions

        Returns:
          kl_div: float
        """
        p = p + 1e-10  # Avoid log(0)
        q = q + 1e-10
        kl = (p * torch.log(p / q)).sum()
        return kl.item()

    def measure_drift(self, kv_edge, kv_cloud):
        """
        Measure semantic drift between edge and cloud KV-Cache.

        Args:
          kv_edge, kv_cloud: List of (K, V) tuples

        Returns:
          drift: float (KL divergence)
        """
        drift_per_layer = []

        for (k_edge, v_edge), (k_cloud, v_cloud) in zip(kv_edge, kv_cloud):
            # Compute attention distributions
            p_edge = F.softmax(k_edge.mean(dim=(0, 1)), dim=-1)  # (head_dim,)
            p_cloud = F.softmax(k_cloud.mean(dim=(0, 1)), dim=-1)

            # KL divergence
            kl = self.compute_kl_divergence(p_edge, p_cloud)
            drift_per_layer.append(kl)

        # Average drift across layers
        drift = np.mean(drift_per_layer)
        self.drift_history.append(drift)

        return drift

    def should_reset(self, current_drift):
        """
        Decide if reset is needed.

        Args:
          current_drift: float

        Returns:
          reset: bool
        """
        # Compute accumulated drift with contraction weighting
        # 理论基础: ||e_t||_eff ≤ ρ·||e_{t-1}||_eff + ||ε_t||
        if len(self.drift_history) == 0:
            accumulated_drift = current_drift
        else:
            accumulated_drift = sum(
                d * (self.rho ** (len(self.drift_history) - i))
                for i, d in enumerate(self.drift_history)
            )

        # Trigger reset if accumulated drift exceeds threshold
        if accumulated_drift > self.reset_threshold:
            self.drift_history = []  # Clear history after reset
            return True

        return False
```

---

## 3. End-to-End Pipeline

### 3.1 Edge Device Main Loop

```python
class EdgeDevice:
    """
    Edge device main loop for SSC transmission.
    """

    def __init__(self, config):
        self.perception = PerceptionModule()
        self.model = EdgeFoundationModel(model_path=config['model_path'])
        self.indexer = SemanticIndexer(top_k=config['top_k'])
        self.encoder = TokenEncoder(
            quantization=config['quantization'],
            compression=config['compression']
        )
        self.transmitter = WirelessTransmitter(config['channel'])

    def run(self, camera_stream):
        """
        Main loop: capture → encode → transmit.

        Args:
          camera_stream: Iterator yielding frames
        """
        for frame_id, frame in enumerate(camera_stream):
            # Step 1: Perception
            image_tensor = self.perception.process_frame(frame)

            # Step 2: Foundation model inference
            kv_cache = self.model.forward_with_cache(image_tensor)

            # Step 3: Semantic indexing (select important tokens)
            selected_indices, selected_kv = self.indexer.select_top_k_tokens(kv_cache)

            # Step 4: Encode token
            token_data = {
                'token_id': f"frame_{frame_id}",
                'modality': 'vision',
                'timestamp': time.time_ns() // 1000,  # microseconds
                'kv_cache': selected_kv,
                'metadata': {'frame_id': frame_id}
            }
            packet = self.encoder.encode(token_data)

            # Step 5: Transmit
            self.transmitter.send(packet)

            print(f"[Edge] Frame {frame_id}: Transmitted {len(packet)} bytes")
```

### 3.2 Cloud Server Main Loop

```python
class CloudServer:
    """
    Cloud server main loop for SSC reception.
    """

    def __init__(self, config):
        self.receiver = WirelessReceiver(config['channel'])
        self.decoder = TokenDecoder(
            quantization=config['quantization'],
            compression=config['compression']
        )
        self.projector = KVCacheProjector(d_source=512, d_target=4096)
        self.model = CloudFoundationModel(model_path=config['cloud_model'])
        self.drift_monitor = SemanticDriftMonitor(reset_threshold=0.10)

    def run(self):
        """
        Main loop: receive → decode → reason.
        """
        while True:
            # Step 1: Receive packet
            packet = self.receiver.receive()
            if packet is None:
                continue

            # Step 2: Decode token
            token_data = self.decoder.decode(packet)

            # Step 3: Project KV-Cache to cloud dimension
            kv_source = token_data['kv_cache']
            kv_target = self.projector.project_kv_cache(kv_source)

            # Step 4: Inject into cloud model
            response = self.model.generate_with_cache(kv_target)

            # Step 5: Monitor drift
            drift = self.drift_monitor.measure_drift(kv_source, kv_target)
            if self.drift_monitor.should_reset(drift):
                print(f"[Cloud] Drift {drift:.3f} exceeds threshold, requesting reset")
                self.request_full_resync()

            print(f"[Cloud] Received token {token_data['token_id']}, response: {response}")
```

---

## 4. Configuration Files

### 4.1 Edge Config (`edge_config.yaml`)

```yaml
# Edge Device Configuration
device:
  name: "jetson_nano"
  gpu: "cuda:0"
  power_budget_watts: 10

model:
  name: "MobileVLM-3B"
  path: "models/mobilevlm_3b"
  kv_dim: 512
  num_layers: 24

indexer:
  method: "dsa_lightning"
  top_k: 32
  context_aware: true

encoder:
  quantization: "fp8"  # fp32, fp16, fp8, int4
  compression: "zstd"  # zstd, none
  compression_level: 3

transmission:
  protocol: "5g_nr"
  bandwidth_mhz: 100
  frequency_ghz: 28
  max_packet_size: 1500
```

### 4.2 Cloud Config (`cloud_config.yaml`)

```yaml
# Cloud Server Configuration
device:
  name: "a100_server"
  gpu: "cuda:0"
  num_gpus: 1

model:
  name: "GPT-4V"
  path: "models/gpt4v"
  kv_dim: 4096
  num_layers: 96

projector:
  checkpoint: "models/projector_512_to_4096.pt"
  d_source: 512
  d_target: 4096
  use_residual: true

decoder:
  quantization: "fp8"
  compression: "zstd"

drift_monitor:
  reset_threshold: 0.10    # 理论值: 0.05(strict)/0.10(moderate)/0.15(lenient)
  contraction_param: 0.98  # ρ parameter from Error Contraction Property
  check_interval_steps: 10
```

---

## 5. Training Scripts

### 5.1 Projector Training

```python
# train_projector.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_projector(config):
    """
    Train KV-Cache projector.
    """
    # Load dataset
    dataset = PairedKVCacheDataset(
        small_model=config['small_model'],
        large_model=config['large_model'],
        num_samples=config['num_samples']
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    projector = KVCacheProjector(
        d_source=config['d_source'],
        d_target=config['d_target']
    ).to('cuda')

    # Optimizer
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # Loss function
    mse_loss = nn.MSELoss()

    # Training loop
    for epoch in range(config['epochs']):
        total_loss = 0

        for batch_idx, (kv_small, kv_large) in enumerate(tqdm(dataloader)):
            kv_small = kv_small.to('cuda')
            kv_large = kv_large.to('cuda')

            # Forward pass
            kv_proj = projector(kv_small)

            # Compute loss
            loss = mse_loss(kv_proj, kv_large)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                projector.state_dict(),
                f"checkpoints/projector_epoch{epoch+1}.pt"
            )

if __name__ == "__main__":
    config = {
        'small_model': 'MobileVLM-3B',
        'large_model': 'GPT-4V',
        'd_source': 512,
        'd_target': 4096,
        'num_samples': 100000,
        'epochs': 50
    }
    train_projector(config)
```

---

## 6. Deployment Checklist

### Edge Device Setup

- [ ] Flash Jetson Nano with JetPack 5.1
- [ ] Install PyTorch 2.0 (ARM64)
- [ ] Download MobileVLM-3B model weights
- [ ] Configure camera (MIPI CSI or USB)
- [ ] Install 5G modem drivers
- [ ] Test end-to-end latency (<50ms)

### Cloud Server Setup

- [ ] Provision A100 GPU instance
- [ ] Install CUDA 12.1 + PyTorch 2.0
- [ ] Download GPT-4V model weights
- [ ] Load pre-trained projector checkpoint
- [ ] Configure network interface (public IP)
- [ ] Set up monitoring (Prometheus + Grafana)

### Network Configuration

- [ ] Configure 5G NR parameters (bandwidth, frequency)
- [ ] Set up packet loss emulation (tc netem)
- [ ] Configure SNR variation (USRP or ns-3)
- [ ] Test end-to-end connectivity

---

**Status**: Implementation specification complete. Ready for coding phase.
**Next**: Begin implementation with edge perception module.
