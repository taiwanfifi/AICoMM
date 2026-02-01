# Phase 3: Experimental Design & Validation Framework

**Status**: 🚧 In Progress
**Target**: Validate theoretical predictions with empirical results
**Goal**: Demonstrate SSC achieves 85-90% of theoretical bounds in real-world scenarios

---

## 1. Experimental Objectives

### 1.1 Primary Research Questions

**RQ1: Bandwidth Efficiency**
- Does SSC achieve 99%+ bandwidth savings vs. traditional methods?
- How does compression ratio scale with task complexity?
- What is the rate-distortion trade-off in practice?

**RQ2: Task Success Rate**
- Does SSC maintain >90% task success with 10% packet loss?
- How does quantization level (FP32/FP16/FP8/INT4) affect accuracy?
- What is the minimum bandwidth for τ_success ≥ 0.9?

**RQ3: Latency & Real-time Performance**
- Does end-to-end latency stay <50ms for edge scenarios?
- How does KV-Cache alignment overhead scale with dimension mismatch?
- Can SSC meet autonomous vehicle deadlines (100ms)?

**RQ4: Robustness & Generalization**
- Does SSC generalize across datasets (VIRAT → UCF-Crime → COCO)?
- How does performance degrade with channel SNR variation (0-30dB)?
- Does the system recover gracefully from semantic drift?

**RQ5: Theoretical Validation**
- Do empirical results match theoretical predictions (±5% error)?
- Is DSA approximation error within proven bounds?
- Does drift accumulation follow exponential decay model?

---

## 2. Experimental Scenarios

### 2.1 Scenario 1: UAV Fire Detection

**Application**: Wildfire monitoring with drone swarm
**Dataset**: VIRAT Video Dataset + Custom fire annotations
**Models**:
- Edge: MobileVLM (512-dim KV-Cache, 3.2B params)
- Cloud: GPT-4V (4096-dim KV-Cache, 1.76T params)

**Setup**:
```
UAV (Edge Device: Jetson Nano)
  ├─ Camera: 1920×1080 @ 30fps
  ├─ Fire Detection Model: MobileVLM-3B
  ├─ Semantic Indexer: DSA Lightning (top-k=32)
  └─ Wireless: 5G NR (bandwidth: 100MHz, SNR: 20dB)
       ↓
Ground Station (Cloud Server: A100 GPU)
  ├─ Cloud Model: GPT-4V
  ├─ KV-Cache Projector: 512→4096 dim
  └─ Task Executor: Fire location triangulation
```

**Evaluation Metrics**:
1. **Bandwidth**: Avg bits/frame transmitted
2. **Latency**: Detection-to-response time
3. **Task Success**: Fire localization error <10m
4. **Energy**: mJ per event transmission

**Baselines**:
- **B1**: H.264 video streaming (1080p @ 30fps)
- **B2**: CLIP image embeddings (every 1s)
- **B3**: Text prompts (GPT-4V captioning)
- **B4**: Full KV-Cache transmission (no compression)

**Expected Results** (from theoretical predictions):
| Method | Bandwidth (Mbps) | Latency (ms) | Task Success (%) | Energy (J/event) |
|--------|------------------|--------------|------------------|------------------|
| H.264 baseline | 5.2 | 120 | 92 | 0.19 |
| CLIP embeddings | 2.1 | 95 | 89 | 0.08 |
| Text prompts | 0.8 | 150 | 85 | 0.03 |
| **SSC (Ours)** | **0.02** | **18** | **92** | **0.012** |

---

### 2.2 Scenario 2: V2V Autonomous Driving Coordination

**Application**: Vehicle-to-Vehicle hazard sharing
**Dataset**: nuScenes + COCO Traffic
**Models**:
- Vehicle A: MobileNet-SSD (edge)
- Vehicle B: ResNet-101 (edge)
- Cloud: GPT-4V (coordination)

**Setup**:
```
Vehicle A (Lead Vehicle)
  ├─ LiDAR + Camera fusion
  ├─ Detect: Pedestrian crossing ahead
  ├─ Semantic Token: {HAZARD_PEDESTRIAN, location, velocity}
  └─ C-V2X broadcast (latency budget: 100ms)
       ↓
Vehicle B (Following Vehicle, 50m behind)
  ├─ Receive token → Update world model
  ├─ Decision: Pre-emptive braking
  └─ Task: Avoid pedestrian collision
```

**Evaluation Metrics**:
1. **End-to-end latency**: Token generation → Decision execution
2. **Collision avoidance rate**: % of hazards successfully avoided
3. **False positive rate**: % of unnecessary braking events
4. **Bandwidth utilization**: Avg bits/hazard event

**Stress Tests**:
- **Dense traffic**: 20 vehicles, 5 hazards/second
- **Network congestion**: Packet loss 5-20%
- **Multi-modal fusion**: Vision + LiDAR + Radar

**Expected Results**:
- Latency: <50ms (vs. 120ms for H.264)
- Collision avoidance: >95% (vs. 88% for delayed video)
- Bandwidth: 0.5 Mbps (vs. 15 Mbps for video streaming)

---

### 2.3 Scenario 3: Smart Factory Anomaly Detection

**Application**: Industrial equipment monitoring
**Dataset**: MVTec Anomaly Detection + Custom vibration data
**Models**:
- Edge: EfficientNet-B0 (visual) + 1D-CNN (vibration)
- Cloud: GPT-4V (root cause analysis)

**Setup**:
```
Edge Sensors (50 cameras + 100 vibration sensors)
  ├─ Normal operation: Silence (no transmission)
  ├─ Anomaly detected: Generate semantic token
  │   {ANOMALY_VIBRATION, machine_id, severity, timestamp}
  └─ 5G NR uplink (shared 100Mbps among 50 devices)
       ↓
Cloud Analytics (A100 GPU)
  ├─ Aggregate anomalies from multiple sensors
  ├─ Root cause analysis (GPT-4V reasoning)
  └─ Maintenance scheduling
```

**Evaluation Metrics**:
1. **Detection latency**: Anomaly-to-alert time
2. **Precision/Recall**: Anomaly detection accuracy
3. **Bandwidth efficiency**: Bits/hour/sensor
4. **Scalability**: Performance with 50→500 sensors

**Challenge**:
- **Extreme sparsity**: Normal operation = 0 transmission
- **Burst load**: Mass anomaly during equipment failure
- **Multi-modality**: Vision + Audio + Vibration fusion

**Expected Results**:
- Bandwidth: 0.001 Mbps/sensor (vs. 2 Mbps for video)
- Detection latency: 12ms (vs. 200ms for video streaming)
- Scalability: Linear (SSC) vs. Quadratic (traditional)

---

## 3. Experimental Protocol

### 3.1 Hardware Configuration

**Edge Devices**:
```yaml
Device: NVIDIA Jetson Nano
CPU: Quad-core ARM A57 @ 1.43GHz
GPU: 128-core Maxwell
RAM: 4GB LPDDR4
Storage: 64GB eMMC
Power: 5W typical, 10W max
```

**Cloud Server**:
```yaml
Device: NVIDIA A100 GPU Server
CPU: AMD EPYC 7742 (64 cores)
GPU: NVIDIA A100 (80GB HBM2e)
RAM: 512GB DDR4
Storage: 2TB NVMe SSD
Power: 400W TDP
```

**Network Emulation**:
```yaml
Tool: Mininet-WiFi + ns-3
Bandwidth: 10-100 Mbps (configurable)
Latency: 5-50ms (configurable)
Packet Loss: 0-20% (uniform/burst)
SNR: 0-30dB (AWGN/Rayleigh fading)
```

### 3.2 Dataset Preparation

**Fire Detection**:
- VIRAT Video Dataset: 8.5 hours, 29 scenarios
- Custom annotations: Fire bounding boxes (5,200 frames)
- Train/Val/Test split: 70/15/15

**Autonomous Driving**:
- nuScenes: 1,000 scenes, 1.4M camera images
- COCO Traffic subset: 5,000 images with pedestrian/vehicle
- Train/Val/Test split: 80/10/10

**Smart Factory**:
- MVTec AD: 5,354 images, 15 object categories
- Custom vibration: 100 hours, 10 machines
- Train/Val/Test split: 60/20/20

### 3.3 Model Training

**KV-Cache Projector Training**:
```python
# Training configuration
optimizer = AdamW(lr=1e-4, weight_decay=0.01)
loss_fn = MSELoss() + 0.1 * DistillationLoss()
batch_size = 32
epochs = 50
warmup_steps = 1000

# Data: Paired (small_kv, large_kv) from pre-training
dataset = PairedKVCacheDataset(
    small_model="MobileVLM-3B",
    large_model="GPT-4V",
    num_samples=100_000
)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        kv_small, kv_large = batch
        kv_proj = projector(kv_small)
        loss = loss_fn(kv_proj, kv_large)
        loss.backward()
        optimizer.step()
```

**Semantic Indexer (DSA) Configuration**:
```python
# DeepSeek DSA parameters
top_k = 32  # Select top-32 tokens per layer
num_layers = 24
threshold_confidence = 0.7

# Context-aware adjustment
def adaptive_top_k(scene_complexity):
    if scene_complexity == "high":  # Fire detected
        return 64  # More tokens
    elif scene_complexity == "medium":  # Smoke/motion
        return 32
    else:  # Normal operation
        return 8  # Minimal tokens
```

### 3.4 Evaluation Pipeline

**Step 1: Baseline Collection**
```bash
# Run H.264 baseline
python eval_baseline.py \
  --method h264 \
  --dataset virat \
  --bitrate 5000 \
  --output results/h264_baseline.json

# Run CLIP baseline
python eval_baseline.py \
  --method clip \
  --dataset virat \
  --model openai/clip-vit-large-patch14 \
  --output results/clip_baseline.json
```

**Step 2: SSC Evaluation**
```bash
# Run SSC with different configurations
for quant in fp32 fp16 fp8 int4; do
  for top_k in 8 16 32 64; do
    python eval_ssc.py \
      --quantization $quant \
      --top_k $top_k \
      --dataset virat \
      --output results/ssc_${quant}_k${top_k}.json
  done
done
```

**Step 3: Robustness Testing**
```bash
# Packet loss sweep
for loss_rate in 0 0.05 0.10 0.15 0.20; do
  python eval_robustness.py \
    --packet_loss $loss_rate \
    --dataset virat \
    --output results/robustness_loss${loss_rate}.json
done

# SNR sweep
for snr_db in 0 5 10 15 20 25 30; do
  python eval_robustness.py \
    --snr_db $snr_db \
    --dataset virat \
    --output results/robustness_snr${snr_db}.json
done
```

**Step 4: Cross-dataset Generalization**
```bash
# Train on VIRAT, test on UCF-Crime
python eval_generalization.py \
  --train_dataset virat \
  --test_datasets ucf_crime,coco,shanghaitech \
  --output results/generalization.json
```

---

## 4. Metrics & Analysis

### 4.1 Primary Metrics

**Bandwidth Efficiency**:
```python
def bandwidth_efficiency(method_results, baseline_results):
    """
    Compute bandwidth savings percentage.

    Returns:
      savings: % reduction (0-100)
      compression_ratio: baseline_bw / method_bw
    """
    baseline_bw = baseline_results['avg_bandwidth_mbps']
    method_bw = method_results['avg_bandwidth_mbps']

    savings = (1 - method_bw / baseline_bw) * 100
    compression_ratio = baseline_bw / method_bw

    return {
        'savings_percent': savings,
        'compression_ratio': compression_ratio,
        'baseline_bw': baseline_bw,
        'method_bw': method_bw
    }
```

**Task Success Rate**:
```python
def task_success_rate(predictions, ground_truth, threshold=10.0):
    """
    Compute task success rate for fire localization.

    Args:
      predictions: List of (x, y) fire locations
      ground_truth: List of (x, y) true locations
      threshold: Max error in meters for success

    Returns:
      success_rate: % of predictions within threshold
    """
    successes = 0
    for pred, gt in zip(predictions, ground_truth):
        error = np.linalg.norm(np.array(pred) - np.array(gt))
        if error <= threshold:
            successes += 1

    return successes / len(predictions)
```

**Latency Breakdown**:
```python
def latency_breakdown(trace):
    """
    Analyze end-to-end latency components.

    Returns:
      breakdown: {
        'encoding': ms,
        'transmission': ms,
        'decoding': ms,
        'kv_alignment': ms,
        'inference': ms,
        'total': ms
      }
    """
    return {
        'encoding': trace['t_encode_end'] - trace['t_encode_start'],
        'transmission': trace['t_recv'] - trace['t_send'],
        'decoding': trace['t_decode_end'] - trace['t_recv'],
        'kv_alignment': trace['t_proj_end'] - trace['t_proj_start'],
        'inference': trace['t_infer_end'] - trace['t_infer_start'],
        'total': trace['t_infer_end'] - trace['t_encode_start']
    }
```

### 4.2 Secondary Metrics

**Energy Consumption**:
```python
def energy_per_event(power_trace, event_timestamps):
    """
    Compute energy consumption per semantic event transmission.

    Args:
      power_trace: Time series of power consumption (watts)
      event_timestamps: List of (start, end) timestamps

    Returns:
      energy_joules: List of energy per event
    """
    energies = []
    for start, end in event_timestamps:
        # Integrate power over time
        duration = end - start
        avg_power = np.mean(power_trace[start:end])
        energy = avg_power * duration
        energies.append(energy)

    return energies
```

**Semantic Drift Accumulation**:
```python
def measure_drift(edge_kv_trace, cloud_kv_trace, reset_points):
    """
    Measure semantic drift between edge and cloud KV-Cache.

    Returns:
      drift_curve: Time series of KL(P_edge || P_cloud)
      reset_triggered: List of reset event timestamps
    """
    drift_values = []

    for t in range(len(edge_kv_trace)):
        # Compute KL divergence at each timestep
        p_edge = softmax(edge_kv_trace[t])
        p_cloud = softmax(cloud_kv_trace[t])
        drift_t = kl_divergence(p_edge, p_cloud)
        drift_values.append(drift_t)

    return {
        'drift_curve': drift_values,
        'max_drift': max(drift_values),
        'avg_drift': np.mean(drift_values),
        'reset_points': reset_points
    }
```

### 4.3 Theoretical Validation

**Compare Empirical vs. Theoretical Predictions**:
```python
def validate_theory(empirical_results, theoretical_bounds):
    """
    Check if empirical results match theoretical predictions.

    Target: Error < 5%
    """
    validations = {}

    # Bandwidth efficiency
    emp_bw = empirical_results['avg_bandwidth_mbps']
    theo_bw = theoretical_bounds['optimal_rate_mbps']
    bw_error = abs(emp_bw - theo_bw) / theo_bw * 100
    validations['bandwidth_error_percent'] = bw_error

    # Task success rate
    emp_success = empirical_results['task_success_rate']
    theo_success = theoretical_bounds['predicted_success_rate']
    success_error = abs(emp_success - theo_success) / theo_success * 100
    validations['success_rate_error_percent'] = success_error

    # Drift accumulation
    emp_drift = empirical_results['max_drift']
    theo_drift = theoretical_bounds['drift_upper_bound']
    drift_error = abs(emp_drift - theo_drift) / theo_drift * 100
    validations['drift_error_percent'] = drift_error

    # Overall validation
    validations['all_within_5_percent'] = all(
        e < 5.0 for e in [bw_error, success_error, drift_error]
    )

    return validations
```

---

## 5. Ablation Studies

### 5.1 Component Ablation

**Test each component's contribution**:

| Configuration | Description | Expected Impact |
|--------------|-------------|-----------------|
| **Full SSC** | All components enabled | Baseline |
| **No DSA** | Random token selection | Task success ↓15%, Bandwidth ↑3x |
| **No Quantization** | FP32 only | Bandwidth ↑4x, Success ↔ |
| **No Compression** | Raw tokens | Bandwidth ↑2.5x |
| **No KV Projector** | Direct injection | Cloud task success ↓25% |
| **No Delta Streaming** | Full state every time | Bandwidth ↑10x |

### 5.2 Hyperparameter Sensitivity

**Top-k selection**:
```python
top_k_values = [4, 8, 16, 32, 64, 128]
# Expected: Sweet spot at k=32 (task success 92%, bandwidth 0.02 Mbps)
```

**Quantization levels**:
```python
quant_levels = ['fp32', 'fp16', 'fp8', 'int4']
# Expected: FP8 optimal (90% success, 4x compression vs FP32)
```

**Reset threshold**:
```python
reset_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
# Expected: τ=0.3 optimal (minimize resets while keeping drift <5%)
```

---

## 6. Expected Outcomes

### 6.1 Key Results (Predictions)

**Hypothesis 1**: SSC achieves 99%+ bandwidth savings
- **Expected**: 0.02 Mbps (SSC) vs. 5.2 Mbps (H.264) = 99.6% savings ✅
- **Validation**: Measure actual bandwidth in all 3 scenarios

**Hypothesis 2**: Task success rate >90% with aggressive compression
- **Expected**: 92% with FP8 quantization + top-k=32
- **Validation**: Compare against ground truth annotations

**Hypothesis 3**: End-to-end latency <50ms
- **Expected**: 18ms (encoding) + 12ms (transmission) + 8ms (decoding) = 38ms
- **Validation**: Profile latency breakdown in real hardware

**Hypothesis 4**: Robust to 10% packet loss
- **Expected**: Task success degrades <3% (92% → 89%)
- **Validation**: Network emulation with controlled packet loss

**Hypothesis 5**: Cross-dataset generalization
- **Expected**: VIRAT→UCF-Crime accuracy drop <5%
- **Validation**: Train on one dataset, test on others

### 6.2 Publication-Ready Figures

**Figure 1: Rate-Distortion Curve**
- X-axis: Bandwidth (Mbps, log scale)
- Y-axis: Task Success Rate (%)
- Curves: SSC (FP32/FP16/FP8/INT4), H.264, CLIP, Text
- Annotation: Theoretical bound (dashed line)

**Figure 2: Latency Breakdown**
- Stacked bar chart comparing SSC vs. baselines
- Components: Encoding, Transmission, Decoding, Inference
- Highlight: SSC transmission = 12ms vs. H.264 = 95ms

**Figure 3: Robustness to Packet Loss**
- X-axis: Packet loss rate (0-20%)
- Y-axis: Task success rate (%)
- Curves: SSC, H.264, CLIP
- Annotation: SSC degrades gracefully (RAG recovery)

**Figure 4: Scalability (Smart Factory)**
- X-axis: Number of sensors (10-500)
- Y-axis: Total bandwidth (Mbps)
- Curves: SSC (linear), Video (quadratic)
- Annotation: SSC enables 10x more sensors

**Figure 5: Theoretical vs. Empirical Validation**
- Scatter plot: Theoretical prediction (x) vs. Empirical result (y)
- Metrics: Bandwidth, Latency, Drift, Task Success
- Target: All points within ±5% diagonal line

---

## 7. Risk Mitigation

### 7.1 Potential Issues & Contingencies

**Issue 1: Hardware unavailability**
- **Risk**: Jetson Nano out of stock, A100 access limited
- **Mitigation**: Use Raspberry Pi 4 + RTX 3090 as alternatives
- **Impact**: Latency may increase 20-30%, but trends remain valid

**Issue 2: Dataset annotation quality**
- **Risk**: Fire annotations insufficient for training
- **Mitigation**: Use data augmentation + weak supervision
- **Backup**: Switch to pre-labeled datasets (COCO, ImageNet)

**Issue 3: Network emulation accuracy**
- **Risk**: Mininet-WiFi doesn't model 5G NR accurately
- **Mitigation**: Validate with real 5G testbed (if available)
- **Backup**: Use conservative parameters (higher loss, lower SNR)

**Issue 4: Projector training instability**
- **Risk**: KV-Cache dimension mismatch causes gradient explosion
- **Mitigation**: Use gradient clipping + layer normalization
- **Backup**: Use simpler linear projection (no residual)

**Issue 5: Theoretical-empirical gap >5%**
- **Risk**: Real-world results deviate from theoretical predictions
- **Mitigation**: Refine theory with empirical correction factors
- **Explanation**: Document sources of discrepancy (non-idealities)

---

## 8. Timeline & Milestones

### Week 1-2: Setup & Baseline
- [ ] Configure Jetson Nano + A100 environment
- [ ] Install datasets (VIRAT, nuScenes, MVTec)
- [ ] Run baseline experiments (H.264, CLIP, Text)
- [ ] Deliverable: `results/baselines.json`

### Week 3-4: SSC Implementation
- [ ] Train KV-Cache Projector (512→4096)
- [ ] Implement DSA Lightning Indexer
- [ ] Integrate Protobuf encoding + ZSTD compression
- [ ] Deliverable: End-to-end SSC pipeline

### Week 5-6: Core Experiments
- [ ] Run SSC on all 3 scenarios
- [ ] Collect bandwidth, latency, task success metrics
- [ ] Compare against baselines
- [ ] Deliverable: `results/ssc_core.json`

### Week 7: Robustness & Ablation
- [ ] Packet loss sweep (0-20%)
- [ ] SNR variation (0-30dB)
- [ ] Ablation studies (component removal)
- [ ] Deliverable: `results/robustness.json`

### Week 8: Analysis & Validation
- [ ] Theoretical vs. empirical comparison
- [ ] Generate publication figures
- [ ] Write experimental section draft
- [ ] Deliverable: Paper Section V (Evaluation)

---

## 9. Code Repository Structure

```
AI-Comm/
├── experiments/
│   ├── baselines/
│   │   ├── h264_baseline.py
│   │   ├── clip_baseline.py
│   │   └── text_baseline.py
│   ├── ssc/
│   │   ├── edge_encoder.py          # Semantic token generation
│   │   ├── cloud_decoder.py         # KV-Cache reconstruction
│   │   ├── kv_projector.py          # 512→4096 alignment
│   │   └── ssc_pipeline.py          # End-to-end system
│   ├── robustness/
│   │   ├── packet_loss.py
│   │   ├── snr_variation.py
│   │   └── generalization.py
│   └── ablation/
│       ├── ablate_dsa.py
│       ├── ablate_quantization.py
│       └── ablate_projector.py
├── datasets/
│   ├── virat/
│   ├── nuscenes/
│   └── mvtec/
├── models/
│   ├── mobilevlm_3b/
│   ├── gpt4v/
│   └── projector_checkpoints/
├── results/
│   ├── baselines.json
│   ├── ssc_core.json
│   ├── robustness.json
│   └── ablation.json
└── scripts/
    ├── run_all_experiments.sh
    ├── analyze_results.py
    └── generate_figures.py
```

---

## 10. Success Criteria

### Minimum Acceptable Results (Paper Acceptance Threshold)

✅ **Bandwidth**: SSC < 0.1 Mbps (vs. H.264 5.2 Mbps) → 98%+ savings
✅ **Task Success**: >85% with FP8 quantization
✅ **Latency**: <100ms end-to-end (edge scenario)
✅ **Robustness**: <10% degradation at 10% packet loss
✅ **Theory-Practice Gap**: <10% error on key metrics

### Target Results (Strong Accept)

🎯 **Bandwidth**: SSC < 0.05 Mbps → 99%+ savings
🎯 **Task Success**: >90% with FP8 quantization
🎯 **Latency**: <50ms end-to-end
🎯 **Robustness**: <5% degradation at 10% packet loss
🎯 **Theory-Practice Gap**: <5% error on all metrics

### Stretch Goals (Best Paper Candidate)

🏆 **Real-world Deployment**: Live demo on 5G testbed
🏆 **Cross-dataset**: <2% accuracy drop (VIRAT→UCF-Crime)
🏆 **Scalability**: Support 500+ sensors in smart factory
🏆 **Open-source**: Release code + pre-trained models

---

## References

**Datasets**:
- VIRAT Video Dataset: https://viratdata.org/
- nuScenes: https://www.nuscenes.org/
- MVTec Anomaly Detection: https://www.mvtec.com/company/research/datasets/mvtec-ad
- COCO: https://cocodataset.org/

**Baselines**:
- H.264 Reference: JM reference software
- CLIP: OpenAI CLIP-ViT-Large-Patch14
- C2C (Cache-to-Cache): https://github.com/C2C-Communication
- CacheGen: https://github.com/CacheGen

**Tools**:
- Network Emulation: Mininet-WiFi, ns-3
- Video Processing: FFmpeg, OpenCV
- Deep Learning: PyTorch 2.0, Hugging Face Transformers

---

**Status**: Phase 3 experimental design complete. Ready to begin implementation.
**Next**: Hardware setup + baseline experiments (Week 1-2).
