# Phase 3: Experimental Validation - Progress Tracker

**Date**: 2026-01-24
**Status**: 🚧 **Planning Complete, Ready for Implementation**
**Duration**: 8 weeks (estimated)

---

## Executive Summary

Phase 3 focuses on empirical validation of the SSC framework through comprehensive experiments across three real-world scenarios:
1. **UAV Fire Detection** (wildfire monitoring)
2. **V2V Autonomous Driving** (collision avoidance)
3. **Smart Factory** (anomaly detection)

**Goals**:
- ✅ Validate theoretical predictions (error <5%)
- ✅ Demonstrate 99%+ bandwidth savings vs. baselines
- ✅ Prove task success >90% with aggressive compression
- ✅ Show robustness to packet loss and channel noise
- ✅ Cross-dataset generalization

**Current Maturity**: 85-90/100 (theoretical)
**Target After Phase 3**: 90-95/100 (empirical validation complete, paper-ready)

---

## Progress Overview

### Phase 3 Deliverables Status

| Deliverable | Status | Progress | ETA |
|------------|--------|----------|-----|
| **3.1 Experimental Design** | ✅ Complete | 100% | 2026-01-24 |
| **3.2 Implementation Spec** | ✅ Complete | 100% | 2026-01-24 |
| **3.3 Hardware Setup** | ⏸️ Pending | 0% | Week 1 |
| **3.4 Dataset Preparation** | ⏸️ Pending | 0% | Week 1-2 |
| **3.5 Baseline Experiments** | ⏸️ Pending | 0% | Week 2 |
| **3.6 SSC Implementation** | ⏸️ Pending | 0% | Week 3-4 |
| **3.7 Core Experiments** | ⏸️ Pending | 0% | Week 5-6 |
| **3.8 Robustness Testing** | ⏸️ Pending | 0% | Week 7 |
| **3.9 Analysis & Paper** | ⏸️ Pending | 0% | Week 8 |

**Overall Progress**: 20% (Planning & Design) ✅

---

## Phase 3 Timeline

### Week 1-2: Setup & Baseline (Days 1-14)

**Objectives**:
- Configure hardware (Jetson Nano + A100 Server)
- Prepare datasets (VIRAT, nuScenes, MVTec)
- Run baseline experiments (H.264, CLIP, Text)

**Detailed Tasks**:

#### Week 1 Tasks
- [ ] **Day 1-2**: Hardware Procurement & Setup
  - [ ] Order/acquire Jetson Nano Developer Kit
  - [ ] Provision A100 GPU cloud instance (AWS/Azure/GCP)
  - [ ] Install JetPack 5.1 on Jetson
  - [ ] Install CUDA 12.1 + PyTorch 2.0 on A100
  - [ ] Verify GPU connectivity (`nvidia-smi`)

- [ ] **Day 3-4**: Network Configuration
  - [ ] Set up network emulation (Mininet-WiFi or tc netem)
  - [ ] Configure 5G NR parameters (100MHz bandwidth)
  - [ ] Test edge-cloud connectivity (ping, iperf)
  - [ ] Baseline latency measurement (<10ms RTT target)

- [ ] **Day 5-7**: Dataset Download & Preprocessing
  - [ ] Download VIRAT Video Dataset (8.5 hours, ~100GB)
  - [ ] Download nuScenes (1.4M images, ~400GB)
  - [ ] Download MVTec AD (5,354 images, ~5GB)
  - [ ] Extract frames, create train/val/test splits
  - [ ] Annotate fire events in VIRAT (if needed)

#### Week 2 Tasks
- [ ] **Day 8-10**: Baseline Implementation
  - [ ] Implement H.264 video streaming pipeline
    - Edge: FFmpeg encoding (1080p @ 30fps, bitrate 5Mbps)
    - Cloud: FFmpeg decoding + object detection
    - Measure: bandwidth, latency, task success
  - [ ] Implement CLIP baseline
    - Edge: CLIP ViT-L/14 encoding (1 frame/sec)
    - Cloud: CLIP zero-shot classification
    - Measure: bandwidth (512-dim embeddings), accuracy
  - [ ] Implement Text prompts baseline
    - Edge: GPT-4V captioning
    - Cloud: Text-based reasoning
    - Measure: bandwidth, latency, task success

- [ ] **Day 11-14**: Baseline Evaluation
  - [ ] Run baselines on VIRAT fire detection scenario
  - [ ] Collect metrics:
    ```json
    {
      "h264": {
        "bandwidth_mbps": 5.2,
        "latency_ms": 120,
        "task_success_rate": 0.92,
        "energy_joules_per_event": 0.19
      },
      "clip": {...},
      "text": {...}
    }
    ```
  - [ ] Save results to `results/baselines.json`
  - [ ] Generate preliminary comparison plots

**Week 1-2 Deliverable**: ✅ Baseline results documented, ready for SSC comparison

---

### Week 3-4: SSC Implementation (Days 15-28)

**Objectives**:
- Implement complete SSC pipeline
- Train KV-Cache Projector
- End-to-end integration testing

**Detailed Tasks**:

#### Week 3 Tasks
- [ ] **Day 15-17**: Edge Components
  - [ ] Implement `PerceptionModule` (camera input, preprocessing)
  - [ ] Load MobileVLM-3B model
  - [ ] Implement `forward_with_cache()` to extract KV-Cache
  - [ ] Test: Verify KV-Cache shape (24 layers × 512-dim)

- [ ] **Day 18-20**: Semantic Indexer
  - [ ] Implement DSA Lightning attention scoring
  - [ ] Implement top-k token selection (k=32)
  - [ ] Add context-aware adaptive k
  - [ ] Test: Compare selected vs. random tokens (attention scores)

- [ ] **Day 21**: Token Encoder
  - [ ] Implement Protobuf serialization
  - [ ] Implement FP8 quantization
  - [ ] Implement ZSTD compression
  - [ ] Test: Measure compression ratio (target: >10x)

#### Week 4 Tasks
- [ ] **Day 22-24**: KV-Cache Projector Training
  - [ ] Create paired dataset (MobileVLM ↔ GPT-4V KV-Cache)
    - Run both models on 10,000 images from COCO
    - Save paired (KV_small, KV_large) to disk
  - [ ] Train projector (50 epochs, ~12 hours on A100)
  - [ ] Validate: Measure MSE loss (target: <0.01)
  - [ ] Save checkpoint: `models/projector_512_to_4096.pt`

- [ ] **Day 25-27**: Cloud Components
  - [ ] Implement `TokenDecoder` (decompress, deserialize, dequantize)
  - [ ] Load GPT-4V model
  - [ ] Implement KV-Cache injection into GPT-4V
  - [ ] Test: Verify cloud task execution with projected KV

- [ ] **Day 28**: End-to-End Integration
  - [ ] Connect edge and cloud over network
  - [ ] Test full pipeline: Camera → Edge → Network → Cloud → Response
  - [ ] Measure end-to-end latency (target: <50ms)
  - [ ] Debug any issues (serialization, dimension mismatch, etc.)

**Week 3-4 Deliverable**: ✅ Working SSC pipeline, end-to-end latency <50ms

---

### Week 5-6: Core Experiments (Days 29-42)

**Objectives**:
- Run SSC on all 3 scenarios
- Collect comprehensive metrics
- Compare against baselines

**Detailed Tasks**:

#### Week 5 Tasks
- [ ] **Day 29-31**: Scenario 1 - UAV Fire Detection
  - [ ] Configure: VIRAT dataset, fire annotations
  - [ ] Run SSC with configurations:
    ```yaml
    experiments:
      - quantization: fp32, top_k: 32
      - quantization: fp16, top_k: 32
      - quantization: fp8, top_k: 32
      - quantization: int4, top_k: 32
      - quantization: fp8, top_k: [8, 16, 64, 128]
    ```
  - [ ] Collect metrics for each config:
    - Bandwidth (Mbps)
    - Latency (ms): encoding, transmission, decoding, total
    - Task success rate (fire localization error <10m)
    - Energy consumption (J per event)
  - [ ] Save results: `results/ssc_fire_detection.json`

- [ ] **Day 32-34**: Scenario 2 - V2V Autonomous Driving
  - [ ] Configure: nuScenes dataset, pedestrian hazards
  - [ ] Run SSC with same configurations as Scenario 1
  - [ ] Additional metrics:
    - Collision avoidance rate
    - False positive rate
    - End-to-end latency (detection → braking decision)
  - [ ] Stress test: 20 vehicles, 5 hazards/sec, 10% packet loss
  - [ ] Save results: `results/ssc_v2v.json`

- [ ] **Day 35**: Data Analysis
  - [ ] Aggregate results across scenarios
  - [ ] Compute averages, std deviations
  - [ ] Identify optimal configuration (likely FP8, k=32)

#### Week 6 Tasks
- [ ] **Day 36-38**: Scenario 3 - Smart Factory
  - [ ] Configure: MVTec AD dataset, anomaly detection
  - [ ] Run SSC focusing on sparsity (normal = no transmission)
  - [ ] Metrics:
    - Bandwidth per sensor per hour
    - Anomaly detection precision/recall
    - Scalability: test with 50, 100, 500 simulated sensors
  - [ ] Save results: `results/ssc_smart_factory.json`

- [ ] **Day 39-41**: Comparison Analysis
  - [ ] Create comparison tables:
    | Metric | H.264 | CLIP | Text | SSC (Ours) |
    |--------|-------|------|------|------------|
    | Bandwidth (Mbps) | 5.2 | 2.1 | 0.8 | **0.02** |
    | Latency (ms) | 120 | 95 | 150 | **18** |
    | Task Success (%) | 92 | 89 | 85 | **92** |
    | Energy (J/event) | 0.19 | 0.08 | 0.03 | **0.012** |

  - [ ] Generate figures:
    - Rate-distortion curve (bandwidth vs. task success)
    - Latency breakdown (stacked bar chart)
    - Energy efficiency comparison

- [ ] **Day 42**: Theoretical Validation
  - [ ] Compare empirical results vs. theoretical predictions
  - [ ] Compute errors:
    ```python
    bandwidth_error = |empirical_bw - theoretical_bw| / theoretical_bw
    # Target: <5%
    ```
  - [ ] Document discrepancies and explanations

**Week 5-6 Deliverable**: ✅ Complete experimental results, theory validated

---

### Week 7: Robustness & Ablation (Days 43-49)

**Objectives**:
- Test robustness to packet loss, channel noise
- Cross-dataset generalization
- Component ablation studies

**Detailed Tasks**:

#### Robustness Tests
- [ ] **Day 43-44**: Packet Loss Sweep
  - [ ] Configure network emulation: loss rates [0%, 5%, 10%, 15%, 20%]
  - [ ] Run SSC on VIRAT fire detection for each loss rate
  - [ ] Measure task success degradation
  - [ ] Expected: <3% degradation at 10% loss (92% → 89%)
  - [ ] Save: `results/robustness_packet_loss.json`

- [ ] **Day 45**: Channel SNR Variation
  - [ ] Configure AWGN noise: SNR [0, 5, 10, 15, 20, 25, 30] dB
  - [ ] Run SSC on VIRAT
  - [ ] Measure task success vs. SNR curve
  - [ ] Save: `results/robustness_snr.json`

#### Generalization Tests
- [ ] **Day 46**: Cross-Dataset Evaluation
  - [ ] Train on VIRAT, test on:
    - UCF-Crime (violence detection)
    - COCO (general object detection)
    - ShanghaiTech (crowd counting)
  - [ ] Measure accuracy drop (target: <5%)
  - [ ] Save: `results/generalization.json`

#### Ablation Studies
- [ ] **Day 47-49**: Component Ablation
  - [ ] Run experiments with components disabled:
    1. **No DSA**: Random token selection instead of attention-based
       - Expected: Task success ↓15%, Bandwidth ↑3x
    2. **No Quantization**: FP32 only
       - Expected: Bandwidth ↑4x, Success ↔
    3. **No Compression**: Raw Protobuf without ZSTD
       - Expected: Bandwidth ↑2.5x
    4. **No KV Projector**: Skip alignment (dimension mismatch error expected)
       - Measure degradation with direct injection (if possible)
    5. **No Delta Streaming**: Transmit full KV-Cache every time
       - Expected: Bandwidth ↑10x
  - [ ] Save: `results/ablation.json`

**Week 7 Deliverable**: ✅ Robustness analysis, ablation study complete

---

### Week 8: Analysis & Paper Writing (Days 50-56)

**Objectives**:
- Finalize all figures and tables
- Write Experimental Evaluation section (Section V)
- Prepare conference submission draft

**Detailed Tasks**:

- [ ] **Day 50-51**: Figure Generation
  - [ ] **Figure 1**: Rate-Distortion Curves
    - X-axis: Bandwidth (Mbps, log scale)
    - Y-axis: Task Success Rate (%)
    - Lines: SSC (FP32/FP16/FP8/INT4), H.264, CLIP, Text, Theoretical Bound
  - [ ] **Figure 2**: Latency Breakdown
    - Stacked bar chart: Encoding + Transmission + Decoding + Inference
    - Compare SSC vs. baselines
  - [ ] **Figure 3**: Robustness to Packet Loss
    - X-axis: Packet loss rate (%)
    - Y-axis: Task success rate (%)
    - Lines: SSC, H.264, CLIP
  - [ ] **Figure 4**: Scalability (Smart Factory)
    - X-axis: Number of sensors
    - Y-axis: Total bandwidth (Mbps)
    - Lines: SSC (linear), Video (quadratic)
  - [ ] **Figure 5**: Theoretical vs. Empirical
    - Scatter plot: Predicted (x) vs. Actual (y)
    - Points: Bandwidth, Latency, Drift, Task Success
    - Target: All within ±5% diagonal

- [ ] **Day 52-53**: Table Creation
  - [ ] **Table 1**: Experimental Configuration
    - Hardware specs, model parameters, dataset statistics
  - [ ] **Table 2**: Main Results Comparison
    - SSC vs. baselines across 3 scenarios
  - [ ] **Table 3**: Ablation Study Results
    - Impact of each component on performance
  - [ ] **Table 4**: Cross-Dataset Generalization
    - Accuracy on train vs. test datasets

- [ ] **Day 54-55**: Paper Section V Writing
  - [ ] **V.A Experimental Setup**
    - Hardware, datasets, baselines, evaluation metrics
  - [ ] **V.B Main Results**
    - Bandwidth efficiency (99.6% savings)
    - Task success (92% with FP8)
    - Latency (18ms end-to-end)
    - Energy efficiency (0.012J per event)
  - [ ] **V.C Robustness Analysis**
    - Packet loss, SNR variation, cross-dataset results
  - [ ] **V.D Ablation Study**
    - Validate each component's contribution
  - [ ] **V.E Theoretical Validation**
    - Theory-practice error <5% on all metrics
  - [ ] **V.F Discussion**
    - Insights, limitations, future work

- [ ] **Day 56**: Final Review & Integration
  - [ ] Integrate Section V into full paper draft
  - [ ] Cross-check all references, citations
  - [ ] Verify figure/table numbering
  - [ ] Spell check, grammar check
  - [ ] Generate PDF for advisor review

**Week 8 Deliverable**: ✅ Section V complete, paper ready for advisor review

---

## Key Metrics Targets

### Primary Metrics (Must Achieve)

| Metric | Theoretical Prediction | Empirical Target | Acceptance Threshold |
|--------|------------------------|------------------|---------------------|
| **Bandwidth Savings** | 99.6% (vs. H.264) | 99%+ | 98%+ |
| **Task Success Rate** | 92% (FP8, k=32) | 90%+ | 85%+ |
| **End-to-End Latency** | 38ms | <50ms | <100ms |
| **Robustness (10% loss)** | 89% (3% degradation) | 87%+ | 80%+ |
| **Theory-Practice Error** | N/A | <5% | <10% |

### Secondary Metrics (Nice to Have)

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Energy per Event** | <0.02J | <0.01J |
| **Cross-Dataset Accuracy Drop** | <5% | <2% |
| **Scalability (sensors)** | 100 | 500 |
| **Compression Ratio** | 10x | 20x |

---

## Risk Assessment & Mitigation

### High-Risk Items (Probability: Medium-High)

**Risk 1: Hardware Unavailability**
- **Impact**: Delays of 1-2 weeks
- **Probability**: 30%
- **Mitigation**:
  - Pre-order Jetson Nano (or use Raspberry Pi 4 as backup)
  - Secure A100 access early (AWS reserved instance or academic cluster)
- **Contingency**: Use lighter models (MobileNet instead of MobileVLM)

**Risk 2: Theoretical-Empirical Gap >10%**
- **Impact**: Need to revise theory or explain discrepancy
- **Probability**: 40%
- **Mitigation**:
  - Run preliminary experiments early (Week 3)
  - Identify sources of error (quantization, channel noise)
- **Contingency**: Document gap as "real-world correction factors"

**Risk 3: Dataset Annotation Quality**
- **Impact**: Unreliable task success metrics
- **Probability**: 20%
- **Mitigation**:
  - Use multiple annotators for fire detection
  - Cross-validate with existing labeled datasets
- **Contingency**: Switch to fully pre-labeled datasets (COCO)

### Medium-Risk Items (Probability: Low-Medium)

**Risk 4: Network Emulation Accuracy**
- **Impact**: Results may not reflect real 5G performance
- **Probability**: 25%
- **Mitigation**:
  - Validate emulation with real 5G testbed (if available)
  - Use conservative parameters (higher loss, lower SNR)
- **Contingency**: Note limitation in paper, propose future real-world validation

**Risk 5: Projector Training Instability**
- **Impact**: Poor KV-Cache alignment, low task success
- **Probability**: 15%
- **Mitigation**:
  - Use gradient clipping, careful hyperparameter tuning
  - Start with simpler linear projection
- **Contingency**: Use dimension padding/truncation instead of learned projection

---

## Resource Requirements

### Compute Resources

| Resource | Specification | Duration | Cost Estimate |
|----------|--------------|----------|---------------|
| **Edge Device** | Jetson Nano 4GB | 8 weeks | $150 (one-time) |
| **Cloud GPU** | A100 80GB (AWS p4d.24xlarge) | 200 hours | $3,200 ($16/hour) |
| **Storage** | 1TB SSD (datasets + results) | 8 weeks | $100 (one-time) |
| **Network** | 5G emulation (software) | 8 weeks | $0 (open-source) |
| **Total** | - | - | **~$3,500** |

**Cost Optimization**:
- Use academic GPU cluster if available (free)
- Run experiments overnight to maximize GPU utilization
- Use spot instances for non-critical experiments (50% discount)

### Human Resources

| Role | Time Commitment | Responsibilities |
|------|----------------|------------------|
| **PhD Student** | 40 hours/week × 8 weeks | Implementation, experiments, paper writing |
| **Advisor** | 2 hours/week × 8 weeks | Weekly meetings, feedback, paper review |
| **Collaborator (Optional)** | 10 hours/week × 4 weeks | Dataset annotation, baseline implementation |

---

## Deliverables Checklist

### Week 1-2 Deliverables
- [ ] `results/baselines.json` - Baseline metrics (H.264, CLIP, Text)
- [ ] `results/baseline_comparison.pdf` - Preliminary comparison plots
- [ ] `docs/hardware_setup.md` - Hardware configuration documentation

### Week 3-4 Deliverables
- [ ] `models/projector_512_to_4096.pt` - Trained KV-Cache Projector
- [ ] `experiments/ssc/ssc_pipeline.py` - Complete SSC implementation
- [ ] `results/end_to_end_latency.json` - Latency breakdown measurements

### Week 5-6 Deliverables
- [ ] `results/ssc_fire_detection.json` - Scenario 1 results
- [ ] `results/ssc_v2v.json` - Scenario 2 results
- [ ] `results/ssc_smart_factory.json` - Scenario 3 results
- [ ] `results/comparison_table.md` - SSC vs. baselines comparison

### Week 7 Deliverables
- [ ] `results/robustness_packet_loss.json` - Packet loss robustness
- [ ] `results/robustness_snr.json` - SNR variation robustness
- [ ] `results/generalization.json` - Cross-dataset results
- [ ] `results/ablation.json` - Component ablation study

### Week 8 Deliverables
- [ ] `paper/figures/` - All publication-quality figures (PDF/PNG)
- [ ] `paper/tables/` - All LaTeX tables
- [ ] `paper/section_5_evaluation.tex` - Experimental section draft
- [ ] `paper/full_draft.pdf` - Complete paper PDF for review

---

## Success Criteria

### Minimum Viable Results (Accept Threshold)

✅ **Technical Success**:
- SSC achieves 98%+ bandwidth savings vs. H.264
- Task success >85% with FP8 quantization
- End-to-end latency <100ms
- Robust to 10% packet loss (<10% degradation)
- Theory-practice error <10%

✅ **Academic Success**:
- Results support all theoretical claims
- Sufficient novelty vs. SOTA (C2C, CacheGen)
- Clear ablation study validating design choices
- Cross-dataset generalization demonstrated

### Target Results (Strong Accept)

🎯 **Technical Excellence**:
- SSC achieves 99%+ bandwidth savings
- Task success >90% with FP8 quantization
- End-to-end latency <50ms
- Theory-practice error <5%
- Real-world 5G testbed validation (if possible)

🎯 **Academic Excellence**:
- Results exceed theoretical predictions
- Open-source code + pre-trained models released
- Live demo at conference (fire detection on drone)
- Multiple dataset evaluations

### Stretch Goals (Best Paper Consideration)

🏆 **Exceptional Impact**:
- Deploy on real UAV swarm in collaboration with industry
- Demonstrate 10x scalability (500+ sensors)
- Achieve <1% theory-practice error
- Create new benchmark dataset for semantic communication
- Multi-modal fusion (Vision + LiDAR + Audio)

---

## Next Steps

### Immediate Actions (This Week)

1. **Confirm hardware access**:
   - [ ] Order Jetson Nano or confirm availability
   - [ ] Secure A100 GPU access (cloud or academic cluster)

2. **Begin dataset downloads** (can run in background):
   - [ ] VIRAT: `wget https://viratdata.org/downloads/...`
   - [ ] nuScenes: `wget https://nuscenes.org/data/...`
   - [ ] MVTec AD: `wget https://www.mvtec.com/...`

3. **Set up development environment**:
   - [ ] Create conda environment: `conda create -n ssc python=3.10`
   - [ ] Install dependencies: `pip install torch transformers protobuf zstandard`

4. **Baseline implementation** (quick win):
   - [ ] Start with H.264 baseline (simplest)
   - [ ] Measure bandwidth on a few sample videos
   - [ ] Verify evaluation pipeline works

### Decision Points

**Decision 1: Hardware Selection** (Week 1)
- Option A: Jetson Nano (official, $150, may have stock issues)
- Option B: Raspberry Pi 4 (backup, $80, readily available)
- **Recommendation**: Order both, use Jetson if available

**Decision 2: Cloud Provider** (Week 1)
- Option A: AWS p4d.24xlarge (A100, $16/hour, reliable)
- Option B: Academic GPU cluster (free, may have queue times)
- Option C: Azure NC A100 v4 (similar pricing)
- **Recommendation**: Academic cluster first, AWS as backup

**Decision 3: Dataset Scope** (Week 2)
- Option A: Full datasets (VIRAT 8.5 hours, nuScenes 1.4M images)
- Option B: Subset (VIRAT 2 hours, nuScenes 100K images)
- **Recommendation**: Start with subset, expand if needed

---

## Advisor Discussion Topics

### Items to Confirm with Advisor

1. **Budget approval**: $3,500 for cloud GPU + hardware
2. **Timeline feasibility**: 8 weeks realistic for full experiments?
3. **Scope**: Focus on 3 scenarios or reduce to 2 (fire + V2V)?
4. **Publication target**: INFOCOM 2027 (deadline Oct 2026) or ICC 2027?
5. **Collaboration**: Need help with dataset annotation?

### Questions to Ask Advisor

1. Do we have access to real 5G testbed for validation?
2. Should we prioritize real-world demo over comprehensive experiments?
3. Any concerns about theoretical-empirical gap?
4. Open-source release timing (before or after publication)?
5. Industry collaboration opportunities (UAV companies, auto manufacturers)?

---

## Summary

**Phase 3 Status**: Planning 100% complete ✅

**Ready to Start**:
- ✅ Experimental design documented (`experimental-design.md`)
- ✅ Implementation specification ready (`ssc-pipeline-spec.md`)
- ✅ Timeline and milestones defined
- ✅ Risk mitigation strategies in place

**Pending User Input**:
- Hardware procurement approval
- Budget confirmation
- Timeline acceptance
- Scope adjustment (if needed)

**Estimated Completion**: 8 weeks from start date

**Expected Outcome**: Research maturity 90-95/100, paper ready for INFOCOM submission

---

**Next Action**: Await user confirmation to begin Week 1 tasks (hardware setup + dataset preparation).

**Alternative**: If hardware access is delayed, can proceed with simulation-based experiments using synthetic data to validate pipeline before full evaluation.

---

**Phase 3 Planning Complete** ✅
**Ready to Execute** 🚀
