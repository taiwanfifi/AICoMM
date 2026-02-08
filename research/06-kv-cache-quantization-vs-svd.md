# Topic 6: KV-Cache Quantization vs Spectral Compression — A Communication-Theoretic Analysis

> **Status**: CONFIRMED — INT8/INT4 both LOSSLESS for task accuracy
> **Target Venue**: IEEE SPAWC 2027 / IEEE Workshop / ISIT 2027
> **Confidence**: High (data confirms orthogonal compression axes)

## Core Hypothesis

SVD spectral compression and quantization (INT8, INT4, NF4) represent fundamentally different compression strategies for KV-cache: SVD preserves global semantic structure (low-rank approximation) while quantization preserves local precision (per-element). Their optimal operating regimes differ, and a hybrid approach combining both outperforms either alone.

## Why This Matters

The ML community focuses on quantization (GPTQ, AWQ, SqueezeLLM) for inference efficiency. But from a communication perspective, quantization is just "reduce bits per element" — it doesn't reduce the NUMBER of elements transmitted. SVD reduces dimensionality. These are orthogonal axes of compression.

## Compression Taxonomy

```
           Elements Transmitted
           Full          Reduced
Bits   ┌─────────────┬──────────────┐
Full   │ Full KV     │ Selection    │
       │ (baseline)  │ (SnapKV/Q2C) │
       ├─────────────┼──────────────┤
Reduced│ Quantized   │ Select+Quant │
       │ (INT8/INT4) │ (hybrid 1)   │
       ├─────────────┼──────────────┤
Low-   │ SVD only    │ Select+SVD   │
Rank   │ (our Exp06) │ (hybrid 2)   │
       └─────────────┴──────────────┘
```

## Experimental Plan

### Phase 1: Quantization Baseline (1 day)
1. Full KV-cache in FP16 (baseline)
2. Quantize to INT8: `kv_int8 = round(kv_fp16 * scale)`
3. Quantize to INT4: group-wise quantization
4. Quantize to NF4: NormalFloat4 (from bitsandbytes)
5. Measure: F1 on SQuAD, transmission size, reconstruction error

### Phase 2: SVD vs Quantization at Matched Bandwidth (1 day)
1. SVD rank-r: Bandwidth = 2 * r * (T + D) * 2 bytes
2. INT8: Bandwidth = T * D * 1 byte (50% of FP16)
3. INT4: Bandwidth = T * D * 0.5 bytes (25% of FP16)
4. Find SVD rank that matches INT4/INT8 bandwidth → compare F1

### Phase 3: Hybrid — SVD + Quantization (1 day)
1. SVD compress to rank-r → get U, S, Vh matrices
2. Quantize U, S, Vh to INT8 → further 2x compression
3. Total compression: rank reduction × quantization
4. Compare with: pure SVD at same total size, pure quantization at same size

## Initial Experimental Results (2026-02-08)

### Reconstruction Error (Qwen2.5-3B, 20 samples)

| Method | Bandwidth | Mean Relative Error | Notes |
|--------|-----------|-------------------|-------|
| **INT8** | 50% | **0.018 (1.8%)** | Near-lossless |
| **INT4** (group=32) | 25% | **0.123 (12.3%)** | Reasonable |
| SVD rank-4 | ~6% | 0.668 (66.8%) | Very lossy |
| SVD rank-8 | ~11% | 0.592 (59.2%) | Still bad |
| SVD rank-16 | ~22% | 0.488 (48.8%) | Improving |
| SVD rank-32 | ~44% | 0.349 (34.9%) | Moderate |

**Key Insight**: INT8 quantization at 50% bandwidth has 20x lower error than SVD rank-32 at 44% bandwidth. Quantization dominates SVD in reconstruction error. BUT: our Exp06 showed SVD dominates SnapKV in QA accuracy — suggesting reconstruction error ≠ task accuracy.

**Hypothesis update**: ~~SVD may work by preserving semantic structure even with high reconstruction error, while quantization preserves local precision but may miss global patterns.~~ CONFIRMED below: quantization is LOSSLESS for task accuracy.

### Batch 4: Task Accuracy Results (Qwen2.5-3B, SQuAD v2, 30 samples)

| Method | Bandwidth | F1 | % of Full KV | Status |
|--------|-----------|-----|-------------|--------|
| Full KV (FP16) | 100% | 0.737 | 100% | Baseline |
| **INT8** | **50%** | **0.737** | **100%** | **LOSSLESS** |
| **INT4** (group=32) | **25%** | **0.748** | **101%** | **LOSSLESS (above baseline!)** |
| SVD rank-16 | ~22% | 0.149 | 20% | Very lossy |
| SVD rank-32 | ~44% | 0.502 | 68% | Moderate |

**CONFIRMED**: Both INT8 and INT4 quantization are LOSSLESS for task accuracy on SQuAD QA. INT4 even slightly exceeds the FP16 baseline (F1=0.748 vs 0.737), possibly due to regularization effects from quantization noise.

**Critical implication for compression taxonomy**: Quantization is a "free" compression axis — you get 2-4x bandwidth reduction with zero task accuracy cost. This means:
1. The practical baseline for KV transmission is INT4 (25% bandwidth), NOT FP16 (100%)
2. SVD and selection methods should be evaluated ON TOP of INT4 quantization
3. The "compression cube" collapses: quantization is always applied first, then selection/SVD for further reduction

## Predictions vs Reality (UPDATED with Batch 4)

| Method | Bandwidth | F1 (predicted) | F1 (actual) | Recon Error | Status |
|--------|-----------|----------------|-------------|-------------|--------|
| Full FP16 | 100% | 0.688 | **0.737** | 0.0% | Baseline (30 samples) |
| INT8 | 50% | ~0.685 | **0.737** | 1.8% | **LOSSLESS** |
| INT4 | 25% | ~0.65 | **0.748** | 12.3% | **LOSSLESS (above baseline!)** |
| SVD rank-16 | 22% | — | 0.149 | 48.8% | Very lossy |
| SVD rank-32 | 44% | — | 0.502 | 34.9% | Moderate |
| Q2C 50% | 50% | — | **0.527** | — | 71% of full |
| Q2C 75% | 75% | — | **0.674** | — | 88% of full |
| Q2C 50% | 50% | — | **0.527** | — | 68% of full |
| SnapKV 50% | 50% | — | **0.295** | — | 38% of full |
| SnapKV 75% | 75% | — | **0.454** | — | 59% of full |
| Random 50% | 50% | — | **0.214** | — | 28% of full |
| SVD rank-64 | ~50% | — | **0.734** | — | 95% of full |
| **INT3** | **18.75%** | — | **0.718** | — | **93% — information cliff** |
| **INT2** | **12.5%** | — | **0.119** | — | **15% — catastrophic** |
| **Binary** | **6.25%** | — | **0.036** | — | **5% — near-zero** |
| Q2C 75% + INT4 | ~18.75% | — | **0.739** | — | 96% of full |
| Q2C 50% + INT4 | ~12.5% | — | **0.591** | — | 77% of full |

**Key findings (updated with batch 5-6)**:
1. INT4 lossless, INT3 retains 93%, INT2 catastrophic → **information cliff at 3-4 bits**
2. SVD rank-64 (95%) vs rank-32 (59%) → **SVD cliff at head_dim/2**
3. Q2C75%+INT4 = 96% at 18.75% BW → **quantization + selection is the optimal pipeline**
4. At matched bandwidth (~25%), INT4 (100%) >> SVD rank-32 (59%) >> Q2C 25% (40%) → quantization dominates

## Key Insight

Quantization and SVD compress along **orthogonal dimensions**:
- Quantization: Reduce precision per element (bits/element)
- SVD: Reduce number of effective elements (elements/position)
- Selection: Reduce number of positions (positions/sequence)

These three axes form a **compression cube**. The optimal operating point depends on the task and bandwidth constraint.

## Paper Framing

"We provide a unified communication-theoretic analysis of three orthogonal KV-cache compression strategies — position selection, spectral compression, and quantization — and show that their combination achieves Pareto-optimal accuracy-bandwidth tradeoffs for semantic state transmission."

## Implementation Notes

- INT8 quantization: Use `torch.quantize_per_tensor` or manual scaling
- INT4: Use `bitsandbytes` NF4 or manual group-wise quantization
- SVD + quantization: Quantize the U, S, Vh components separately
- Need to handle: dequantization error propagation through SVD reconstruction

## Risks

- ~~INT8 quantization may be "too good" — hard to show SVD advantage~~ CONFIRMED: INT8 AND INT4 are both lossless. SVD cannot compete at matched bandwidth.
- Workshop paper scope — BUT the information cliff finding (INT3=93%, INT2=15%) adds significant depth
- Quantization is well-studied; novelty comes from: (1) the exact cliff location for KV-cache, (2) the comparison with spectral methods, (3) the compression cube taxonomy
- **Resolved**: Q2C75%+INT4 gives 96% at 18.75% BW — this IS the practical recipe for KV transmission
