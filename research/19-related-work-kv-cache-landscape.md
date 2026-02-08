# Topic 19: Related Work — KV Cache Compression Landscape (2024-2026)

> **Status**: REFERENCE DOCUMENT — Literature survey for paper related work section
> **Last Updated**: 2026-02-08
> **Purpose**: Map the competitive landscape, identify differentiation points, support paper writing

---

## 1. DeepSeek MLA — Multi-Head Latent Attention

**Papers**: [DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434), [DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)

**Core Mechanism**: Training-time architectural change — low-rank joint compression of K and V into a latent vector.

| Parameter | Value |
|-----------|-------|
| Latent dimension (d_c) | 512 |
| Decoupled RoPE dim (d_h^R) | 64 |
| KV cache per token | (512 + 64) × layers |
| Reduction vs MHA | **93.3%** (57x fewer elements) |
| KV cache per token (V3) | 70 KB vs LLaMA-3.1 405B's 516 KB |

**Key Technical Details**:
- Down-projection W^DKV: d → d_c, then up-projection W^UK, W^UV: d_c → d_h × n_h
- Decoupled RoPE: RoPE applied to separate small vectors (not compressed part), avoids incompatibility
- W^UK can be absorbed into W^Q during inference (no extra compute)
- Performance **matches or exceeds MHA** while using GQA-2.25-equivalent cache

**Relationship to Our Work**:
- **Different paradigm**: MLA is training-time, we are inference-time post-hoc
- **Not a competitor**: Our methods apply to ANY existing model without retraining
- **Complementary**: MLA-compressed cache could still benefit from our Q2C selection for further reduction
- **Cite as**: Architectural approach to KV reduction; motivates the importance of the problem

---

## 2. DeepSeek Sparse Attention (DSA) — V3.2

**Paper**: [DeepSeek-V3.2 (arXiv:2512.02556)](https://arxiv.org/abs/2512.02556), [vLLM Blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)

**Core Mechanism**: Runtime sparse attention with Lightning Indexer + Top-k token selection.

| Component | Details |
|-----------|---------|
| Lightning Indexer | FP8 scorer, computes query-context relevance |
| Token selection | Top-2048 tokens per query |
| Complexity | O(L²) → O(Lk) where k ≪ L |
| Extra cache | Separate indexer K cache per token |

**Key Details**:
- Per-query dynamic selection (adapts at each decoding step)
- Does **NOT reduce KV cache memory** — reduces compute only
- Separate from MLA (orthogonal optimization)
- Fine-grained: different queries attend to different subsets

**Relationship to Our Work**:
- **Closely related mechanism**: DSA's token selection ≈ our Q2C/SnapKV/H2O
- **Different objective**: DSA reduces compute; we reduce transmission bandwidth
- **Key difference**: DSA selects per-query (dynamic), Q2C selects per-task (static set for transmission)
- **Our advantage**: We compare 4 selection strategies (Q2C, SnapKV, H2O, Random) across 6 models, 4 tasks
- **Cite as**: State-of-the-art in inference-time token selection; validates that sparse attention is effective

---

## 3. CacheGen (SIGCOMM 2024) — **Most Direct Competitor**

**Paper**: [CacheGen (SIGCOMM'24)](https://dl.acm.org/doi/10.1145/3651890.3672274), [arXiv:2310.07240](https://arxiv.org/abs/2310.07240)

**Core**: KV cache compression + network streaming for fast context loading.

### Technical Approach
1. **Delta encoding**: Split into 10-token groups, encode deltas from anchor token (2.4-2.9x lower variance)
2. **Layer-wise quantization**: Early layers get more bits, deeper layers get fewer bits
3. **Arithmetic coding**: Per-channel-layer probability distributions
4. **Adaptive bandwidth**: Per-chunk quality-bandwidth tradeoff at runtime

### Results
| Metric | Value |
|--------|-------|
| Compression ratio | 3.5-4.3x vs 8-bit quantization |
| TTFT improvement | 3.2-3.7x |
| Quality loss | < 2% accuracy (LongChat), < 0.1% F1 (TriviaQA) |
| Models tested | Llama-7B/13B/34B/70B, Mistral-7B |
| Bandwidth range | 0.4-400 Gbps |

### KEY FINDING (aligned with ours):
> "Output quality is more sensitive to losses in shallow layers than deep layers"

This matches our Layer 0 bottleneck discovery exactly.

### Differentiation — What We Have That CacheGen Doesn't

| Aspect | CacheGen | Our Work |
|--------|----------|----------|
| **Selection** | None — compresses ALL tokens | Q2C task-aware selection (proven >> H2O > SnapKV > Random) |
| **Layer diagnosis** | Uniform layer-wise bit allocation (heuristic) | Precise bottleneck identification (Layer 0 for Qwen-7B, none for Yi/Mistral) |
| **Mixed-precision** | Graded bits across layers | Binary: FP16 for bottleneck, INT4 for rest (simpler, more effective) |
| **Model characterization** | Tested on Llama/Mistral only | 6 model families, model-specific fragility analysis |
| **Task analysis** | LongChat + TriviaQA + NarrativeQA | SQuAD + TriviaQA + HotpotQA + MMLU (extractive → reasoning spectrum) |
| **Context scaling** | Not studied | Controlled needle-in-haystack 512-4096 tokens |
| **Cross-architecture** | Not studied | GQA head count hypothesis tested and refuted |
| **Communication framing** | KV cache streaming for TTFT | Agent-to-agent semantic communication protocol |
| **Delta encoding** | Core pipeline step (anchor-based, 10-token groups) | **TESTED AND REFUTED** (Batches 22+23): direct INT4 (72.2%) strictly beats CacheGen's anchor method (67.6%) and all other delta variants (12-68.5%) |

### What CacheGen Has That We Don't (Yet)
- ~~Delta encoding (adjacent token similarity)~~ — **FULLY TESTED AND REFUTED (Batches 22+23)**:
  - **Batch 22** tested sequential delta (worst case): CATASTROPHIC at INT4 (12.0% vs 72.2% direct). Variance reduces for keys (2.7-583x) but entropy INCREASES (+30% bpe). Values show <1x variance reduction in deep layers.
  - **Batch 23** tested CacheGen's ACTUAL method — anchor-based delta within 10-token groups — and all other grouped variants (grouped-sequential gs=4/10, anchor gs=4/10, mixed-precision variants). **Result: Direct INT4 (72.2%) is STRICTLY SUPERIOR to ALL delta variants at INT4:**
    - Sequential delta: 12.0%
    - Grouped-sequential gs=10: 66.0%, gs=4: 68.5%
    - Anchor gs=10 (CacheGen's method): 67.6%, gs=4: 58.6%
  - Even with mixed-precision: direct=100.8% vs anchor10=93.4% vs grp10=96.1%
  - Root causes: (1) error accumulation through cumulative reconstruction, (2) value variance NOT reduced (<1x across all delta modes), (3) deltas are more uniformly distributed despite lower variance
  - **Conclusion**: Delta encoding is counterproductive at every bit-width, every group size, and every grouping strategy. CacheGen's core compression step adds complexity while hurting quality. This comparison is now COMPLETE — no further delta experiments needed.
- Arithmetic coding (entropy-optimal)
- Adaptive bandwidth negotiation
- Real system implementation with CUDA kernels
- Latency measurements on real networks

---

## 4. PALU (ICLR 2025) — Low-Rank KV Compression

**Paper**: [PALU (ICLR'25)](https://arxiv.org/abs/2407.21118)

**Core**: Post-training low-rank decomposition of KV projection matrices.

| Metric | Value |
|--------|-------|
| Compression ratio | 7.59x (low-rank only), 11.4x (+ quantization) |
| Speedup | 1.89x (RoPE attention), 2.91x (+ quantization) |
| Models | Mistral-7B, LongChat-7B |
| vs KIVI | Similar accuracy, +30% compression from low-rank |
| Perplexity | 1.19 (vs KVQuant's higher) |

**Relationship to Our Work**:
- PALU's low-rank ≈ our SVD experiments (Topic 06)
- They use learned projection, we use post-hoc SVD
- Our SVD results: rank-64 = 95%, rank-32 = 59% (cliff at head_dim/2)
- PALU achieves better compression because they jointly optimize the projection
- **Cite as**: State-of-the-art in low-rank KV compression; we compare against SVD as a compression axis

---

## 5. xKV — Cross-Layer KV Compression

**Paper**: [xKV (OpenReview)](https://openreview.net/forum?id=CSooB1sE2m)

**Core**: Exploit layer-to-layer KV similarity for cross-layer sharing.

| Metric | Value |
|--------|-------|
| Compression ratio | Up to 8x |
| Accuracy loss | ~2% |
| Approach | Share KV cache across similar layers |

**Relationship to Our Work**:
- Related to our Topic 11 (layer-heterogeneous compression)
- xKV shares cache across layers; we use different precision per layer
- Our layer probing shows early layers most informative (consistent with xKV's finding that adjacent layers are similar)
- **Complementary**: Could combine xKV's sharing with our mixed-precision

---

## 6. MHA2MLA — Post-Training MLA Conversion

**Paper**: [MHA2MLA (ACL'25, arXiv:2502.14837)](https://arxiv.org/abs/2502.14837)

**Core**: Convert existing MHA models to MLA without full retraining.

| Metric | Value |
|--------|-------|
| KV cache reduction | 92.19% (Llama2-7B) |
| Quality loss | 0.5% on LongBench |
| Training data needed | 0.3-0.6% of original |
| Method | Partial-RoPE removal + joint SVD on existing KV weights |

**Relationship to Our Work**:
- More aggressive than our approach (92% vs our ~75% with Q2C 75% + INT4)
- But requires fine-tuning (even if minimal), we are zero-shot
- Validates that low-rank structure exists in KV cache (supports our SVD findings)

---

## 7. Other Relevant Methods

### H2O — Heavy Hitter Oracle (NeurIPS 2023)
- Our baseline method
- Cumulative attention-based eviction
- We show H2O collapses on multi-hop (43-63%) while Q2C maintains 85-91%

### SnapKV (2024)
- Our baseline method
- Recent-attention-based selection
- Matches Q2C only on HotpotQA multi-hop; loses on extractive tasks

### KIVI — 2-bit KV Cache Quantization
- Per-channel key quantization, per-token value quantization
- 2.35-3.47x throughput
- More aggressive than our INT4, but we provide model-specific analysis

### KVQuant — Activation-Aware Quantization
- Custom CUDA kernels, 1.2-1.7x throughput
- We use simpler round-to-nearest quantization but provide richer analysis

### Expected Attention (2025)
- Estimates future attention to decide eviction
- Targets reasoning models with long chain-of-thought
- More sophisticated than our methods but much higher overhead

---

## 8. KV Cache Compression Survey Taxonomy (2025)

From [arXiv:2508.06297](https://arxiv.org/abs/2508.06297):

| Category | Methods | Our Contribution |
|----------|---------|-----------------|
| **Selective** (token eviction) | H2O, SnapKV, NACL, StreamingLLM | Q2C is our novel task-aware method |
| **Quantization** | KIVI, KVQuant, QAQ | Model-specific characterization + mixed-precision recipe |
| **Attention** (architectural) | MLA, MQA, GQA | We study GQA's effect on quantization robustness |
| **Low-rank** | PALU, SVD | SVD comparison in our pipeline |
| **Hybrid** | CacheGen, GEAR | Our pipeline = Q2C selection + mixed-precision quantization |
| **Cross-layer** | xKV, SqueezeAttention | Related to our layer-heterogeneous findings |

---

## 9. Gap Analysis — Our Unique Contributions

### What NO existing paper provides:

1. **Task-aware selection for KV transmission** (Q2C) — CacheGen has no selection, DSA isn't for transmission
2. **Model-specific quantization fragility diagnosis** — 6 models, refuted KV head count hypothesis
3. **Layer 0 bottleneck with binary mixed-precision** — CacheGen does graded, we do targeted
4. **Cross-task quantization characterization** — MMLU=100%, TriviaQA=98%, SQuAD=77%, HotpotQA=63%
5. **Controlled context-length scaling** — needle-in-haystack design, monotonic INT4 degradation
6. **Agent-to-agent communication framing** — no existing KV paper frames this as semantic communication

### Where we need to strengthen:

1. ~~**Delta encoding comparison**~~ — **FULLY DONE (Batches 22+23)**: CacheGen's variance reduction CONFIRMED for keys (523-735x), but delta encoding is STRICTLY INFERIOR to direct quantization at every bit-width, group size, and strategy tested. CacheGen's actual method (anchor gs=10) achieves 67.6% vs direct INT4's 72.2%. Sequential delta is catastrophic (12.0%). With mixed-precision: direct=100.8% vs anchor=93.4%. Delta encoding is counterproductive — comparison COMPLETE.
2. **Real network latency measurements** — CacheGen has these, we don't
3. **Entropy coding** — CacheGen uses arithmetic coding on top of quantization; another compression layer
4. **System implementation** — CacheGen has CUDA kernels; we're pure Python/PyTorch
5. **Larger models** — CacheGen tests up to 70B; our max is 14B

---

## 10. Recommended Paper Positioning

### For IEEE INFOCOM/ICC (Networking + ML):

**Title direction**: "Task-Aware KV-Cache Compression for Bandwidth-Efficient Collaborative LLM Inference"

**Positioning**: CacheGen showed KV cache can be compressed for streaming (3.5-4.3x). We go further:
1. Add task-aware selection (Q2C) → additional 2-4x on top of quantization
2. Diagnose and fix model-specific fragility → practical deployment guide
3. Validate across 6 architectures, 4 tasks, 4 context lengths → comprehensive characterization

**Story**: "CacheGen compresses blindly; we compress intelligently by understanding what matters for the task."
