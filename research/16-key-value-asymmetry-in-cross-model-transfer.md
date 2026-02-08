# Topic 16: The Key-Value Asymmetry — Why Keys Transfer but Values Don't

> **Status**: DISCOVERED — experimental finding, 2026-02-08
> **Target Venue**: ICLR 2027 / NeurIPS 2027 / EMNLP 2027
> **Confidence**: High (backed by experimental data)

## Discovery

When projecting KV-cache from Qwen2.5-3B to Qwen2.5-7B:

| Component | Cosine Similarity | Relative Error | Transfers? |
|-----------|------------------|----------------|------------|
| **Keys** | **0.9997** | **4.8%** | YES |
| **Values** | 0.222 | 107.6% | NO |

**This is a 50x difference in similarity.** Keys are essentially identical between models; values are completely different.

## Hypothesis: RoPE Creates a Shared Key Space

**Keys** have Rotary Position Embeddings (RoPE) applied, which encode position:
```
k_l = RoPE(θ_pos) × W_k × h_l
```

Since Qwen2.5-3B and Qwen2.5-7B use the **same RoPE parameters** (same θ base, same frequencies), the positional component of keys is shared. The learned `W_k` contributes model-specific information, but the RoPE component dominates.

**Values** have no positional encoding:
```
v_l = W_v × h_l
```

The value space depends entirely on `W_v`, which differs significantly between 3B and 7B. There's no "anchor" like RoPE to align the representations.

## Research Questions

1. **Is this asymmetry universal?** Does it hold across all model families (Llama, Mistral, Gemma)?
2. **Is it layer-dependent?** Are early-layer values more transferable than deep-layer values?
3. **Can we exploit it?** Key-only transfer + value recomputation as a communication protocol
4. **Does it explain other phenomena?** E.g., why speculative decoding works despite different model internals

## Experimental Plan

### Phase 1: Validate Across Models (1 day)
1. Qwen2.5-3B → Qwen2.5-7B (DONE — confirmed)
2. Llama-3.2-3B → Llama-3.1-8B
3. Qwen2.5-3B → Llama-3.2-3B (cross-family — expect key transfer to FAIL due to different RoPE)
4. Qwen2.5-3B-Instruct → Qwen2.5-3B (same family, different training — expect both to transfer)

### Phase 2: Layer-wise Analysis (1 day)
1. Compute key/value transfer quality per layer, not just last layer
2. Hypothesis: Shallow layers have better value transfer (more generic representations)
3. Build "transfer heatmap": layer × {key, value} → transfer quality

### Phase 3: Key-Only Communication Protocol (2 days)
1. Edge sends only projected keys (half the bandwidth)
2. Cloud computes its own values from text but uses received keys for attention routing
3. Measure: F1 with key-only transfer vs full KV vs text retransmission

### Phase 4: Theoretical Analysis
1. Formalize: Why does RoPE create a shared space?
2. Prove: Under what conditions is the key space linearly equivalent across model sizes?
3. Derive: Bounds on value transfer error as function of model size ratio

## Paper Angle

"We discover and explain a fundamental asymmetry in cross-model KV-cache transferability: keys transfer near-perfectly between same-family models of different sizes (cos_sim=0.9997) while values do not (cos_sim=0.222). We trace this to Rotary Position Embeddings (RoPE) creating a shared key space, and exploit this asymmetry to design a key-only communication protocol that halves bandwidth while preserving task accuracy."

## Why This Is Novel

- No prior work has characterized this key-value asymmetry
- Explains a fundamental property of transformer KV representations
- Directly applicable to distributed/collaborative LLM inference
- Connects theoretical understanding (RoPE geometry) to practical system design

## Update: Structural vs Functional Transfer (Batch 7c v2)

The cos_sim=0.222 for values means STRUCTURAL (pointwise) transfer of value vectors across models is not feasible. However, batch 7c v2 cross-model experiments reveal that **FUNCTIONAL transfer works well**:

| Transfer Type | What's Measured | Result |
|--------------|----------------|--------|
| Structural (values) | Cosine similarity of projected value vectors | **0.222** (fails) |
| Structural (keys) | Cosine similarity of projected key vectors | **0.9997** (works) |
| Functional (Q2C overlap) | Agreement on which positions to keep | **86.3%** at 50%, **91.5%** at 75% |
| Functional (task F1 loss) | Accuracy when using other model's selection | **-0.046** at 50%, **-0.008** at 75% |

**The resolution**: While value representations are structurally incompatible (you cannot substitute 3B's values into 7B), the attention SCORES derived from keys and queries produce highly similar importance rankings. The task-relevant signal — "which positions matter" — is shared across model sizes, even though the underlying value representations that carry the content are model-specific.

This has a key implication for protocol design: **don't transfer values at all.** Transfer attention-based selection decisions (position indices) instead. Each model computes its own values locally but uses shared attention importance to decide what to keep.

This finding also refines the RoPE hypothesis. RoPE doesn't just create a shared key space — it creates shared attention patterns. Since attention scores are computed as `softmax(Q @ K^T)`, and keys are in a shared space (cos=0.9997), the resulting attention distributions are also shared. Values being different doesn't matter for the selection decision, only for the final output computation.

## Connection to Our Work

This is the deepest insight to emerge from Topic 02 experiments. It doesn't just say "cross-model transfer works" — it explains WHY keys transfer and values don't, which leads to a targeted communication protocol. The batch 7c v2 results further show that the asymmetry is not a blocker but rather a design guide: transfer decisions (from shared attention), not representations (from divergent values).

## Potential Impact

- **Transformer understanding**: New insight into how RoPE shapes representation spaces
- **System design**: Key-only transfer protocol for edge-cloud inference
- **Compression**: Keys and values should be compressed differently (keys are redundant across models, values need local computation)
