# Topic 10: Privacy-Preserving KV-Cache Sharing via Differential Privacy and Federated Compression

> **Status**: Hypothesis — important practical concern
> **Target Venue**: IEEE S&P Workshop / USENIX Security 2027 / CCS Workshop
> **Confidence**: Medium (privacy + ML is hot, but KV-cache privacy is unexplored)

## Core Hypothesis

KV-cache contains recoverable information about the original text. Sharing KV-cache between agents without privacy protection leaks sensitive content. We can design differentially private KV-cache compression that preserves task utility while preventing content reconstruction.

## Why This Matters

If Agent A sends KV-cache to Agent B:
- B can potentially reconstruct A's original text (or close approximation)
- In healthcare/legal/finance, this is a privacy violation
- Even compressed KV still contains semantic information

**The question**: Can we add noise/distortion to KV-cache such that:
1. Task accuracy is preserved (B can still answer questions)
2. Content privacy is protected (B cannot reconstruct original text)

## Privacy Attack Model

### Attack: KV-Cache Inversion
Given KV-cache K, attacker tries to recover original text X:
1. **Embedding inversion**: Project KV vectors back to vocabulary space
2. **Generation attack**: Use K as prefix, generate text that "extends" from K
3. **Probing attack**: Train classifier on K to extract specific attributes (names, dates, etc.)

### Defense: DP-KV Compression
Add calibrated noise to KV-cache before transmission:
```
K_private = f(K) + Laplace(0, Δf/ε)
where:
  f(K) = SVD compression (already removes information)
  Δf = sensitivity of f
  ε = privacy budget
```

## Experimental Plan

### Phase 1: Privacy Attack Baseline (2 days)
1. Given a KV-cache, try to recover the original text
2. Method: Use the model to generate from KV-cache prefix
3. Measure: BLEU/ROUGE between generated and original text
4. Try at different compression levels: Does SVD already help privacy?

### Phase 2: DP-KV Compression (3 days)
1. Add Gaussian noise to SVD components: `S_private = S + N(0, σ²)`
2. Add noise to selection scores: privatize which positions are selected
3. Measure utility-privacy tradeoff: F1(ε) curve
4. Compare: Noise on raw KV vs noise on SVD components vs noise on selection

### Phase 3: Federated KV Aggregation (1 week)
1. N agents each have private documents
2. Want to create a shared "knowledge base" KV-cache
3. Federated aggregation: Average KV-caches with DP noise
4. Measure: Aggregated KV utility vs individual privacy

## Key Insight

SVD compression ALREADY provides some privacy protection:
- Rank-r approximation discards fine-grained details
- Lower rank → more privacy but less utility
- This is like adding a "structural noise" that removes high-frequency information

**Hypothesis**: SVD rank serves as a natural privacy knob — we can characterize the privacy-utility tradeoff purely through the SVD rank parameter.

## Formal Framework

### Privacy Definition
(ε, δ)-differential privacy for KV-cache mechanism M:
```
P[M(K₁) ∈ S] ≤ e^ε × P[M(K₂) ∈ S] + δ
for all neighboring KV-caches K₁, K₂ (differing by one input token)
```

### Utility Definition
```
U(M) = E[F1(M(K), Q)] — expected task accuracy under mechanism M
```

### Tradeoff
```
max U(M) s.t. M satisfies (ε, δ)-DP
```

## Paper Angle

"We identify a novel privacy risk in KV-cache sharing for collaborative LLM inference and propose DP-KV, a differentially private compression mechanism that provides formal privacy guarantees while maintaining practical task utility. We show that SVD compression acts as a natural privacy mechanism, and characterize the privacy-utility-bandwidth three-way tradeoff."

## Risks

- Privacy definitions for KV-cache may be non-standard
- DP noise may destroy too much utility
- Privacy attacks may be weak (hard to show meaningful risk)
- Competitive space (DP + ML is crowded)
