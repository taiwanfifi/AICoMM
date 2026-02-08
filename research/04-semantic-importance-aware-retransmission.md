# Topic 4: Semantic Importance-Aware Retransmission for KV-Cache Communication

> **Status**: Hypothesis — novel protocol-layer contribution
> **Target Venue**: IEEE ICC 2027 / IEEE Globecom 2027
> **Confidence**: Medium (interesting angle, needs formalization)

## Core Hypothesis

In KV-cache transmission over unreliable channels, not all positions are equally important. A semantic-importance-aware ARQ (Automatic Repeat reQuest) protocol that prioritizes retransmission of high-attention positions achieves better accuracy under packet loss than traditional retransmission strategies.

## Motivation

Traditional ARQ retransmits lost packets in order. But in KV-cache:
- Some positions carry critical answer information (high Q2C attention)
- Some positions are nearly redundant (low attention, high SVD residual)
- Losing a high-importance position → catastrophic accuracy drop
- Losing a low-importance position → negligible impact

**Key insight**: We can use Q2C attention scores as a "semantic importance map" to guide retransmission priority.

## Protocol Design

```
Sender                          Receiver
  |                                |
  |--- KV-cache packets (ranked) →|
  |    [P1: rank-1 SVD of top Q2C positions]
  |    [P2: rank-1 SVD of mid Q2C positions]
  |    [P3: rank-2 refinement of top positions]
  |    ...                         |
  |                                |
  |←-- NACK for lost packets ------|
  |                                |
  |--- Retransmit by importance →  |
  |    (top-Q2C first, not FIFO)   |
```

## Experimental Plan

### Phase 1: Importance Analysis
1. Compute Q2C attention scores for each context position
2. Simulate dropping positions: measure F1 vs % positions lost
3. Compare: random drop vs drop-lowest-attention vs drop-highest-attention
4. **Expected**: Dropping low-attention positions has minimal F1 impact

### Phase 2: Packet Loss Simulation
1. Divide KV-cache into packets (e.g., 16 positions per packet)
2. Simulate i.i.d. packet loss at rates: 1%, 5%, 10%, 20%, 30%
3. Compare recovery strategies:
   - **No retransmission**: Use what arrived
   - **FIFO retransmission**: Retransmit in order
   - **Importance-first**: Retransmit highest-attention packets first
   - **SVD-first**: Retransmit most energetic SVD components first
4. Metric: F1 at fixed latency budget (limited retransmission rounds)

### Phase 3: Unequal Error Protection
1. Assign different FEC (Forward Error Correction) levels based on importance
2. High-importance positions: more redundancy (lower loss probability)
3. Low-importance positions: less redundancy (accept some loss)
4. Compare with uniform FEC at same total overhead

## Mathematical Formulation

### Importance-Weighted Loss
```
L_semantic = Σ_i α_i × 1[position i lost]
where α_i = softmax(attention_score_i) = semantic importance of position i
```

### Optimal Retransmission Order
```
Given: Lost positions L = {l_1, ..., l_k}
       Retransmission budget: B packets
Solve: max_{S ⊆ L, |S|≤B} Σ_{i ∈ S} α_i
→ Simply sort by α_i and retransmit top-B
```

### Unequal Error Protection
```
Given: Total FEC budget F
       N positions with importance α_1 ≥ ... ≥ α_N
Allocate: FEC_i ∝ α_i such that Σ FEC_i = F
→ Water-filling solution
```

## Relation to Our Work

- Q2C attention scores from Exp04 directly provide the importance ranking
- SVD energy distribution from Exp02/06 provides spectral importance
- This paper "completes the protocol" with reliability mechanisms

## Novelty

- Traditional semantic communication reliability uses JSCC — we use explicit importance ranking
- No prior work combines attention-based importance with ARQ for LLM KV-cache
- Bridges ML (attention scores) with networking (retransmission protocols)

## Risks

- Simulated channel model may be too simple for comm venues
- Real packet loss is bursty, not i.i.d.
- Overhead of importance metadata may negate gains
