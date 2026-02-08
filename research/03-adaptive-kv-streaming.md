# Topic 3: Adaptive KV-Cache Streaming Protocol with Channel-Aware Rank Selection

> **Status**: Hypothesis — extends Topic 1
> **Target Venue**: IEEE INFOCOM 2027 / IEEE TWC
> **Confidence**: Medium-High (natural extension, but needs simulation)

## Core Hypothesis

A semantic transport protocol that dynamically adjusts KV-cache compression granularity (SVD rank, selection ratio) based on real-time channel conditions achieves near-optimal accuracy-latency tradeoffs across varying network conditions, outperforming static compression policies.

## Why This Matters

In real networks, bandwidth fluctuates. A fixed compression ratio is suboptimal:
- Good channel: should send more KV-cache (higher accuracy)
- Bad channel: should aggressively compress (maintain latency SLA)
- Interruption: should have meaningful partial results at any point

## Protocol Design

### Control Plane
```
1. Task Negotiation: Sender/Receiver agree on task type
2. Model Alignment: Verify compatible KV-cache formats
3. Channel Estimation: Receiver reports estimated bandwidth
4. Rank Selection: Sender computes optimal SVD rank for given bandwidth
5. Progressive Transmission: Send SVD components in decreasing importance
```

### Data Plane — Progressive KV Transmission
```
Step 1: Send rank-1 approximation (U_1, S_1, Vh_1) — coarsest sketch
Step 2: Send rank-2 residual — refine
Step 3: Send rank-4 residual — further refine
...
Step K: Send full resolution — lossless
```

Key insight: SVD naturally supports **progressive refinement**. The receiver can start generation at ANY point during transmission and improve as more data arrives.

## Experimental Plan

### Phase 1: Progressive SVD Evaluation
1. For each SQuAD sample, compute F1 at ranks 1, 2, 4, 8, 16, 32, 64
2. Plot: F1 curve over cumulative bandwidth → "progressive quality curve"
3. Compare with: sending random positions progressively

### Phase 2: Channel Simulation
1. Model channel as time-varying bandwidth: B(t) ~ Markov chain
2. States: {very_low, low, medium, high, very_high} → maps to SVD ranks
3. Compare policies:
   - **Static**: Fixed rank, may miss deadline or waste bandwidth
   - **Adaptive**: Select rank based on B(t) estimation
   - **Progressive**: Always send, stop when deadline reached
4. Metric: Average F1 under latency SLA constraints

### Phase 3: Multi-Agent Scenario
1. N agents share a base station with shared bandwidth
2. Each agent pair wants to transmit KV-cache
3. Bandwidth allocation: Who gets how much? → Optimization problem
4. Compare: Equal split vs priority-based (task urgency) vs proportional fair

## Key Innovation

**This is the "protocol paper" the advisor wants.** It's not just compression — it's:
- Channel-aware adaptation
- Progressive transmission with anytime utility
- Multi-user resource allocation
- Task-priority scheduling

## Formulation

### Single-Link Optimization
```
max_r  F1(r)
s.t.   Size(SVD_rank_r) ≤ B(t) × T_deadline
       T_encode(r) + T_transmit(r) + T_decode(r) ≤ T_deadline
```

### Multi-Agent Resource Allocation
```
max_{r_1,...,r_N}  Σ_i w_i × F1_i(r_i)
s.t.  Σ_i Size(SVD_rank_{r_i}) ≤ B_total × T
      r_i ∈ {1, 2, 4, 8, 16, 32, 64}
```

This is a variant of the knapsack problem → solvable with DP or greedy.

## Relation to Existing Results

We already have the F1-vs-rank curve from Exp06! This is the "progressive quality curve" needed for the protocol simulation. The protocol paper builds directly on Topic 1's compression results.

## Risks

- Channel simulation may be seen as simplistic by comm reviewers
- Need to show advantage over simple "just retransmit text" baseline
- Multi-agent scenario may be too ambitious for first paper
