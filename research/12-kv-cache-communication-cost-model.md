# Topic 12: Communication Cost Model for KV-Cache Sharing in 6G Agent Networks

> **Status**: Hypothesis — networking-focused paper
> **Target Venue**: IEEE INFOCOM 2027 / IEEE TWC / IEEE JSAC
> **Confidence**: Medium-High (fills the "protocol gap" in our research)

## Core Hypothesis

We can build a comprehensive cost model for KV-cache communication that accounts for computation, transmission, and reconstruction costs, enabling optimal decision-making: when to share KV-cache vs. retransmit text vs. recompute locally.

## The Decision Problem

When Agent A has context and Agent B needs it:

| Strategy | Compute Cost (A) | Bandwidth Cost | Compute Cost (B) | Latency |
|----------|-----------------|----------------|-------------------|---------|
| Text retransmission | 0 | C_text | C_prefill | High |
| Full KV share | C_prefill | C_kv_full | 0 | Medium |
| Compressed KV | C_prefill + C_compress | C_kv_compressed | C_decompress | Low |
| Partial KV + text | C_prefill + C_select | C_partial + C_text_remainder | C_partial_prefill | Variable |

**When is each optimal?** Depends on:
- Available bandwidth B
- Agent B's compute capability
- Latency requirement T_max
- Accuracy requirement F1_min

## Formal Cost Model

### Total Cost
```
C_total = α × C_compute + β × C_bandwidth + γ × C_latency
where α, β, γ are system-specific weights
```

### Computation Cost
```
C_prefill(n) = 2 × n × d² × L  (FLOPs for n-token prefill, d = hidden dim, L = layers)
C_compress(n, r) = O(n × d × r × L)  (SVD compression)
C_decompress(n, r) = O(n × r × L)  (SVD reconstruction)
```

### Bandwidth Cost
```
C_text = n × bytes_per_token  (typically 2-4 bytes with tokenization)
C_kv_full = 2 × L × H_kv × n × d_head × 2  (FP16 KV-cache)
C_kv_svd(r) = 2 × L × H_kv × (n×r + r + r×d_head) × 2
C_kv_select(p) = 2 × L × H_kv × (p×n) × d_head × 2
```

### Latency Model
```
T_total = T_compute + T_transmit + T_reconstruct
T_transmit = Size / Bandwidth
T_compute = FLOPs / ThroughputGPU
```

## Experimental Plan

### Phase 1: Cost Measurement (1 day)
1. Measure actual computation time for prefill at n = {256, 512, 1024, 2048, 4096}
2. Measure SVD compression time for ranks = {4, 8, 16, 32, 64}
3. Measure KV-cache sizes in bytes for all configurations
4. Build lookup tables: T_compute(n, model), Size(n, r), etc.

### Phase 2: Decision Boundary Analysis (2 days)
1. For given (B, T_max, F1_min), compute optimal strategy
2. Plot decision boundaries in (bandwidth, latency_budget) space
3. Identify regimes:
   - High bandwidth, low latency → full KV
   - Low bandwidth, high latency → compressed KV
   - Very low bandwidth → text retransmission
   - High compute at B → recompute locally

### Phase 3: Network Simulation (3 days)
1. Simulate 6G network with variable bandwidth (10 Mbps - 10 Gbps)
2. Multiple agent pairs communicating simultaneously
3. Dynamic strategy selection based on current network state
4. Compare: static policy vs adaptive policy vs oracle
5. Use ns-3 or custom Python simulator

## Key Results to Show

### Figure 1: Decision Boundary Map
```
Bandwidth (Mbps)
  10000 ┤████████████████████
        │████ Full KV ████████
   1000 ┤████████████████████
        │██ Compressed KV ████
    100 ┤████████████████████
        │█ Text Retransmit ███
     10 ┤████████████████████
        │ Local Recompute ████
      1 ┤████████████████████
        └─────────────────────
         256  512  1024  2048  4096
              Context Length (tokens)
```

### Figure 2: Accuracy-Cost Pareto
Different strategies at different operating points.

### Table 1: Strategy Selection Rules
Practical guidelines for system designers.

## Paper Contribution

1. First comprehensive cost model for KV-cache communication
2. Closed-form decision boundaries for strategy selection
3. Adaptive protocol that selects optimal strategy based on network conditions
4. Validation through network simulation with realistic 6G parameters

## Connection to Advisor's Vision

This IS the "protocol design paper" — it answers:
- WHEN to use KV-cache communication (vs alternatives)
- HOW MUCH to compress (optimization)
- HOW to adapt to channel conditions (protocol)

## Risks

- Cost model parameters are hardware-specific → generalization concerns
- Network simulation may be too simple for comm reviewers
- Need realistic 6G channel models (mmWave, THz)
