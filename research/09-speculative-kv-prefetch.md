# Topic 9: Speculative KV-Cache Prefetching for Predictive Semantic Communication

> **Status**: Hypothesis — novel system-level contribution
> **Target Venue**: IEEE INFOCOM 2027 / MobiCom 2027
> **Confidence**: Medium (interesting system, but complex to implement)

## Core Hypothesis

In multi-turn agent interactions, the receiver can predict what context the sender will process next (based on conversation history and task structure) and speculatively request KV-cache prefetching. This reduces perceived latency by overlapping computation with communication.

## Scenario

```
Turn 1: User asks about Chapter 1 → Agent A reads Ch.1, sends KV to Agent B
Turn 2: User likely asks about Chapter 2 → Agent A SPECULATIVELY computes Ch.2 KV
        While B answers Turn 1, A pre-computes and pre-sends Ch.2 KV
Turn 3: User indeed asks about Ch.2 → B already has the KV! Near-zero latency
```

## Why This Matters

In sequential document analysis, reading patterns are predictable:
- Documents are read sequentially (Ch.1 → Ch.2 → Ch.3)
- Follow-up questions are topically related
- Agent workflows follow known patterns (read → analyze → decide)

Speculative prefetching exploits this predictability to hide communication latency.

## Technical Design

### Prediction Model
1. **Sequential predictor**: Next chunk = current chunk + 1 (trivial but effective)
2. **Topic predictor**: Predict next query topic from conversation history
3. **Task graph predictor**: Follow known workflow DAG

### Prefetch Policy
```
p_prefetch = P(next_query involves chunk_i | history)
if p_prefetch > threshold:
    prefetch KV-cache for chunk_i at low priority
```

### Resource Management
- Prefetched KV-cache stored in receiver's "KV buffer"
- Buffer has limited capacity → eviction policy needed
- Correct predictions: instant access (cache hit)
- Wrong predictions: wasted bandwidth + compute (cache miss)

## Experimental Plan

### Phase 1: Predictability Analysis
1. Analyze SQuAD: Given question N, how predictable is question N+1?
2. Analyze multi-turn QA datasets (CoQA, QuAC)
3. Measure: position overlap between consecutive questions' KV importance
4. Compute: hit rate of simple prefetch policies

### Phase 2: Prefetch Simulation
1. Simulate multi-turn QA with 10-turn conversations
2. Agent A has the document; Agent B asks questions
3. Compare policies:
   - **No prefetch**: Compute on demand → full latency
   - **Sequential prefetch**: Always prefetch next chunk → amortized latency
   - **Adaptive prefetch**: Predict + prefetch based on confidence
4. Metrics: Average latency, bandwidth waste, hit rate, F1

### Phase 3: KV-Cache Incremental Update
1. If successive queries share overlapping context, transmit only the delta
2. KV-cache delta = new positions + updated attention weights
3. Measure: How much bandwidth does incremental update save?

## Metrics

| Metric | Definition |
|--------|-----------|
| Prefetch hit rate | % of queries where prefetched KV was useful |
| Latency reduction | Time saved vs on-demand computation |
| Bandwidth overhead | Extra bytes transmitted for wrong predictions |
| Net efficiency | (Latency saved × hit rate) / bandwidth overhead |

## Connection to Networking

This directly maps to:
- **TCP prefetching / predictive caching** (CDN world)
- **Speculative execution** (CPU architecture → transferred to distributed LLM)
- **Proactive resource allocation** (5G/6G network slicing)

## Paper Angle

"We introduce speculative KV-cache prefetching for multi-turn LLM collaboration, showing that conversational predictability enables the receiver to pre-warm its KV-cache, reducing time-to-first-token by up to Nx in sequential document analysis tasks."

## Risks

- Predictability may be task-specific — hard to generalize
- Buffer management adds complexity
- Wasted bandwidth on wrong predictions
- Need convincing multi-turn scenarios
