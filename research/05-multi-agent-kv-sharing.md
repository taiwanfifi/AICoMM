# Topic 5: Multi-Agent KV-Cache Sharing for Collaborative Document Understanding

> **Status**: Hypothesis — ambitious, high-novelty direction
> **Target Venue**: ACL 2027 / EMNLP 2027 / NeurIPS 2027
> **Confidence**: Medium (novel scenario, feasibility unclear)

## Core Hypothesis

When multiple LLM agents need to understand the same document but answer different questions, sharing a common compressed KV-cache "base" with task-specific "refinement layers" is more efficient than each agent encoding the document independently.

## Scenario

```
        [Long Document]
             |
      [Agent A: Encoder]
      Computes full KV-cache
             |
    ┌────────┼────────┐
    ↓        ↓        ↓
[Agent B] [Agent C] [Agent D]
Q: "Who?" Q: "When?" Q: "Why?"
```

Instead of A, B, C, D each processing the document:
- Agent A processes once → produces KV-cache
- Transmit compressed KV to B, C, D
- Each agent adds task-specific attention refinement
- **Savings**: 4x compute reduction (only 1 prefill instead of 4)

## Why This Matters

Real-world multi-agent scenarios:
1. **Legal discovery**: Multiple lawyers analyze same case file, different aspects
2. **Medical consult**: Multiple specialists read same patient record
3. **Intelligence analysis**: Multiple analysts process same report
4. **Customer service**: Multiple agents handle different aspects of same ticket

## Experimental Design

### Phase 1: Shared Base Evaluation
1. One document, N different questions
2. Agent A computes full KV-cache on document
3. Agents B-D receive compressed KV and answer their questions
4. Compare: each agent's accuracy vs independent processing
5. Measure: total compute, total bandwidth, accuracy per agent

### Phase 2: Task-Specific Refinement
1. After receiving base KV, each agent runs Q2C with their own question
2. This "refines" which positions are important for THEIR task
3. Compare strategies:
   - **Shared base only**: Same compressed KV for all
   - **Shared base + task refinement**: Q2C re-ranking per agent
   - **Independent**: Each agent processes from scratch

### Phase 3: Incremental Updates
1. Document changes (new paragraph added)
2. Agent A computes incremental KV update (delta)
3. Transmit only the delta to all other agents
4. Compare: delta update vs full re-encode

## Key Technical Challenges

1. **Different questions need different context positions**: Q2C scores differ per question
2. **Base compression trades off between tasks**: Can't optimize for all questions simultaneously
3. **Rank allocation**: Which layers/positions to include in "universal base"?

## Proposed Solution: Universal Base + Task-Specific Mask

```
Universal Base = Top-k positions by AVERAGE attention across diverse question templates
Task Mask = Q2C re-ranking specific to each agent's question
Final = Base KV × Task Mask
```

### Optimization
```
max_{S_base}  Σ_q∈Q  F1(S_base ∪ S_task(q), q)
s.t.  |S_base| ≤ B_shared
      |S_task(q)| ≤ B_task  for all q
```

## Metrics

| Metric | What it measures |
|--------|-----------------|
| Per-agent F1 | Quality of shared vs independent |
| Total compute (FLOPs) | Efficiency of sharing |
| Total bandwidth | Communication cost |
| Marginal cost per new agent | Scalability |
| Time-to-answer (wall clock) | Practical latency |

## Dataset Requirements

Need questions that share context but differ in focus:
- SQuAD: Multiple questions per passage (natural fit!)
- Natural Questions: Different aspects of same Wikipedia article
- Custom: Generate diverse questions per document using GPT-4

## Paper Angle

"We introduce multi-agent KV-cache sharing, where a single agent's document encoding is compressed and distributed to multiple agents for diverse downstream tasks, achieving near-independent accuracy at a fraction of the compute and communication cost."

## Risks

- Shared base may be too generic → accuracy loss
- Multi-agent coordination overhead may negate savings
- Need to demonstrate practical scenarios convincingly
