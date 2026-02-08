# Topic 14: Knowledge Distillation via KV-Cache Transfer — Teaching Small Models with Large Model Representations

> **Status**: Hypothesis — high novelty, connects to Topic 2
> **Target Venue**: NeurIPS 2027 / ICML 2027
> **Confidence**: Medium (very novel, but feasibility uncertain)

## Core Hypothesis

A large model's KV-cache, when projected into a small model's representation space, can serve as an efficient "teaching signal" that improves the small model's task performance WITHOUT fine-tuning — a form of inference-time knowledge distillation.

## The Idea

Traditional knowledge distillation (KD):
```
Large model → soft labels → Train small model on soft labels
```

KV-cache distillation (ours):
```
Large model → KV-cache → Project to small model space → Small model uses projected KV
```

**Key difference**: No training required. The small model directly uses the large model's "processed understanding" of the context at inference time.

## Scenario

```
Cloud (70B model):
  - Processes document once
  - Extracts KV-cache
  - Projects KV to 3B format
  - Sends to edge

Edge (3B model):
  - Receives projected KV
  - Answers questions as if it were a 70B model
  - No fine-tuning needed
```

## Why This Could Work

1. KV-cache captures contextual understanding — larger models understand better
2. The small model's decoder is still effective — it just needs better context representation
3. This is like giving a student the professor's lecture notes instead of making them read the textbook

## Experimental Plan

### Phase 1: Same-Family Transfer (2 days)
1. Qwen2.5-7B processes SQuAD context → KV-cache_7B
2. Linear projection: KV-cache_7B → KV-cache_3B_format
3. Inject into Qwen2.5-3B → measure F1
4. Compare: F1 with projected 7B KV vs 3B's own KV vs 7B's own performance

### Phase 2: Cross-Family Transfer (3 days)
1. Qwen2.5-7B → project → Llama-3.2-3B format
2. Different architecture, different tokenization
3. Need: shared vocabulary mapping + KV projection
4. Much harder but much more interesting if it works

### Phase 3: Scaling Analysis
1. How does transfer quality scale with model size ratio?
   - 3B → 3B (same size) = baseline
   - 7B → 3B (2.3x)
   - 14B → 3B (4.7x)
   - 70B → 3B (23x) = can a tiny model benefit from a huge model's KV?
2. Is there a "transfer ceiling" — beyond which bigger source doesn't help?

## Connection to Communication

This directly instantiates the advisor's vision:
- Cloud model = powerful agent with deep understanding
- Edge model = lightweight agent needing guidance
- KV-cache transfer = semantic state synchronization between agents
- Projection = cross-model protocol adaptation

## Metrics

| Metric | What it measures |
|--------|-----------------|
| F1 improvement | How much does projected KV help vs own KV |
| Projection fidelity | CKA/CCA between projected and true KV |
| Compute savings | FLOPs saved by not running large model locally |
| Quality retention | % of large model's accuracy preserved in small model |

## Paper Angle

"We introduce KV-Cache Distillation, a training-free inference-time knowledge transfer mechanism where a large model's KV-cache is projected into a small model's representation space, enabling the small model to answer questions with near-large-model accuracy at a fraction of the cost."

## Risks

- Projection may fail entirely (KV spaces too different)
- Even if CKA is high, generation quality may degrade
- Tokenization differences between models create alignment issues
- Conceptually similar to speculative decoding (need clear differentiation)
