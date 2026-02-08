# Topic 7: How Task Type Shapes Attention: A Systematic Analysis of KV-Cache Compressibility

> **Status**: Hypothesis — data-driven paper
> **Target Venue**: EMNLP 2027 / ACL Findings 2027
> **Confidence**: High (empirical study, many insights possible)

## Core Hypothesis

Different NLP tasks (QA, summarization, translation, reasoning) produce fundamentally different attention patterns, leading to different optimal KV-cache compression strategies. A task-type classifier can predict the best compression method without running all methods.

## Motivation

Our Exp04 shows Q2C works well for extractive QA. But would it work for:
- **Summarization**: No single "answer span" — diffuse attention needed
- **Translation**: Monotonic alignment — sequential positions important
- **Multi-hop reasoning**: Multiple evidence pieces — scattered positions
- **Dialogue**: Recent context more important — recency bias

If attention patterns differ systematically by task, we need task-aware protocol negotiation.

## Experimental Plan

### Phase 1: Task-Specific Attention Analysis (2 days)
1. **Extractive QA** (SQuAD): Where does the model attend?
2. **Abstractive Summarization** (CNN/DailyMail): Attention distribution
3. **Multi-hop QA** (HotpotQA): Multiple evidence attention
4. **Dialogue** (PersonaChat or MultiWOZ): Recency vs long-range
5. For each: compute attention entropy, attention sparsity, top-k coverage

### Phase 2: Compression Method × Task Matrix (3 days)
For each (task, method) combination:

| Task \ Method | Random | SnapKV | Q2C | SVD | Q2C+SVD |
|---------------|--------|--------|-----|-----|---------|
| Extractive QA | | | | | |
| Summarization | | | | | |
| Multi-hop QA | | | | | |
| Dialogue | | | | | |

Measure F1/ROUGE/BLEU at 30% and 50% retention.

### Phase 3: Task-Aware Method Selection
1. Extract attention features: entropy, sparsity, effective rank, top-k coverage
2. Train a simple classifier: features → best compression method
3. Evaluate: Does predicted method match oracle-best method?
4. Protocol implication: Receiver tells sender task type → sender picks method

## Expected Findings

1. **QA**: Sparse attention (few key positions) → Selection methods win
2. **Summarization**: Dense attention → SVD wins (preserves global structure)
3. **Multi-hop**: Multi-modal attention → Hybrid wins
4. **Dialogue**: Recency-biased → Sliding window + SVD

## Key Metrics

- Attention entropy per layer per task
- Position importance distribution (Gini coefficient)
- Optimal compression method per task
- Accuracy of task-type → method predictor
- Protocol overhead of task negotiation

## Paper Contribution

1. First systematic study of how task type affects KV-cache compressibility
2. Task-type aware compression selection algorithm
3. Protocol design implication: Control plane should include task negotiation
4. Practical guidelines: "For QA, use Q2C at 50%; for summarization, use SVD rank-32"

## Datasets Needed

| Task | Dataset | Metric | Samples |
|------|---------|--------|---------|
| Extractive QA | SQuAD v2 | F1 | 100 |
| Abstractive Summarization | CNN/DailyMail | ROUGE-L | 100 |
| Multi-hop QA | HotpotQA | F1 | 100 |
| Dialogue | MultiWOZ | BLEU / Success | 100 |
| Translation | WMT | BLEU | 100 |

## Risks

- Empirical study without strong theoretical contribution
- Task boundaries may be fuzzy (QA vs reading comprehension vs reasoning)
- Need large enough model to show meaningful differences
