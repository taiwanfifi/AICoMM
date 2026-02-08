# Topic 15: KV-Cache as External Memory for Continual Learning in Agent Networks

> **Status**: Speculative — long-term research direction
> **Target Venue**: AAAI 2027 / AAMAS 2027
> **Confidence**: Low-Medium (ambitious, conceptual stage)

## Core Hypothesis

Compressed KV-cache fragments can serve as an efficient external memory system for LLM agents, where past processed contexts are stored as compressed KV representations and retrieved for future tasks, enabling continual learning without weight updates.

## The Idea

```
Time T1: Agent processes Document A → Compress KV_A → Store in KV Memory Bank
Time T2: Agent processes Document B → Compress KV_B → Store in KV Memory Bank
Time T3: New question about A+B → Retrieve KV_A, KV_B → Combine → Answer
```

This is like RAG but at the KV-cache level instead of the text level:
- **RAG**: Retrieve text chunks → re-encode → answer
- **KV-RAG**: Retrieve compressed KV → inject directly → answer (skip re-encoding!)

## Why This Matters

1. **Latency**: KV retrieval skips the expensive prefill phase
2. **Quality**: KV representations are richer than text embeddings
3. **Scalability**: Compressed KV is smaller than storing full model states
4. **Multi-agent**: Agents can share KV memory banks

## Technical Design

### KV Memory Bank
```
Memory = {
  (key_embedding_1, compressed_kv_1, metadata_1),
  (key_embedding_2, compressed_kv_2, metadata_2),
  ...
}
```

### Retrieval
```
Given query Q:
1. Compute query embedding: e_q = embed(Q)
2. Retrieve top-k: {kv_i : similarity(e_q, key_i) > threshold}
3. Decompress: kv_full = SVD_reconstruct(kv_compressed)
4. Concatenate: kv_combined = concat(kv_1, ..., kv_k)
5. Generate answer from kv_combined
```

### Memory Management
- **Insertion**: New KV entries compressed and indexed
- **Eviction**: LRU or importance-based eviction when memory full
- **Merging**: Similar KV entries merged to save space
- **Update**: KV entries refreshed when source document changes

## Experimental Plan

### Phase 1: KV Retrieval Accuracy (2 days)
1. Process 100 SQuAD passages → store 100 compressed KV entries
2. Given a question, retrieve relevant KV entry
3. Inject KV → answer question
4. Compare: KV-RAG vs text-RAG vs full re-encode

### Phase 2: Multi-Document KV Fusion (3 days)
1. Question requires information from multiple passages
2. Retrieve and combine multiple KV entries
3. Challenge: How to combine KV from different contexts?
4. Options: Concatenation, attention-weighted fusion, position re-indexing

### Phase 3: Agent Memory Scaling
1. Scale to 1000+ KV entries
2. Measure: Retrieval latency, answer quality, memory footprint
3. Compare with FAISS text retrieval + re-encode baseline

## Key Technical Challenges

1. **KV Concatenation**: Combining KV from different contexts → position ID conflicts
2. **Staleness**: Stored KV may become outdated as knowledge changes
3. **Retrieval Quality**: Text embedding may not align with KV utility
4. **Memory Overhead**: Even compressed KV is larger than text embeddings

## Paper Contribution

1. KV-RAG: A new retrieval-augmented generation paradigm at the KV-cache level
2. Compression-aware memory management for KV banks
3. Multi-source KV fusion algorithm
4. Comparison with text-based RAG showing latency and quality tradeoffs

## Risks

- KV concatenation from different contexts may confuse the model
- Position encoding conflicts when combining KV entries
- Text-based RAG may be "good enough" — hard to show KV-RAG advantage
- Memory overhead may be prohibitive for large-scale deployment
