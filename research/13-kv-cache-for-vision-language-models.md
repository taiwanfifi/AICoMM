# Topic 13: KV-Cache Compression for Vision-Language Models in Edge-Cloud Collaboration

> **Status**: Hypothesis — extends text to multimodal
> **Target Venue**: CVPR 2027 / ECCV 2027 / IEEE TMM
> **Confidence**: Medium (large impact if works, but VLM KV-cache is complex)

## Core Hypothesis

Vision-Language Models (VLMs) like LLaVA and Qwen-VL produce KV-cache with distinct visual and textual components. Visual KV tokens (from image patches) have different compressibility properties than text KV tokens, and modality-aware compression significantly outperforms modality-agnostic approaches.

## Why This Matters

Edge-cloud VLM scenario:
```
Edge Camera/Sensor → Edge VLM (7B) processes image + prompt
                   → Transmits compressed KV-cache
                   → Cloud VLM (72B) answers complex question
```

VLMs are the most bandwidth-hungry case:
- An image produces 256-576 visual tokens → large KV-cache
- Visual tokens may be highly redundant (large patches of sky, etc.)
- Text tokens carry task specification → less compressible

## Key Research Questions

1. Do visual KV tokens have lower effective rank than text tokens?
2. Can we compress visual tokens more aggressively than text tokens?
3. Does Q2C attention differentiate between visual and text importance?
4. Can we skip visual tokens entirely if the question is text-only?

## Experimental Plan

### Phase 1: VLM KV-Cache Analysis (2 days)
1. Load Qwen2-VL-7B (fits in 98GB VRAM)
2. Process image + text prompt → extract KV-cache
3. Separate visual vs text tokens in KV-cache
4. Analyze: effective rank, sparsity, attention patterns per modality

### Phase 2: Modality-Aware Compression (3 days)
1. **Uniform**: Same compression for visual and text tokens
2. **Visual-heavy**: Compress visual tokens more (higher rank/more selection)
3. **Text-preserving**: Keep all text tokens, compress only visual
4. **Adaptive**: Use attention scores to decide per-token compression

### Phase 3: Edge-Cloud VLM Pipeline
1. Edge: Qwen2-VL-2B processes image
2. Compress visual KV-cache
3. Transmit to cloud: Qwen2-VL-7B
4. Cloud answers question using received KV
5. Measure: accuracy vs bandwidth, latency improvement

## Expected Findings

1. Visual tokens have ~2x lower effective rank than text tokens → more compressible
2. For text-only questions about an image, visual token importance follows long-tail distribution
3. Modality-aware compression achieves same accuracy at 30-50% less bandwidth
4. Critical visual tokens (objects mentioned in question) must be preserved

## Technical Challenges

- VLM KV-cache structure differs from text-only LLM
- Visual tokens may use different position encoding
- Cross-attention between modalities complicates independent compression
- Need VLM evaluation metrics (VQA accuracy, not just F1)

## Paper Contribution

1. First analysis of VLM KV-cache compressibility by modality
2. Modality-aware compression algorithm
3. Edge-cloud VLM collaboration protocol
4. Benchmarks on VQA tasks

## Risks

- VLM architecture complexity
- Cross-model VLM transfer even harder than text-only
- Need significant compute for VLM experiments
- Competitive with concurrent VLM efficiency work
