# Task-Aware KV-Cache Compression and Adaptive Transport for Edge-Cloud LLM Inference

Code, data, and papers for our research on KV-cache compression and transport protocols for collaborative LLM inference across edge-cloud boundaries.

**Authors**: Wei-Lun Cheng and Wanjiun Liao, Department of Electrical Engineering, National Taiwan University

## Papers

### Paper A: Task-Aware KV-Cache Compression

**"Task-Aware KV-Cache Compression for Bandwidth-Efficient Collaborative LLM Inference"**

We propose a compression pipeline combining Q2C (Query-to-Context) attention-based token selection with diagnostic mixed-precision quantization. Systematic evaluation across 7 model families (1.1B--14B parameters), 4 NLP tasks, and context lengths from 512 to 4096 tokens reveals that:

- **Q2C outperforms SnapKV by 29--47%** and H2O by 92--128% at 25% token retention
- **INT8 quantization is universally lossless** across all 7 models
- **INT4 fragility is model-specific**, not architecture-determined (Yi-6B: 100%, Qwen-7B: 77% with identical GQA config)
- **Mixed-precision** with bottleneck layer protection achieves lossless quality at 3.6x compression
- **Delta encoding (CacheGen) degrades quality** when combined with quantization

Source: [`papers/paper-A/main.tex`](papers/paper-A/main.tex)

### Paper B: Scout Protocol

**"Scout: Bandwidth-Adaptive KV-Cache Transport for Heterogeneous Edge-Cloud LLM Inference"**

We propose Scout, an adaptive transport protocol that exploits cross-model attention alignment to eliminate KV-cache transmission entirely:

- **Scout mode** transmits position indices instead of KV data (28,800--98,800x payload reduction: 9.7 MB to 336 bytes)
- **82--83% position overlap** between edge and cloud model selections within the Qwen2.5 family
- **Attention focusing effect**: a 7B scout model *improves* 14B cloud quality by 10.2% (p=0.018)
- **Adaptive policy engine** achieves 98--107% quality with up to 100% deadline compliance
- **Model-aware multi-agent allocation** converts 0% to 100% deadline compliance under congestion

Source: [`papers/paper-B/main.tex`](papers/paper-B/main.tex) | Figures: [`papers/paper-B/figures/`](papers/paper-B/figures/)

## Repository Structure

```
papers/
  paper-A/                  # KV-cache compression paper (IEEE format, 7 pages)
  paper-B/                  # Scout protocol paper (IEEE format, 7 pages)
    figures/                # 4 publication-ready figures (PDF + PNG)

experiments/
  scripts/                  # 39 GPU experiment scripts (batches 2--30)
  results/                  # 25 JSON result files with per-sample data
  figures/                  # Figure generation scripts
  poc/                      # Proof-of-concept experiments

research/                   # 19 research directions and ideas
```

## Key Results

### Q2C Token Selection vs. Baselines (Paper A, SQuAD v2, 25% retention)

| Model | Q2C | SnapKV | H2O | Random |
|-------|-----|--------|-----|--------|
| Qwen-7B | **0.428** | 0.292 | 0.205 | 0.193 |
| Qwen-14B | **0.360** | 0.279 | 0.160 | 0.192 |
| Mistral-7B | **0.294** | 0.205 | 0.129 | 0.104 |
| Qwen-3B | **0.390** | 0.272 | 0.203 | 0.130 |

### Scout Cross-Model Transfer (Paper B, Qwen2.5 family)

| Edge to Cloud | Position Overlap (75%) | Scout F1 | Cloud Own F1 | Effect |
|--------------|----------------------|----------|-------------|--------|
| 3B to 7B | 82% | 0.490 | 0.603 | -19% |
| 3B to 14B | 83% | 0.541 | 0.648 | -17% |
| 7B to 14B | 83% | **0.714** | 0.648 | **+10.2%** |

### Operating Modes (Paper B)

| Mode | Payload | Quality | TX @ 100 Mbps |
|------|---------|---------|--------------|
| Full BF16 | 9.7 MB | 100% | 775 ms |
| INT8 | 4.7 MB | 99.6% | 388 ms |
| INT4 | 2.3 MB | 96.2% | 195 ms |
| Mixed INT4 | 2.6 MB | 107% | 216 ms |
| Scout | 336 B | 81--110% | 0.03 ms |

## Hardware

All GPU experiments ran on an NVIDIA RTX PRO 6000 (Blackwell, 102 GB VRAM) with PyTorch and HuggingFace Transformers in BF16 precision with eager attention.

## Models Evaluated

| Model | Parameters | Layers | KV Heads | Architecture |
|-------|-----------|--------|----------|-------------|
| Qwen2.5-3B | 3B | 36 | 2 | GQA |
| Qwen2.5-7B | 7B | 28 | 4 | GQA |
| Qwen2.5-14B | 14B | 48 | 8 | GQA |
| Yi-1.5-6B-Chat | 6B | 32 | 4 | GQA |
| Mistral-7B-v0.3 | 7B | 32 | 8 | GQA |
| Phi-3.5-mini | 3.8B | 32 | 32 | MHA |
| Pythia-2.8B | 2.8B | 32 | 32 | MHA |

## Building the Papers

```bash
cd papers/paper-A && pdflatex main.tex
cd papers/paper-B && pdflatex main.tex
```

## Citation

```bibtex
@article{cheng2026kvcache,
  title={Task-Aware KV-Cache Compression for Bandwidth-Efficient Collaborative LLM Inference},
  author={Cheng, Wei-Lun and Liao, Wanjiun},
  year={2026}
}

@article{cheng2026scout,
  title={Scout: Bandwidth-Adaptive KV-Cache Transport for Heterogeneous Edge-Cloud LLM Inference},
  author={Cheng, Wei-Lun and Liao, Wanjiun},
  year={2026}
}
```

## License

This research is provided for academic and educational purposes.
