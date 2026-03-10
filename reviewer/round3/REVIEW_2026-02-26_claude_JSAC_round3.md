# JSAC Review -- Scout: Cross-Model Attention Transfer for Bandwidth-Adaptive Edge-Cloud LLM Inference

**Reviewer**: Claude Opus 4.6 (independent adversarial review)
**Date**: 2026-02-26
**Venue**: IEEE Journal on Selected Areas in Communications (JSAC)
**Paper length**: ~12 pages (1348 LaTeX lines), 9 figures, 12 tables, 36 references

---

## 1. High-Level Assessment

This paper proposes Scout, a protocol that replaces KV-cache transmission in edge-cloud LLM inference with position-index transmission. The central insight is that models within the same family (e.g., Qwen 7B and 14B) attend to similar context positions, so a small edge model can identify important positions and send only their indices (336 bytes) instead of the full KV-cache (9.7 MB). The paper wraps this into a 5-mode adaptive protocol that selects compression strategy based on real-time bandwidth, and extends to multi-agent scenarios with model-aware allocation.

The research question is well-defined and practically motivated. The approach makes engineering sense: if attention patterns align across model scales, transmitting metadata instead of data is a clean design. However, the paper conflates several loosely coupled contributions (attention alignment observation, compression characterization, adaptive protocol, multi-agent allocation) into a single submission, which dilutes the depth of each. For JSAC, the networking contribution -- the adaptive protocol and multi-agent allocation -- is notably thin relative to the ML characterization work, which is a significant concern.

---

## 2. Novelty Check

### Closest Prior Work

| Paper | Venue | Relationship |
|-------|-------|-------------|
| CacheGen (Liu et al., 2024) | SIGCOMM '24 | KV-cache compression + streaming for LLM serving; 3.5x compression via delta encoding + arithmetic coding. **Most directly comparable system.** |
| Cache-to-Cache / C2C (2025) | arXiv:2510.03215 | **Critical overlap**: cross-model KV-cache communication between Sharer/Receiver LLMs from same/different families (Qwen, Llama, Gemma). Uses learned neural fusers rather than position indices. |
| SnapKV (Li et al., 2024) | NeurIPS '24 | Observation-window attention-based KV selection. Paper shows Q2C matches SnapKV. |
| H2O (Zhang et al., 2023) | NeurIPS '23 | Heavy-hitter attention eviction. Paper shows Q2C > H2O at aggressive retention. |
| Quest (Tang et al., 2024) | ICML '24 | Query-aware page-level sparsity for single-model serving; closest to Q2C scoring. |
| Splitwise (Patel et al., 2024) | ISCA '24 | Phase splitting for LLM serving; datacenter setting. |
| PerLLM (2024) | arXiv | Personalized inference scheduling with edge-cloud collaboration. |
| SLICE (2025) | arXiv | SLO-driven scheduling for LLM inference on edge. |
| Adaptive layer splitting (2025) | Frontiers of IT & EE | RL-based adaptive layer splitting for wireless LLM inference. |
| EdgeLLM (2024) | IEEE TMC | Speculative decoding for on-device inference. |

### Delta over existing work

The core novelty is the **observation that same-family models share attention patterns** and the specific application of **transmitting position indices rather than KV data** for bandwidth reduction. This is a genuine and useful insight. However:

1. **Cache-to-Cache (C2C, Oct 2025)** is a significant concurrent/prior work that the paper does not cite. C2C explores the same core idea -- cross-model KV-cache communication between same-family and cross-family LLMs -- but through learned neural fusers rather than position-index transmission. C2C tests the same model families (Qwen, Llama, Gemma) in the same Sharer-Receiver configurations. While the technical approaches differ (learned projection vs. training-free index transfer), the intellectual space is shared. **Not citing C2C is a significant omission** that reviewers at JSAC will flag.

2. The **Q2C scoring method** is essentially last-layer attention aggregation across query tokens -- a straightforward instantiation of attention-based selection that the paper itself shows is statistically identical to SnapKV. The novelty of Q2C itself is minimal; its value is as a component of the scout protocol.

3. The **adaptive protocol** is a simple threshold-based mode selector (Eq. 7-8): enumerate 5 modes, pick the highest-quality one that fits the deadline. This is a direct adaptation of ABR logic (Pensieve-style) to a discrete set with 5 choices. The algorithmic contribution is thin for a systems venue.

4. The **multi-agent allocation** is proportional bandwidth splitting based on KV-cache size, a well-known heuristic. No queuing theory, no game-theoretic analysis, no online learning.

**Overall novelty assessment**: The attention alignment observation is novel and empirically solid. The system design around it is competent but shallow. For JSAC, which expects deep networking/systems contributions, the protocol and allocation components are underdeveloped.

---

## 3. Experimental Validity

| Check | Assessment |
|-------|-----------|
| **Data leakage** | No evidence of contamination. SQuAD v2 is used in standard fashion with fixed seeds. Prompt format is documented. |
| **Baseline fairness** | Mostly fair. CacheGen is compared only on compression ratio, not on actual end-to-end quality; the authors did not run CacheGen themselves. Table 7 compares ratios but not quality at matched compression levels. |
| **Statistical significance** | Strong. n=200 with 95% CIs, paired t-tests, Bonferroni correction. This is above-average rigor for a systems paper. |
| **Metric gaming** | Token-F1 is appropriate for extractive QA but limited for generation tasks. ROUGE-1 for XSum is used but only briefly. No BERTScore or human evaluation. |
| **Reproducibility** | Good. Seeds, sample sizes, prompt formats, and hardware are specified. Scripts and JSON results are claimed available. |
| **Ablation completeness** | Partial. Q2C vs. all-layer and eager overhead are ablated. Missing: ablation of retention ratio selection strategy, ablation of EWMA alpha, sensitivity to bandwidth estimation error. |

### Specific experimental concerns

**E1. The "28,800x compression" claim is misleading.**
Scout does not compress the KV-cache -- it eliminates KV-cache transmission and replaces it with cloud re-prefill. The 28,800x number compares 9.7 MB (KV data) to 336 bytes (position indices), but **the cloud must re-do the full prefill** (57 ms compute, Table 5). This is not compression in the traditional sense; it is compute-bandwidth substitution. The paper acknowledges this in Section IV but the abstract and introduction present it as "compression," which is misleading. CacheGen achieves 3.5x actual compression of KV data; Scout achieves 0x data compression + full re-computation. These are not comparable on the same axis.

**E2. Protocol simulation is simplistic.**
The 5G trace evaluation (Table 10) uses a lookup-table mode selector with EWMA bandwidth estimation. This is not a real protocol implementation -- there is no actual sender/receiver, no packet-level simulation, no retransmission handling, no handoff modeling, no scheduling delay. The "2,000 requests" are generated synthetically against trace bandwidth values. For JSAC, which expects rigorous networking evaluation, this is insufficient. The paper honestly notes this limitation (Section VII-F), but it significantly weakens the networking contribution.

**E3. Multi-agent results are from a simplified simulation.**
Table 11 shows multi-agent results for "500 rounds," but the simulation assumes synchronized requests, no queuing, and static bandwidth. Real multi-agent scenarios involve bursty arrivals, heterogeneous deadlines, and interference-driven bandwidth variations. The claim "model-aware allocation converts 0% to 100% deadline compliance" is technically correct only in this idealized setting.

**E4. Context lengths are short for the motivating scenario.**
The paper motivates the problem with 1024-token KV-caches (201 MB for 14B) but the actual experiments use contexts of 100-500 tokens (~170 average). The long-context experiment (Table 9) goes to 4096 tokens but only for the 3B->7B pair at 4K (due to VRAM constraints). For the flagship 7B->14B pair, overlap is only measured at 1K and 2K tokens. The paper's headline numbers (9.7 MB, 201 MB) describe a scenario that is not fully evaluated experimentally.

**E5. Scout quality claim of "~100%" requires careful qualification.**
Table 2 shows Scout quality as "~100%" with a footnote "depends on model pair; 7B->14B achieves 99.4% at 75% retention." But Table 3 shows that at 75% retention, the 7B->14B pair has a scout F1 of 0.661 vs. cloud own F1 of 0.664, which is 99.5%. However, the **absolute F1** of 0.661 at 75% retention vs. 0.731 at 100% retention (full KV, no eviction) means scout actually achieves ~90% of the no-eviction baseline. The "~100%" claim is relative to the cloud's own eviction-based selection, not to the full-KV baseline. This conflation appears throughout the paper and could mislead readers.

**E6. Llama 3B->8B has borderline significance (p=0.060).**
The paper states scout is "statistically indistinguishable" across all three architectures at 75% retention. For Llama 3B->8B, p=0.060 is borderline at alpha=0.05. The paper honestly notes this ("Llama's p-value is close to the threshold") but then claims the result "confirms" generalization. With n=200, p=0.06 is suggestive of a real effect size that a larger sample would likely confirm as significant. The honest framing should be that 2 of 3 families show clear non-significance while Llama shows a marginal trend.

---

## 4. Methodology Critique

### Fundamental issues

**M1. Scout requires the cloud to have the original tokens.**
In scout mode, the cloud re-prefills from the original context tokens (Algorithm 1, line 11: "Cloud: prefill with M^cloud(C, Q)"). This means the cloud already has or receives the full text prompt. If the cloud has the prompt, why not simply run its own prefill without any edge guidance? The paper addresses this in Section IV-A ("Why not simply retransmit the original text?") by arguing that the position mask saves per-step decode attention. But the benefit is O(n) -> O(rn) attention per decode step, which at 75% retention and 1K context is a modest saving of ~256 positions per head per token. **The core bandwidth saving of scout comes from not transmitting KV-cache, but the cloud must still receive the prompt text** (which for 1K tokens is ~4 KB in UTF-8 -- still tiny, but the paper's framing implies the edge-to-cloud payload is *only* 336 bytes, which ignores prompt transmission).

**M2. The scout paradigm assumes same-family models at edge and cloud.**
This is a strong deployment constraint. In practice, edge devices may run whatever model fits their hardware (e.g., Phi-3, TinyLlama) while cloud providers run whatever is best for the task (e.g., Llama 70B, GPT-4). The requirement for same-family, same-tokenizer, same-RoPE-base limits practical applicability. The cross-family experiment (Qwen->Mistral) shows degraded overlap (73% vs. 84-92%), and Mistral's low baseline F1 (0.258) makes that experiment hard to interpret.

**M3. The problem formulation (Eq. 1-2) is standard and underspecified.**
The optimization problem is a weighted sum of quality subject to bandwidth and deadline constraints. This is a textbook resource allocation problem. The paper does not derive any analytical properties (convexity, optimal structure, price of anarchy for the multi-agent case). The mode selection is brute-force enumeration over 5 choices. For JSAC, this formulation needs substantially more depth -- online learning under uncertainty, competitive ratio analysis, or at minimum a formal proof of optimality for the greedy allocation.

**M4. Bandwidth estimation is assumed, not validated.**
The EWMA estimator (Eq. 9, alpha=0.3) with 0.8x conservative factor is stated but not evaluated. How sensitive is the protocol to estimation error? What happens when bandwidth drops suddenly (common in mmWave 5G with blockage)? The Lumos5G traces provide ground truth bandwidth, but the paper doesn't show estimation error or its impact on mode selection mistakes.

**M5. The "attention regularization" effect is speculative.**
The paper repeatedly notes that scout can *improve* cloud quality beyond the cloud's own selection (Section VII-F, "Discussion"). The explanation is that "the scout's focused attention mask removes noisy positions." This is an interesting hypothesis but is not rigorously tested. A proper test would be: compare cloud with random position removal vs. cloud with scout-guided removal vs. cloud with its own selection, controlling for retention ratio. Without this, the "regularization" claim is post-hoc rationalization.

### Assumptions

- **A1**: Same-family models share tokenizer and RoPE base. Stated and justified.
- **A2**: Attention alignment transfers from edge to cloud. Empirically validated across 3 families.
- **A3**: Single-layer Q2C scoring is sufficient. Ablated (Pearson r > 0.99 with all-layer).
- **A4**: EWMA bandwidth estimation is accurate enough for mode selection. **Not validated.**
- **A5**: Synchronized multi-agent requests. **Unrealistic for production.**

---

## 5. Writing & Presentation

### Strengths

- **Clarity**: The paper is well-organized and clearly written. The 9-section structure (system model -> attention alignment -> protocol -> compression -> transport -> experiments -> related work) is logical and easy to follow.
- **Statistical rigor**: Consistent use of n=200, 95% CIs, paired t-tests, and Bonferroni correction. This is exemplary for a systems paper.
- **Honest limitations**: Section VII-F lists 6 concrete limitations. This is refreshing.
- **Comprehensive related work**: 36 references covering KV eviction, quantization, streaming, collaborative inference, speculative decoding, cross-layer sharing, hybrid inference, adaptive streaming, and architectural compression.
- **Reproducibility statement**: Scripts and JSON results are available.

### Weaknesses

- **Overclaiming in abstract/intro**: "28,800x compression" frames compute-bandwidth substitution as compression. "100% deadline compliance" is achieved by falling back to a different operating mode (scout or local). These claims, while technically defensible, are presented in a way that oversells the contribution.
- **Missing critical citation**: Cache-to-Cache (C2C, arXiv:2510.03215, Oct 2025) explores the same intellectual space of cross-model KV-cache communication and tests the same model families. Not citing it is a serious omission.
- **Table/figure density**: 12 tables + 9 figures + 1 algorithm in ~12 pages is aggressive. Some tables could be consolidated (e.g., Tables 3 and 5 overlap; Tables 8 and 9 are related).
- **Terminology inconsistency**: The paper uses "compression" for scout mode (which transmits indices, not compressed data), "quality" interchangeably for F1 and percentage-of-baseline, and "lossless" for INT8 (which has measurable loss, just <1%).
- **Section V (Compression Operating Points) is largely independent of the scout contribution.** It characterizes quantization across models -- valuable work, but it could be a separate paper. Including it here dilutes focus.

---

## 6. Scoring

| Dimension | Score (0-100) | Weight | Justification |
|-----------|:---:|:---:|---|
| Novelty | 68 | 25% | Attention alignment observation is novel; Q2C is incremental over SnapKV; protocol design is ABR-adaptation; multi-agent allocation is textbook. C2C concurrent work reduces novelty. |
| Experimental rigor | 75 | 25% | Strong statistical methodology (n=200, CIs, Bonferroni). Weakened by: short contexts, simplified protocol simulation, no real sender-receiver, CacheGen not directly compared on quality. |
| Technical correctness | 72 | 20% | Core attention alignment results appear correct. "28,800x compression" claim is misleading framing. Bandwidth estimation not validated. Regularization effect uncontrolled. |
| Writing quality | 80 | 15% | Clear, well-organized, honest limitations. Overclaims in abstract. Missing C2C citation. Dense tables. |
| Impact potential | 70 | 15% | Practically relevant problem. Same-family constraint limits deployment. Protocol contribution is too thin for JSAC. Compression characterization is useful but not venue-appropriate as primary contribution. |

**Weighted total: 72.3**

This places the paper in the **"Significant issues, major revision required (weak reject)"** range (60-74). The core observation about attention alignment is interesting and well-validated, but the networking contribution is insufficient for JSAC. The paper reads more like a strong ML-systems characterization study (suitable for MLSys or a workshop) than a deep networking protocol paper.

---

## 7. Actionable Recommendations

### Must Fix (blocking for publication at JSAC)

1. **Cite and discuss Cache-to-Cache (C2C, arXiv:2510.03215)**. This is the most directly related concurrent work. Explain how Scout differs: training-free vs. learned fusers, bandwidth reduction vs. quality improvement, index transfer vs. projected KV-cache transfer.

2. **Reframe the "28,800x compression" claim.** Scout does not compress KV-cache data; it eliminates transmission by trading for cloud re-computation. Call it "payload reduction" or "bandwidth substitution," not compression. Clearly state the total cost: 336 bytes + cloud re-prefill (57 ms) + original prompt transmission.

3. **Deepen the networking contribution.** For JSAC, the protocol and transport components need substantial strengthening. Options:
   - Implement a packet-level simulation (e.g., ns-3) with realistic PHY/MAC, handoff, retransmission.
   - Formulate the multi-agent problem as an online optimization with competitive ratio guarantees.
   - Model bandwidth estimation error and show how mode selection degrades gracefully.
   - Add queuing-theoretic analysis for asynchronous multi-agent arrivals.

4. **Clarify what "~100% quality" means throughout.** Clearly distinguish between: (a) scout F1 vs. cloud's own eviction-based selection (what "~100%" refers to), and (b) scout F1 vs. full-KV no-eviction baseline (which is ~90% at 75% retention). Tables 2, 3, and 10 all conflate these.

### Should Fix (significant improvements)

5. **Address the "cloud already has the prompt" question more rigorously.** Quantify the decode-time attention savings of masked prefill vs. full prefill as a function of context length and generation length. Show where the crossover point is -- at what context length does the 25% attention reduction become meaningful?

6. **Validate bandwidth estimation.** Run the protocol simulation with realistic estimation error (not ground-truth bandwidth). Show mode selection accuracy and quality degradation under estimation noise.

7. **Extend long-context evaluation to the flagship pair.** The 7B->14B pair is only measured at 1K and 2K. Use techniques to avoid the VRAM bottleneck (e.g., gradient checkpointing, offloading) or use a proxy metric (e.g., overlap without extracting full attention tensors).

8. **Run CacheGen for direct comparison.** CacheGen code is publicly available. Compare Scout vs. CacheGen at matched quality levels (not just compression ratios) on the same task and dataset.

9. **Control for the "regularization" effect.** Compare cloud performance under: (a) its own top-k selection, (b) scout's top-k selection, (c) random top-k selection at the same retention ratio. This would distinguish genuine attention regularization from noise.

### Nice to Have (minor suggestions)

10. **Consolidate Section V (Compression Operating Points) or move to appendix.** It is largely independent of the scout contribution and dilutes the paper's focus. The quantization characterization is valuable but belongs in a separate paper or supplementary material for this venue.

11. **Add a figure showing the complete protocol message sequence** (handshake, mode advertisement, data/index transmission, response) with timing annotations. The current TikZ system diagram (Fig. 1) shows architecture but not protocol dynamics.

12. **Discuss heterogeneous deadlines.** The current formulation assumes a single T_max for all agents. In practice, different applications (chatbot vs. document analysis) have very different latency requirements.

13. **Report wall-clock times for edge prefill on actual edge hardware.** The paper uses A100/3090 GPUs; real edge devices (Jetson, Apple Neural Engine) would have very different prefill latencies for 3B-7B models.

14. **Table 1 footnote clarification.** The "KV@1K" column says "Per-direction (K or V); full KV-cache is 2x." But the text (line 72) computes the full size. Ensure the column header and body text use consistent conventions.

---

## 8. Potential Research Directions

1. **Learned position transfer.** Train a lightweight adapter (similar to C2C's fuser but much smaller -- perhaps a linear projection from edge Q2C scores to cloud Q2C scores) that maps edge attention to cloud attention. This could close the 3B->14B quality gap while remaining bandwidth-efficient (transmit adapter parameters once, then only indices per request).

2. **Position indices as a semantic hash.** Formalize the set of selected positions as a compact semantic representation of the query-context pair. Explore information-theoretic bounds: what is the minimum number of bits needed to convey "which positions matter" to the cloud, and how does this scale with context length?

3. **Dynamic retention ratio.** Instead of fixed 75% retention, learn a per-query retention ratio based on query difficulty (estimated from edge model confidence). Easy queries may need only 25% retention; hard queries may need 90%.

4. **Cross-family alignment via tokenizer bridging.** The 73% cross-family overlap is limited by different tokenizers. Explore lightweight tokenizer alignment (e.g., BPE merge tables, shared subword vocabularies) to enable cross-family scout operation without same-family constraint.

5. **Integration with speculative decoding.** Scout identifies important positions (prefill-time optimization); speculative decoding accelerates generation (decode-time optimization). Combining both -- scout-guided prefill with speculative decoding -- could provide end-to-end latency gains.

6. **Real-deployment measurement study.** Deploy scout on actual edge hardware (Jetson Orin, Raspberry Pi 5 with NPU) connected to a cloud GPU via real 5G. Measure end-to-end latency, reliability, and quality under real wireless conditions. This alone could be a strong JSAC contribution.

7. **Formal competitive ratio for the adaptive protocol.** Analyze the worst-case quality ratio of the greedy mode selector vs. the omniscient optimal (which knows future bandwidth). This would provide formal guarantees that JSAC reviewers expect from protocol contributions.

---

## Summary Verdict

The paper presents a genuinely interesting observation -- that same-family LLMs attend to similar context positions -- and builds a practical protocol around it. The experimental methodology is statistically rigorous and above average for this type of work. However, for JSAC specifically, the paper has two critical weaknesses: (1) the networking/protocol contribution is too shallow (threshold-based mode selection, proportional bandwidth allocation, lookup-table simulation), and (2) the framing overstates the contribution (calling compute-bandwidth substitution "28,800x compression"). The missing citation to Cache-to-Cache (C2C) is a serious gap in the related work. A major revision that deepens the protocol contribution, reframes the compression claim, and cites C2C would make this competitive for a top networking venue.

**Recommendation: Major revision (weak reject in current form)**
