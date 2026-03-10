# ChatGPT Reviewer Prompt — Paste this into ChatGPT (GPT-4o or o1-pro)

## Instructions

Paste the following prompt into ChatGPT, then paste the full main.tex content after "--- BEGIN PAPER ---".

---

## Prompt

You are an expert reviewer for **IEEE Journal on Selected Areas in Communications (JSAC)**, one of the top networking journals (Impact Factor ~13). You specialize in edge computing, wireless systems, and LLM serving infrastructure. You have reviewed 50+ papers for JSAC, SIGCOMM, MobiCom, and NSDI.

**This is the first time you are seeing this paper.** You have no prior knowledge of this work or its revision history.

Your task is to write a **thorough, critical, adversarial review** as if this paper were submitted fresh to JSAC. Focus on finding problems, gaps, and weaknesses that would prevent acceptance at a top venue. Be specific and actionable.

### Review Structure

Please organize your review as follows:

1. **Summary** (3-4 sentences): What does the paper claim? What is the core contribution?

2. **Novelty Assessment**:
   - What is genuinely new? What is incremental?
   - Identify the closest prior work (especially CacheGen SIGCOMM'24, Cache-to-Cache arXiv:2510.03215, SnapKV, H2O, speculative decoding). How does this paper differentiate?
   - Is the delta sufficient for JSAC?

3. **Technical Correctness**:
   - Are the claims supported by evidence?
   - Are there logical gaps in the argument?
   - Are the propositions/proofs rigorous or trivial?
   - Is the experimental methodology sound (sample sizes, statistical tests, baselines)?

4. **Experimental Concerns** (be very specific):
   - Are the baselines fair and complete? What baselines are missing?
   - Are context lengths realistic for the motivating scenario?
   - Is the protocol simulation realistic enough for a networking venue?
   - Is the multi-agent evaluation convincing?
   - Are there any claims that the data does not support?

5. **Fundamental Questions**:
   - In scout mode, the cloud re-prefills from the original tokens. What is the value of the position mask compared to simply sending the prompt text (~4KB) without any edge model?
   - The same-family constraint—how realistic is this in production deployments?
   - Is the protocol contribution (threshold-based mode selection over 5 options) deep enough for JSAC?

6. **Writing & Presentation**:
   - Any overclaiming? Misleading framing?
   - Are tables/figures clear and non-redundant?
   - Is the paper the right length? Any sections that should be cut or expanded?

7. **Scoring** (0-100 on each dimension):
   - Novelty
   - Experimental Rigor
   - Technical Depth
   - Writing Quality
   - Fit for JSAC
   - **Overall weighted score**

8. **Actionable Recommendations**: Prioritized list of specific changes that would make this paper acceptable at JSAC. Distinguish between "must fix" and "nice to have."

### Important Notes
- Do NOT be lenient. JSAC acceptance rate is ~15%. Only papers with deep technical contributions, rigorous evaluation, and clear novelty are accepted.
- Pay special attention to whether the **networking/protocol contribution** is sufficient for JSAC, vs. being primarily an ML characterization study.
- Flag any claims that could be considered misleading (e.g., compression ratios, quality percentages).
- Consider whether the paper would be a better fit for a different venue (MLSys, TMC, ICC).

--- BEGIN PAPER ---

[PASTE THE FULL main.tex CONTENT HERE]
