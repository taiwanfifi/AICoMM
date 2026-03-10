# Reviewer Protocol

Independent critical review of research projects — papers, experiments, and codebases.
For writing style → `REF_explainer.md`. For remote experiment ops → `REF_remote_ops.md`.

## Activation

When the user says "activate reviewer", "review this", or points to a project folder:

1. **Identify target** — user will provide a folder path, project name, or drop a directory. That directory is the review scope.
2. **Read everything** — scan all source code, papers, experiment scripts, results, configs, READMEs. Understand the full picture before writing a single word.
3. **Ignore prior reviews** — if a `reviewer/` folder already exists with previous reviews, do NOT read them. They will bias your judgement. Start fresh from the primary sources.
4. **Create output folder** — if `reviewer/` does not exist in the target project, create it. If it already exists, add your new review alongside existing files (do not overwrite).
5. **Write review** — produce a new markdown file inside `reviewer/` with the current date in the filename (e.g., `REVIEW_2026-02-12.md`).

## Review Scope

Read-only on the main project. Never modify source code, papers, or experiment files. All output goes into `reviewer/` only.

```
target_project/
├── code/papers/experiments/   ← READ ONLY, do not touch
└── reviewer/
    ├── REVIEW_2026-02-12.md   ← your output goes here
    └── (previous reviews)     ← ignore these
```

## Review Structure

Every review MUST follow this template:

### 1. High-Level Assessment (1 paragraph)

What is this project trying to do? Does the approach make sense at a fundamental level? Is the research question well-defined?

### 2. Novelty Check

- Search for at least **40 peer-reviewed papers** in the same domain.
- Identify the closest prior work. Has this exact experiment been done before?
- What is the delta (incremental contribution) over existing work?
- Is the novelty sufficient for a top-tier venue (ACL, EMNLP, NeurIPS, ICML, CVPR, etc.)?

### 3. Experimental Validity

Question everything. Assume nothing is correct until verified.

| Check | What to look for |
|-------|-----------------|
| Data leakage | Train/test contamination, ground truth in input |
| Baseline fairness | Are comparisons apples-to-apples? Same data, same compute budget? |
| Statistical significance | N too small? No error bars? No p-values? Cherry-picked results? |
| Metric gaming | Are metrics appropriate? Could a trivial baseline score well? |
| Reproducibility | Are hyperparameters, seeds, and configs fully specified? |
| Ablation completeness | Are all components justified? What happens if you remove each one? |

### 4. Methodology Critique

- Are there fundamental flaws in the approach?
- Are assumptions stated and justified?
- Are there confounding variables?
- Is the evaluation methodology sound?

### 5. Writing & Presentation

- Is the paper clearly written?
- Are claims supported by evidence?
- Are figures and tables informative?
- Are limitations honestly discussed?

### 6. Scoring

| Dimension | Score (0-100) | Weight |
|-----------|:---:|:---:|
| Novelty | — | 25% |
| Experimental rigor | — | 25% |
| Technical correctness | — | 20% |
| Writing quality | — | 15% |
| Impact potential | — | 15% |
| **Weighted total** | **—** | **100%** |

Provide a single overall score (0-100) with explicit justification for each dimension.

**Calibration guide:**

| Range | Meaning |
|-------|---------|
| 90-100 | Top-tier venue accept (strong accept) |
| 75-89 | Competitive but needs revision (weak accept / borderline) |
| 60-74 | Significant issues, major revision required (weak reject) |
| 40-59 | Fundamental problems (reject) |
| 0-39 | Not publishable in current form (strong reject) |

### 7. Actionable Recommendations

Separate into:
- **Must fix** — blocking issues that prevent publication
- **Should fix** — significant improvements that strengthen the paper
- **Nice to have** — minor suggestions

### 8. Potential Research Directions

Based on everything you've read, identify:
- Gaps the current work exposes
- Natural extensions or follow-up experiments
- Unexplored angles that could become independent contributions

## Review Principles

1. **Be adversarial, not hostile** — your job is to find real problems, not to tear things down for sport. Every critique must be constructive and come with a suggestion.
2. **Think in layers** — first check high-level logic (is the research question sound?), then methodology, then implementation details. Don't get lost in code style while the experimental design is flawed.
3. **Verify claims with data** — if the paper says "significant improvement", check the numbers yourself. Recompute metrics if needed.
4. **Challenge assumptions** — "we assume X" should trigger "is X actually true? What if it isn't?"
5. **Distinguish fatal vs cosmetic** — a missing ablation is fatal; a typo in a figure caption is cosmetic. Prioritize accordingly.
6. **No anchoring** — do not read previous reviews before forming your own opinion. Prior reviews can anchor your judgement and cause you to miss things others missed too.

## GPU / Compute Requests

If the review reveals that re-running experiments would strengthen the assessment:
- Specify exactly what experiment you'd run and why
- State the compute requirement (VRAM, estimated time)
- The user will provision resources via `REF_remote_ops.md` workflow
