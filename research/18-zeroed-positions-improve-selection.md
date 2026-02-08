# Topic 18: Zeroing vs Masking — Generation Path Matters More Than Strategy

> **Status**: RESOLVED — was a generation path artifact; NEW finding: manual_generate > model.generate for selection
> **Target Venue**: Methodology note (not standalone paper)
> **Confidence**: High (controlled experiment with 50 samples, same generation path)

## Original Discovery (Batch 6)

| Method | Gen Path | F1 |
|--------|----------|-----|
| Q2C 50% (attention mask) | `model.generate()` | 0.527 |
| Q2C 50% + INT4 (zero + quantize) | `manual_generate()` | **0.591** |

This appeared to show that zeroing > masking, but the generation paths differed.

## Controlled Experiment (Batch 7 — Topic 18 Verification)

All methods below use the **SAME generation path** (`manual_generate_with_mask()`):

### 50% Retention

| Method | F1 | Description |
|--------|-----|-------------|
| **mask_only** | **0.626** | Mask unselected, KV untouched |
| **zero_mask** | **0.626** | Zero unselected + mask |
| zero_only | 0.605 | Zero unselected, no mask |
| zero_int4 | 0.591 | Zero + INT4 quantize, no mask |
| mask_int4 | 0.581 | Mask + INT4 quantize all KV |

### 75% Retention

| Method | F1 | Description |
|--------|-----|-------------|
| **mask_only** | **0.735** | Mask unselected, KV untouched |
| **zero_mask** | **0.735** | Zero unselected + mask |
| zero_only | 0.730 | Zero unselected, no mask |
| zero_int4 | 0.739 | Zero + INT4, no mask |
| mask_int4 | 0.720 | Mask + INT4 all KV |

## Key Findings

### 1. Zeroing vs Masking: NO DIFFERENCE when both applied
`mask_only == zero_mask` at both 50% (0.626 = 0.626) and 75% (0.735 = 0.735). When attention mask blocks a position, zeroing its KV has no additional effect.

### 2. Masking > Zeroing (when only one applied)
At 50%: mask_only (0.626) > zero_only (0.605). Masking is better because it completely prevents attention to unselected positions. Zeroing leaves positions in the softmax denominator.

### 3. THE REAL FINDING: Generation path matters enormously
| Gen Path | Method | F1 |
|----------|--------|-----|
| `model.generate()` + attention mask | Q2C 50% | 0.527 |
| `manual_generate_with_mask()` | Q2C 50% mask | **0.626** |

**+19% F1 difference** from generation path alone! `model.generate()` handles pre-populated attention masks differently from our manual loop. This means:
- All our selection results using `model.generate()` (batch 5) UNDERESTIMATE the true selection accuracy
- The "true" Q2C 50% performance is ~0.626, not 0.527
- This changes the Q2C dominance story: Q2C50% at 0.626 is 81% of full (not 68%)

### 4. Quantization adds small noise to selection
- mask_only (0.626) > mask_int4 (0.581): INT4 hurts slightly when combined with selection
- zero_int4 (0.591) < zero_only (0.605): same pattern
- But at 75%: zero_int4 (0.739) > zero_only (0.730) — noise is negligible at high retention

## Implications

### For Our Paper
1. **Report selection results with manual_generate path** for consistency with quantization results
2. The combined pipeline story is still valid: Q2C 75% + INT4 = 0.739 (96% of full at 18.75% BW)
3. Need to re-evaluate if the difference between `model.generate()` and `manual_generate()` affects our method rankings (Q2C vs H2O vs SnapKV)

### For the Community
- `model.generate()` may handle partial attention masks suboptimally
- Researchers comparing KV selection methods should be careful about generation path consistency

## Status: RESOLVED

The original "zeroing improves selection" observation was a **generation path artifact**. The real finding is that `manual_generate()` significantly outperforms `model.generate()` for selection-based methods. This is an important methodological note but not a standalone paper topic.

This topic is now a **methodology appendix** for Topic 01, not a separate paper.
