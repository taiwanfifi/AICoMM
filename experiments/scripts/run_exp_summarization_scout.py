#!/usr/bin/env python3
"""
Experiment E1: Summarization Scout Evaluation

Test scout protocol on a GENERATION task (XSum summarization) to demonstrate
generalization beyond extractive QA. This addresses the most common reviewer
concern across ALL reviews: "only tested on QA tasks."

Model pair: Qwen 7B → 14B (the pair where scout improves cloud on QA)
Dataset: XSum (BBC article → one-sentence summary)
Metrics: ROUGE-1, ROUGE-2, ROUGE-L
n=200 samples

Conditions per sample:
  - Cloud full-KV summary (baseline)
  - Cloud own-Q2C selection summary
  - Scout (edge 7B Q2C) selection summary
  - Position overlap between edge and cloud selections

VRAM: 28 GB peak (14B), sequential model loading
GPU time: ~3-4 hours on A100
Dependencies: pip install rouge-score datasets
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    prefill_and_score, select_positions, generate_with_mask,
    confidence_interval_95, paired_ttest, overlap_percentage,
    save_results, make_timestamp, HF_MODELS, SEED,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_summarization_scout.log')

NUM_SAMPLES = 200
RETENTIONS = [0.75, 0.50, 0.25]
MAX_NEW_TOKENS = 128  # Summaries can be longer than QA answers


# =========================================================================
# ROUGE evaluation
# =========================================================================

def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


# =========================================================================
# Dataset loading
# =========================================================================

def load_xsum_samples(num_samples: int, seed: int = SEED):
    """Load XSum samples with deterministic selection."""
    from datasets import load_dataset

    logger.info("Loading XSum dataset...")
    dataset = load_dataset('EdinburghNLP/xsum', split='test')

    # Filter for reasonable length articles (not too short, not too long)
    valid = []
    for s in dataset:
        doc = s['document']
        # XSum articles: 200-800 words is the sweet spot
        word_count = len(doc.split())
        if 100 < word_count < 1000 and len(s['summary'].strip()) > 10:
            valid.append({
                'article': doc,
                'reference_summary': s['summary'].strip(),
            })

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(valid), size=min(num_samples, len(valid)), replace=False)
    samples = [valid[i] for i in indices]

    logger.info(f"Loaded {len(samples)} XSum samples (from {len(valid)} valid, seed={seed})")
    return samples


# =========================================================================
# Prompt formatting
# =========================================================================

def format_summarization_prompt(article: str) -> str:
    """Format a summarization prompt for base models."""
    return f"Article: {article}\n\nSummarize the article above in one sentence.\nSummary:"


def get_article_end_pos(tokenizer, article: str) -> int:
    """Get the token position where the article text ends."""
    article_part = f"Article: {article}\n\n"
    article_ids = tokenizer.encode(article_part)
    return len(article_ids)


# =========================================================================
# Main experiment
# =========================================================================

def run_summarization_scout(edge_name: str, cloud_name: str, samples: list):
    """
    Run scout experiment for summarization task.

    Phase 1: Load edge model, compute Q2C selections
    Phase 2: Load cloud model, compute own selections + scout-guided summaries
    Phase 3: Statistical analysis with ROUGE
    """
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"Summarization Scout: {edge_short} -> {cloud_short}")
    logger.info(f"{'='*70}")

    # ---- Phase 1: Edge model ----
    logger.info(f"\nPhase 1: Edge model ({edge_short})")
    edge_model, edge_tok = load_model(edge_name)

    edge_selections = {}
    prompts = []
    article_ends = []

    for i, sample in enumerate(samples):
        article = sample['article']
        prompt = format_summarization_prompt(article)
        prompts.append(prompt)
        article_end = get_article_end_pos(edge_tok, article)
        article_ends.append(article_end)

        try:
            kv, q2c, inputs, seq_len = prefill_and_score(
                edge_model, edge_tok, prompt, article_end
            )

            # Edge selections at each retention level
            edge_selections[i] = {}
            for ret in RETENTIONS:
                edge_selections[i][ret] = select_positions(q2c, ret).tolist()

        except Exception as e:
            logger.error(f"  Edge sample {i} failed: {e}")
            edge_selections[i] = {r: [] for r in RETENTIONS}

        if (i + 1) % 25 == 0:
            logger.info(f"  Edge [{i+1}/{len(samples)}] Q2C computed")

    free_model(edge_model)
    del edge_tok

    # ---- Phase 2: Cloud model ----
    logger.info(f"\nPhase 2: Cloud model ({cloud_short})")
    cloud_model, cloud_tok = load_model(cloud_name)

    results = []

    for i, sample in enumerate(samples):
        reference = sample['reference_summary']
        prompt = prompts[i]
        article_end = article_ends[i]

        try:
            kv, cloud_q2c, inputs, seq_len = prefill_and_score(
                cloud_model, cloud_tok, prompt, article_end
            )

            # Cloud full-KV summary
            with torch.no_grad():
                gen_ids = cloud_model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                )
            cloud_full_summary = cloud_tok.decode(
                gen_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            cloud_full_rouge = compute_rouge(cloud_full_summary, reference)

            sample_result = {
                'sample_idx': i,
                'reference': reference,
                'cloud_full_summary': cloud_full_summary,
                'cloud_full_rouge': cloud_full_rouge,
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                cloud_selected = select_positions(cloud_q2c, ret)
                edge_selected = np.array(edge_selections[i].get(ret, []))

                # Overlap
                overlap = 0.0
                if len(cloud_selected) > 0 and len(edge_selected) > 0:
                    overlap = overlap_percentage(cloud_selected, edge_selected)

                # Generate with cloud's own selection
                cloud_own_summary = generate_with_mask(
                    cloud_model, cloud_tok, inputs['input_ids'],
                    cloud_selected, article_end, seq_len,
                    max_new=MAX_NEW_TOKENS
                )
                cloud_own_rouge = compute_rouge(cloud_own_summary, reference)

                # Generate with scout selection
                if len(edge_selected) > 0:
                    scout_summary = generate_with_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        edge_selected, article_end, seq_len,
                        max_new=MAX_NEW_TOKENS
                    )
                    scout_rouge = compute_rouge(scout_summary, reference)
                else:
                    scout_summary = ""
                    scout_rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

                sample_result['conditions'][ret_key] = {
                    'cloud_own_rouge': cloud_own_rouge,
                    'scout_rouge': scout_rouge,
                    'overlap_pct': overlap,
                    'n_selected': len(cloud_selected),
                    'context_len': len(cloud_q2c),
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Cloud sample {i} failed: {e}")
            results.append({
                'sample_idx': i,
                'reference': reference,
                'cloud_full_rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'conditions': {},
                'error': str(e),
            })

        if (i + 1) % 25 == 0:
            valid = [r for r in results if '50%' in r.get('conditions', {})]
            if valid:
                avg_scout_r1 = np.mean([r['conditions']['50%']['scout_rouge']['rouge1'] for r in valid])
                avg_own_r1 = np.mean([r['conditions']['50%']['cloud_own_rouge']['rouge1'] for r in valid])
                logger.info(f"  Cloud [{i+1}/{len(samples)}] own_R1@50%={avg_own_r1:.3f} "
                           f"scout_R1@50%={avg_scout_r1:.3f}")

        # Checkpoint every 50 samples
        if (i + 1) % 50 == 0:
            _save_checkpoint(edge_name, cloud_name, results, i + 1, len(samples))

    free_model(cloud_model)

    # ---- Phase 3: Statistical analysis ----
    valid = [r for r in results if r.get('conditions')]
    logger.info(f"\nValid samples: {len(valid)}/{len(results)}")

    summary = {
        'edge_model': edge_name,
        'cloud_model': cloud_name,
        'num_samples': len(samples),
        'num_valid': len(valid),
        'seed': SEED,
        'cloud_full_rouge1': float(np.mean([r['cloud_full_rouge']['rouge1'] for r in valid])),
        'cloud_full_rouge2': float(np.mean([r['cloud_full_rouge']['rouge2'] for r in valid])),
        'cloud_full_rougeL': float(np.mean([r['cloud_full_rouge']['rougeL'] for r in valid])),
        'retention_results': {},
    }

    for ret in RETENTIONS:
        ret_key = f"{int(ret*100)}%"
        ret_results = [r for r in valid if ret_key in r.get('conditions', {})]
        if not ret_results:
            continue

        # Extract per-sample scores
        own_r1 = [r['conditions'][ret_key]['cloud_own_rouge']['rouge1'] for r in ret_results]
        own_r2 = [r['conditions'][ret_key]['cloud_own_rouge']['rouge2'] for r in ret_results]
        own_rL = [r['conditions'][ret_key]['cloud_own_rouge']['rougeL'] for r in ret_results]

        scout_r1 = [r['conditions'][ret_key]['scout_rouge']['rouge1'] for r in ret_results]
        scout_r2 = [r['conditions'][ret_key]['scout_rouge']['rouge2'] for r in ret_results]
        scout_rL = [r['conditions'][ret_key]['scout_rouge']['rougeL'] for r in ret_results]

        overlaps = [r['conditions'][ret_key]['overlap_pct'] for r in ret_results]

        # Paired t-tests (scout vs cloud-own) for each ROUGE metric
        t1, p1 = paired_ttest(scout_r1, own_r1)
        t2, p2 = paired_ttest(scout_r2, own_r2)
        tL, pL = paired_ttest(scout_rL, own_rL)

        summary['retention_results'][ret_key] = {
            'cloud_own_rouge1_mean': float(np.mean(own_r1)),
            'cloud_own_rouge1_ci95': float(confidence_interval_95(own_r1)),
            'cloud_own_rouge2_mean': float(np.mean(own_r2)),
            'cloud_own_rougeL_mean': float(np.mean(own_rL)),

            'scout_rouge1_mean': float(np.mean(scout_r1)),
            'scout_rouge1_ci95': float(confidence_interval_95(scout_r1)),
            'scout_rouge2_mean': float(np.mean(scout_r2)),
            'scout_rouge2_ci95': float(confidence_interval_95(scout_r2)),
            'scout_rougeL_mean': float(np.mean(scout_rL)),
            'scout_rougeL_ci95': float(confidence_interval_95(scout_rL)),

            'scout_vs_own_gap_rouge1': float(np.mean(scout_r1) - np.mean(own_r1)),
            'scout_vs_own_gap_rouge2': float(np.mean(scout_r2) - np.mean(own_r2)),
            'scout_vs_own_gap_rougeL': float(np.mean(scout_rL) - np.mean(own_rL)),

            'paired_ttest_p_rouge1': float(p1),
            'paired_ttest_p_rouge2': float(p2),
            'paired_ttest_p_rougeL': float(pL),

            'overlap_pct_mean': float(np.mean(overlaps)),
            'overlap_pct_std': float(np.std(overlaps)),
        }

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {edge_short} -> {cloud_short} (Summarization)")
    logger.info(f"  Cloud full-KV: R1={summary['cloud_full_rouge1']:.3f} "
                f"R2={summary['cloud_full_rouge2']:.3f} RL={summary['cloud_full_rougeL']:.3f}")
    for ret_key, rd in summary['retention_results'].items():
        logger.info(f"  {ret_key}: own_R1={rd['cloud_own_rouge1_mean']:.3f} "
                    f"scout_R1={rd['scout_rouge1_mean']:.3f} "
                    f"gap_R1={rd['scout_vs_own_gap_rouge1']:+.3f} "
                    f"p_R1={rd['paired_ttest_p_rouge1']:.4f} "
                    f"overlap={rd['overlap_pct_mean']:.1f}%")

    return {'summary': summary, 'per_sample': results}


def _save_checkpoint(edge_name, cloud_name, results, completed, total):
    """Save intermediate checkpoint."""
    from exp_utils import RESULTS_DIR
    ckpt = {
        'metadata': {'edge': edge_name, 'cloud': cloud_name,
                     'experiment': 'summarization_scout',
                     'completed': completed, 'total': total},
        'per_sample': results,
    }
    path = RESULTS_DIR / 'exp_summarization_scout_checkpoint.json'
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=str)


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info(f"Experiment E1: Summarization Scout (n={NUM_SAMPLES})")
    logger.info("=" * 70)

    samples = load_xsum_samples(NUM_SAMPLES)
    start = time.time()

    # Primary pair: 7B → 14B (where scout helps on QA)
    edge_name = HF_MODELS['7B']
    cloud_name = HF_MODELS['14B']
    result = run_summarization_scout(edge_name, cloud_name, samples)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    output = {
        'metadata': {
            'experiment': 'summarization_scout',
            'description': 'Scout protocol on XSum summarization (7B→14B)',
            'task': 'xsum_summarization',
            'edge_model': edge_name,
            'cloud_model': cloud_name,
            'num_samples': NUM_SAMPLES,
            'retentions': RETENTIONS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'seed': SEED,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summary': result['summary'],
        'per_sample': result['per_sample'],
    }

    save_results(output, f'exp_summarization_scout_{make_timestamp()}.json')

    # Also save just the summary for quick reference
    summary_output = {
        'metadata': output['metadata'],
        'summary': result['summary'],
    }
    save_results(summary_output, f'exp_summarization_scout_summary_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
