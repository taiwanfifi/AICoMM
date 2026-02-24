#!/usr/bin/env python3
"""
Experiment: Cross-Family Same-Tokenizer Overlap (Llama 3 + Gemma 2)

Tests whether the 83% attention overlap finding from Qwen generalizes to
other model families. Uses same-family pairs (shared tokenizer):

  Pair 1: Llama-3.2-3B → Llama-3.1-8B   (Llama 3 tokenizer, 128K vocab)
  Pair 2: Gemma-2-2B   → Gemma-2-9B     (Gemma tokenizer, 256K vocab)

Metrics per pair per retention (75%, 50%, 25%):
  - Edge baseline F1
  - Cloud full-KV F1
  - Cloud own-selection F1
  - Scout-selection F1
  - Position overlap %
  - Paired t-test (scout vs cloud-own)
  - 95% confidence intervals

VRAM: ~20 GB peak (9B bfloat16), sequential model loading with unloading
GPU time: ~6-8 hours total
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt, get_context_end_pos,
    prefill_and_score, select_positions, generate_with_mask,
    compute_f1, confidence_interval_95, paired_ttest,
    bonferroni_correction, overlap_percentage,
    save_results, make_timestamp, log_gpu_info,
    RESULTS_DIR,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_cross_family_overlap.log')

NUM_SAMPLES = 200
RETENTIONS = [0.75, 0.50, 0.25]

# Same-family pairs (shared tokenizer within each pair)
FAMILY_PAIRS = [
    {
        'name': 'Llama3-3B→8B',
        'family': 'Llama-3',
        'edge': 'unsloth/Llama-3.2-3B',  # ungated mirror (identical weights+tokenizer)
        'cloud': 'meta-llama/Llama-3.1-8B',
    },
    {
        'name': 'Gemma2-2B→9B',
        'family': 'Gemma-2',
        'edge': 'google/gemma-2-2b',
        'cloud': 'google/gemma-2-9b',
    },
]


def run_scout_pair(edge_name: str, cloud_name: str, pair_name: str, samples: list):
    """
    Run scout experiment for one edge→cloud pair.

    Phase 1: Load edge model, compute Q2C selections + baseline answers
    Phase 2: Unload edge, load cloud model, compute own selections + scout answers
    Phase 3: Statistical analysis
    """
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"Scout: {edge_short} → {cloud_short} ({pair_name})")
    logger.info(f"{'='*70}")

    # ---- Phase 1: Edge model ----
    logger.info(f"\nPhase 1: Edge model ({edge_short})")
    edge_model, edge_tok = load_model(edge_name)

    edge_selections = {}  # {sample_idx: {ret: positions}}
    edge_baselines = []
    prompts = []
    context_ends = []

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['gold_answer']

        prompt = format_qa_prompt(context, question)
        prompts.append(prompt)
        context_end = get_context_end_pos(edge_tok, context)
        context_ends.append(context_end)

        try:
            kv, q2c, inputs, seq_len = prefill_and_score(
                edge_model, edge_tok, prompt, context_end
            )

            # Edge baseline answer
            with torch.no_grad():
                gen_ids = edge_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            edge_answer = edge_tok.decode(
                gen_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            edge_f1 = compute_f1(edge_answer, gold)
            edge_baselines.append({'answer': edge_answer, 'f1': edge_f1})

            # Selections at each retention
            edge_selections[i] = {}
            for ret in RETENTIONS:
                edge_selections[i][ret] = select_positions(q2c, ret).tolist()

        except Exception as e:
            logger.error(f"  Edge sample {i} failed: {e}")
            edge_baselines.append({'answer': '', 'f1': 0.0})
            edge_selections[i] = {r: [] for r in RETENTIONS}

        if (i + 1) % 25 == 0:
            avg_f1 = np.mean([b['f1'] for b in edge_baselines])
            logger.info(f"  Edge [{i+1}/{len(samples)}] avg F1={avg_f1:.3f}")

        # Checkpoint edge every 50 samples
        if (i + 1) % 50 == 0:
            _save_checkpoint(pair_name, 'edge', edge_baselines, edge_selections, i + 1, len(samples))

    logger.info(f"  Edge done. Avg F1={np.mean([b['f1'] for b in edge_baselines]):.3f}")
    free_model(edge_model)
    del edge_tok
    logger.info(f"  Edge model unloaded. VRAM freed.")

    # ---- Phase 2: Cloud model ----
    logger.info(f"\nPhase 2: Cloud model ({cloud_short})")
    cloud_model, cloud_tok = load_model(cloud_name)

    results = []

    for i, sample in enumerate(samples):
        gold = sample['gold_answer']
        prompt = prompts[i]
        context_end = context_ends[i]

        try:
            kv, cloud_q2c, inputs, seq_len = prefill_and_score(
                cloud_model, cloud_tok, prompt, context_end
            )

            # Cloud full-KV answer
            with torch.no_grad():
                gen_ids = cloud_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            cloud_full_answer = cloud_tok.decode(
                gen_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            cloud_full_f1 = compute_f1(cloud_full_answer, gold)

            sample_result = {
                'sample_idx': i,
                'gold': gold,
                'edge_f1': edge_baselines[i]['f1'],
                'cloud_full_f1': cloud_full_f1,
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                cloud_selected = select_positions(cloud_q2c, ret)
                edge_selected = np.array(edge_selections[i].get(ret, []))

                # Overlap
                if len(cloud_selected) > 0 and len(edge_selected) > 0:
                    overlap = overlap_percentage(cloud_selected, edge_selected)
                else:
                    overlap = 0.0

                # Generate with cloud's own selection
                cloud_own_answer = generate_with_mask(
                    cloud_model, cloud_tok, inputs['input_ids'],
                    cloud_selected, context_end, seq_len
                )
                cloud_own_f1 = compute_f1(cloud_own_answer, gold)

                # Generate with scout selection
                if len(edge_selected) > 0:
                    scout_answer = generate_with_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        edge_selected, context_end, seq_len
                    )
                    scout_f1 = compute_f1(scout_answer, gold)
                else:
                    scout_answer = ""
                    scout_f1 = 0.0

                sample_result['conditions'][ret_key] = {
                    'cloud_own_f1': cloud_own_f1,
                    'scout_f1': scout_f1,
                    'overlap_pct': overlap,
                    'n_selected': len(cloud_selected),
                    'context_len': len(cloud_q2c),
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Cloud sample {i} failed: {e}")
            results.append({
                'sample_idx': i, 'gold': gold,
                'edge_f1': edge_baselines[i]['f1'],
                'cloud_full_f1': 0.0,
                'conditions': {},
                'error': str(e),
            })

        if (i + 1) % 25 == 0:
            valid = [r for r in results if '50%' in r.get('conditions', {})]
            if valid:
                avg_scout = np.mean([r['conditions']['50%']['scout_f1'] for r in valid])
                avg_own = np.mean([r['conditions']['50%']['cloud_own_f1'] for r in valid])
                avg_overlap = np.mean([r['conditions']['75%']['overlap_pct'] for r in valid])
                logger.info(f"  Cloud [{i+1}/{len(samples)}] own@50%={avg_own:.3f} "
                           f"scout@50%={avg_scout:.3f} overlap@75%={avg_overlap:.1f}%")

        # Checkpoint every 25 samples (per REF_remote_ops protocol)
        if (i + 1) % 25 == 0:
            _save_cloud_checkpoint(pair_name, results, i + 1, len(samples))

    free_model(cloud_model)
    del cloud_tok
    logger.info(f"  Cloud model unloaded. VRAM freed.")

    # ---- Phase 3: Statistical analysis ----
    valid = [r for r in results if r.get('conditions')]
    logger.info(f"\nValid samples: {len(valid)}/{len(results)}")

    summary = {
        'edge_model': edge_name,
        'cloud_model': cloud_name,
        'pair_name': pair_name,
        'num_samples': len(samples),
        'num_valid': len(valid),
        'seed': 42,
        'edge_baseline_f1': float(np.mean([r['edge_f1'] for r in valid])),
        'cloud_full_f1': float(np.mean([r['cloud_full_f1'] for r in valid])),
        'retention_results': {},
    }

    p_values = []

    for ret in RETENTIONS:
        ret_key = f"{int(ret*100)}%"
        ret_results = [r for r in valid if ret_key in r.get('conditions', {})]
        if not ret_results:
            continue

        own_f1s = [r['conditions'][ret_key]['cloud_own_f1'] for r in ret_results]
        scout_f1s = [r['conditions'][ret_key]['scout_f1'] for r in ret_results]
        overlaps = [r['conditions'][ret_key]['overlap_pct'] for r in ret_results]

        t_stat, p_val = paired_ttest(scout_f1s, own_f1s)
        p_values.append(p_val)

        summary['retention_results'][ret_key] = {
            'cloud_own_f1_mean': float(np.mean(own_f1s)),
            'cloud_own_f1_ci95': float(confidence_interval_95(own_f1s)),
            'scout_f1_mean': float(np.mean(scout_f1s)),
            'scout_f1_ci95': float(confidence_interval_95(scout_f1s)),
            'overlap_pct_mean': float(np.mean(overlaps)),
            'overlap_pct_std': float(np.std(overlaps)),
            'scout_vs_own_gap': float(np.mean(scout_f1s) - np.mean(own_f1s)),
            'paired_ttest_t': float(t_stat),
            'paired_ttest_p': float(p_val),
        }

    if p_values:
        bonferroni_alpha = bonferroni_correction(p_values)
        summary['bonferroni_alpha'] = float(bonferroni_alpha)
        for i, ret in enumerate(RETENTIONS):
            ret_key = f"{int(ret*100)}%"
            if ret_key in summary['retention_results'] and i < len(p_values):
                summary['retention_results'][ret_key]['significant_bonferroni'] = bool(
                    p_values[i] < bonferroni_alpha
                )

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {pair_name}")
    logger.info(f"  Edge baseline: {summary['edge_baseline_f1']:.3f}")
    logger.info(f"  Cloud full-KV: {summary['cloud_full_f1']:.3f}")
    for ret_key, rd in summary['retention_results'].items():
        sig = " *" if rd.get('significant_bonferroni') else ""
        logger.info(f"  {ret_key}: own={rd['cloud_own_f1_mean']:.3f} "
                    f"scout={rd['scout_f1_mean']:.3f} "
                    f"gap={rd['scout_vs_own_gap']:+.3f} "
                    f"p={rd['paired_ttest_p']:.4f}{sig} "
                    f"overlap={rd['overlap_pct_mean']:.1f}%")

    return {'summary': summary, 'per_sample': results}


def _save_checkpoint(pair_name, phase, edge_baselines, edge_selections, completed, total):
    """Save edge phase checkpoint."""
    ckpt = {
        'pair_name': pair_name,
        'phase': phase,
        'completed': completed,
        'total': total,
        'edge_baselines': edge_baselines,
        'edge_selections': {str(k): {str(r): v for r, v in sel.items()}
                           for k, sel in edge_selections.items()},
    }
    path = RESULTS_DIR / f'ckpt_{pair_name.replace("→", "_")}_{phase}_{completed}.json'
    import json
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=str)
    logger.info(f"  [CHECKPOINT] {path.name}")


def _save_cloud_checkpoint(pair_name, results, completed, total):
    """Save cloud phase checkpoint."""
    ckpt = {
        'pair_name': pair_name,
        'phase': 'cloud',
        'completed': completed,
        'total': total,
        'per_sample': results,
    }
    path = RESULTS_DIR / f'ckpt_{pair_name.replace("→", "_")}_{completed}.json'
    import json
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=str)
    logger.info(f"  [CHECKPOINT] {path.name}")


def main():
    setup_hf_cache()
    log_gpu_info()

    logger.info("=" * 70)
    logger.info(f"Cross-Family Overlap Experiment (n={NUM_SAMPLES})")
    logger.info(f"Families: Llama 3 (3B→8B), Gemma 2 (2B→9B)")
    logger.info("=" * 70)

    samples = load_squad_samples(NUM_SAMPLES)
    start = time.time()

    all_results = {}
    for pair in FAMILY_PAIRS:
        pair_start = time.time()
        all_results[pair['name']] = run_scout_pair(
            pair['edge'], pair['cloud'], pair['name'], samples
        )
        pair_elapsed = time.time() - pair_start
        logger.info(f"\n  {pair['name']} completed in {pair_elapsed/60:.1f} minutes")

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # ---- Combined summary ----
    logger.info(f"\n{'='*70}")
    logger.info("COMBINED SUMMARY — Cross-Family Overlap")
    logger.info(f"{'Pair':20s} | {'Ret':5s} | {'Own F1':8s} | {'Scout F1':9s} | "
                f"{'Gap':7s} | {'p-val':8s} | {'Overlap':8s}")
    logger.info("-" * 75)
    for pair_name, data in all_results.items():
        for ret_key, rd in data['summary']['retention_results'].items():
            logger.info(f"{pair_name:20s} | {ret_key:5s} | {rd['cloud_own_f1_mean']:8.3f} | "
                       f"{rd['scout_f1_mean']:9.3f} | {rd['scout_vs_own_gap']:+7.3f} | "
                       f"{rd['paired_ttest_p']:8.4f} | {rd['overlap_pct_mean']:7.1f}%")

    # Compare with Qwen baseline
    logger.info(f"\nReference (Qwen 7B→14B from previous exp):")
    logger.info(f"  75%: overlap=83.7%, scout-own gap=−0.009, p=0.88")
    logger.info(f"  50%: overlap=69.5%, scout-own gap=−0.021, p=0.22")
    logger.info(f"  25%: overlap=55.3%, scout-own gap=−0.025, p=0.14")

    output = {
        'metadata': {
            'experiment': 'cross_family_overlap',
            'description': f'Cross-family same-tokenizer overlap test with n={NUM_SAMPLES}',
            'families': ['Llama-3', 'Gemma-2'],
            'pairs': [p['name'] for p in FAMILY_PAIRS],
            'num_samples': NUM_SAMPLES,
            'seed': 42,
            'retentions': RETENTIONS,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summaries': {k: v['summary'] for k, v in all_results.items()},
        'per_sample': {k: v['per_sample'] for k, v in all_results.items()},
    }

    save_results(output, f'exp_cross_family_overlap_{make_timestamp()}.json')
    logger.info("\nDone! Results saved.")


if __name__ == '__main__':
    main()
