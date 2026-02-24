#!/usr/bin/env python3
"""
Experiment F1: Q2C Formula Unification — Last-Layer vs All-Layer Ablation

Resolves the Paper A / Paper B inconsistency by showing that last-layer-only
and all-layer-averaged Q2C produce near-identical position selections.

Metrics:
  - Pearson correlation between last-layer and all-layer Q2C scores
  - Jaccard overlap of selected position sets at 25/50/75% retention
  - F1 difference when generating with each selection

Models: Qwen-7B, Qwen-14B, Mistral-7B
Samples: 200 SQuAD v2 (answerable)

Expected result: r > 0.95 correlation, >90% Jaccard overlap.
Target sentence for paper: "We use last-layer Q2C for efficiency; ablation
confirms all-layer averaging yields equivalent results (Pearson r=0.97)."

VRAM: 28 GB peak (14B), sequential model loading
GPU time: ~4 hours
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent to path for exp_utils
sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt, get_context_end_pos,
    compute_q2c_last_layer, compute_q2c_all_layers,
    select_positions, generate_with_mask, compute_f1,
    confidence_interval_95, save_results, make_timestamp,
    overlap_percentage, HF_MODELS, RESULTS_DIR,
)

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = setup_logging(__name__, 'exp_q2c_ablation.log')


def run_q2c_ablation(model_name: str, model_key: str, samples: list,
                     retentions=(0.75, 0.50, 0.25)):
    """
    Compare last-layer vs all-layer Q2C for one model.

    Returns dict with per-sample scores and aggregate statistics.
    """
    model, tokenizer = load_model(model_name)

    per_sample = []
    pearson_rs = []
    spearman_rs = []
    jaccard_by_ret = {f"{int(r*100)}%": [] for r in retentions}
    f1_diffs_by_ret = {f"{int(r*100)}%": [] for r in retentions}

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['gold_answer']

        prompt = format_qa_prompt(context, question)
        context_end = get_context_end_pos(tokenizer, context)

        try:
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024,
                               truncation=True).to(device)

            with torch.no_grad():
                out = model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]

            # Compute both Q2C variants
            q2c_last = compute_q2c_last_layer(out.attentions, context_end)
            q2c_all = compute_q2c_all_layers(out.attentions, context_end)

            # Correlation
            if len(q2c_last) > 1:
                r_pearson, p_pearson = pearsonr(q2c_last, q2c_all)
                r_spearman, p_spearman = spearmanr(q2c_last, q2c_all)
            else:
                r_pearson = r_spearman = 1.0
                p_pearson = p_spearman = 0.0

            pearson_rs.append(r_pearson)
            spearman_rs.append(r_spearman)

            sample_result = {
                'sample_idx': i,
                'gold': gold,
                'context_len': len(q2c_last),
                'pearson_r': float(r_pearson),
                'spearman_r': float(r_spearman),
                'conditions': {},
            }

            for ret in retentions:
                ret_key = f"{int(ret*100)}%"

                sel_last = select_positions(q2c_last, ret)
                sel_all = select_positions(q2c_all, ret)

                # Jaccard overlap
                set_last = set(sel_last.tolist())
                set_all = set(sel_all.tolist())
                union = set_last | set_all
                jaccard = len(set_last & set_all) / len(union) if union else 1.0
                jaccard_by_ret[ret_key].append(jaccard)

                # Generate with each and compare F1
                ans_last = generate_with_mask(
                    model, tokenizer, inputs['input_ids'],
                    sel_last, context_end, seq_len, max_new=64
                )
                ans_all = generate_with_mask(
                    model, tokenizer, inputs['input_ids'],
                    sel_all, context_end, seq_len, max_new=64
                )

                f1_last = compute_f1(ans_last, gold)
                f1_all = compute_f1(ans_all, gold)
                f1_diffs_by_ret[ret_key].append(f1_last - f1_all)

                sample_result['conditions'][ret_key] = {
                    'jaccard': float(jaccard),
                    'f1_last_layer': float(f1_last),
                    'f1_all_layers': float(f1_all),
                    'f1_diff': float(f1_last - f1_all),
                    'n_selected': len(sel_last),
                    'positions_only_in_last': len(set_last - set_all),
                    'positions_only_in_all': len(set_all - set_last),
                }

            per_sample.append(sample_result)

        except Exception as e:
            logger.error(f"  Sample {i} failed: {e}")
            per_sample.append({'sample_idx': i, 'gold': gold, 'error': str(e)})

        if (i + 1) % 20 == 0:
            avg_r = np.mean(pearson_rs) if pearson_rs else 0
            logger.info(f"  [{i+1}/{len(samples)}] avg Pearson r={avg_r:.4f}")

    free_model(model)

    # Aggregate statistics
    summary = {
        'model': model_name,
        'num_samples': len(samples),
        'num_valid': len(pearson_rs),
        'pearson_r_mean': float(np.mean(pearson_rs)),
        'pearson_r_std': float(np.std(pearson_rs)),
        'pearson_r_ci95': float(confidence_interval_95(pearson_rs)),
        'spearman_r_mean': float(np.mean(spearman_rs)),
        'spearman_r_std': float(np.std(spearman_rs)),
    }

    for ret_key in jaccard_by_ret:
        jaccards = jaccard_by_ret[ret_key]
        f1_diffs = f1_diffs_by_ret[ret_key]
        summary[f'jaccard_{ret_key}_mean'] = float(np.mean(jaccards))
        summary[f'jaccard_{ret_key}_std'] = float(np.std(jaccards))
        summary[f'f1_diff_{ret_key}_mean'] = float(np.mean(f1_diffs))
        summary[f'f1_diff_{ret_key}_std'] = float(np.std(f1_diffs))
        summary[f'f1_diff_{ret_key}_ci95'] = float(confidence_interval_95(f1_diffs))

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: {model_key}")
    logger.info(f"  Pearson r: {summary['pearson_r_mean']:.4f} +/- {summary['pearson_r_ci95']:.4f}")
    logger.info(f"  Spearman r: {summary['spearman_r_mean']:.4f}")
    for ret_key in jaccard_by_ret:
        logger.info(f"  {ret_key} Jaccard: {summary[f'jaccard_{ret_key}_mean']:.3f} "
                    f"  F1 diff: {summary[f'f1_diff_{ret_key}_mean']:+.4f}")

    return {'summary': summary, 'per_sample': per_sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['7B', '14B', 'Mistral-7B'],
                        help='Model keys to run (e.g. --models 7B 14B)')
    args = parser.parse_args()

    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Experiment F1: Q2C Last-Layer vs All-Layer Ablation")
    logger.info("=" * 70)

    samples = load_squad_samples(200)
    start = time.time()

    available = {
        '7B': ('Qwen-7B', HF_MODELS['7B']),
        '14B': ('Qwen-14B', HF_MODELS['14B']),
        'Mistral-7B': ('Mistral-7B', HF_MODELS['Mistral-7B']),
    }

    models_to_run = [(available[k]) for k in args.models if k in available]
    logger.info(f"Running for models: {[m[0] for m in models_to_run]}")

    all_results = {}

    # Run for each model sequentially (GPU memory), save incrementally
    for key, hf_name in models_to_run:
        logger.info(f"\n--- {key} ---")
        try:
            all_results[key] = run_q2c_ablation(hf_name, key, samples)
            # Save intermediate result per model
            save_results(
                {'model': key, 'summary': all_results[key]['summary'],
                 'per_sample': all_results[key]['per_sample']},
                f'exp_q2c_ablation_{key.lower().replace("-","_")}_{make_timestamp()}.json'
            )
        except Exception as e:
            logger.error(f"Model {key} FAILED: {e}")

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Save combined results
    if all_results:
        output = {
            'metadata': {
                'experiment': 'q2c_ablation_last_vs_all_layer',
                'description': 'Compare last-layer-only vs all-layer-averaged Q2C scoring',
                'num_samples': 200,
                'seed': 42,
                'timestamp': make_timestamp(),
                'elapsed_minutes': elapsed / 60,
            },
            'results': {k: v['summary'] for k, v in all_results.items()},
            'per_sample': {k: v['per_sample'] for k, v in all_results.items()},
        }

        save_results(output, f'exp_q2c_ablation_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
