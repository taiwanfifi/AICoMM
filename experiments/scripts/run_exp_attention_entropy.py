#!/usr/bin/env python3
"""
Experiment A1: Attention Entropy Analysis

Compute per-head attention entropy for Qwen 3B, 7B, 14B on the same 200 samples.

Hypothesis: Smaller models (3B) have LOWER entropy (more concentrated attention)
compared to larger models (14B). This proves the "attention focusing" effect:
the smaller scout model's selection is actually MORE decisive, which explains
why scout-guided selection can IMPROVE the larger model's performance.

Metrics:
  - Per-head Shannon entropy of attention distributions
  - Per-layer mean/std entropy
  - Cross-model entropy comparison (paired by sample)
  - Entropy of Q2C score distributions

VRAM: 28 GB peak (14B), sequential loading
GPU time: ~4 hours
"""

import sys
import time
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt, get_context_end_pos,
    compute_q2c_last_layer, confidence_interval_95, paired_ttest,
    save_results, make_timestamp, HF_MODELS,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_attention_entropy.log')

NUM_SAMPLES = 200


def compute_attention_entropy(attn_weights):
    """
    Compute Shannon entropy of attention distribution per head.

    Args:
        attn_weights: tensor of shape [heads, seq_len, seq_len]
            attn_weights[h, i, j] = attention from position i to position j

    Returns:
        per_head_entropy: array of shape [heads], mean entropy per head
        per_position_entropy: array of shape [seq_len], mean entropy per query position
    """
    # attn_weights: [heads, query_len, key_len]
    # For each query position, the attention over keys sums to 1
    # Entropy = -sum(p * log(p))

    eps = 1e-10
    attn_clamped = attn_weights.float().clamp(min=eps)
    log_attn = torch.log2(attn_clamped)

    # Entropy per head per query position
    entropy = -(attn_weights.float() * log_attn).sum(dim=-1)  # [heads, query_len]

    per_head_entropy = entropy.mean(dim=-1).cpu().numpy()  # [heads]
    per_position_entropy = entropy.mean(dim=0).cpu().numpy()  # [query_len]

    return per_head_entropy, per_position_entropy


def compute_q2c_entropy(q2c_scores):
    """
    Compute entropy of the Q2C score distribution (after normalization).

    Lower entropy = more concentrated selection = more decisive scout.
    """
    eps = 1e-10
    p = q2c_scores / (q2c_scores.sum() + eps)
    p = np.clip(p, eps, None)
    return -np.sum(p * np.log2(p))


def run_entropy_analysis(model_name, model_key, samples):
    """
    Compute attention entropy for one model on all samples.

    Returns per-layer and per-head entropy statistics.
    """
    model, tokenizer = load_model(model_name)

    per_sample_results = []
    all_layer_entropies = []  # [sample][layer] -> mean head entropy
    all_q2c_entropies = []
    all_last_layer_head_entropies = []

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']

        prompt = format_qa_prompt(context, question)
        context_end = get_context_end_pos(tokenizer, context)

        try:
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024,
                               truncation=True).to(device)

            with torch.no_grad():
                out = model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)

            # Per-layer entropy
            layer_entropies = []
            for layer_idx, layer_attn in enumerate(out.attentions):
                attn = layer_attn[0]  # [heads, seq, seq]
                per_head_e, per_pos_e = compute_attention_entropy(attn)
                layer_entropies.append(float(per_head_e.mean()))

            all_layer_entropies.append(layer_entropies)

            # Last layer: per-head entropy (for head-level analysis)
            last_attn = out.attentions[-1][0]
            last_head_e, _ = compute_attention_entropy(last_attn)
            all_last_layer_head_entropies.append(last_head_e)

            # Context-specific: entropy of attention from query to context positions
            query_to_context = last_attn[:, ce:, :ce]  # [heads, query_len, context_len]
            if query_to_context.shape[1] > 0 and query_to_context.shape[2] > 0:
                # Normalize per query position
                q2c_attn = query_to_context / (query_to_context.sum(dim=-1, keepdim=True) + 1e-10)
                q2c_head_e, _ = compute_attention_entropy(q2c_attn)
                q2c_head_entropy_mean = float(q2c_head_e.mean())
            else:
                q2c_head_entropy_mean = 0.0

            # Q2C score distribution entropy
            q2c_scores = compute_q2c_last_layer(out.attentions, ce)
            q2c_dist_entropy = compute_q2c_entropy(q2c_scores)
            all_q2c_entropies.append(q2c_dist_entropy)

            per_sample_results.append({
                'sample_idx': i,
                'context_len': ce,
                'seq_len': seq_len,
                'num_layers': len(out.attentions),
                'num_heads': last_attn.shape[0],
                'mean_layer_entropy': float(np.mean(layer_entropies)),
                'last_layer_entropy': layer_entropies[-1],
                'q2c_head_entropy': q2c_head_entropy_mean,
                'q2c_dist_entropy': float(q2c_dist_entropy),
            })

        except Exception as e:
            logger.error(f"  Sample {i} failed: {e}")
            per_sample_results.append({'sample_idx': i, 'error': str(e)})

        if (i + 1) % 25 == 0:
            valid = [r for r in per_sample_results if 'mean_layer_entropy' in r]
            if valid:
                avg_e = np.mean([r['mean_layer_entropy'] for r in valid])
                avg_q2c = np.mean([r['q2c_dist_entropy'] for r in valid])
                logger.info(f"  [{i+1}/{len(samples)}] avg_entropy={avg_e:.3f} "
                           f"q2c_dist_entropy={avg_q2c:.3f}")

    free_model(model)

    # Aggregate statistics
    valid = [r for r in per_sample_results if 'mean_layer_entropy' in r]

    # Per-layer statistics
    n_layers = len(all_layer_entropies[0]) if all_layer_entropies else 0
    layer_stats = []
    for l in range(n_layers):
        layer_vals = [le[l] for le in all_layer_entropies if len(le) > l]
        layer_stats.append({
            'layer': l,
            'mean_entropy': float(np.mean(layer_vals)),
            'std_entropy': float(np.std(layer_vals)),
        })

    summary = {
        'model': model_name,
        'model_key': model_key,
        'num_valid': len(valid),
        'num_layers': n_layers,
        'num_heads': valid[0]['num_heads'] if valid else 0,
        'overall_mean_entropy': float(np.mean([r['mean_layer_entropy'] for r in valid])),
        'overall_mean_entropy_ci95': float(confidence_interval_95(
            [r['mean_layer_entropy'] for r in valid])),
        'last_layer_entropy_mean': float(np.mean([r['last_layer_entropy'] for r in valid])),
        'last_layer_entropy_ci95': float(confidence_interval_95(
            [r['last_layer_entropy'] for r in valid])),
        'q2c_head_entropy_mean': float(np.mean([r['q2c_head_entropy'] for r in valid])),
        'q2c_dist_entropy_mean': float(np.mean(all_q2c_entropies)),
        'q2c_dist_entropy_ci95': float(confidence_interval_95(all_q2c_entropies)),
        'per_layer_stats': layer_stats,
    }

    logger.info(f"\n  {model_key}: overall_entropy={summary['overall_mean_entropy']:.3f} "
                f"last_layer={summary['last_layer_entropy_mean']:.3f} "
                f"q2c_dist={summary['q2c_dist_entropy_mean']:.3f}")

    return {
        'summary': summary,
        'per_sample': per_sample_results,
        'q2c_entropies': all_q2c_entropies,
        'mean_layer_entropies': [r['mean_layer_entropy'] for r in valid],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['3B', '7B', '14B'],
                        help='Model sizes to run (e.g. --models 3B 7B)')
    args = parser.parse_args()

    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Experiment A1: Attention Entropy Analysis")
    logger.info("=" * 70)

    samples = load_squad_samples(NUM_SAMPLES)
    start = time.time()

    available_models = {
        '3B': ('Qwen-3B', HF_MODELS['3B']),
        '7B': ('Qwen-7B', HF_MODELS['7B']),
        '14B': ('Qwen-14B', HF_MODELS['14B']),
    }

    models = [(available_models[k]) for k in args.models if k in available_models]
    logger.info(f"Running for models: {[m[0] for m in models]}")

    all_results = {}
    for key, hf_name in models:
        logger.info(f"\n--- {key} ---")
        all_results[key] = run_entropy_analysis(hf_name, key, samples)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # ---- Cross-model comparison ----
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-MODEL ENTROPY COMPARISON")
    logger.info(f"{'Model':12s} | {'Overall H':>10s} | {'Last Layer':>11s} | "
                f"{'Q2C Dist H':>11s} | {'#Layers':>8s} | {'#Heads':>7s}")
    logger.info("-" * 70)
    for key in all_results:
        s = all_results[key]['summary']
        logger.info(f"{key:12s} | {s['overall_mean_entropy']:10.3f} | "
                    f"{s['last_layer_entropy_mean']:11.3f} | "
                    f"{s['q2c_dist_entropy_mean']:11.3f} | "
                    f"{s['num_layers']:8d} | {s['num_heads']:7d}")

    # Paired tests between models on per-sample entropy
    logger.info("\nPaired t-tests (attention entropy):")
    model_keys = list(all_results.keys())
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            k1, k2 = model_keys[i], model_keys[j]
            e1 = all_results[k1]['mean_layer_entropies']
            e2 = all_results[k2]['mean_layer_entropies']
            min_len = min(len(e1), len(e2))
            t, p = paired_ttest(e1[:min_len], e2[:min_len])
            direction = "lower" if np.mean(e1[:min_len]) < np.mean(e2[:min_len]) else "higher"
            logger.info(f"  {k1} vs {k2}: t={t:.3f} p={p:.6f} ({k1} entropy is {direction})")

    logger.info("\nPaired t-tests (Q2C distribution entropy):")
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            k1, k2 = model_keys[i], model_keys[j]
            e1 = all_results[k1]['q2c_entropies']
            e2 = all_results[k2]['q2c_entropies']
            min_len = min(len(e1), len(e2))
            t, p = paired_ttest(e1[:min_len], e2[:min_len])
            direction = "lower" if np.mean(e1[:min_len]) < np.mean(e2[:min_len]) else "higher"
            logger.info(f"  {k1} vs {k2}: t={t:.3f} p={p:.6f} ({k1} Q2C entropy is {direction})")

    output = {
        'metadata': {
            'experiment': 'attention_entropy',
            'description': 'Per-head attention entropy for 3B/7B/14B (attention focusing effect)',
            'num_samples': NUM_SAMPLES,
            'seed': 42,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summaries': {k: v['summary'] for k, v in all_results.items()},
        'per_sample': {k: v['per_sample'] for k, v in all_results.items()},
    }

    save_results(output, f'exp_attention_entropy_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
