#!/usr/bin/env python3
"""
Experiment S3: Cross-Family Scout

Test scout protocol ACROSS model families using character-level tokenizer
alignment. Critical for JSAC: proves scout is not Qwen-specific.

Pairs:
  - Qwen-7B -> Mistral-7B (same size, different family)
  - Qwen-3B -> Yi-6B-Chat (different size AND family)

Tokenizer alignment algorithm:
  1. Tokenize text with edge tokenizer -> edge_tokens, edge_spans
  2. Tokenize text with cloud tokenizer -> cloud_tokens, cloud_spans
  3. For each selected edge position, find character span
  4. Map character span to overlapping cloud token positions
  5. Apply cloud positions as attention mask

n=200 SQuAD samples, 25/50/75% retention

VRAM: 14 GB peak (7B models), sequential loading
GPU time: ~6 hours
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt,
    compute_q2c_last_layer, select_positions, generate_with_mask,
    compute_f1, confidence_interval_95, paired_ttest,
    overlap_percentage, align_positions_cross_tokenizer,
    save_results, make_timestamp, HF_MODELS,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_scout_cross_family.log')

NUM_SAMPLES = 200
RETENTIONS = [0.75, 0.50, 0.25]


def get_context_text_and_prompt(context, question):
    """Return the raw context text and the formatted prompt."""
    prompt = format_qa_prompt(context, question)
    context_text = f"Context: {context}\n"
    return context_text, prompt


def run_cross_family_pair(edge_name, cloud_name, samples):
    """
    Run cross-family scout experiment for one pair.

    Key difference from same-family: tokenizer alignment via character spans.
    """
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"Cross-Family Scout: {edge_short} -> {cloud_short}")
    logger.info(f"{'='*70}")

    # ---- Phase 1: Edge model ----
    logger.info(f"\nPhase 1: Edge model ({edge_short})")
    edge_model, edge_tok = load_model(edge_name)

    edge_data = []  # Per-sample: selections, baseline, raw context text

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['gold_answer']

        context_text, prompt = get_context_text_and_prompt(context, question)

        try:
            device = next(edge_model.parameters()).device
            inputs = edge_tok(prompt, return_tensors="pt", max_length=1024,
                              truncation=True).to(device)

            # Compute context end position
            context_ids = edge_tok.encode(context_text)
            context_end = len(context_ids)

            with torch.no_grad():
                out = edge_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)
            q2c = compute_q2c_last_layer(out.attentions, ce)

            # Edge baseline
            with torch.no_grad():
                gen_ids = edge_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            edge_answer = edge_tok.decode(
                gen_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            edge_f1 = compute_f1(edge_answer, gold)

            # Selections at each retention
            selections = {}
            for ret in RETENTIONS:
                selections[ret] = select_positions(q2c, ret).tolist()

            edge_data.append({
                'selections': selections,
                'edge_f1': edge_f1,
                'context_text': context_text,
                'context_end': ce,
            })

        except Exception as e:
            logger.error(f"  Edge sample {i} failed: {e}")
            edge_data.append({
                'selections': {r: [] for r in RETENTIONS},
                'edge_f1': 0.0,
                'context_text': context_text,
                'context_end': 0,
                'error': str(e),
            })

        if (i + 1) % 25 == 0:
            avg_f1 = np.mean([d['edge_f1'] for d in edge_data])
            logger.info(f"  Edge [{i+1}/{len(samples)}] avg F1={avg_f1:.3f}")

    # Keep edge tokenizer for alignment
    edge_tok_for_align = edge_tok

    free_model(edge_model)

    # ---- Phase 2: Cloud model ----
    logger.info(f"\nPhase 2: Cloud model ({cloud_short})")
    cloud_model, cloud_tok = load_model(cloud_name)

    results = []

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['gold_answer']

        context_text = edge_data[i]['context_text']
        _, prompt = get_context_text_and_prompt(context, question)

        try:
            device = next(cloud_model.parameters()).device
            inputs = cloud_tok(prompt, return_tensors="pt", max_length=1024,
                               truncation=True).to(device)

            # Cloud's own context end
            cloud_context_ids = cloud_tok.encode(context_text)
            cloud_context_end = len(cloud_context_ids)

            with torch.no_grad():
                out = cloud_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(cloud_context_end, seq_len - 1)
            cloud_q2c = compute_q2c_last_layer(out.attentions, ce)

            # Cloud full-KV baseline
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
                'edge_f1': edge_data[i]['edge_f1'],
                'cloud_full_f1': cloud_full_f1,
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                # Cloud's own selection (in cloud token space)
                cloud_selected = select_positions(cloud_q2c, ret)

                # Edge's selection needs cross-tokenizer alignment
                edge_positions = np.array(edge_data[i]['selections'].get(ret, []))

                if len(edge_positions) > 0:
                    # Align edge positions to cloud tokenizer via character spans
                    # Use the raw context text for alignment
                    aligned_positions = align_positions_cross_tokenizer(
                        edge_tok_for_align, cloud_tok,
                        context_text,
                        edge_positions
                    )

                    # Ensure aligned positions are within bounds
                    aligned_positions = aligned_positions[aligned_positions < ce]
                else:
                    aligned_positions = np.array([], dtype=np.int64)

                # Overlap between cloud's own selection and aligned edge selection
                overlap = overlap_percentage(cloud_selected, aligned_positions) if len(cloud_selected) > 0 and len(aligned_positions) > 0 else 0.0

                # Generate with cloud's own selection
                cloud_own_answer = generate_with_mask(
                    cloud_model, cloud_tok, inputs['input_ids'],
                    cloud_selected, ce, seq_len
                )
                cloud_own_f1 = compute_f1(cloud_own_answer, gold)

                # Generate with aligned scout selection
                if len(aligned_positions) > 0:
                    scout_answer = generate_with_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        aligned_positions, ce, seq_len
                    )
                    scout_f1 = compute_f1(scout_answer, gold)
                else:
                    scout_answer = ""
                    scout_f1 = 0.0

                sample_result['conditions'][ret_key] = {
                    'cloud_own_f1': cloud_own_f1,
                    'scout_f1': scout_f1,
                    'overlap_pct': overlap,
                    'n_cloud_selected': len(cloud_selected),
                    'n_edge_selected': len(edge_positions),
                    'n_aligned_positions': len(aligned_positions),
                    'alignment_expansion': float(len(aligned_positions) / max(len(edge_positions), 1)),
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Cloud sample {i} failed: {e}")
            results.append({
                'sample_idx': i, 'gold': gold,
                'edge_f1': edge_data[i]['edge_f1'],
                'cloud_full_f1': 0.0,
                'conditions': {},
                'error': str(e),
            })

        if (i + 1) % 25 == 0:
            valid = [r for r in results if '50%' in r.get('conditions', {})]
            if valid:
                avg_overlap = np.mean([r['conditions']['50%']['overlap_pct'] for r in valid])
                avg_scout = np.mean([r['conditions']['50%']['scout_f1'] for r in valid])
                logger.info(f"  Cloud [{i+1}/{len(samples)}] overlap@50%={avg_overlap:.1f}% "
                           f"scout_f1@50%={avg_scout:.3f}")

    free_model(cloud_model)

    # ---- Phase 3: Analysis ----
    valid = [r for r in results if r.get('conditions')]

    summary = {
        'edge_model': edge_name,
        'cloud_model': cloud_name,
        'num_samples': len(samples),
        'num_valid': len(valid),
        'cross_family': True,
        'edge_baseline_f1': float(np.mean([r['edge_f1'] for r in valid])),
        'cloud_full_f1': float(np.mean([r['cloud_full_f1'] for r in valid])),
        'retention_results': {},
    }

    for ret in RETENTIONS:
        ret_key = f"{int(ret*100)}%"
        ret_data = [r for r in valid if ret_key in r.get('conditions', {})]
        if not ret_data:
            continue

        own_f1s = [r['conditions'][ret_key]['cloud_own_f1'] for r in ret_data]
        scout_f1s = [r['conditions'][ret_key]['scout_f1'] for r in ret_data]
        overlaps = [r['conditions'][ret_key]['overlap_pct'] for r in ret_data]
        expansions = [r['conditions'][ret_key]['alignment_expansion'] for r in ret_data]

        t_stat, p_val = paired_ttest(scout_f1s, own_f1s)

        summary['retention_results'][ret_key] = {
            'cloud_own_f1_mean': float(np.mean(own_f1s)),
            'cloud_own_f1_ci95': float(confidence_interval_95(own_f1s)),
            'scout_f1_mean': float(np.mean(scout_f1s)),
            'scout_f1_ci95': float(confidence_interval_95(scout_f1s)),
            'overlap_pct_mean': float(np.mean(overlaps)),
            'overlap_pct_std': float(np.std(overlaps)),
            'alignment_expansion_mean': float(np.mean(expansions)),
            'scout_vs_own_gap': float(np.mean(scout_f1s) - np.mean(own_f1s)),
            'paired_ttest_t': float(t_stat),
            'paired_ttest_p': float(p_val),
        }

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {edge_short} -> {cloud_short} (cross-family)")
    for ret_key, rd in summary['retention_results'].items():
        logger.info(f"  {ret_key}: own={rd['cloud_own_f1_mean']:.3f} "
                    f"scout={rd['scout_f1_mean']:.3f} "
                    f"gap={rd['scout_vs_own_gap']:+.3f} "
                    f"p={rd['paired_ttest_p']:.4f} "
                    f"overlap={rd['overlap_pct_mean']:.1f}% "
                    f"expansion={rd['alignment_expansion_mean']:.2f}x")

    return {'summary': summary, 'per_sample': results}


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Experiment S3: Cross-Family Scout")
    logger.info("=" * 70)

    samples = load_squad_samples(NUM_SAMPLES)
    start = time.time()

    pairs = [
        ('Qwen-7B->Mistral-7B', HF_MODELS['7B'], HF_MODELS['Mistral-7B']),
        ('Qwen-3B->Yi-6B', HF_MODELS['3B'], HF_MODELS['Yi-6B']),
    ]

    all_results = {}
    for name, edge, cloud in pairs:
        all_results[name] = run_cross_family_pair(edge, cloud, samples)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    output = {
        'metadata': {
            'experiment': 'scout_cross_family',
            'description': 'Cross-family scout with tokenizer alignment',
            'num_samples': NUM_SAMPLES,
            'seed': 42,
            'retentions': RETENTIONS,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summaries': {k: v['summary'] for k, v in all_results.items()},
        'per_sample': {k: v['per_sample'] for k, v in all_results.items()},
    }

    save_results(output, f'exp_scout_cross_family_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
