#!/usr/bin/env python3
"""
Experiment E2: Long Context 4K Scout Evaluation

Extend scout protocol testing to 4096-token contexts.
Previous experiments only tested up to 2048 tokens (1K and 2K).

Strategy for handling output_attentions OOM:
  - 3B→7B at 4K: full analysis (both models extract attentions; fits 80GB)
  - 7B→14B at 4K: edge (7B) extracts attentions; cloud (14B) generates only
    (14B output_attentions at 4K would need ~148 GB → impossible)

Needle-in-haystack approach: embed SQuAD Q&A in long distractor context.
n=100 samples

VRAM requirements (A100 80GB):
  - 3B at 4K with attentions: ~42 GB (model 6GB + attn 36GB)
  - 7B at 4K with attentions: ~63 GB (model 14GB + attn 49GB)
  - 14B at 4K without attentions: ~35 GB (model 28GB + KV 7GB)

GPU time: ~4-6 hours on A100 80GB
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt, get_context_end_pos,
    compute_q2c_last_layer, select_positions, generate_with_mask,
    compute_f1, confidence_interval_95, paired_ttest, overlap_percentage,
    save_results, make_timestamp, HF_MODELS, RESULTS_DIR,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_long_context_4k.log')

NUM_SAMPLES = 100
TARGET_LENGTH = 4096  # tokens
RETENTIONS = [0.75, 0.50, 0.25]


def build_long_context(sample, distractor_pool, tokenizer, target_tokens):
    """
    Build a long context by embedding the answer-bearing paragraph among
    distractor paragraphs from SQuAD.

    Returns the full context string.
    """
    rng = np.random.RandomState(hash(sample['context']) % (2**31))

    answer_para = sample['context']
    answer_para_tokens = len(tokenizer.encode(answer_para))

    distractors = []
    current_tokens = answer_para_tokens
    available = [d for d in distractor_pool if d['context'] != answer_para]
    rng.shuffle(available)

    for d in available:
        d_tokens = len(tokenizer.encode(d['context']))
        if current_tokens + d_tokens > target_tokens * 1.1:
            break
        distractors.append(d['context'])
        current_tokens += d_tokens

    # If not enough, repeat some distractors
    if current_tokens < target_tokens * 0.5:
        while current_tokens < target_tokens * 0.8 and available:
            d = available[rng.randint(len(available))]
            d_tokens = len(tokenizer.encode(d['context']))
            distractors.append(d['context'])
            current_tokens += d_tokens

    # Insert answer paragraph at random position
    insert_pos = rng.randint(0, max(1, len(distractors)))
    all_paragraphs = distractors[:insert_pos] + [answer_para] + distractors[insert_pos:]

    full_context = "\n\n".join(all_paragraphs)
    return full_context


def run_pair_with_attentions(edge_name, cloud_name, samples, distractor_pool):
    """
    Run scout at 4K where BOTH models can extract attentions.
    Used for 3B→7B pair (fits 80GB).

    Returns full analysis: overlap + quality comparison.
    """
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"4K Full Analysis: {edge_short} -> {cloud_short}")
    logger.info(f"{'='*70}")

    # ---- Phase 1: Edge model ----
    logger.info(f"\nPhase 1: Edge model ({edge_short})")
    edge_model, edge_tok = load_model(edge_name)

    edge_data = []
    prompts = []
    context_ends = []

    for i, sample in enumerate(samples):
        long_context = build_long_context(sample, distractor_pool, edge_tok, TARGET_LENGTH)
        prompt = format_qa_prompt(long_context, sample['question'])
        prompts.append(prompt)
        context_end = get_context_end_pos(edge_tok, long_context)
        context_ends.append(context_end)

        try:
            device = next(edge_model.parameters()).device
            inputs = edge_tok(prompt, return_tensors="pt",
                              max_length=TARGET_LENGTH + 256,
                              truncation=True).to(device)
            actual_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                out = edge_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)
            q2c = compute_q2c_last_layer(out.attentions, ce)

            selections = {}
            for ret in RETENTIONS:
                selections[ret] = select_positions(q2c, ret).tolist()

            edge_data.append({
                'selections': selections,
                'actual_tokens': actual_len,
                'context_end': ce,
                'seq_len': seq_len,
            })

            # Free attention tensors immediately
            del out
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  Edge sample {i} @ 4K failed: {e}")
            edge_data.append({
                'selections': {r: [] for r in RETENTIONS},
                'actual_tokens': 0, 'context_end': 0, 'seq_len': 0,
                'error': str(e),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Edge [{i+1}/{len(samples)}]")

    free_model(edge_model)
    del edge_tok

    # ---- Phase 2: Cloud model ----
    logger.info(f"\nPhase 2: Cloud model ({cloud_short})")
    cloud_model, cloud_tok = load_model(cloud_name)

    results = []

    for i, sample in enumerate(samples):
        gold = sample['gold_answer']
        prompt = prompts[i]
        context_end = context_ends[i]

        try:
            device = next(cloud_model.parameters()).device
            inputs = cloud_tok(prompt, return_tensors="pt",
                               max_length=TARGET_LENGTH + 256,
                               truncation=True).to(device)

            with torch.no_grad():
                out = cloud_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)
            cloud_q2c = compute_q2c_last_layer(out.attentions, ce)

            # Free attention tensors immediately
            del out
            torch.cuda.empty_cache()

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
                'cloud_full_f1': cloud_full_f1,
                'actual_tokens': inputs['input_ids'].shape[1],
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                cloud_selected = select_positions(cloud_q2c, ret)
                edge_selected = np.array(edge_data[i]['selections'].get(ret, []))

                overlap = 0.0
                if len(cloud_selected) > 0 and len(edge_selected) > 0:
                    overlap = overlap_percentage(cloud_selected, edge_selected)

                cloud_own_answer = generate_with_mask(
                    cloud_model, cloud_tok, inputs['input_ids'],
                    cloud_selected, ce, seq_len
                )
                cloud_own_f1 = compute_f1(cloud_own_answer, gold)

                scout_f1 = 0.0
                if len(edge_selected) > 0:
                    scout_answer = generate_with_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        edge_selected, ce, seq_len
                    )
                    scout_f1 = compute_f1(scout_answer, gold)

                sample_result['conditions'][ret_key] = {
                    'cloud_own_f1': cloud_own_f1,
                    'scout_f1': scout_f1,
                    'overlap_pct': overlap,
                    'n_selected': len(cloud_selected),
                    'context_len': len(cloud_q2c),
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Cloud sample {i} @ 4K failed: {e}")
            results.append({
                'sample_idx': i, 'gold': gold,
                'cloud_full_f1': 0.0, 'conditions': {},
                'error': str(e),
            })

        if (i + 1) % 10 == 0:
            valid = [r for r in results if '50%' in r.get('conditions', {})]
            if valid:
                avg_o = np.mean([r['conditions']['50%']['overlap_pct'] for r in valid])
                logger.info(f"  Cloud [{i+1}/{len(samples)}] overlap@50%={avg_o:.1f}%")

    free_model(cloud_model)

    return analyze_results(edge_name, cloud_name, results, full_analysis=True)


def run_pair_scout_only(edge_name, cloud_name, samples, distractor_pool):
    """
    Run scout at 4K where only the EDGE model extracts attentions.
    Cloud model generates but does NOT extract attentions (OOM prevention).
    Used for 7B→14B pair.

    Reports: scout F1 vs full-KV F1 (no cloud-own baseline possible).
    """
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"4K Scout-Only: {edge_short} -> {cloud_short}")
    logger.info(f"(Cloud attentions skipped to prevent OOM)")
    logger.info(f"{'='*70}")

    # ---- Phase 1: Edge model ----
    logger.info(f"\nPhase 1: Edge model ({edge_short})")
    edge_model, edge_tok = load_model(edge_name)

    edge_data = []
    prompts = []
    context_ends = []

    for i, sample in enumerate(samples):
        long_context = build_long_context(sample, distractor_pool, edge_tok, TARGET_LENGTH)
        prompt = format_qa_prompt(long_context, sample['question'])
        prompts.append(prompt)
        context_end = get_context_end_pos(edge_tok, long_context)
        context_ends.append(context_end)

        try:
            device = next(edge_model.parameters()).device
            inputs = edge_tok(prompt, return_tensors="pt",
                              max_length=TARGET_LENGTH + 256,
                              truncation=True).to(device)
            actual_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                out = edge_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)
            q2c = compute_q2c_last_layer(out.attentions, ce)

            selections = {}
            for ret in RETENTIONS:
                selections[ret] = select_positions(q2c, ret).tolist()

            edge_data.append({
                'selections': selections,
                'actual_tokens': actual_len,
                'context_end': ce,
                'seq_len': seq_len,
            })

            del out
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  Edge sample {i} @ 4K failed: {e}")
            edge_data.append({
                'selections': {r: [] for r in RETENTIONS},
                'actual_tokens': 0, 'context_end': 0, 'seq_len': 0,
                'error': str(e),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Edge [{i+1}/{len(samples)}]")

    free_model(edge_model)
    del edge_tok

    # ---- Phase 2: Cloud model (NO attention extraction) ----
    logger.info(f"\nPhase 2: Cloud model ({cloud_short}) - generate only, no attentions")
    cloud_model, cloud_tok = load_model(cloud_name)

    # Disable eager attention since we don't need output_attentions
    # This allows using SDPA for faster generation
    # (But model was loaded with eager; that's fine, we just won't request attentions)

    results = []

    for i, sample in enumerate(samples):
        gold = sample['gold_answer']
        prompt = prompts[i]
        context_end = context_ends[i]

        try:
            device = next(cloud_model.parameters()).device
            inputs = cloud_tok(prompt, return_tensors="pt",
                               max_length=TARGET_LENGTH + 256,
                               truncation=True).to(device)
            seq_len = inputs['input_ids'].shape[1]
            ce = min(context_end, seq_len - 1)

            # Cloud full-KV answer (no attention extraction)
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
                'cloud_full_f1': cloud_full_f1,
                'actual_tokens': seq_len,
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                edge_selected = np.array(edge_data[i]['selections'].get(ret, []))

                # Generate with scout selection
                scout_f1 = 0.0
                if len(edge_selected) > 0:
                    scout_answer = generate_with_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        edge_selected, ce, seq_len
                    )
                    scout_f1 = compute_f1(scout_answer, gold)

                sample_result['conditions'][ret_key] = {
                    'scout_f1': scout_f1,
                    'cloud_own_f1': None,  # Cannot compute (OOM)
                    'overlap_pct': None,    # Cannot compute (no cloud Q2C)
                    'n_selected': len(edge_selected),
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Cloud sample {i} @ 4K failed: {e}")
            results.append({
                'sample_idx': i, 'gold': gold,
                'cloud_full_f1': 0.0, 'conditions': {},
                'error': str(e),
            })

        if (i + 1) % 10 == 0:
            valid = [r for r in results if '50%' in r.get('conditions', {})]
            if valid:
                avg_sf = np.mean([r['conditions']['50%']['scout_f1'] for r in valid])
                avg_cf = np.mean([r['cloud_full_f1'] for r in valid])
                logger.info(f"  Cloud [{i+1}/{len(samples)}] full={avg_cf:.3f} scout@50%={avg_sf:.3f}")

    free_model(cloud_model)

    return analyze_results(edge_name, cloud_name, results, full_analysis=False)


def analyze_results(edge_name, cloud_name, results, full_analysis=True):
    """Compute summary statistics."""
    valid = [r for r in results if r.get('conditions')]
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    summary = {
        'edge_model': edge_name,
        'cloud_model': cloud_name,
        'target_length': TARGET_LENGTH,
        'num_valid': len(valid),
        'full_analysis': full_analysis,
        'cloud_full_f1_mean': float(np.mean([r['cloud_full_f1'] for r in valid])),
        'cloud_full_f1_ci95': float(confidence_interval_95([r['cloud_full_f1'] for r in valid])),
        'avg_actual_tokens': float(np.mean([r.get('actual_tokens', 0) for r in results])),
        'retention_results': {},
    }

    for ret in RETENTIONS:
        ret_key = f"{int(ret*100)}%"
        rd = [r for r in valid if ret_key in r.get('conditions', {})]
        if not rd:
            continue

        scout_f1s = [r['conditions'][ret_key]['scout_f1'] for r in rd]

        ret_summary = {
            'scout_f1_mean': float(np.mean(scout_f1s)),
            'scout_f1_ci95': float(confidence_interval_95(scout_f1s)),
        }

        if full_analysis:
            own_f1s = [r['conditions'][ret_key]['cloud_own_f1'] for r in rd]
            overlaps = [r['conditions'][ret_key]['overlap_pct'] for r in rd]

            t_stat, p_val = paired_ttest(scout_f1s, own_f1s)

            ret_summary.update({
                'cloud_own_f1_mean': float(np.mean(own_f1s)),
                'cloud_own_f1_ci95': float(confidence_interval_95(own_f1s)),
                'overlap_pct_mean': float(np.mean(overlaps)),
                'overlap_pct_std': float(np.std(overlaps)),
                'scout_vs_own_gap': float(np.mean(scout_f1s) - np.mean(own_f1s)),
                'paired_ttest_p': float(p_val),
            })
        else:
            # Scout vs full-KV comparison
            full_f1s = [r['cloud_full_f1'] for r in rd]
            t_stat, p_val = paired_ttest(scout_f1s, full_f1s)

            ret_summary.update({
                'scout_vs_full_gap': float(np.mean(scout_f1s) - np.mean(full_f1s)),
                'scout_vs_full_p': float(p_val),
            })

        summary['retention_results'][ret_key] = ret_summary

    # Log
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {edge_short} -> {cloud_short} @ {TARGET_LENGTH} tokens")
    logger.info(f"  Cloud full-KV F1: {summary['cloud_full_f1_mean']:.3f} ± {summary['cloud_full_f1_ci95']:.3f}")
    logger.info(f"  Avg actual tokens: {summary['avg_actual_tokens']:.0f}")
    for ret_key, rd in summary['retention_results'].items():
        if full_analysis:
            logger.info(f"  {ret_key}: own={rd['cloud_own_f1_mean']:.3f} "
                        f"scout={rd['scout_f1_mean']:.3f} "
                        f"gap={rd['scout_vs_own_gap']:+.3f} "
                        f"p={rd['paired_ttest_p']:.4f} "
                        f"overlap={rd['overlap_pct_mean']:.1f}%")
        else:
            logger.info(f"  {ret_key}: scout={rd['scout_f1_mean']:.3f} "
                        f"vs_full={rd['scout_vs_full_gap']:+.3f} "
                        f"p={rd['scout_vs_full_p']:.4f}")

    return {'summary': summary, 'per_sample': results}


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info(f"Experiment E2: Long Context 4K Scout (n={NUM_SAMPLES})")
    logger.info(f"Target: {TARGET_LENGTH} tokens")
    logger.info("=" * 70)

    # Load samples (seed=123, different from n=200 experiments)
    samples = load_squad_samples(NUM_SAMPLES, seed=123)

    # Distractor pool for building long contexts
    from datasets import load_dataset
    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    distractor_pool = [s for s in dataset if len(s['answers']['text']) > 0]

    start = time.time()
    all_results = {}

    # Pair 1: 3B→7B — full analysis (both fit on 80GB at 4K)
    logger.info("\n\n>>> Pair 1: 3B → 7B (full overlap analysis)")
    all_results['3B->7B'] = run_pair_with_attentions(
        HF_MODELS['3B'], HF_MODELS['7B'], samples, distractor_pool
    )

    # Pair 2: 7B→14B — scout-only (14B can't extract attentions at 4K)
    logger.info("\n\n>>> Pair 2: 7B → 14B (scout quality only)")
    all_results['7B->14B'] = run_pair_scout_only(
        HF_MODELS['7B'], HF_MODELS['14B'], samples, distractor_pool
    )

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Combined summary
    logger.info(f"\n{'='*70}")
    logger.info("COMBINED RESULTS @ 4K TOKENS")
    logger.info("=" * 70)
    for pair_name, data in all_results.items():
        s = data['summary']
        logger.info(f"\n{pair_name} (n={s['num_valid']}, avg tokens={s['avg_actual_tokens']:.0f}):")
        for ret_key, rd in s['retention_results'].items():
            if s['full_analysis']:
                logger.info(f"  {ret_key}: overlap={rd['overlap_pct_mean']:.1f}% "
                            f"scout_f1={rd['scout_f1_mean']:.3f}")
            else:
                logger.info(f"  {ret_key}: scout_f1={rd['scout_f1_mean']:.3f} "
                            f"full_f1={s['cloud_full_f1_mean']:.3f}")

    # Compare with previous 1K/2K results
    logger.info(f"\n{'='*70}")
    logger.info("CONTEXT LENGTH PROGRESSION (from memory + this experiment):")
    logger.info("  Previous: @1K overlap=83.3% @2K overlap=82.7% (7B→14B @75% ret)")
    if '3B->7B' in all_results:
        s = all_results['3B->7B']['summary']
        o75 = s['retention_results'].get('75%', {}).get('overlap_pct_mean', 0)
        logger.info(f"  This exp: @4K overlap={o75:.1f}% (3B→7B @75% ret)")

    output = {
        'metadata': {
            'experiment': 'long_context_4k',
            'description': 'Scout protocol at 4096-token context length',
            'target_length': TARGET_LENGTH,
            'num_samples': NUM_SAMPLES,
            'retentions': RETENTIONS,
            'seed': 123,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summaries': {k: v['summary'] for k, v in all_results.items()},
        'per_sample': {k: v['per_sample'] for k, v in all_results.items()},
    }

    save_results(output, f'exp_long_context_4k_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
