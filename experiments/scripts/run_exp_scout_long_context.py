#!/usr/bin/env python3
"""
Experiment S2: Long Context Scout Evaluation

Test scout protocol at longer context lengths: 1K, 2K, 4K, 8K tokens.
Uses needle-in-haystack style: embed SQuAD Q&A pairs into longer contexts
by adding distractor paragraphs from other SQuAD passages.

Key question: Does cross-model attention alignment hold at longer contexts?

Pairs: Qwen 3B->7B, 3B->14B, 7B->14B
n=100 samples per context length (different from S1 to avoid memorization effects)

VRAM: 31 GB peak (14B at 8K context), sequential loading
GPU time: ~12 hours
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt, get_context_end_pos,
    compute_q2c_last_layer, select_positions, generate_with_mask,
    compute_f1, confidence_interval_95, overlap_percentage,
    save_results, make_timestamp, HF_MODELS,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_scout_long_context.log')

NUM_SAMPLES = 100
TARGET_CONTEXT_LENGTHS = [1024, 2048, 4096, 8192]  # in tokens
RETENTIONS = [0.75, 0.50, 0.25]


def build_long_context(sample, distractor_pool, tokenizer, target_tokens):
    """
    Build a long context by embedding the answer-bearing paragraph among
    distractor paragraphs.

    The answer paragraph is placed at a random position among distractors.
    Returns the full context string and the character offset where the
    answer paragraph starts.
    """
    rng = np.random.RandomState(hash(sample['context']) % (2**31))

    answer_para = sample['context']
    answer_para_tokens = len(tokenizer.encode(answer_para))

    # Accumulate distractor paragraphs until we approach target length
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

    if current_tokens < target_tokens * 0.5:
        # Not enough distractors; repeat some
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


def run_long_context_pair(edge_name, cloud_name, samples, distractor_pool):
    """Run scout experiment at multiple context lengths for one pair."""
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"Long Context Scout: {edge_short} -> {cloud_short}")
    logger.info(f"{'='*70}")

    results_by_length = {}

    for target_len in TARGET_CONTEXT_LENGTHS:
        logger.info(f"\n--- Target context: {target_len} tokens ---")

        # ---- Phase 1: Edge model ----
        edge_model, edge_tok = load_model(edge_name)

        edge_data = []  # per-sample edge selections
        prompts = []
        context_ends = []

        for i, sample in enumerate(samples):
            long_context = build_long_context(sample, distractor_pool, edge_tok, target_len)
            question = sample['question']
            gold = sample['gold_answer']

            prompt = format_qa_prompt(long_context, question)
            prompts.append(prompt)
            context_end = get_context_end_pos(edge_tok, long_context)
            context_ends.append(context_end)

            try:
                device = next(edge_model.parameters()).device
                inputs = edge_tok(prompt, return_tensors="pt",
                                  max_length=target_len + 256,
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
                })

            except Exception as e:
                logger.error(f"  Edge sample {i} @ {target_len}tok failed: {e}")
                edge_data.append({
                    'selections': {r: [] for r in RETENTIONS},
                    'actual_tokens': 0,
                    'context_end': 0,
                    'error': str(e),
                })

            if (i + 1) % 25 == 0:
                logger.info(f"  Edge [{i+1}/{len(samples)}]")

        free_model(edge_model)
        del edge_tok

        # ---- Phase 2: Cloud model ----
        cloud_model, cloud_tok = load_model(cloud_name)

        length_results = []

        for i, sample in enumerate(samples):
            gold = sample['gold_answer']
            prompt = prompts[i]
            context_end = context_ends[i]

            try:
                device = next(cloud_model.parameters()).device
                inputs = cloud_tok(prompt, return_tensors="pt",
                                   max_length=target_len + 256,
                                   truncation=True).to(device)

                with torch.no_grad():
                    out = cloud_model(**inputs, use_cache=True, output_attentions=True)

                seq_len = out.attentions[-1].shape[-1]
                ce = min(context_end, seq_len - 1)
                cloud_q2c = compute_q2c_last_layer(out.attentions, ce)

                # Cloud full-KV
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

                    overlap = overlap_percentage(cloud_selected, edge_selected) if len(cloud_selected) > 0 and len(edge_selected) > 0 else 0.0

                    cloud_own_answer = generate_with_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        cloud_selected, ce, seq_len
                    )
                    cloud_own_f1 = compute_f1(cloud_own_answer, gold)

                    if len(edge_selected) > 0:
                        scout_answer = generate_with_mask(
                            cloud_model, cloud_tok, inputs['input_ids'],
                            edge_selected, ce, seq_len
                        )
                        scout_f1 = compute_f1(scout_answer, gold)
                    else:
                        scout_f1 = 0.0

                    sample_result['conditions'][ret_key] = {
                        'cloud_own_f1': cloud_own_f1,
                        'scout_f1': scout_f1,
                        'overlap_pct': overlap,
                        'n_selected': len(cloud_selected),
                        'context_len': len(cloud_q2c),
                    }

                length_results.append(sample_result)

            except Exception as e:
                logger.error(f"  Cloud sample {i} @ {target_len}tok failed: {e}")
                length_results.append({
                    'sample_idx': i, 'gold': gold,
                    'cloud_full_f1': 0.0, 'conditions': {},
                    'error': str(e),
                })

            if (i + 1) % 25 == 0:
                valid = [r for r in length_results if '50%' in r.get('conditions', {})]
                if valid:
                    avg_o = np.mean([r['conditions']['50%']['overlap_pct'] for r in valid])
                    logger.info(f"  Cloud [{i+1}/{len(samples)}] @{target_len}tok "
                               f"overlap@50%={avg_o:.1f}%")

        free_model(cloud_model)

        # Aggregate for this context length
        valid = [r for r in length_results if r.get('conditions')]
        length_summary = {
            'target_tokens': target_len,
            'avg_actual_tokens': float(np.mean([r.get('actual_tokens', 0) for r in length_results])),
            'n_valid': len(valid),
            'cloud_full_f1': float(np.mean([r['cloud_full_f1'] for r in valid])) if valid else 0,
        }
        for ret in RETENTIONS:
            ret_key = f"{int(ret*100)}%"
            ret_data = [r for r in valid if ret_key in r.get('conditions', {})]
            if ret_data:
                length_summary[f'{ret_key}_overlap'] = float(np.mean(
                    [r['conditions'][ret_key]['overlap_pct'] for r in ret_data]))
                length_summary[f'{ret_key}_scout_f1'] = float(np.mean(
                    [r['conditions'][ret_key]['scout_f1'] for r in ret_data]))
                length_summary[f'{ret_key}_own_f1'] = float(np.mean(
                    [r['conditions'][ret_key]['cloud_own_f1'] for r in ret_data]))

        results_by_length[str(target_len)] = {
            'summary': length_summary,
            'per_sample': length_results,
        }

        logger.info(f"\n  Summary @{target_len} tokens:")
        for ret in RETENTIONS:
            ret_key = f"{int(ret*100)}%"
            overlap_val = length_summary.get(f'{ret_key}_overlap', 0)
            scout_val = length_summary.get(f'{ret_key}_scout_f1', 0)
            logger.info(f"    {ret_key}: overlap={overlap_val:.1f}% scout_f1={scout_val:.3f}")

    return results_by_length


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Experiment S2: Long Context Scout")
    logger.info("=" * 70)

    # Load samples (different seed to avoid overlap with S1)
    samples = load_squad_samples(NUM_SAMPLES, seed=123)

    # Load distractor pool (larger set for building long contexts)
    from datasets import load_dataset
    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    distractor_pool = [s for s in dataset if len(s['answers']['text']) > 0]

    start = time.time()

    pairs = [
        ('3B->7B', HF_MODELS['3B'], HF_MODELS['7B']),
        ('3B->14B', HF_MODELS['3B'], HF_MODELS['14B']),
        ('7B->14B', HF_MODELS['7B'], HF_MODELS['14B']),
    ]

    all_results = {}
    for name, edge, cloud in pairs:
        all_results[name] = run_long_context_pair(edge, cloud, samples, distractor_pool)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Summary: overlap vs context length
    logger.info(f"\n{'='*70}")
    logger.info("OVERLAP vs CONTEXT LENGTH (50% retention)")
    logger.info(f"{'Pair':10s} | {'1K':>6s} | {'2K':>6s} | {'4K':>6s} | {'8K':>6s}")
    logger.info("-" * 50)
    for name, data in all_results.items():
        overlaps = []
        for tlen in TARGET_CONTEXT_LENGTHS:
            summary = data.get(str(tlen), {}).get('summary', {})
            overlaps.append(summary.get('50%_overlap', 0))
        logger.info(f"{name:10s} | {overlaps[0]:6.1f} | {overlaps[1]:6.1f} | "
                    f"{overlaps[2]:6.1f} | {overlaps[3]:6.1f}")

    output = {
        'metadata': {
            'experiment': 'scout_long_context',
            'description': 'Scout at 1K/2K/4K/8K context lengths',
            'num_samples': NUM_SAMPLES,
            'target_lengths': TARGET_CONTEXT_LENGTHS,
            'retentions': RETENTIONS,
            'seed': 123,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'results': {pair: {tlen: data[tlen]['summary']
                          for tlen in data}
                   for pair, data in all_results.items()},
        'per_sample': {pair: {tlen: data[tlen]['per_sample']
                             for tlen in data}
                      for pair, data in all_results.items()},
    }

    save_results(output, f'exp_scout_long_context_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
