#!/usr/bin/env python3
"""
Experiment S4: Multi-Task Scout Evaluation

Test scout protocol across multiple QA tasks to show generalization.

Tasks:
  - SQuAD v2 (extractive QA)
  - HotpotQA (multi-hop QA)
  - TriviaQA (trivia/knowledge QA)

Model pair: Qwen 7B -> 14B (the most interesting pair: scout improves cloud)
n=100 samples per task

VRAM: 28 GB (14B), sequential loading
GPU time: ~6 hours
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    format_qa_prompt, get_context_end_pos,
    prefill_and_score, select_positions, generate_with_mask,
    compute_f1, confidence_interval_95, paired_ttest,
    overlap_percentage, save_results, make_timestamp, HF_MODELS,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_scout_multitask.log')

NUM_SAMPLES_PER_TASK = 100
RETENTIONS = [0.75, 0.50, 0.25]


def load_task_samples(task_name, num_samples, seed=42):
    """
    Load samples from a QA task dataset.

    Returns list of dicts: {context, question, gold_answer}
    """
    from datasets import load_dataset
    rng = np.random.RandomState(seed)

    if task_name == 'squad_v2':
        dataset = load_dataset('rajpurkar/squad_v2', split='validation')
        answerable = [s for s in dataset if len(s['answers']['text']) > 0]
        indices = rng.choice(len(answerable), size=min(num_samples, len(answerable)),
                             replace=False)
        samples = []
        for i in indices:
            s = answerable[i]
            samples.append({
                'context': s['context'],
                'question': s['question'],
                'gold_answer': s['answers']['text'][0],
            })

    elif task_name == 'hotpotqa':
        dataset = load_dataset('hotpot_qa', 'distractor', split='validation')
        # Filter for samples with reasonable context length
        valid = []
        for s in dataset:
            # Combine supporting facts into context
            context_parts = []
            for title, sents in zip(s['context']['title'], s['context']['sentences']):
                context_parts.append(f"{title}: {' '.join(sents)}")
            context = " ".join(context_parts)
            if 100 < len(context) < 3000 and len(s['answer']) > 0:
                valid.append({
                    'context': context,
                    'question': s['question'],
                    'gold_answer': s['answer'],
                })

        indices = rng.choice(len(valid), size=min(num_samples, len(valid)),
                             replace=False)
        samples = [valid[i] for i in indices]

    elif task_name == 'triviaqa':
        dataset = load_dataset('trivia_qa', 'rc', split='validation')
        valid = []
        for s in dataset:
            # Use the search context (Wikipedia)
            if s['entity_pages']['wiki_context']:
                context = s['entity_pages']['wiki_context'][0][:2000]
            elif s['search_results']['search_context']:
                context = s['search_results']['search_context'][0][:2000]
            else:
                continue

            if len(context) > 100 and len(s['answer']['aliases']) > 0:
                valid.append({
                    'context': context,
                    'question': s['question'],
                    'gold_answer': s['answer']['aliases'][0],
                })

        indices = rng.choice(len(valid), size=min(num_samples, len(valid)),
                             replace=False)
        samples = [valid[i] for i in indices]

    else:
        raise ValueError(f"Unknown task: {task_name}")

    logger.info(f"Loaded {len(samples)} samples for {task_name}")
    return samples


def run_scout_task(edge_name, cloud_name, task_name, samples):
    """Run scout experiment for one task."""
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n--- {task_name}: {edge_short} -> {cloud_short} ---")

    # ---- Edge model ----
    edge_model, edge_tok = load_model(edge_name)

    edge_selections = {}
    edge_baselines = []
    prompts = []
    context_ends = []

    for i, sample in enumerate(samples):
        prompt = format_qa_prompt(sample['context'], sample['question'])
        prompts.append(prompt)
        context_end = get_context_end_pos(edge_tok, sample['context'])
        context_ends.append(context_end)
        gold = sample['gold_answer']

        try:
            kv, q2c, inputs, seq_len = prefill_and_score(
                edge_model, edge_tok, prompt, context_end
            )

            with torch.no_grad():
                gen_ids = edge_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            edge_answer = edge_tok.decode(
                gen_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            edge_f1 = compute_f1(edge_answer, gold)
            edge_baselines.append({'f1': edge_f1})

            edge_selections[i] = {}
            for ret in RETENTIONS:
                edge_selections[i][ret] = select_positions(q2c, ret).tolist()

        except Exception as e:
            logger.error(f"  Edge sample {i} ({task_name}) failed: {e}")
            edge_baselines.append({'f1': 0.0})
            edge_selections[i] = {r: [] for r in RETENTIONS}

        if (i + 1) % 25 == 0:
            logger.info(f"  Edge [{i+1}/{len(samples)}] {task_name}")

    free_model(edge_model)
    del edge_tok

    # ---- Cloud model ----
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

            # Cloud full-KV
            with torch.no_grad():
                gen_ids = cloud_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            cloud_full_f1 = compute_f1(
                cloud_tok.decode(gen_ids[0][inputs['input_ids'].shape[1]:],
                                 skip_special_tokens=True).strip(),
                gold
            )

            sample_result = {
                'sample_idx': i,
                'edge_f1': edge_baselines[i]['f1'],
                'cloud_full_f1': cloud_full_f1,
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                cloud_selected = select_positions(cloud_q2c, ret)
                edge_selected = np.array(edge_selections[i].get(ret, []))

                overlap = overlap_percentage(cloud_selected, edge_selected) if len(cloud_selected) > 0 and len(edge_selected) > 0 else 0.0

                cloud_own_f1 = compute_f1(
                    generate_with_mask(cloud_model, cloud_tok, inputs['input_ids'],
                                       cloud_selected, context_end, seq_len),
                    gold
                )

                if len(edge_selected) > 0:
                    scout_f1 = compute_f1(
                        generate_with_mask(cloud_model, cloud_tok, inputs['input_ids'],
                                           edge_selected, context_end, seq_len),
                        gold
                    )
                else:
                    scout_f1 = 0.0

                sample_result['conditions'][ret_key] = {
                    'cloud_own_f1': cloud_own_f1,
                    'scout_f1': scout_f1,
                    'overlap_pct': overlap,
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Cloud sample {i} ({task_name}) failed: {e}")
            results.append({
                'sample_idx': i,
                'edge_f1': edge_baselines[i]['f1'],
                'cloud_full_f1': 0.0,
                'conditions': {},
                'error': str(e),
            })

        if (i + 1) % 25 == 0:
            logger.info(f"  Cloud [{i+1}/{len(samples)}] {task_name}")

    free_model(cloud_model)

    # Analyze
    valid = [r for r in results if r.get('conditions')]
    task_summary = {
        'task': task_name,
        'n_valid': len(valid),
        'edge_baseline_f1': float(np.mean([r['edge_f1'] for r in valid])),
        'cloud_full_f1': float(np.mean([r['cloud_full_f1'] for r in valid])),
        'retention_results': {},
    }

    for ret in RETENTIONS:
        ret_key = f"{int(ret*100)}%"
        rd = [r for r in valid if ret_key in r.get('conditions', {})]
        if not rd:
            continue

        own_f1s = [r['conditions'][ret_key]['cloud_own_f1'] for r in rd]
        scout_f1s = [r['conditions'][ret_key]['scout_f1'] for r in rd]
        overlaps = [r['conditions'][ret_key]['overlap_pct'] for r in rd]

        t_stat, p_val = paired_ttest(scout_f1s, own_f1s)

        task_summary['retention_results'][ret_key] = {
            'cloud_own_f1_mean': float(np.mean(own_f1s)),
            'scout_f1_mean': float(np.mean(scout_f1s)),
            'scout_f1_ci95': float(confidence_interval_95(scout_f1s)),
            'overlap_pct_mean': float(np.mean(overlaps)),
            'scout_vs_own_gap': float(np.mean(scout_f1s) - np.mean(own_f1s)),
            'paired_ttest_p': float(p_val),
        }

    logger.info(f"\n  {task_name} results:")
    for ret_key, rd in task_summary['retention_results'].items():
        logger.info(f"    {ret_key}: own={rd['cloud_own_f1_mean']:.3f} "
                    f"scout={rd['scout_f1_mean']:.3f} "
                    f"gap={rd['scout_vs_own_gap']:+.3f} "
                    f"p={rd['paired_ttest_p']:.4f}")

    return {'summary': task_summary, 'per_sample': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['squad_v2', 'hotpotqa', 'triviaqa'],
                        help='Tasks to run (e.g. --tasks squad_v2 hotpotqa)')
    args = parser.parse_args()

    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Experiment S4: Multi-Task Scout")
    logger.info("=" * 70)

    edge_name = HF_MODELS['7B']
    cloud_name = HF_MODELS['14B']

    tasks = args.tasks
    start = time.time()

    all_results = {}
    for task in tasks:
        samples = load_task_samples(task, NUM_SAMPLES_PER_TASK)
        all_results[task] = run_scout_task(edge_name, cloud_name, task, samples)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Cross-task comparison
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-TASK COMPARISON (7B -> 14B)")
    logger.info(f"{'Task':12s} | {'Ret':5s} | {'Own F1':8s} | {'Scout F1':9s} | {'Gap':7s} | {'p-val':8s}")
    logger.info("-" * 65)
    for task, data in all_results.items():
        for ret_key, rd in data['summary']['retention_results'].items():
            logger.info(f"{task:12s} | {ret_key:5s} | {rd['cloud_own_f1_mean']:8.3f} | "
                       f"{rd['scout_f1_mean']:9.3f} | {rd['scout_vs_own_gap']:+7.3f} | "
                       f"{rd['paired_ttest_p']:8.4f}")

    output = {
        'metadata': {
            'experiment': 'scout_multitask',
            'description': 'Scout on SQuAD, HotpotQA, TriviaQA (7B->14B)',
            'edge_model': edge_name,
            'cloud_model': cloud_name,
            'tasks': tasks,
            'num_samples_per_task': NUM_SAMPLES_PER_TASK,
            'retentions': RETENTIONS,
            'seed': 42,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summaries': {k: v['summary'] for k, v in all_results.items()},
        'per_sample': {k: v['per_sample'] for k, v in all_results.items()},
    }

    save_results(output, f'exp_scout_multitask_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
