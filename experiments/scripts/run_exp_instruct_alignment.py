#!/usr/bin/env python3
"""
Experiment E3: Instruction-Tuned Model Alignment

Test if attention alignment holds for instruction-tuned (chat) models,
which are more production-relevant than base models.

Model pair: Qwen2.5-7B-Instruct → Qwen2.5-14B-Instruct
Dataset: SQuAD v2, n=200 (same samples as base model experiment S1 for comparison)
Prompt format: ChatML (system + user message with context + question)

Metrics:
  - Position overlap (instruct vs instruct, same as base experiment)
  - Scout F1 (instruct models)
  - Cross-comparison: instruct overlap vs base overlap
  - Paired t-test with Bonferroni correction

VRAM: 28 GB peak (14B), sequential model loading
GPU time: ~3-4 hours on A100
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, get_context_end_pos,
    compute_q2c_last_layer, select_positions, generate_with_mask,
    compute_f1, confidence_interval_95, paired_ttest,
    bonferroni_correction, overlap_percentage,
    save_results, make_timestamp, SEED,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_instruct_alignment.log')

NUM_SAMPLES = 200
RETENTIONS = [0.75, 0.50, 0.25]

# Instruct model names
HF_MODELS_INSTRUCT = {
    '7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
    '14B-Instruct': 'Qwen/Qwen2.5-14B-Instruct',
}


# =========================================================================
# ChatML prompt formatting
# =========================================================================

def format_qa_prompt_instruct(tokenizer, context: str, question: str) -> str:
    """Format a QA prompt using ChatML for Qwen2.5-Instruct models."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the given context. Give only the answer, nothing else."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def get_context_end_pos_instruct(tokenizer, context: str) -> int:
    """
    Get the token position where context ends in the ChatML-formatted prompt.

    The context is inside the user message. We tokenize everything up to and
    including the context text to find where it ends.
    """
    # Build the prefix that includes everything up to the context
    # ChatML format: <|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\nContext: {context}\n
    prefix_text = f"Context: {context}\n"

    # Build the full prefix including system message
    messages_prefix = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the given context. Give only the answer, nothing else."},
        {"role": "user", "content": prefix_text},
    ]
    # Apply chat template to get the formatted prefix
    full_prefix = tokenizer.apply_chat_template(
        messages_prefix, tokenize=False, add_generation_prompt=False
    )

    # The context ends at the end of the "Context: {context}\n" part within the user message
    # Tokenize everything up to "Question:"
    messages_context = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the given context. Give only the answer, nothing else."},
    ]
    sys_formatted = tokenizer.apply_chat_template(
        messages_context, tokenize=False, add_generation_prompt=False
    )

    # Tokenize: system part + user start + "Context: {context}\n"
    # Since chat template formats differ, we use a marker approach
    marker = "<<CONTEXT_END_MARKER>>"
    messages_with_marker = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the given context. Give only the answer, nothing else."},
        {"role": "user", "content": f"Context: {context}\n{marker}Question: placeholder"},
    ]
    full_with_marker = tokenizer.apply_chat_template(
        messages_with_marker, tokenize=False, add_generation_prompt=False
    )

    # Find the marker position and tokenize everything before it
    marker_pos = full_with_marker.find(marker)
    if marker_pos == -1:
        # Fallback: use a simpler estimation
        logger.warning("Marker not found, using fallback context_end estimation")
        context_part = f"Context: {context}\n"
        return len(tokenizer.encode(context_part))

    prefix_text = full_with_marker[:marker_pos]
    context_end_pos = len(tokenizer.encode(prefix_text))
    return context_end_pos


# =========================================================================
# Main experiment
# =========================================================================

def run_instruct_scout(edge_name, cloud_name, samples):
    """
    Run scout experiment with instruction-tuned models.

    Same structure as run_exp_scout_n200.py but with ChatML prompts.
    """
    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]

    logger.info(f"\n{'='*70}")
    logger.info(f"Instruct Scout: {edge_short} -> {cloud_short}")
    logger.info(f"{'='*70}")

    # ---- Phase 1: Edge model ----
    logger.info(f"\nPhase 1: Edge model ({edge_short})")
    edge_model, edge_tok = load_model(edge_name)

    edge_selections = {}
    edge_baselines = []
    prompts = []
    context_ends = []

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['gold_answer']

        prompt = format_qa_prompt_instruct(edge_tok, context, question)
        prompts.append(prompt)
        context_end = get_context_end_pos_instruct(edge_tok, context)
        context_ends.append(context_end)

        try:
            device = next(edge_model.parameters()).device
            inputs = edge_tok(prompt, return_tensors="pt",
                              max_length=1024, truncation=True).to(device)

            with torch.no_grad():
                out = edge_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)
            q2c = compute_q2c_last_layer(out.attentions, ce)

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

            del out
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  Edge sample {i} failed: {e}")
            edge_baselines.append({'answer': '', 'f1': 0.0})
            edge_selections[i] = {r: [] for r in RETENTIONS}

        if (i + 1) % 25 == 0:
            avg_f1 = np.mean([b['f1'] for b in edge_baselines])
            logger.info(f"  Edge [{i+1}/{len(samples)}] avg F1={avg_f1:.3f}")

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
                               max_length=1024, truncation=True).to(device)

            with torch.no_grad():
                out = cloud_model(**inputs, use_cache=True, output_attentions=True)

            seq_len = out.attentions[-1].shape[-1]
            ce = min(context_end, seq_len - 1)
            cloud_q2c = compute_q2c_last_layer(out.attentions, ce)

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
                'edge_f1': edge_baselines[i]['f1'],
                'cloud_full_f1': cloud_full_f1,
                'conditions': {},
            }

            for ret in RETENTIONS:
                ret_key = f"{int(ret*100)}%"

                cloud_selected = select_positions(cloud_q2c, ret)
                edge_selected = np.array(edge_selections[i].get(ret, []))

                overlap = 0.0
                if len(cloud_selected) > 0 and len(edge_selected) > 0:
                    overlap = overlap_percentage(cloud_selected, edge_selected)

                # Generate with cloud's own selection
                cloud_own_answer = generate_with_mask(
                    cloud_model, cloud_tok, inputs['input_ids'],
                    cloud_selected, ce, seq_len
                )
                cloud_own_f1 = compute_f1(cloud_own_answer, gold)

                # Generate with scout selection
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
                logger.info(f"  Cloud [{i+1}/{len(samples)}] own@50%={avg_own:.3f} "
                           f"scout@50%={avg_scout:.3f}")

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
        'model_type': 'instruct',
        'num_samples': len(samples),
        'num_valid': len(valid),
        'seed': SEED,
        'edge_baseline_f1': float(np.mean([r['edge_f1'] for r in valid])),
        'cloud_full_f1': float(np.mean([r['cloud_full_f1'] for r in valid])),
        'cloud_full_f1_ci95': float(confidence_interval_95([r['cloud_full_f1'] for r in valid])),
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

    # Bonferroni correction
    if p_values:
        bonf_alpha = bonferroni_correction(p_values)
        summary['bonferroni_alpha'] = float(bonf_alpha)
        for i, ret in enumerate(RETENTIONS):
            ret_key = f"{int(ret*100)}%"
            if ret_key in summary['retention_results']:
                summary['retention_results'][ret_key]['significant_bonferroni'] = bool(
                    p_values[i] < bonf_alpha
                )

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {edge_short} -> {cloud_short} (Instruct)")
    logger.info(f"  Edge baseline F1: {summary['edge_baseline_f1']:.3f}")
    logger.info(f"  Cloud full-KV F1: {summary['cloud_full_f1']:.3f}")
    for ret_key, rd in summary['retention_results'].items():
        sig = "*" if rd.get('significant_bonferroni') else ""
        logger.info(f"  {ret_key}: own={rd['cloud_own_f1_mean']:.3f} "
                    f"scout={rd['scout_f1_mean']:.3f} "
                    f"gap={rd['scout_vs_own_gap']:+.3f} "
                    f"p={rd['paired_ttest_p']:.4f}{sig} "
                    f"overlap={rd['overlap_pct_mean']:.1f}%")

    # Cross-reference with base model results
    logger.info(f"\n  === Comparison with base model (from memory) ===")
    logger.info(f"  Base 7B→14B @75%: overlap≈84%, scout improves (+0.088 @50%)")
    logger.info(f"  Instruct 7B→14B @75%: overlap="
                f"{summary['retention_results'].get('75%', {}).get('overlap_pct_mean', 0):.1f}%")

    return {'summary': summary, 'per_sample': results}


def _save_checkpoint(edge_name, cloud_name, results, completed, total):
    """Save intermediate checkpoint."""
    from exp_utils import RESULTS_DIR
    ckpt = {
        'metadata': {'edge': edge_name, 'cloud': cloud_name,
                     'experiment': 'instruct_alignment',
                     'completed': completed, 'total': total},
        'per_sample': results,
    }
    path = RESULTS_DIR / 'exp_instruct_alignment_checkpoint.json'
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=str)


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info(f"Experiment E3: Instruction-Tuned Alignment (n={NUM_SAMPLES})")
    logger.info("=" * 70)

    # Same samples as base model experiment (seed=42) for direct comparison
    samples = load_squad_samples(NUM_SAMPLES, seed=SEED)
    start = time.time()

    edge_name = HF_MODELS_INSTRUCT['7B-Instruct']
    cloud_name = HF_MODELS_INSTRUCT['14B-Instruct']
    result = run_instruct_scout(edge_name, cloud_name, samples)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    output = {
        'metadata': {
            'experiment': 'instruct_alignment',
            'description': 'Scout with Qwen2.5-Instruct models (7B→14B) on SQuAD v2',
            'model_type': 'instruct',
            'edge_model': edge_name,
            'cloud_model': cloud_name,
            'num_samples': NUM_SAMPLES,
            'retentions': RETENTIONS,
            'seed': SEED,
            'timestamp': make_timestamp(),
            'elapsed_minutes': elapsed / 60,
        },
        'summary': result['summary'],
        'per_sample': result['per_sample'],
    }

    save_results(output, f'exp_instruct_alignment_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
