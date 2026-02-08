#!/usr/bin/env python3
"""
Batch 28: Scout Model Protocol — End-to-End Validation

Core idea: Small edge model (3B) selects positions via Q2C, transmits only
position INDICES to cloud. Cloud model (7B/14B) runs its own prefill, applies
edge's selection mask, and generates answer.

Bandwidth savings: Full KV ~10-30 MB → Position indices ~2 KB (5000x reduction)
But cloud must run its own prefill (~18-57ms, negligible vs TX time).

Experiments:
  1. Same-family scout: Qwen-3B → Qwen-7B, Qwen-3B → Qwen-14B
  2. Position overlap analysis at 25/50/75% retention
  3. End-to-end F1 comparison: cloud-own-selection vs scout-selection vs full-KV
  4. Bandwidth + latency analysis

Target: Paper B — Adaptive Semantic Transport Protocol
"""

import os
import sys
import json
import time
import re
import string
import logging
from pathlib import Path
from datetime import datetime

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/dev/shm/hf_7b'

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch28_scout.log')
    ]
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Utilities
# =========================================================================
def normalize_answer(s):
    """SQuAD v2 normalization."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def compute_f1(pred, gold):
    """Normalized token-F1 (SQuAD v2 style)."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def get_kv_layer(cache, layer_idx, component='key'):
    """Extract key or value tensor from cache object."""
    if hasattr(cache, 'layers'):
        layer = cache.layers[layer_idx]
        return layer.keys if component == 'key' else layer.values
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx] if component == 'key' else cache.value_cache[layer_idx]
    pair = cache[layer_idx]
    return pair[0] if component == 'key' else pair[1]


def num_layers(cache):
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    if hasattr(cache, 'key_cache'):
        return len(cache.key_cache)
    return len(cache)


def compute_q2c_scores(model, tokenizer, prompt, context_end_pos):
    """
    Run prefill and compute Q2C attention scores for context positions.
    Returns: kv_cache, q2c_scores (for context positions only), context_len
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)

    # Get last layer attention: [batch, heads, seq, seq]
    attn = out.attentions[-1][0]  # [heads, seq, seq]
    seq_len = attn.shape[-1]

    # Context positions: 0 to context_end_pos-1
    # Query positions: context_end_pos to seq_len-1
    # Q2C score for position j = sum of attention from query positions to j
    q2c = attn[:, context_end_pos:, :context_end_pos].sum(dim=(0, 1))  # [context_len]

    return out.past_key_values, q2c.float().cpu().numpy(), inputs, seq_len


def select_positions(q2c_scores, retention_ratio):
    """Select top-k positions by Q2C score."""
    n = len(q2c_scores)
    k = max(1, int(n * retention_ratio))
    indices = np.argsort(q2c_scores)[-k:]
    return np.sort(indices)


def generate_with_selection_mask(model, tokenizer, input_ids, selected_positions,
                                  context_len, seq_len, max_new=64):
    """
    Generate answer with attention mask that excludes unselected context positions.

    Uses model.generate() with a 2D attention mask. The model recomputes the
    forward pass with the mask, preserving RoPE encoding for all positions.
    """
    device = next(model.parameters()).device

    # Build attention mask: 1 for selected + query positions, 0 for unselected context
    attn_mask = torch.zeros(1, seq_len, device=device, dtype=torch.long)
    for pos in selected_positions:
        attn_mask[0, pos] = 1
    # All non-context positions (question, instructions, special tokens)
    attn_mask[0, context_len:seq_len] = 1

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
        )

    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


# =========================================================================
# Main Experiment
# =========================================================================
def run_scout_experiment(edge_name, cloud_name, num_samples=50, retentions=[0.75, 0.50, 0.25]):
    """
    Run scout model experiment: edge selects, cloud generates.

    Args:
        edge_name: HuggingFace model name for edge (scout)
        cloud_name: HuggingFace model name for cloud
        num_samples: Number of SQuAD samples
        retentions: List of retention ratios to test
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    edge_short = edge_name.split('/')[-1]
    cloud_short = cloud_name.split('/')[-1]
    logger.info(f"\n{'='*70}")
    logger.info(f"Scout Experiment: {edge_short} → {cloud_short}")
    logger.info(f"{'='*70}")

    # Load dataset
    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]

    # Use deterministic sample selection
    rng = np.random.RandomState(42)
    indices = rng.choice(len(answerable), size=min(num_samples, len(answerable)), replace=False)
    samples = [answerable[i] for i in indices]
    logger.info(f"Using {len(samples)} samples (seed=42)")

    # ---- Phase 1: Edge model (3B) — compute Q2C selections ----
    logger.info(f"\nPhase 1: Loading edge model ({edge_short})...")
    edge_model = AutoModelForCausalLM.from_pretrained(
        edge_name, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    edge_model.eval()
    edge_tok = AutoTokenizer.from_pretrained(edge_name, trust_remote_code=True)
    if edge_tok.pad_token is None:
        edge_tok.pad_token = edge_tok.eos_token

    edge_selections = {}  # {sample_idx: {ret: selected_positions}}
    edge_q2c_scores = []
    edge_baselines = []   # Edge model's own full-KV answers
    prompts = []
    context_end_positions = []

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['answers']['text'][0]

        # Build prompt (simple format matching Paper A experiments)
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)

        # Find where context ends in token space
        context_part = f"Context: {context}\n"
        context_ids = edge_tok.encode(context_part)
        context_end = len(context_ids)
        context_end_positions.append(context_end)

        # Compute Q2C scores
        try:
            kv_cache, q2c, inputs, seq_len = compute_q2c_scores(
                edge_model, edge_tok, prompt, context_end
            )
            edge_q2c_scores.append(q2c)

            # Edge's own full-KV answer (baseline)
            with torch.no_grad():
                gen_ids = edge_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            edge_answer = edge_tok.decode(gen_ids[0][inputs['input_ids'].shape[1]:],
                                          skip_special_tokens=True).strip()
            edge_f1 = compute_f1(edge_answer, gold)
            edge_baselines.append({
                'answer': edge_answer, 'f1': edge_f1, 'gold': gold
            })

            # Compute selections at each retention level
            edge_selections[i] = {}
            for ret in retentions:
                selected = select_positions(q2c, ret)
                edge_selections[i][ret] = selected.tolist()

        except Exception as e:
            logger.error(f"  Sample {i} edge failed: {e}")
            edge_q2c_scores.append(None)
            edge_baselines.append({'answer': '', 'f1': 0.0, 'gold': gold})
            edge_selections[i] = {ret: [] for ret in retentions}

        if (i + 1) % 10 == 0:
            avg_f1 = np.mean([b['f1'] for b in edge_baselines])
            logger.info(f"  Edge [{i+1}/{len(samples)}] avg F1={avg_f1:.3f}")

    # Free edge model
    del edge_model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ---- Phase 2: Cloud model — compute own selections + generate answers ----
    logger.info(f"\nPhase 2: Loading cloud model ({cloud_short})...")
    cloud_model = AutoModelForCausalLM.from_pretrained(
        cloud_name, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    cloud_model.eval()
    cloud_tok = AutoTokenizer.from_pretrained(cloud_name, trust_remote_code=True)
    if cloud_tok.pad_token is None:
        cloud_tok.pad_token = cloud_tok.eos_token

    results = []

    for i, sample in enumerate(samples):
        gold = sample['answers']['text'][0]
        prompt = prompts[i]
        context_end = context_end_positions[i]

        try:
            # Cloud's own Q2C scores
            kv_cache, cloud_q2c, inputs, seq_len = compute_q2c_scores(
                cloud_model, cloud_tok, prompt, context_end
            )

            # Cloud full-KV answer (upper bound)
            with torch.no_grad():
                gen_ids = cloud_model.generate(
                    **inputs, max_new_tokens=64, do_sample=False
                )
            cloud_full_answer = cloud_tok.decode(
                gen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
            ).strip()
            cloud_full_f1 = compute_f1(cloud_full_answer, gold)

            sample_result = {
                'sample_idx': i,
                'gold': gold,
                'edge_f1': edge_baselines[i]['f1'],
                'edge_answer': edge_baselines[i]['answer'],
                'cloud_full_f1': cloud_full_f1,
                'cloud_full_answer': cloud_full_answer,
                'conditions': {}
            }

            for ret in retentions:
                ret_key = f"{int(ret*100)}%"

                # Cloud's own selection
                cloud_selected = select_positions(cloud_q2c, ret)

                # Edge's selection for this sample
                edge_selected = np.array(edge_selections[i].get(ret, []))

                # Position overlap
                if len(cloud_selected) > 0 and len(edge_selected) > 0:
                    overlap = len(set(cloud_selected) & set(edge_selected))
                    overlap_pct = overlap / len(cloud_selected) * 100
                else:
                    overlap_pct = 0.0

                # Generate with cloud's own selection
                cloud_own_answer = generate_with_selection_mask(
                    cloud_model, cloud_tok, inputs['input_ids'],
                    cloud_selected, context_end, seq_len, max_new=64
                )
                cloud_own_f1 = compute_f1(cloud_own_answer, gold)

                # Generate with edge's (scout) selection
                if len(edge_selected) > 0:
                    scout_answer = generate_with_selection_mask(
                        cloud_model, cloud_tok, inputs['input_ids'],
                        edge_selected, context_end, seq_len, max_new=64
                    )
                    scout_f1 = compute_f1(scout_answer, gold)
                else:
                    scout_answer = ""
                    scout_f1 = 0.0

                sample_result['conditions'][ret_key] = {
                    'cloud_own_f1': cloud_own_f1,
                    'cloud_own_answer': cloud_own_answer,
                    'scout_f1': scout_f1,
                    'scout_answer': scout_answer,
                    'overlap_pct': overlap_pct,
                    'n_selected': len(cloud_selected),
                    'context_len': len(cloud_q2c),
                }

            results.append(sample_result)

        except Exception as e:
            logger.error(f"  Sample {i} cloud failed: {e}")
            results.append({
                'sample_idx': i, 'gold': gold,
                'edge_f1': edge_baselines[i]['f1'],
                'cloud_full_f1': 0.0,
                'conditions': {},
                'error': str(e)
            })

        if (i + 1) % 10 == 0:
            valid = [r for r in results if '50%' in r.get('conditions', {})]
            if valid:
                avg_full = np.mean([r['cloud_full_f1'] for r in valid])
                avg_own = np.mean([r['conditions']['50%']['cloud_own_f1'] for r in valid])
                avg_scout = np.mean([r['conditions']['50%']['scout_f1'] for r in valid])
                avg_overlap = np.mean([r['conditions']['50%']['overlap_pct'] for r in valid])
                logger.info(f"  Cloud [{i+1}/{len(samples)}] full={avg_full:.3f} "
                          f"own@50%={avg_own:.3f} scout@50%={avg_scout:.3f} "
                          f"overlap={avg_overlap:.1f}%")

        # Save checkpoint every 10 samples
        if (i + 1) % 10 == 0:
            checkpoint = {
                'metadata': {
                    'edge_model': edge_name,
                    'cloud_model': cloud_name,
                    'num_samples': len(samples),
                    'completed': i + 1,
                    'retentions': retentions,
                    'normalized_f1': True,
                    'seed': 42,
                },
                'per_sample': results,
            }
            ckpt_path = RESULTS_DIR / f'batch28_scout_{edge_short}_{cloud_short}_checkpoint.json'
            with open(ckpt_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

    # Free cloud model
    del cloud_model
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Phase 3: Compute summary statistics ----
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: {edge_short} → {cloud_short}")
    logger.info(f"{'='*50}")

    valid_results = [r for r in results if r.get('conditions')]

    summary = {
        'edge_model': edge_name,
        'cloud_model': cloud_name,
        'num_samples': len(samples),
        'num_valid': len(valid_results),
        'normalized_f1': True,
        'seed': 42,
        'edge_baseline_f1': float(np.mean([r['edge_f1'] for r in valid_results])),
        'cloud_full_f1': float(np.mean([r['cloud_full_f1'] for r in valid_results])),
        'retention_results': {},
    }

    for ret in retentions:
        ret_key = f"{int(ret*100)}%"
        ret_results = [r['conditions'][ret_key] for r in valid_results if ret_key in r.get('conditions', {})]
        if ret_results:
            summary['retention_results'][ret_key] = {
                'cloud_own_f1': float(np.mean([r['cloud_own_f1'] for r in ret_results])),
                'scout_f1': float(np.mean([r['scout_f1'] for r in ret_results])),
                'overlap_pct': float(np.mean([r['overlap_pct'] for r in ret_results])),
                'scout_vs_own_gap': float(
                    np.mean([r['scout_f1'] for r in ret_results]) -
                    np.mean([r['cloud_own_f1'] for r in ret_results])
                ),
            }
            logger.info(f"  {ret_key}: cloud_own={summary['retention_results'][ret_key]['cloud_own_f1']:.3f} "
                       f"scout={summary['retention_results'][ret_key]['scout_f1']:.3f} "
                       f"overlap={summary['retention_results'][ret_key]['overlap_pct']:.1f}% "
                       f"gap={summary['retention_results'][ret_key]['scout_vs_own_gap']:+.3f}")

    logger.info(f"  Edge baseline: {summary['edge_baseline_f1']:.3f}")
    logger.info(f"  Cloud full-KV: {summary['cloud_full_f1']:.3f}")

    # Bandwidth analysis
    # Assume context_len ~ 170 tokens (SQuAD average from Paper A)
    avg_context = np.mean([r['conditions']['50%']['context_len']
                           for r in valid_results if '50%' in r.get('conditions', {})])

    cloud_config = {
        'Qwen2.5-7B': {'layers': 28, 'kv_heads': 4, 'head_dim': 128},
        'Qwen2.5-14B': {'layers': 48, 'kv_heads': 8, 'head_dim': 128},
    }
    cfg = cloud_config.get(cloud_short, cloud_config.get('Qwen2.5-7B'))

    kv_size_bf16 = 2 * cfg['layers'] * cfg['kv_heads'] * avg_context * cfg['head_dim'] * 2
    idx_size_50 = int(avg_context * 0.5) * 4  # int32 indices

    summary['bandwidth_analysis'] = {
        'avg_context_tokens': float(avg_context),
        'full_kv_bf16_bytes': float(kv_size_bf16),
        'full_kv_bf16_mb': float(kv_size_bf16 / 1e6),
        'indices_50pct_bytes': float(idx_size_50),
        'compression_ratio': float(kv_size_bf16 / max(idx_size_50, 1)),
        'tx_time_100mbps_full_ms': float(kv_size_bf16 * 8 / 100e6 * 1000),
        'tx_time_100mbps_idx_ms': float(idx_size_50 * 8 / 100e6 * 1000),
    }

    logger.info(f"\n  Bandwidth Analysis (avg {avg_context:.0f} tokens):")
    logger.info(f"    Full KV (BF16): {summary['bandwidth_analysis']['full_kv_bf16_mb']:.1f} MB")
    logger.info(f"    Position indices (50%): {idx_size_50} bytes")
    logger.info(f"    Compression ratio: {summary['bandwidth_analysis']['compression_ratio']:.0f}x")
    logger.info(f"    TX time @100Mbps: full={summary['bandwidth_analysis']['tx_time_100mbps_full_ms']:.0f}ms "
               f"idx={summary['bandwidth_analysis']['tx_time_100mbps_idx_ms']:.2f}ms")

    # Save final results
    output = {
        'metadata': summary,
        'per_sample': results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f'batch28_scout_{edge_short}_{cloud_short}_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n[SAVED] → {result_path}")

    return summary


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Torch: {torch.__version__}")
    logger.info(f"Start time: {datetime.now()}")

    start = time.time()

    # Experiment 1: Qwen-3B → Qwen-7B (same family, same tokenizer)
    summary_3b_7b = run_scout_experiment(
        edge_name="Qwen/Qwen2.5-3B",
        cloud_name="Qwen/Qwen2.5-7B",
        num_samples=50,
        retentions=[0.75, 0.50, 0.25],
    )

    # Experiment 2: Qwen-3B → Qwen-14B (same family, bigger gap)
    summary_3b_14b = run_scout_experiment(
        edge_name="Qwen/Qwen2.5-3B",
        cloud_name="Qwen/Qwen2.5-14B",
        num_samples=50,
        retentions=[0.75, 0.50, 0.25],
    )

    # Experiment 3: Qwen-7B → Qwen-14B (smaller gap, should have higher overlap)
    summary_7b_14b = run_scout_experiment(
        edge_name="Qwen/Qwen2.5-7B",
        cloud_name="Qwen/Qwen2.5-14B",
        num_samples=50,
        retentions=[0.75, 0.50, 0.25],
    )

    elapsed = time.time() - start
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL BATCH 28 DONE in {elapsed/60:.1f} minutes")
    logger.info(f"{'='*70}")

    # Print combined summary
    for name, s in [("3B→7B", summary_3b_7b), ("3B→14B", summary_3b_14b), ("7B→14B", summary_7b_14b)]:
        logger.info(f"\n{name}:")
        logger.info(f"  Edge baseline: {s['edge_baseline_f1']:.3f}")
        logger.info(f"  Cloud full-KV: {s['cloud_full_f1']:.3f}")
        for ret_key, ret_data in s['retention_results'].items():
            logger.info(f"  {ret_key}: own={ret_data['cloud_own_f1']:.3f} "
                       f"scout={ret_data['scout_f1']:.3f} "
                       f"overlap={ret_data['overlap_pct']:.1f}%")
