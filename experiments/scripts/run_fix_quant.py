#!/usr/bin/env python3
"""
Fix for Paper A unified quantization experiment.

Bug: original run_reviewer_fixes.py passed past_key_values=None to generate(),
completely ignoring the quantized cache. All 4 quant methods gave identical F1.

Fix: manual greedy generation loop using the quantized cache.

Only re-runs quantization part (selection results from EXP 2 are valid).
Only Qwen-7B (14B OOMs on 40GB A100 with output_attentions).
"""

import sys
import os
import time
import math
import json
import gc
from pathlib import Path
from datetime import datetime

os.environ['HF_HOME'] = '/workspace/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/hf_cache/datasets'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    load_squad_samples, format_qa_prompt, get_context_end_pos,
    compute_f1, confidence_interval_95, paired_ttest,
    save_results, make_timestamp, num_layers, get_kv_layer,
    HF_MODELS, RESULTS_DIR,
)

import torch
import numpy as np

logger = setup_logging('fix_quant', 'fix_quant.log')
TIMESTAMP = make_timestamp()


def quantize_tensor_symmetric(tensor, bits):
    max_val = (1 << (bits - 1)) - 1
    abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / max_val
    quantized = (tensor / scale).round().clamp(-max_val, max_val)
    return quantized * scale


def quantize_kv_cache(cache, bits, sensitive_layers=None):
    n = num_layers(cache)
    for layer_idx in range(n):
        if sensitive_layers is not None:
            layer_bits = 8 if layer_idx in sensitive_layers else 4
        else:
            layer_bits = bits

        k = get_kv_layer(cache, layer_idx, 'key')
        v = get_kv_layer(cache, layer_idx, 'value')
        k_q = quantize_tensor_symmetric(k, layer_bits)
        v_q = quantize_tensor_symmetric(v, layer_bits)

        if hasattr(cache, 'layers'):
            cache.layers[layer_idx].keys = k_q
            cache.layers[layer_idx].values = v_q
        elif hasattr(cache, 'key_cache'):
            cache.key_cache[layer_idx] = k_q
            cache.value_cache[layer_idx] = v_q


def identify_sensitive_layers(model, tokenizer, text, top_pct=0.25):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    n = num_layers(cache)
    errors = []
    for layer_idx in range(n):
        k = get_kv_layer(cache, layer_idx, 'key')
        v = get_kv_layer(cache, layer_idx, 'value')
        k_err = (k - quantize_tensor_symmetric(k, 4)).abs().mean().item()
        v_err = (v - quantize_tensor_symmetric(v, 4)).abs().mean().item()
        errors.append(k_err + v_err)
    n_sensitive = max(1, int(n * top_pct))
    return set(np.argsort(errors)[-n_sensitive:].tolist())


def manual_generate(model, tokenizer, input_ids, cache, max_new_tokens=64):
    """
    Greedy generation using a pre-computed (possibly quantized) KV cache.

    This is the FIX: instead of model.generate(past_key_values=None),
    we manually decode token-by-token from the quantized cache.
    """
    # Get logits for the last position from cache
    # We need one forward pass with the full input to get logits aligned with cache
    # But cache already has all positions, so we just need the last token's logit
    with torch.no_grad():
        # Forward the last token position to get logits (cache has everything before it)
        last_token = input_ids[:, -1:]
        out = model(input_ids=last_token, past_key_values=cache, use_cache=True)

    cache = out.past_key_values
    next_logits = out.logits[:, -1, :]

    generated_ids = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == eos_id:
            break
        generated_ids.append(tok_id)
        with torch.no_grad():
            step_out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = step_out.past_key_values
        next_logits = step_out.logits[:, -1, :]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("FIX: Paper A Quantization (manual generation from quantized cache)")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 70)

    samples = load_squad_samples(200, seed=42)
    quant_methods = ['BF16', 'INT8', 'INT4', 'Mixed-INT4']

    model_name = HF_MODELS['7B']
    logger.info(f"Loading {model_name}...")
    model, tokenizer = load_model(model_name)

    # Identify sensitive layers
    logger.info("Identifying sensitive layers...")
    sensitive = identify_sensitive_layers(model, tokenizer, samples[0]['context'])
    logger.info(f"Sensitive layers: {sorted(sensitive)}")

    per_sample = []
    start = time.time()

    for i, s in enumerate(samples):
        prompt = format_qa_prompt(s['context'], s['question'])
        gold = s['gold_answer']

        try:
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024,
                               truncation=True).to(device)
            input_ids = inputs['input_ids']

            # Full baseline (no quantization, normal generate)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            full_ans = tokenizer.decode(gen[0][input_ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            full_f1 = compute_f1(full_ans, gold)

            sr = {
                'sample_idx': i,
                'full_f1': full_f1,
                'quantization': {},
            }

            for qm in quant_methods:
                # Fresh prefill for clean cache (prefill ALL tokens except last)
                with torch.no_grad():
                    out = model(input_ids=input_ids[:, :-1], use_cache=True)
                cache = out.past_key_values

                # Apply quantization
                if qm == 'INT8':
                    quantize_kv_cache(cache, 8)
                elif qm == 'INT4':
                    quantize_kv_cache(cache, 4)
                elif qm == 'Mixed-INT4':
                    quantize_kv_cache(cache, 4, sensitive_layers=sensitive)
                # BF16 = no quantization

                # Generate from quantized cache (THE FIX)
                qans = manual_generate(model, tokenizer, input_ids, cache, max_new_tokens=64)
                qf1 = compute_f1(qans, gold)
                sr['quantization'][qm] = {'f1': qf1, 'answer': qans[:100]}

            per_sample.append(sr)

        except Exception as e:
            logger.error(f"Sample {i}: {e}")
            per_sample.append({'sample_idx': i, 'error': str(e)})

        if (i + 1) % 25 == 0:
            valid = [r for r in per_sample if 'quantization' in r]
            if valid:
                bf16_mean = np.mean([r['quantization']['BF16']['f1'] for r in valid])
                int8_mean = np.mean([r['quantization']['INT8']['f1'] for r in valid])
                int4_mean = np.mean([r['quantization']['INT4']['f1'] for r in valid])
                mix4_mean = np.mean([r['quantization']['Mixed-INT4']['f1'] for r in valid])
                logger.info(f"  [{i+1}/200] BF16={bf16_mean:.3f} INT8={int8_mean:.3f} "
                             f"INT4={int4_mean:.3f} Mixed={mix4_mean:.3f}")

    elapsed = time.time() - start
    free_model(model)

    # Summarize
    valid = [r for r in per_sample if 'quantization' in r]
    summary = {
        'model': model_name,
        'n_valid': len(valid),
        'full_f1_mean': float(np.mean([r['full_f1'] for r in valid])),
        'sensitive_layers': sorted(sensitive),
        'elapsed_minutes': elapsed / 60,
    }

    for qm in quant_methods:
        f1s = [r['quantization'][qm]['f1'] for r in valid]
        summary[f'{qm}_mean'] = float(np.mean(f1s))
        summary[f'{qm}_ci95'] = float(confidence_interval_95(f1s))
        summary[f'{qm}_pct'] = float(np.mean(f1s) / summary['full_f1_mean'] * 100) if summary['full_f1_mean'] > 0 else 0.0

    # Paired t-tests: BF16 vs each quantization
    bf16_f1s = [r['quantization']['BF16']['f1'] for r in valid]
    for qm in ['INT8', 'INT4', 'Mixed-INT4']:
        qf1s = [r['quantization'][qm]['f1'] for r in valid]
        _, p = paired_ttest(bf16_f1s, qf1s)
        summary[f'{qm}_vs_BF16_p'] = float(p)

    logger.info(f"\n{'='*60}")
    logger.info(f"Qwen-7B Quantization Results (n={len(valid)}):")
    logger.info(f"  Full F1: {summary['full_f1_mean']:.3f}")
    for qm in quant_methods:
        pval_str = ""
        if qm != 'BF16':
            pval_str = f"  p={summary[f'{qm}_vs_BF16_p']:.4f}"
        logger.info(f"  {qm:12s}: F1={summary[f'{qm}_mean']:.3f} "
                     f"({summary[f'{qm}_pct']:.1f}%){pval_str}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")

    output = {
        'metadata': {
            'experiment': 'paper_a_quant_fix',
            'description': 'Fixed quantization: manual generation from quantized cache (Qwen-7B, n=200)',
            'bug_fixed': 'original used past_key_values=None in generate(), ignoring quantized cache',
            'model': model_name,
            'n_samples': 200,
            'seed': 42,
            'timestamp': TIMESTAMP,
        },
        'summary': summary,
        'per_sample': per_sample,
    }
    save_results(output, f'exp_paper_a_quant_fix_{TIMESTAMP}.json')
    logger.info("DONE")


if __name__ == '__main__':
    main()
