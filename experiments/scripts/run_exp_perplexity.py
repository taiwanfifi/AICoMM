#!/usr/bin/env python3
"""
Experiment F2: Perplexity Evaluation

Standard WikiText-2 perplexity benchmark for KV-cache quantization schemes.
This is the metric all reviewers expect and was missing from Papers A/B.

Conditions:
  - Full BF16 (baseline)
  - INT8 (symmetric per-channel quantization)
  - INT4 (symmetric per-channel quantization)
  - Mixed-INT4 (sensitive layers at INT8, rest at INT4)

Models: Qwen-7B, Qwen-14B, Mistral-7B, Yi-6B-Chat

Approach: Sliding window perplexity on WikiText-2 test set.
We quantize the KV-cache after prefill, then measure how well the model
predicts the next tokens using the quantized cache.

VRAM: 28 GB peak (14B), sequential model loading
GPU time: ~2 hours
"""

import sys
import time
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    save_results, make_timestamp, get_kv_layer, num_layers, HF_MODELS,
)

import torch
import numpy as np

logger = setup_logging(__name__, 'exp_perplexity.log')


# =========================================================================
# KV-Cache Quantization
# =========================================================================

def quantize_tensor_symmetric(tensor, bits):
    """
    Symmetric per-channel quantization.

    Args:
        tensor: [batch, heads, seq, dim] KV tensor
        bits: 4 or 8

    Returns:
        dequantized tensor (simulated quantization)
    """
    max_val = (1 << (bits - 1)) - 1

    # Per-channel: quantize along last dimension
    abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / max_val

    quantized = (tensor / scale).round().clamp(-max_val, max_val)
    dequantized = quantized * scale

    return dequantized


def identify_sensitive_layers(model, tokenizer, calibration_text, top_pct=0.25):
    """
    Identify sensitive layers for mixed-precision quantization.

    Strategy: layers where INT4 quantization error is highest get INT8.
    Uses a calibration text to measure quantization error per layer.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(calibration_text, return_tensors="pt",
                       max_length=512, truncation=True).to(device)

    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    cache = out.past_key_values
    n_layers = num_layers(cache)

    errors = []
    for layer_idx in range(n_layers):
        k = get_kv_layer(cache, layer_idx, 'key')
        v = get_kv_layer(cache, layer_idx, 'value')

        k_q4 = quantize_tensor_symmetric(k, 4)
        v_q4 = quantize_tensor_symmetric(v, 4)

        k_err = (k - k_q4).abs().mean().item()
        v_err = (v - v_q4).abs().mean().item()
        errors.append(k_err + v_err)

    # Top top_pct layers with highest error are sensitive
    n_sensitive = max(1, int(n_layers * top_pct))
    sensitive_indices = np.argsort(errors)[-n_sensitive:]
    return set(sensitive_indices.tolist())


def quantize_kv_cache(cache, bits, sensitive_layers=None):
    """
    Apply quantization to KV-cache (in-place simulation via dequantized values).

    For mixed-INT4: sensitive_layers get INT8, rest get INT4.
    """
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

        # Write back dequantized values
        if hasattr(cache, 'layers'):
            cache.layers[layer_idx].keys = k_q
            cache.layers[layer_idx].values = v_q
        elif hasattr(cache, 'key_cache'):
            cache.key_cache[layer_idx] = k_q
            cache.value_cache[layer_idx] = v_q


# =========================================================================
# Perplexity computation
# =========================================================================

def compute_sliding_window_ppl(model, tokenizer, text, stride=512, max_length=1024):
    """
    Compute perplexity using sliding window approach.

    Standard method: process text in overlapping windows and average NLL.
    """
    device = next(model.parameters()).device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, prev_end)

        input_window = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_ids=input_window)

        logits = outputs.logits
        # Shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_window[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Only count losses for the non-overlapping part
        offset = target_begin - begin
        if offset > 0:
            token_losses = token_losses[offset - 1:]  # -1 because shift

        nlls.append(token_losses)
        prev_end = end

        if end >= seq_len:
            break

    all_nlls = torch.cat(nlls)
    ppl = torch.exp(all_nlls.mean()).item()
    return ppl, len(all_nlls)


def compute_ppl_with_quantized_kv(model, tokenizer, text, quant_config,
                                   prefix_len=256, eval_len=256):
    """
    Compute perplexity with quantized KV-cache.

    Strategy:
    1. Prefill with prefix_len tokens to build KV-cache
    2. Quantize the KV-cache
    3. Continue generation/evaluation with quantized cache
    4. Measure NLL on the eval_len tokens following the prefix

    This isolates the effect of KV-cache quantization on generation quality.
    """
    device = next(model.parameters()).device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    if seq_len < prefix_len + eval_len:
        # If text is too short, adjust
        prefix_len = min(prefix_len, seq_len // 2)
        eval_len = min(eval_len, seq_len - prefix_len)

    if prefix_len < 10 or eval_len < 10:
        return float('nan'), 0

    prefix_ids = input_ids[:, :prefix_len]
    eval_ids = input_ids[:, prefix_len:prefix_len + eval_len]

    # Step 1: Prefill to build KV-cache
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)

    cache = prefix_out.past_key_values

    # Step 2: Quantize KV-cache
    if quant_config['method'] != 'none':
        quantize_kv_cache(
            cache,
            bits=quant_config.get('bits', 8),
            sensitive_layers=quant_config.get('sensitive_layers'),
        )

    # Step 3: Evaluate NLL on subsequent tokens using quantized cache
    with torch.no_grad():
        eval_out = model(input_ids=eval_ids, past_key_values=cache)

    logits = eval_out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = eval_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    ppl = math.exp(loss.item())
    n_tokens = eval_len - 1
    return ppl, n_tokens


def run_perplexity_eval(model_name: str, model_key: str, texts: list,
                        sensitive_layers: set = None):
    """
    Run perplexity evaluation for one model across all quantization conditions.
    """
    model, tokenizer = load_model(model_name)

    # Identify sensitive layers using first text as calibration
    if sensitive_layers is None:
        logger.info("  Identifying sensitive layers for mixed-INT4...")
        sensitive_layers = identify_sensitive_layers(model, tokenizer, texts[0])
        logger.info(f"  Sensitive layers: {sorted(sensitive_layers)}")

    conditions = [
        {'name': 'BF16', 'method': 'none'},
        {'name': 'INT8', 'method': 'quant', 'bits': 8},
        {'name': 'INT4', 'method': 'quant', 'bits': 4},
        {'name': 'Mixed-INT4', 'method': 'quant', 'bits': 4,
         'sensitive_layers': sensitive_layers},
    ]

    results = {}

    for cond in conditions:
        ppls = []
        total_tokens = 0

        for j, text in enumerate(texts):
            ppl, n_tok = compute_ppl_with_quantized_kv(
                model, tokenizer, text, cond,
                prefix_len=256, eval_len=256,
            )
            if not math.isnan(ppl):
                ppls.append(ppl)
                total_tokens += n_tok

        avg_ppl = float(np.mean(ppls)) if ppls else float('nan')
        std_ppl = float(np.std(ppls)) if ppls else float('nan')

        results[cond['name']] = {
            'mean_ppl': avg_ppl,
            'std_ppl': std_ppl,
            'n_segments': len(ppls),
            'total_tokens': total_tokens,
        }

        logger.info(f"  {cond['name']:12s}: PPL={avg_ppl:.2f} +/- {std_ppl:.2f} "
                    f"({len(ppls)} segments, {total_tokens} tokens)")

    free_model(model)

    return {
        'model': model_name,
        'model_key': model_key,
        'sensitive_layers': sorted(sensitive_layers),
        'conditions': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['7B', '14B', 'Mistral-7B', 'Yi-6B'],
                        help='Model keys to run (e.g. --models 7B 14B)')
    args = parser.parse_args()

    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Experiment F2: Perplexity Evaluation (WikiText-2)")
    logger.info("=" * 70)

    # Load WikiText-2
    from datasets import load_dataset
    wikitext = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Concatenate into segments of ~512 tokens for sliding window
    full_text = "\n".join([t for t in wikitext['text'] if t.strip()])

    # Split into manageable chunks (each ~2048 chars for prefix+eval)
    chunk_size = 4000  # chars, roughly 1000 tokens
    texts = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        if len(chunk) > 500:  # Skip very short chunks
            texts.append(chunk)
    texts = texts[:100]  # Use 100 chunks

    logger.info(f"Using {len(texts)} text segments from WikiText-2")

    start = time.time()
    all_results = {}

    available = {
        '7B': ('Qwen-7B', HF_MODELS['7B']),
        '14B': ('Qwen-14B', HF_MODELS['14B']),
        'Mistral-7B': ('Mistral-7B', HF_MODELS['Mistral-7B']),
        'Yi-6B': ('Yi-6B-Chat', HF_MODELS['Yi-6B']),
    }

    models = [(available[k]) for k in args.models if k in available]
    logger.info(f"Running for models: {[m[0] for m in models]}")

    for key, hf_name in models:
        logger.info(f"\n--- {key} ---")
        try:
            all_results[key] = run_perplexity_eval(hf_name, key, texts)
        except Exception as e:
            logger.error(f"Model {key} FAILED: {e}")

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Build comparison table
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("PERPLEXITY COMPARISON TABLE")
        logger.info(f"{'Model':15s} | {'BF16':>8s} | {'INT8':>8s} | {'INT4':>8s} | {'Mixed':>8s}")
        logger.info("-" * 60)
        for key in all_results:
            r = all_results[key]['conditions']
            logger.info(f"{key:15s} | {r['BF16']['mean_ppl']:8.2f} | {r['INT8']['mean_ppl']:8.2f} | "
                        f"{r['INT4']['mean_ppl']:8.2f} | {r['Mixed-INT4']['mean_ppl']:8.2f}")

        output = {
            'metadata': {
                'experiment': 'perplexity_wikitext2',
                'description': 'WikiText-2 perplexity with KV-cache quantization',
                'n_text_segments': len(texts),
                'timestamp': make_timestamp(),
                'elapsed_minutes': elapsed / 60,
            },
            'results': all_results,
        }

        save_results(output, f'exp_perplexity_{make_timestamp()}.json')


if __name__ == '__main__':
    main()
