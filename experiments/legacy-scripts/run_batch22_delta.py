#!/usr/bin/env python3
"""
Batch 22: Delta Encoding Analysis — Directly Addresses CacheGen's Core Claim

CacheGen (SIGCOMM'24) finds adjacent tokens have 2.4-2.9x lower variance as deltas.
They use: Delta Encode → Layer-wise Quantize → Arithmetic Code.
We use: Q2C Select → Quantize.

KEY QUESTION: How much ADDITIONAL compression does delta encoding provide?
And does Q2C selection + delta outperform CacheGen's approach?

Experiments:
1. Measure delta variance reduction (replicate CacheGen's finding)
2. Compare compression effectiveness: direct quant vs delta+quant
3. Test Q2C + delta encoding (novel pipeline)
4. Per-layer analysis of delta effectiveness

Model: Qwen2.5-7B (our most-studied model)
"""
import os, sys, json, time, logging, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/dev/shm/hf_7b'
os.environ['HF_DATASETS_CACHE'] = '/dev/shm/hf_7b/datasets'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch22.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'Qwen/Qwen2.5-7B'
NUM_SAMPLES = 30


def compute_f1(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not gold_tokens: return 1.0 if not pred_tokens else 0.0
    if not pred_tokens: return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def quantize_pertoken(t, bits):
    if bits >= 16: return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return (t / scale).round().clamp(qmin, qmax) * scale


def delta_encode(t):
    """Delta encoding: first token is anchor, rest are deltas from previous.
    t shape: (batch, heads, seq, dim)"""
    deltas = torch.zeros_like(t)
    deltas[:, :, 0, :] = t[:, :, 0, :]  # anchor
    deltas[:, :, 1:, :] = t[:, :, 1:, :] - t[:, :, :-1, :]  # deltas
    return deltas


def delta_decode(deltas):
    """Reconstruct from delta encoding."""
    t = torch.zeros_like(deltas)
    t[:, :, 0, :] = deltas[:, :, 0, :]
    for i in range(1, deltas.shape[2]):
        t[:, :, i, :] = t[:, :, i-1, :] + deltas[:, :, i, :]
    return t


def measure_delta_stats(kv_cache, num_layers):
    """Measure variance reduction from delta encoding (CacheGen's core claim)."""
    stats = []
    for li in range(num_layers):
        layer = kv_cache.layers[li]
        keys = layer.keys.clone()   # (batch, heads, seq, dim)
        values = layer.values.clone()

        for name, tensor in [('keys', keys), ('values', values)]:
            # Original variance
            orig_var = tensor.var().item()
            # Delta variance
            delta = delta_encode(tensor)
            delta_var = delta[:, :, 1:, :].var().item()  # exclude anchor
            # Variance reduction
            ratio = orig_var / max(delta_var, 1e-12)

            stats.append({
                'layer': li,
                'type': name,
                'orig_var': orig_var,
                'delta_var': delta_var,
                'variance_reduction': ratio,
            })

    return stats


def quantize_with_delta(tensor, bits):
    """Delta encode → quantize deltas → delta decode."""
    deltas = delta_encode(tensor)
    q_deltas = quantize_pertoken(deltas, bits)
    return delta_decode(q_deltas)


def manual_generate(model, tokenizer, past_kv, first_token_id, seq_len, max_new=64):
    generated = [first_token_id]
    cur_len = seq_len
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id: break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_with_compression(model, tokenizer, input_ids, seq_len, method, bits=4, max_new=64):
    """Generate with various compression methods applied to KV cache.

    Methods:
    - 'none': FP16 baseline
    - 'quant': Direct quantization
    - 'delta_quant': Delta encode → quantize → delta decode
    - 'mixed_quant': L0 FP16 + rest INT4
    - 'mixed_delta_quant': L0 FP16 + rest (delta + INT4)
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)

    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    for li in range(len(pkv.layers)):
        layer = pkv.layers[li]

        if method == 'none':
            pass  # FP16

        elif method == 'quant':
            layer.keys.copy_(quantize_pertoken(layer.keys, bits))
            layer.values.copy_(quantize_pertoken(layer.values, bits))

        elif method == 'delta_quant':
            layer.keys.copy_(quantize_with_delta(layer.keys, bits))
            layer.values.copy_(quantize_with_delta(layer.values, bits))

        elif method == 'mixed_quant':
            if li > 0:
                layer.keys.copy_(quantize_pertoken(layer.keys, bits))
                layer.values.copy_(quantize_pertoken(layer.values, bits))

        elif method == 'mixed_delta_quant':
            if li > 0:
                layer.keys.copy_(quantize_with_delta(layer.keys, bits))
                layer.values.copy_(quantize_with_delta(layer.values, bits))

    return manual_generate(model, tokenizer, pkv, first_token_id, seq_len, max_new)


def compute_compression_ratio(tensor, bits, use_delta=False):
    """Estimate effective compression ratio.
    Measures entropy of quantized values to estimate arithmetic coding potential."""
    if use_delta:
        work = delta_encode(tensor)
    else:
        work = tensor.clone()

    # Quantize
    q = quantize_pertoken(work, bits)

    # Estimate entropy (bits per element after arithmetic coding)
    flat = q.flatten().cpu().numpy()
    # Discretize to unique values
    unique, counts = np.unique(np.round(flat, decimals=4), return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    # Original: 16 bits per element
    # Quantized: bits per element
    # After entropy coding: entropy bits per element
    return {
        'original_bpe': 16.0,
        'quantized_bpe': float(bits),
        'entropy_bpe': float(entropy),
        'quant_ratio': 16.0 / bits,
        'entropy_ratio': 16.0 / max(entropy, 0.1),
        'n_unique_values': len(unique),
    }


def prepare_samples(tokenizer, num_samples=30):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    samples = []
    for s in ds:
        if not s['answers']['text']: continue
        ctx = s['context']
        q = s['question']
        prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 400:
            samples.append({
                'prompt': prompt,
                'gold': s['answers']['text'][0],
                'question': q[:100],
                'tok_len': tok_len,
            })
        if len(samples) >= num_samples * 3:
            break
    np.random.seed(42)
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in indices]


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {num_layers} layers")

    samples = prepare_samples(tokenizer, NUM_SAMPLES)
    logger.info(f"Prepared {len(samples)} samples")

    all_results = []
    all_delta_stats = []
    all_compression_stats = []

    methods = ['none', 'quant', 'delta_quant', 'mixed_quant', 'mixed_delta_quant']

    for i, s in enumerate(samples):
        t0 = time.time()
        inputs = tokenizer(s['prompt'], return_tensors='pt', truncation=True, max_length=512).to('cuda')
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        gold = s['gold']

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # Test each compression method at INT4
        for method in methods:
            ans = generate_with_compression(model, tokenizer, input_ids, seq_len, method, bits=4)
            result[method] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # Also test delta_quant at INT8 and INT3
        for bits in [3, 8]:
            ans = generate_with_compression(model, tokenizer, input_ids, seq_len, 'delta_quant', bits=bits)
            result[f'delta_int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}
            ans = generate_with_compression(model, tokenizer, input_ids, seq_len, 'quant', bits=bits)
            result[f'direct_int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # Measure delta stats and compression ratio (first 5 samples only)
        if i < 5:
            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values

            delta_stats = measure_delta_stats(pkv, num_layers)
            all_delta_stats.extend(delta_stats)

            # Compression ratio for layer 0 and layer 14
            for li in [0, num_layers // 2, num_layers - 1]:
                layer = pkv.layers[li]
                for name, tensor in [('keys', layer.keys), ('values', layer.values)]:
                    for use_delta in [False, True]:
                        for bits in [4, 8]:
                            cr = compute_compression_ratio(tensor, bits, use_delta)
                            cr['layer'] = li
                            cr['type'] = name
                            cr['use_delta'] = use_delta
                            cr['bits'] = bits
                            cr['sample_idx'] = i
                            all_compression_stats.append(cr)

            del out, pkv
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        result['time'] = elapsed
        all_results.append(result)

        fp16 = result['none']['f1']
        q4 = result['quant']['f1']
        dq4 = result['delta_quant']['f1']
        mq4 = result['mixed_quant']['f1']
        mdq4 = result['mixed_delta_quant']['f1']
        logger.info(f"  [{i+1}/{len(samples)}] full={fp16:.3f} quant4={q4:.3f} "
                    f"delta4={dq4:.3f} mixed4={mq4:.3f} mix_delta4={mdq4:.3f} ({elapsed:.1f}s)")

    # Summary
    logger.info(f"\n{'#'*70}")
    logger.info(f"DELTA ENCODING ANALYSIS SUMMARY (Qwen2.5-7B, {len(all_results)} samples)")
    logger.info(f"{'#'*70}")

    for method in methods + ['delta_int3', 'direct_int3', 'delta_int8', 'direct_int8']:
        vals = [r[method]['f1'] for r in all_results if method in r]
        if vals:
            mean_f1 = float(np.mean(vals))
            baseline = float(np.mean([r['none']['f1'] for r in all_results]))
            logger.info(f"  {method:25s}: {mean_f1:.4f} ({mean_f1/baseline*100:.1f}%)")

    # Delta stats summary
    if all_delta_stats:
        logger.info(f"\nDelta Variance Reduction by Layer:")
        for name in ['keys', 'values']:
            type_stats = [s for s in all_delta_stats if s['type'] == name]
            by_layer = {}
            for s in type_stats:
                by_layer.setdefault(s['layer'], []).append(s['variance_reduction'])
            logger.info(f"\n  {name}:")
            for li in sorted(by_layer.keys()):
                mean_ratio = float(np.mean(by_layer[li]))
                logger.info(f"    Layer {li:2d}: {mean_ratio:.2f}x reduction")

    # Compression stats summary
    if all_compression_stats:
        logger.info(f"\nEffective Compression (entropy-based):")
        for bits in [4, 8]:
            for use_delta in [False, True]:
                subset = [s for s in all_compression_stats
                         if s['bits'] == bits and s['use_delta'] == use_delta]
                if subset:
                    mean_entropy = float(np.mean([s['entropy_bpe'] for s in subset]))
                    mean_ratio = float(np.mean([s['entropy_ratio'] for s in subset]))
                    label = f"INT{bits}" + ("+delta" if use_delta else "")
                    logger.info(f"  {label:15s}: {mean_entropy:.2f} bpe → {mean_ratio:.1f}x compression")

    # Save
    combined = {
        'metadata': {
            'model': 'Qwen2.5-7B',
            'task': 'SQuAD-v2',
            'experiment': 'delta_encoding_analysis',
            'num_samples': len(all_results),
        },
        'results': all_results,
        'delta_stats': all_delta_stats,
        'compression_stats': all_compression_stats,
    }
    final_path = RESULTS_DIR / f'delta_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"\n[SAVED] -> {final_path}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"Batch 22 COMPLETE in {elapsed:.1f} minutes")

    del model; torch.cuda.empty_cache(); gc.collect()
