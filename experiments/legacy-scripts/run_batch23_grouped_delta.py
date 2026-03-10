#!/usr/bin/env python3
"""
Batch 23: Grouped Delta Encoding — Fair CacheGen Comparison

Batch 22 showed sequential delta + INT4 = catastrophic (12% F1).
BUT CacheGen uses GROUPED delta (10-token groups, anchor per group).
This limits error accumulation to within-group (max 9 steps).

This batch tests CacheGen's ACTUAL approach:
1. Sequential delta (batch 22 confirmed: catastrophic)
2. Grouped delta (group_size=10, CacheGen's default)
3. Grouped delta (group_size=4, more anchors)
4. Anchor delta (each token relative to GROUP ANCHOR, not sequential)
   - This is CacheGen's actual method: delta from anchor, NOT from previous token
   - Zero error accumulation — each delta is independent

The key distinction:
- Sequential: t[i] = t[i-1] + delta[i]  → error accumulates over ENTIRE sequence
- Grouped sequential: t[i] = t[i-1] + delta[i], reset at group boundary → error accumulates over GROUP
- Anchor-based: t[i] = anchor + delta[i]  → NO error accumulation

Model: Qwen2.5-7B (to compare with batch 22)
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch23.log')])
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


# ===== Delta Encoding Variants =====

def sequential_delta_encode(t):
    """Sequential delta: each position is delta from previous.
    Error accumulates over entire sequence."""
    deltas = torch.zeros_like(t)
    deltas[:, :, 0, :] = t[:, :, 0, :]  # anchor
    deltas[:, :, 1:, :] = t[:, :, 1:, :] - t[:, :, :-1, :]
    return deltas

def sequential_delta_decode(deltas):
    """Reconstruct from sequential deltas (cumulative sum)."""
    return deltas.cumsum(dim=2)


def grouped_sequential_delta_encode(t, group_size=10):
    """Grouped sequential delta: reset anchor every group_size positions.
    Error accumulates only within group (max group_size-1 steps)."""
    B, H, S, D = t.shape
    deltas = torch.zeros_like(t)
    for start in range(0, S, group_size):
        end = min(start + group_size, S)
        deltas[:, :, start, :] = t[:, :, start, :]  # group anchor
        if end > start + 1:
            deltas[:, :, start+1:end, :] = t[:, :, start+1:end, :] - t[:, :, start:end-1, :]
    return deltas

def grouped_sequential_delta_decode(deltas, group_size=10):
    """Reconstruct from grouped sequential deltas."""
    B, H, S, D = deltas.shape
    t = torch.zeros_like(deltas)
    for start in range(0, S, group_size):
        end = min(start + group_size, S)
        t[:, :, start, :] = deltas[:, :, start, :]  # anchor
        for i in range(start + 1, end):
            t[:, :, i, :] = t[:, :, i-1, :] + deltas[:, :, i, :]
    return t


def anchor_delta_encode(t, group_size=10):
    """Anchor-based delta: each position is delta from GROUP ANCHOR.
    This is CacheGen's actual method. ZERO error accumulation.
    Each delta is independent — quantization error doesn't propagate."""
    B, H, S, D = t.shape
    deltas = torch.zeros_like(t)
    for start in range(0, S, group_size):
        end = min(start + group_size, S)
        anchor = t[:, :, start:start+1, :]  # (B, H, 1, D)
        deltas[:, :, start, :] = anchor.squeeze(2)  # store anchor as-is
        if end > start + 1:
            deltas[:, :, start+1:end, :] = t[:, :, start+1:end, :] - anchor
    return deltas

def anchor_delta_decode(deltas, group_size=10):
    """Reconstruct from anchor-based deltas.
    Each position = anchor + delta. Independent, no accumulation."""
    B, H, S, D = deltas.shape
    t = torch.zeros_like(deltas)
    for start in range(0, S, group_size):
        end = min(start + group_size, S)
        anchor = deltas[:, :, start:start+1, :]  # (B, H, 1, D)
        t[:, :, start, :] = anchor.squeeze(2)
        if end > start + 1:
            t[:, :, start+1:end, :] = anchor + deltas[:, :, start+1:end, :]
    return t


# ===== Apply compression to KV cache =====

def apply_delta_quant(tensor, bits, delta_method, group_size=10):
    """Apply delta encoding + quantization + delta decoding."""
    if delta_method == 'none':
        return quantize_pertoken(tensor, bits)

    elif delta_method == 'sequential':
        deltas = sequential_delta_encode(tensor)
        q_deltas = quantize_pertoken(deltas, bits)
        return sequential_delta_decode(q_deltas)

    elif delta_method == 'grouped_seq':
        deltas = grouped_sequential_delta_encode(tensor, group_size)
        q_deltas = quantize_pertoken(deltas, bits)
        return grouped_sequential_delta_decode(q_deltas, group_size)

    elif delta_method == 'anchor':
        deltas = anchor_delta_encode(tensor, group_size)
        q_deltas = quantize_pertoken(deltas, bits)
        return anchor_delta_decode(q_deltas, group_size)

    else:
        raise ValueError(f"Unknown delta method: {delta_method}")


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


def generate_with_delta_compression(model, tokenizer, input_ids, seq_len,
                                      delta_method, bits=4, group_size=10,
                                      protect_l0=False, max_new=64):
    """Generate with delta encoding + quantization applied to KV cache."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)

    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    for li in range(len(pkv.layers)):
        layer = pkv.layers[li]

        if protect_l0 and li == 0:
            continue  # keep Layer 0 at FP16

        layer.keys.copy_(apply_delta_quant(layer.keys, bits, delta_method, group_size))
        layer.values.copy_(apply_delta_quant(layer.values, bits, delta_method, group_size))

    return manual_generate(model, tokenizer, pkv, first_token_id, seq_len, max_new)


def measure_variance_reduction(tensor, delta_method, group_size=10):
    """Measure variance reduction for a given delta method."""
    orig_var = tensor.var().item()

    if delta_method == 'sequential':
        deltas = sequential_delta_encode(tensor)
        # Exclude anchors
        delta_vals = deltas[:, :, 1:, :]
    elif delta_method == 'grouped_seq':
        deltas = grouped_sequential_delta_encode(tensor, group_size)
        # Exclude anchors (every group_size positions)
        mask = torch.ones(tensor.shape[2], dtype=torch.bool)
        for start in range(0, tensor.shape[2], group_size):
            mask[start] = False
        delta_vals = deltas[:, :, mask, :]
    elif delta_method == 'anchor':
        deltas = anchor_delta_encode(tensor, group_size)
        mask = torch.ones(tensor.shape[2], dtype=torch.bool)
        for start in range(0, tensor.shape[2], group_size):
            mask[start] = False
        delta_vals = deltas[:, :, mask, :]
    else:
        return {'orig_var': orig_var, 'delta_var': orig_var, 'reduction': 1.0}

    delta_var = delta_vals.var().item()
    return {
        'orig_var': orig_var,
        'delta_var': delta_var,
        'reduction': orig_var / max(delta_var, 1e-12)
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
    all_variance_stats = []

    # Test configurations: (name, delta_method, bits, group_size, protect_l0)
    configs = [
        ('fp16_baseline',     'none',         16, 10, False),
        ('direct_int4',       'none',         4,  10, False),
        ('direct_int8',       'none',         8,  10, False),
        ('seq_delta_int4',    'sequential',   4,  10, False),  # batch 22 replicate
        ('seq_delta_int8',    'sequential',   8,  10, False),
        ('grp10_seq_int4',    'grouped_seq',  4,  10, False),  # CacheGen-like (sequential within groups)
        ('grp10_seq_int8',    'grouped_seq',  8,  10, False),
        ('grp4_seq_int4',     'grouped_seq',  4,   4, False),  # smaller groups
        ('grp4_seq_int8',     'grouped_seq',  8,   4, False),
        ('anchor10_int4',     'anchor',       4,  10, False),  # CacheGen actual (delta from anchor)
        ('anchor10_int8',     'anchor',       8,  10, False),
        ('anchor4_int4',      'anchor',       4,   4, False),  # more anchors
        ('anchor4_int8',      'anchor',       8,   4, False),
        # With Layer 0 protection
        ('mixed_direct_int4', 'none',         4,  10, True),
        ('mixed_anchor10_int4', 'anchor',     4,  10, True),   # best combo?
        ('mixed_grp10_int4',  'grouped_seq',  4,  10, True),
    ]

    for i, s in enumerate(samples):
        t0 = time.time()
        inputs = tokenizer(s['prompt'], return_tensors='pt', truncation=True, max_length=512).to('cuda')
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        gold = s['gold']

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        for name, delta_method, bits, group_size, protect_l0 in configs:
            ans = generate_with_delta_compression(
                model, tokenizer, input_ids, seq_len,
                delta_method=delta_method, bits=bits,
                group_size=group_size, protect_l0=protect_l0)
            f1 = compute_f1(ans, gold)
            result[name] = {'answer': ans[:200], 'f1': f1}

        # Variance stats (first 3 samples only)
        if i < 3:
            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values

            for li in [0, num_layers // 2, num_layers - 1]:
                layer = pkv.layers[li]
                for tensor_name, tensor in [('keys', layer.keys), ('values', layer.values)]:
                    for delta_method in ['sequential', 'grouped_seq', 'anchor']:
                        for gs in [4, 10]:
                            stats = measure_variance_reduction(tensor, delta_method, gs)
                            stats.update({
                                'sample': i, 'layer': li, 'type': tensor_name,
                                'method': delta_method, 'group_size': gs
                            })
                            all_variance_stats.append(stats)

            del out, pkv
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        result['time'] = elapsed
        all_results.append(result)

        fp16 = result['fp16_baseline']['f1']
        d4 = result['direct_int4']['f1']
        sq4 = result['seq_delta_int4']['f1']
        a10_4 = result['anchor10_int4']['f1']
        g10_4 = result['grp10_seq_int4']['f1']
        logger.info(f"  [{i+1}/{len(samples)}] fp16={fp16:.3f} direct4={d4:.3f} "
                    f"seq_d4={sq4:.3f} anchor10_4={a10_4:.3f} grp10_4={g10_4:.3f} ({elapsed:.1f}s)")

    # Summary
    logger.info(f"\n{'#'*70}")
    logger.info(f"GROUPED DELTA ENCODING ANALYSIS (Qwen2.5-7B, {len(all_results)} samples)")
    logger.info(f"{'#'*70}")

    baseline_f1 = float(np.mean([r['fp16_baseline']['f1'] for r in all_results]))

    for name, _, _, _, _ in configs:
        vals = [r[name]['f1'] for r in all_results if name in r]
        if vals:
            mean_f1 = float(np.mean(vals))
            pct = mean_f1 / baseline_f1 * 100 if baseline_f1 > 0 else 0
            logger.info(f"  {name:25s}: {mean_f1:.4f} ({pct:.1f}%)")

    # Variance stats summary
    if all_variance_stats:
        logger.info(f"\nVariance Reduction Comparison:")
        for delta_method in ['sequential', 'grouped_seq', 'anchor']:
            for gs in [4, 10]:
                subset = [s for s in all_variance_stats
                         if s['method'] == delta_method and s['group_size'] == gs]
                if subset:
                    mean_red = float(np.mean([s['reduction'] for s in subset]))
                    key_red = float(np.mean([s['reduction'] for s in subset if s['type'] == 'keys']))
                    val_red = float(np.mean([s['reduction'] for s in subset if s['type'] == 'values']))
                    logger.info(f"  {delta_method} (gs={gs:2d}): keys={key_red:.2f}x vals={val_red:.2f}x overall={mean_red:.2f}x")

    # Save
    combined = {
        'metadata': {
            'model': 'Qwen2.5-7B',
            'task': 'SQuAD-v2',
            'experiment': 'grouped_delta_encoding',
            'num_samples': len(all_results),
            'description': 'Fair comparison with CacheGen: sequential vs grouped vs anchor-based delta encoding',
        },
        'results': all_results,
        'variance_stats': all_variance_stats,
    }
    final_path = RESULTS_DIR / f'grouped_delta_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"\n[SAVED] -> {final_path}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"Batch 23 COMPLETE in {elapsed:.1f} minutes")

    del model; torch.cuda.empty_cache(); gc.collect()
