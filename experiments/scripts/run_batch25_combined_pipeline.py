#!/usr/bin/env python3
"""
Batch 25: Combined Pipeline Exact Measurements + Timing

Critical missing data for paper:
1. Exact F1 numbers for Q2C + quantization combinations (Table 6 currently uses ~estimates)
2. Timing measurements (compression overhead, simulated transmission time)
3. Q2C selection on Qwen-14B and Mistral-7B at 25% retention (expand selection table)

Also adds normalized F1 (strip articles/punctuation) to fix Mistral-7B F1=0.120 issue.
"""
import os, sys, json, time, logging, gc, re, string
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/dev/shm/hf_7b'
os.environ['HF_DATASETS_CACHE'] = '/dev/shm/hf_7b/datasets'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch25.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 50


def normalize_answer(s):
    """Standard SQuAD evaluation: lowercase, remove articles/punctuation/extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def compute_f1(pred, gold):
    """Standard F1 with normalization."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens: return 1.0 if not pred_tokens else 0.0
    if not pred_tokens: return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def compute_f1_raw(pred, gold):
    """Raw F1 without normalization (for comparison)."""
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


def q2c_select_positions(attn_weights, num_context, num_query, retention):
    """Compute Q2C scores and return selected context position indices."""
    # attn_weights shape: (num_heads, seq_len, seq_len) from last layer
    H, S, _ = attn_weights.shape
    n = num_context
    m = num_query

    # Q2C scores: attention from query positions to context positions
    q2c_scores = torch.zeros(n, device=attn_weights.device)
    for h in range(H):
        for qi in range(n, n + m):
            q2c_scores += attn_weights[h, qi, :n]

    # Select top-k positions
    k = max(1, int(retention * n))
    _, selected_idx = q2c_scores.topk(k)
    return sorted(selected_idx.cpu().tolist())


def generate_with_pipeline(model, tokenizer, input_ids, seq_len,
                           quantize_bits=16, mixed_prec=False,
                           selection_retention=1.0, selection_method='none',
                           attn_weights=None, num_context=0, num_query=0,
                           max_new=64, bottleneck_layer=0):
    """Full pipeline: selection + quantization + generation with timing."""
    timings = {}

    # Step 1: Prefill
    t0 = time.time()
    with torch.no_grad():
        if selection_method == 'none' or attn_weights is not None:
            out = model(input_ids=input_ids, use_cache=True,
                       output_attentions=(selection_method != 'none' and attn_weights is None))
        else:
            out = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    timings['prefill'] = time.time() - t0

    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    # Step 2: Selection (if applicable)
    if selection_method != 'none' and selection_retention < 1.0:
        t1 = time.time()
        if attn_weights is None and hasattr(out, 'attentions') and out.attentions is not None:
            attn_weights = out.attentions[-1][0]  # Last layer, first batch
        if attn_weights is not None:
            selected = q2c_select_positions(attn_weights, num_context, num_query, selection_retention)
            # Apply attention mask (keep selected + query positions)
            # For simplicity, we zero out unselected positions in KV
            all_positions = set(range(seq_len))
            query_positions = set(range(num_context, seq_len))
            keep = set(selected) | query_positions
            mask_positions = all_positions - keep
            for li in range(len(pkv.layers)):
                layer = pkv.layers[li]
                for pos in mask_positions:
                    layer.keys[:, :, pos, :] = 0
                    layer.values[:, :, pos, :] = 0
        timings['selection'] = time.time() - t1
    else:
        timings['selection'] = 0.0

    # Step 3: Quantization
    t2 = time.time()
    for li in range(len(pkv.layers)):
        layer = pkv.layers[li]
        if mixed_prec and li == bottleneck_layer:
            pass  # Keep FP16
        else:
            if quantize_bits < 16:
                layer.keys.copy_(quantize_pertoken(layer.keys, quantize_bits))
                layer.values.copy_(quantize_pertoken(layer.values, quantize_bits))
    timings['quantization'] = time.time() - t2

    # Step 4: Compute compressed size
    num_layers = len(pkv.layers)
    H = pkv.layers[0].keys.shape[1]
    D = pkv.layers[0].keys.shape[3]
    S = pkv.layers[0].keys.shape[2]
    if mixed_prec:
        bits_total = (1 * 16 + (num_layers - 1) * quantize_bits) * 2 * H * S * D
    else:
        bits_total = num_layers * quantize_bits * 2 * H * S * D
    original_bits = num_layers * 16 * 2 * H * S * D
    compression_ratio = bits_total / original_bits
    timings['compressed_bytes'] = bits_total // 8
    timings['original_bytes'] = original_bits // 8
    timings['compression_ratio'] = compression_ratio

    # Step 5: Generation
    t3 = time.time()
    answer = manual_generate(model, tokenizer, pkv, first_token_id, seq_len, max_new)
    timings['generation'] = time.time() - t3

    return answer, timings


def prepare_squad(tokenizer, num_samples=50, prompt_template='default'):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    samples = []
    for s in ds:
        if not s['answers']['text']: continue
        ctx = s['context']
        q = s['question']
        if prompt_template == 'default':
            prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer briefly:"
        elif prompt_template == 'instruct':
            prompt = f"<|user|>\nContext: {ctx}\nQuestion: {q}\nAnswer briefly.<|end|>\n<|assistant|>\n"
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 400:
            # Count context and question tokens separately
            ctx_toks = len(tokenizer.encode(f"Context: {ctx}\n"))
            samples.append({
                'prompt': prompt, 'gold': s['answers']['text'][0],
                'tok_len': tok_len, 'ctx_toks': ctx_toks,
                'q_toks': tok_len - ctx_toks
            })
        if len(samples) >= num_samples * 3: break
    np.random.seed(42)
    idx = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in idx]


def run_model_experiment(model_name, model_short, prompt_template, samples_fn,
                         test_selection=True, test_combined=True):
    """Run full experiment battery for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL: {model_short}")
    logger.info(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map='cuda',
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {num_layers} layers")

    samples = samples_fn(tokenizer, NUM_SAMPLES)
    logger.info(f"Prepared {len(samples)} samples")

    # Define pipeline configs
    configs = [
        # name, bits, mixed, retention, selection
        ('full_fp16', 16, False, 1.0, 'none'),
        ('int8', 8, False, 1.0, 'none'),
        ('int4', 4, False, 1.0, 'none'),
        ('mixed_int4', 4, True, 1.0, 'none'),
    ]
    if test_selection:
        configs.extend([
            ('q2c_75_fp16', 16, False, 0.75, 'q2c'),
            ('q2c_50_fp16', 16, False, 0.50, 'q2c'),
            ('q2c_25_fp16', 16, False, 0.25, 'q2c'),
        ])
    if test_combined:
        configs.extend([
            ('q2c_75_int8', 8, False, 0.75, 'q2c'),
            ('q2c_50_int8', 8, False, 0.50, 'q2c'),
            ('q2c_75_mixed4', 4, True, 0.75, 'q2c'),
            ('q2c_50_mixed4', 4, True, 0.50, 'q2c'),
        ])

    all_results = []
    timing_summary = {c[0]: {'prefill': [], 'selection': [], 'quantization': [],
                              'generation': [], 'compressed_bytes': [], 'f1': [], 'f1_raw': []}
                      for c in configs}

    for i, s in enumerate(samples):
        inputs = tokenizer(s['prompt'], return_tensors='pt', truncation=True, max_length=512).to('cuda')
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        gold = s['gold']
        ctx_toks = s.get('ctx_toks', seq_len // 2)
        q_toks = s.get('q_toks', seq_len - ctx_toks)

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        for name, bits, mixed, ret, sel in configs:
            ans, timings = generate_with_pipeline(
                model, tokenizer, input_ids, seq_len,
                quantize_bits=bits, mixed_prec=mixed,
                selection_retention=ret, selection_method=sel,
                num_context=ctx_toks, num_query=q_toks,
                bottleneck_layer=0)

            f1 = compute_f1(ans, gold)
            f1_raw = compute_f1_raw(ans, gold)
            result[name] = {'answer': ans[:200], 'f1': f1, 'f1_raw': f1_raw, 'timings': timings}

            timing_summary[name]['prefill'].append(timings['prefill'])
            timing_summary[name]['selection'].append(timings['selection'])
            timing_summary[name]['quantization'].append(timings['quantization'])
            timing_summary[name]['generation'].append(timings['generation'])
            timing_summary[name]['compressed_bytes'].append(timings.get('compressed_bytes', 0))
            timing_summary[name]['f1'].append(f1)
            timing_summary[name]['f1_raw'].append(f1_raw)

        all_results.append(result)

        fp16 = result['full_fp16']['f1']
        i4 = result['int4']['f1']
        logger.info(f"  [{i+1}/{len(samples)}] fp16={fp16:.3f} int4={i4:.3f} "
                    f"(prefill={result['full_fp16']['timings']['prefill']:.3f}s)")

    # Summary
    baseline_f1 = float(np.mean(timing_summary['full_fp16']['f1']))
    baseline_f1_raw = float(np.mean(timing_summary['full_fp16']['f1_raw']))
    logger.info(f"\n{'-'*60}")
    logger.info(f"SUMMARY — {model_short} (n={len(samples)})")
    logger.info(f"Baseline: F1={baseline_f1:.4f} (normalized), F1_raw={baseline_f1_raw:.4f}")
    logger.info(f"{'-'*60}")

    summary = {}
    for name, _, _, _, _ in configs:
        ts = timing_summary[name]
        mean_f1 = float(np.mean(ts['f1']))
        std_f1 = float(np.std(ts['f1']))
        se_f1 = std_f1 / np.sqrt(len(ts['f1']))
        pct = mean_f1 / baseline_f1 * 100 if baseline_f1 > 0 else 0
        mean_prefill = float(np.mean(ts['prefill']))
        mean_quant = float(np.mean(ts['quantization']))
        mean_sel = float(np.mean(ts['selection']))
        mean_gen = float(np.mean(ts['generation']))
        mean_bytes = float(np.mean(ts['compressed_bytes']))

        summary[name] = {
            'f1_mean': mean_f1, 'f1_std': std_f1, 'f1_se': se_f1, 'f1_pct': pct,
            'f1_raw_mean': float(np.mean(ts['f1_raw'])),
            'prefill_ms': mean_prefill * 1000, 'quant_ms': mean_quant * 1000,
            'selection_ms': mean_sel * 1000, 'generation_ms': mean_gen * 1000,
            'compressed_bytes': mean_bytes,
        }

        # Simulated transmission times at various bandwidths
        for bw_mbps in [10, 50, 100]:
            bw_bytes_per_sec = bw_mbps * 1e6 / 8
            tx_time = mean_bytes / bw_bytes_per_sec if mean_bytes > 0 else 0
            summary[name][f'tx_{bw_mbps}mbps_ms'] = tx_time * 1000

        logger.info(f"  {name:25s}: F1={mean_f1:.4f}±{se_f1:.4f} ({pct:.1f}%) "
                    f"prefill={mean_prefill*1000:.1f}ms quant={mean_quant*1000:.1f}ms "
                    f"size={mean_bytes/1024:.0f}KB")

    result_data = {
        'metadata': {
            'model': model_short, 'model_name': model_name,
            'num_samples': len(samples),
            'normalized_f1': True,
        },
        'summary': summary,
        'per_sample': all_results,
    }

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result_data


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    all_data = {}

    # Qwen-7B: Full pipeline (combined + timing) — this is our primary model
    data = run_model_experiment(
        'Qwen/Qwen2.5-7B', 'Qwen-7B',
        'default',
        lambda tok, n: prepare_squad(tok, n, 'default'),
        test_selection=True, test_combined=True)
    all_data['qwen7b'] = data
    fpath = RESULTS_DIR / f'combined_pipeline_qwen7b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"[SAVED] {fpath}")

    # Mistral-7B: Selection + normalized F1 — fix the F1=0.120 issue
    data = run_model_experiment(
        'mistralai/Mistral-7B-Instruct-v0.3', 'Mistral-7B',
        'default',
        lambda tok, n: prepare_squad(tok, n, 'default'),
        test_selection=True, test_combined=False)
    all_data['mistral7b'] = data
    fpath = RESULTS_DIR / f'combined_pipeline_mistral7b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"[SAVED] {fpath}")

    # Qwen-14B: Selection at 25% — expand selection table
    data = run_model_experiment(
        'Qwen/Qwen2.5-14B', 'Qwen-14B',
        'default',
        lambda tok, n: prepare_squad(tok, n, 'default'),
        test_selection=True, test_combined=False)
    all_data['qwen14b'] = data
    fpath = RESULTS_DIR / f'combined_pipeline_qwen14b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"[SAVED] {fpath}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nBatch 25 COMPLETE in {elapsed:.1f} minutes")
