#!/usr/bin/env python3
"""
Batch 12: Mixed-precision + per-channel quantization — the ultimate compression recipe.

Experiments:
  12a: Mixed-precision (Layer 0 FP16 + rest per-channel INT4) vs uniform
  12b: Per-channel quantization sweep (INT4-INT8) — clean monotonic curves
  12c: Combined pipeline: Q2C selection + mixed-precision quantization

Key insight from batch 11:
- Layer 0 is sole quantization bottleneck (keeps FP16 → recovers full accuracy)
- Per-channel quantization fixes INT6 anomaly (0.748 vs 0.421 per-token)
- Combining these: Layer 0 FP16 + per-channel INT4 for rest → should be lossless
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/dev/shm/hf_7b'
os.environ['HF_DATASETS_CACHE'] = '/dev/shm/hf_7b/datasets'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch12.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)


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


def save_checkpoint(exp_name, idx, results):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    with open(ckpt_path, 'w') as f:
        json.dump({'idx': idx, 'results': results}, f, default=str)


def load_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        logger.info(f"[RESUME] {exp_name} from sample {ckpt['idx'] + 1}")
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


def quantize_tensor_pertoken(t, bits):
    """Standard per-token quantization (amax over head_dim, last axis)."""
    if bits >= 16: return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return (t / scale).round().clamp(qmin, qmax) * scale


def quantize_tensor_perchannel(t, bits):
    """Per-channel quantization (amax over sequence dim, -2 axis).
    Fixes INT6 anomaly on 7B by preserving intra-channel structure."""
    if bits >= 16: return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return (t / scale).round().clamp(qmin, qmax) * scale


def manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new=64):
    generated = [first_token_id]
    cur_len = seq_len
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        mask = torch.cat([mask, torch.ones(1, 1, device='cuda', dtype=torch.long)], dim=1)
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       attention_mask=mask, position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id: break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_mixed_precision(model, tokenizer, input_ids, seq_len, quant_bits, quant_mode='pertoken',
                              layer0_bits=16, max_new=64, selection_mask=None):
    """Generate with mixed-precision quantization.

    Args:
        quant_bits: Default quantization bits for all layers (except layer 0)
        quant_mode: 'pertoken' or 'perchannel'
        layer0_bits: Bits for layer 0 (16=FP16, or lower)
        selection_mask: Optional attention mask for Q2C selection (1=keep, 0=drop)
    """
    quantize_fn = quantize_tensor_perchannel if quant_mode == 'perchannel' else quantize_tensor_pertoken

    if selection_mask is not None:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=selection_mask, use_cache=True)
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)

    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    past_kv = out.past_key_values

    # Quantize each layer
    for layer_idx in range(len(past_kv.layers)):
        bits = layer0_bits if layer_idx == 0 else quant_bits
        if bits < 16:
            layer = past_kv.layers[layer_idx]
            layer.keys.copy_(quantize_fn(layer.keys, bits))
            layer.values.copy_(quantize_fn(layer.values, bits))

    # Generate
    mask = selection_mask.clone() if selection_mask is not None else torch.ones(1, seq_len, device='cuda', dtype=torch.long)
    return manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new)


def compute_q2c_scores(model, tokenizer, input_ids, seq_len):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    q_start, a_start = None, None
    decoded_so_far = ""
    for i, tok in enumerate(tokens):
        decoded_so_far = tokenizer.decode(input_ids[0][:i+1])
        if "Question:" in decoded_so_far and q_start is None:
            q_start = i
        if "Answer:" in decoded_so_far and a_start is None:
            a_start = i
            break

    if q_start is None or a_start is None:
        q_start = int(seq_len * 0.8)
        a_start = seq_len

    context_positions = list(range(0, q_start))
    question_positions = list(range(q_start, a_start))

    if not context_positions or not question_positions:
        return [], context_positions, question_positions

    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        for q_pos in question_positions:
            scores += layer_attn[0, :, q_pos, :].mean(dim=0)

    context_scores = [(pos, scores[pos].item()) for pos in context_positions]
    context_scores.sort(key=lambda x: x[1], reverse=True)
    return context_scores, context_positions, question_positions


def make_selection_mask(seq_len, context_scores, context_positions, question_positions, retention):
    """Create attention mask for Q2C selection at given retention level."""
    n_keep = int(len(context_positions) * retention)
    selected = set(pos for pos, _ in context_scores[:n_keep])
    always_keep = set(question_positions) | set(range(max(0, seq_len-5), seq_len))

    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
    for p in always_keep:
        if p < seq_len: mask[0, p] = 1
    for p in selected:
        if p < seq_len: mask[0, p] = 1
    return mask


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def run_experiment(model, tokenizer, model_name, num_layers, num_samples=50):
    exp_name = f'mixed_precision_{model_name}'
    logger.info(f"\n{'='*60}\nBatch 12: Mixed-Precision + Per-Channel ({model_name}, {num_layers} layers)\n{'='*60}")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    # Define all configurations to test
    configs = [
        # Baselines
        ('full_fp16', {'quant_bits': 16, 'quant_mode': 'pertoken', 'layer0_bits': 16, 'retention': None}),

        # 12b: Per-channel quantization sweep
        ('perchannel_int4', {'quant_bits': 4, 'quant_mode': 'perchannel', 'layer0_bits': 4, 'retention': None}),
        ('perchannel_int5', {'quant_bits': 5, 'quant_mode': 'perchannel', 'layer0_bits': 5, 'retention': None}),
        ('perchannel_int6', {'quant_bits': 6, 'quant_mode': 'perchannel', 'layer0_bits': 6, 'retention': None}),
        ('perchannel_int7', {'quant_bits': 7, 'quant_mode': 'perchannel', 'layer0_bits': 7, 'retention': None}),
        ('perchannel_int8', {'quant_bits': 8, 'quant_mode': 'perchannel', 'layer0_bits': 8, 'retention': None}),

        # Per-token baselines for comparison
        ('pertoken_int4', {'quant_bits': 4, 'quant_mode': 'pertoken', 'layer0_bits': 4, 'retention': None}),
        ('pertoken_int6', {'quant_bits': 6, 'quant_mode': 'pertoken', 'layer0_bits': 6, 'retention': None}),

        # 12a: Mixed-precision (Layer 0 FP16 + rest per-channel INT4)
        ('mixed_L0fp16_rest_pch_int4', {'quant_bits': 4, 'quant_mode': 'perchannel', 'layer0_bits': 16, 'retention': None}),
        ('mixed_L0fp16_rest_ptk_int4', {'quant_bits': 4, 'quant_mode': 'pertoken', 'layer0_bits': 16, 'retention': None}),

        # 12c: Combined pipeline — Q2C selection + mixed-precision
        ('q2c75_mixed_pch_int4', {'quant_bits': 4, 'quant_mode': 'perchannel', 'layer0_bits': 16, 'retention': 0.75}),
        ('q2c50_mixed_pch_int4', {'quant_bits': 4, 'quant_mode': 'perchannel', 'layer0_bits': 16, 'retention': 0.50}),
        ('q2c75_pch_int4', {'quant_bits': 4, 'quant_mode': 'perchannel', 'layer0_bits': 4, 'retention': 0.75}),
        ('q2c50_pch_int4', {'quant_bits': 4, 'quant_mode': 'perchannel', 'layer0_bits': 4, 'retention': 0.50}),

        # Q2C selection only (no quantization) for baseline
        ('q2c75_fp16', {'quant_bits': 16, 'quant_mode': 'pertoken', 'layer0_bits': 16, 'retention': 0.75}),
        ('q2c50_fp16', {'quant_bits': 16, 'quant_mode': 'pertoken', 'layer0_bits': 16, 'retention': 0.50}),
    ]

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # Pre-compute Q2C scores (needed for selection configs)
        context_scores, context_positions, question_positions = None, None, None
        try:
            context_scores, context_positions, question_positions = compute_q2c_scores(
                model, tokenizer, input_ids, seq_len)
        except Exception as e:
            logger.warning(f"Q2C scores failed for sample {i}: {e}")

        for cfg_name, cfg in configs:
            try:
                selection_mask = None
                if cfg['retention'] is not None and context_scores:
                    selection_mask = make_selection_mask(
                        seq_len, context_scores, context_positions, question_positions, cfg['retention'])

                ans = generate_mixed_precision(
                    model, tokenizer, input_ids, seq_len,
                    quant_bits=cfg['quant_bits'],
                    quant_mode=cfg['quant_mode'],
                    layer0_bits=cfg['layer0_bits'],
                    selection_mask=selection_mask)

                result[cfg_name] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}
            except Exception as e:
                logger.warning(f"{cfg_name} failed: {e}")
                result[cfg_name] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        fp16 = result.get('full_fp16', {}).get('f1', -1)
        mixed = result.get('mixed_L0fp16_rest_pch_int4', {}).get('f1', -1)
        combined = result.get('q2c75_mixed_pch_int4', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] fp16={fp16:.3f} mixed={mixed:.3f} combined={combined:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results), 'num_layers': num_layers}
    for cfg_name, _ in configs:
        vals = [r.get(cfg_name, {}).get('f1', 0) for r in results]
        summary[f'{cfg_name}_f1'] = float(np.mean(vals))
        summary[f'{cfg_name}_std'] = float(np.std(vals))

    logger.info(f"\n--- Mixed-Precision Summary ({model_name}) ---")
    fp16_f1 = summary['full_fp16_f1']
    for cfg_name, cfg in configs:
        f1 = summary[f'{cfg_name}_f1']
        pct = f1 / fp16_f1 * 100 if fp16_f1 > 0 else 0

        # Compute effective bandwidth
        ret = cfg.get('retention')
        bits = cfg['quant_bits']
        l0_bits = cfg['layer0_bits']
        if ret:
            bw = ret * ((1/num_layers) * (l0_bits/16) + ((num_layers-1)/num_layers) * (bits/16))
        else:
            bw = (1/num_layers) * (l0_bits/16) + ((num_layers-1)/num_layers) * (bits/16)
        bw_pct = bw * 100

        logger.info(f"  {cfg_name:35s}: F1={f1:.4f} ({pct:5.1f}%) BW={bw_pct:5.1f}%")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': model_name, 'num_layers': num_layers,
                   'configs': {name: str(cfg) for name, cfg in configs},
                   'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    return summary


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    # === 7B ===
    os.environ['HF_HOME'] = '/dev/shm/hf_7b'
    logger.info("Loading Qwen2.5-7B (BF16, eager)...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_7b.config.use_cache = True
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tok_7b.pad_token is None: tok_7b.pad_token = tok_7b.eos_token

    num_layers_7b = model_7b.config.num_hidden_layers
    s7b = run_experiment(model_7b, tok_7b, "qwen25_7b", num_layers_7b, num_samples=50)
    del model_7b; torch.cuda.empty_cache(); gc.collect()

    # === 3B ===
    os.environ['HF_HOME'] = '/workspace/.hf_home'
    logger.info("\nLoading Qwen2.5-3B (FP16, eager)...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_3b.config.use_cache = True
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tok_3b.pad_token is None: tok_3b.pad_token = tok_3b.eos_token

    num_layers_3b = model_3b.config.num_hidden_layers
    s3b = run_experiment(model_3b, tok_3b, "qwen25_3b", num_layers_3b, num_samples=50)
    del model_3b; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 12 COMPLETE in {elapsed:.1f} minutes")

    # Cross-model comparison
    logger.info("\n=== CROSS-MODEL MIXED-PRECISION COMPARISON ===")
    full_3b = s3b.get('full_fp16_f1', 0)
    full_7b = s7b.get('full_fp16_f1', 0)
    for key in ['full_fp16', 'pertoken_int4', 'perchannel_int4', 'perchannel_int6',
                'mixed_L0fp16_rest_pch_int4', 'mixed_L0fp16_rest_ptk_int4',
                'q2c75_fp16', 'q2c75_mixed_pch_int4', 'q2c50_mixed_pch_int4']:
        f3 = s3b.get(f'{key}_f1', 0)
        f7 = s7b.get(f'{key}_f1', 0)
        p3 = f3 / full_3b * 100 if full_3b > 0 else 0
        p7 = f7 / full_7b * 100 if full_7b > 0 else 0
        logger.info(f"  {key:40s}: 3B={f3:.4f} ({p3:5.1f}%) | 7B={f7:.4f} ({p7:5.1f}%)")
