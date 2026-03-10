#!/usr/bin/env python3
"""
Batch 10: 7B quantization sweep — find the information cliff for larger models.

Motivation: Batch 9 showed 7B INT4=0.597 (77%) vs 3B INT4=0.739 (96%).
The 7B model is more sensitive to quantization. We need to find:
1. Where exactly the 7B cliff is (between INT4 and INT8?)
2. Test INT5, INT6, INT7 to find the sweet spot
3. Confirm INT3/INT2 are catastrophic for 7B too

Also test 3B with INT5-INT7 for comparison (we already have INT2,3,4,8 from batch 7).
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch10.log')])
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


def _get_kv_layer(pkv, layer_idx, kv_type):
    layer = pkv.layers[layer_idx]
    return layer.keys if kv_type == 'key' else layer.values


def quantize_tensor(t, bits):
    if bits >= 16:
        return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    t_q = (t / scale).round().clamp(qmin, qmax)
    return t_q * scale


def manual_generate_quantized(model, tokenizer, input_ids, seq_len, quant_bits, max_new=64):
    """Generate answer with quantized KV cache using manual token-by-token loop."""
    # Forward pass to get full KV cache
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    past_kv = out.past_key_values

    # Quantize KV cache in-place
    if quant_bits < 16:
        for layer_idx in range(len(past_kv.layers)):
            layer = past_kv.layers[layer_idx]
            layer.keys.copy_(quantize_tensor(layer.keys, quant_bits))
            layer.values.copy_(quantize_tensor(layer.values, quant_bits))

    # Manual token-by-token generation
    generated = [first_token_id]
    cur_len = seq_len
    full_mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)

    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        new_token_mask = torch.ones(1, 1, device='cuda', dtype=torch.long)
        full_mask = torch.cat([full_mask, new_token_mask], dim=1)

        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       attention_mask=full_mask, position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def run_quant_sweep(model, tokenizer, model_name, model_dtype, num_samples=50):
    exp_name = f'quant_sweep_{model_name}'
    logger.info(f"\n{'='*60}\n{model_name} Quantization Sweep ({num_samples} samples)\n{'='*60}")

    # Test: 2, 3, 4, 5, 6, 7, 8, 16 bits
    bit_levels = [2, 3, 4, 5, 6, 7, 8, 16]

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # Full baseline (model.generate for consistency)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        result['full'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        # Quantized versions
        for bits in bit_levels:
            try:
                ans = manual_generate_quantized(model, tokenizer, input_ids, seq_len, bits)
                result[f'int{bits}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
            except Exception as e:
                logger.warning(f"int{bits} failed: {e}")
                result[f'int{bits}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        f_full = result['full']['f1']
        bits_str = ' '.join(f'{bits}b={result.get(f"int{bits}", {}).get("f1", -1):.2f}' for bits in bit_levels)
        logger.info(f"  [{i+1}/{num_samples}] full={f_full:.3f} {bits_str} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))
    for bits in bit_levels:
        vals = [r.get(f'int{bits}', {}).get('f1', 0) for r in results]
        summary[f'int{bits}_f1'] = float(np.mean(vals))
        summary[f'int{bits}_std'] = float(np.std(vals))

    logger.info(f"\n--- {model_name} Quantization Sweep Summary ---")
    full = summary['full_f1']
    for bits in bit_levels:
        f1 = summary[f'int{bits}_f1']
        pct = f1 / full * 100 if full > 0 else 0
        logger.info(f"  INT{bits}: F1={f1:.4f} ({pct:.1f}% of full)")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': model_name, 'dtype': model_dtype,
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

    # === 7B quantization sweep ===
    logger.info("Loading Qwen2.5-7B (BF16, eager)...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_7b.config.use_cache = True
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tok_7b.pad_token is None: tok_7b.pad_token = tok_7b.eos_token

    s7b = run_quant_sweep(model_7b, tok_7b, "qwen25_7b", "bf16", num_samples=50)
    del model_7b; torch.cuda.empty_cache(); gc.collect()

    # === 3B quantization sweep (INT5-7 only, we have 2,3,4,8 from batch 7) ===
    logger.info("\nLoading Qwen2.5-3B (FP16, eager)...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_3b.config.use_cache = True
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tok_3b.pad_token is None: tok_3b.pad_token = tok_3b.eos_token

    s3b = run_quant_sweep(model_3b, tok_3b, "qwen25_3b", "fp16", num_samples=50)
    del model_3b; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 10 COMPLETE in {elapsed:.1f} minutes")

    # Print comparison
    logger.info("\n=== CROSS-MODEL QUANTIZATION COMPARISON ===")
    full_3b = s3b['full_f1']
    full_7b = s7b['full_f1']
    logger.info(f"{'Bits':>6s} | {'3B F1':>8s} ({'%':>4s}) | {'7B F1':>8s} ({'%':>4s})")
    logger.info("-" * 50)
    for bits in [2, 3, 4, 5, 6, 7, 8, 16]:
        f3 = s3b.get(f'int{bits}_f1', 0)
        f7 = s7b.get(f'int{bits}_f1', 0)
        p3 = f3 / full_3b * 100 if full_3b > 0 else 0
        p7 = f7 / full_7b * 100 if full_7b > 0 else 0
        logger.info(f"  INT{bits:>2d} | {f3:>8.4f} ({p3:4.0f}%) | {f7:>8.4f} ({p7:4.0f}%)")
