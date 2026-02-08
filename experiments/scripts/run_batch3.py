#!/usr/bin/env python3
"""
Batch 3: Corrected quantization F1 (in-place modification + manual generation loop).
Also: Qwen2.5-7B baseline F1 (needed for cross-model comparison).
"""
import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch3.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results/gpu_run')
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


def manual_generate(model, tokenizer, past_kv, seq_len, max_new=64):
    """Generate tokens using manual loop (works with modified DynamicCache)."""
    device = next(model.parameters()).device
    # Start from last position
    # We need a dummy input — use a newline or space token
    # Actually, the last token was already processed, so we generate the NEXT one
    generated = []
    cur_len = seq_len

    # First step: get logits for position after the cached sequence
    # Feed a dummy input (the model already has the KV for all prompt tokens)
    # Use a single forward pass with just position_ids to get next token
    dummy_input = torch.tensor([[tokenizer.encode("\n")[-1]]], device=device)
    # Actually, better: just continue from the cache
    # The issue is we need an input_id for the "current" position
    # The cache covers positions 0..seq_len-1, so we need to predict position seq_len
    # We can use the logits from the last cached position

    # Simpler approach: do a forward pass with the last token again
    # but that's wasteful. Instead, use the logits from initial forward pass.
    # For simplicity, just run the generation.

    # Actually the cleanest approach: run model once more with last token to get first generated token
    last_logits = None
    for step in range(max_new):
        if step == 0:
            # The cache already has all tokens. We need to get logits for the next position.
            # Feed a minimal input to trigger generation from cache.
            # Use an empty forward pass with the cache
            next_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            position_ids = torch.tensor([[cur_len]], device=device)
        else:
            next_input = torch.tensor([[generated[-1]]], device=device)
            position_ids = torch.tensor([[cur_len]], device=device)

        attn_mask = torch.ones(1, cur_len + 1, device=device, dtype=torch.long)

        with torch.no_grad():
            out = model(
                input_ids=next_input,
                past_key_values=past_kv,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)
        cur_len += 1

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def quantize_inplace_int8(pkv):
    """In-place INT8 quantization of DynamicCache."""
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            scale = t.abs().max() / 127.0
            if scale > 0:
                quantized = torch.clamp(torch.round(t / scale), -128, 127) * scale
                tensor.copy_(quantized.to(tensor.dtype))


def quantize_inplace_int4(pkv, group_size=32):
    """In-place INT4 quantization of DynamicCache."""
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            orig_shape = t.shape
            flat = t.reshape(-1)
            pad = (group_size - flat.numel() % group_size) % group_size
            if pad > 0:
                flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
            flat = flat.reshape(-1, group_size)
            scale = flat.abs().max(dim=1, keepdim=True).values / 7.0
            scale = scale.clamp(min=1e-10)
            q = torch.clamp(torch.round(flat / scale), -8, 7)
            dq = (q * scale).reshape(-1)[:t.numel()].reshape(orig_shape)
            tensor.copy_(dq.to(tensor.dtype))


def save_results(name, results, metadata):
    path = RESULTS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump({'metadata': metadata, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] {name} -> {path}")


def run_quantization_f1_v3(model_name="Qwen/Qwen2.5-3B", num_samples=30):
    """Quantization F1: in-place quantize + manual generation loop."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'quant_f1_v3_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import copy

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    # Check for checkpoint
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx = 0
    results = []
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        start_idx = ckpt['idx'] + 1
        results = ckpt['results']
        logger.info(f"Resuming from sample {start_idx}")

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Full KV baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV, make a copy for INT8, another for INT4
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        # INT8: clone KV, quantize in-place, generate
        import copy
        pkv_int8 = copy.deepcopy(out.past_key_values)
        quantize_inplace_int8(pkv_int8)
        int8_answer = manual_generate(model, tokenizer, pkv_int8, seq_len)
        int8_f1 = compute_f1(int8_answer, gold)

        # INT4
        pkv_int4 = copy.deepcopy(out.past_key_values)
        quantize_inplace_int4(pkv_int4)
        int4_answer = manual_generate(model, tokenizer, pkv_int4, seq_len)
        int4_f1 = compute_f1(int4_answer, gold)

        results.append({
            'idx': i, 'gold': gold, 'seq_len': seq_len,
            'full': {'answer': full_answer, 'f1': full_f1},
            'int8': {'answer': int8_answer, 'f1': int8_f1},
            'int4': {'answer': int4_answer, 'f1': int4_f1},
        })

        if (i + 1) % 5 == 0:
            avg = lambda k: np.mean([r[k]['f1'] for r in results])
            logger.info(f"  [{i+1}/{num_samples}] Full={avg('full'):.3f} INT8={avg('int8'):.3f} INT4={avg('int4'):.3f}")
            with open(ckpt_path, 'w') as f:
                json.dump({'idx': i, 'results': results}, f, default=str)

    del model; torch.cuda.empty_cache()

    summary = {
        'model': model_name,
        'num_samples': len(results),
        'full_f1': float(np.mean([r['full']['f1'] for r in results])),
        'int8_f1': float(np.mean([r['int8']['f1'] for r in results])),
        'int4_f1': float(np.mean([r['int4']['f1'] for r in results])),
        'full_std': float(np.std([r['full']['f1'] for r in results])),
        'int8_std': float(np.std([r['int8']['f1'] for r in results])),
        'int4_std': float(np.std([r['int4']['f1'] for r in results])),
    }
    logger.info(f"\nFull: {summary['full_f1']:.3f} +/- {summary['full_std']:.3f}")
    logger.info(f"INT8: {summary['int8_f1']:.3f} +/- {summary['int8_std']:.3f}")
    logger.info(f"INT4: {summary['int4_f1']:.3f} +/- {summary['int4_std']:.3f}")
    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


def run_7b_baseline(num_samples=30):
    """Qwen2.5-7B baseline F1 on SQuAD."""
    exp_name = 'baseline_7b'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    results = []
    for i, sample in enumerate(samples):
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        f1 = compute_f1(answer, gold)
        results.append({'idx': i, 'gold': gold, 'answer': answer, 'f1': f1})

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{num_samples}] F1={np.mean([r['f1'] for r in results]):.3f}")

    del model; torch.cuda.empty_cache()

    summary = {
        'model': 'Qwen/Qwen2.5-7B',
        'num_samples': len(results),
        'f1': float(np.mean([r['f1'] for r in results])),
        'f1_std': float(np.std([r['f1'] for r in results])),
    }
    logger.info(f"\n7B Baseline F1: {summary['f1']:.3f} +/- {summary['f1_std']:.3f}")
    save_results(exp_name, results, summary)
    return summary


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    start = time.time()

    run_quantization_f1_v3("Qwen/Qwen2.5-3B", num_samples=30)
    run_7b_baseline(num_samples=30)

    logger.info(f"\nALL DONE in {(time.time()-start)/60:.1f} minutes")
