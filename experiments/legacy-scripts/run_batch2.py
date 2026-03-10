#!/usr/bin/env python3
"""
Batch 2: Fixed quantization F1 + per-layer CKA + cross-family CKA.
All with checkpointing.
"""

import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch2.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results/gpu_run')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _get_kv_layer(pkv, layer_idx, component='key'):
    if hasattr(pkv, 'layers'):
        layer = pkv.layers[layer_idx]
        return layer.keys if component == 'key' else layer.values
    if hasattr(pkv, 'key_cache') and hasattr(pkv, 'value_cache'):
        return pkv.key_cache[layer_idx] if component == 'key' else pkv.value_cache[layer_idx]
    pair = pkv[layer_idx]
    return pair[0] if component == 'key' else pair[1]


def _num_layers(pkv):
    if hasattr(pkv, 'layers'):
        return len(pkv.layers)
    if hasattr(pkv, 'key_cache'):
        return len(pkv.key_cache)
    return len(pkv)


def compute_f1(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def quantize_int8(tensor):
    t = tensor.float()
    scale = t.abs().max() / 127.0
    if scale == 0: return tensor.clone()
    return (torch.clamp(torch.round(t / scale), -128, 127) * scale).to(tensor.dtype)


def quantize_int4(tensor, group_size=32):
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
    return dq.to(tensor.dtype)


def build_dynamic_cache(kv_pairs, device):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for l, (k, v) in enumerate(kv_pairs):
        cache.update(k.to(device), v.to(device), l)
    return cache


def save_results(name, results, metadata):
    path = RESULTS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump({'metadata': metadata, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] {name} → {path}")


# =========================================================================
# Experiment 1: Fixed Quantization F1
# The bug was passing past_key_values to generate() on same input — double counting.
# Fix: Use quantized KV as the ONLY input (feed dummy token, let KV handle context).
# Actually, the proper approach: encode context → get KV → quantize KV → use for generation.
# =========================================================================
def run_quantization_f1_fixed(num_samples=30):
    """Correct quantization F1: encode context, quantize KV, generate answer."""
    logger.info("\n" + "="*60)
    logger.info("Exp: Quantization F1 (FIXED)")
    logger.info("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    results = []
    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        gold = sample['answers']['text'][0]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Method 1: Full KV baseline — simple generate
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen_ids[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV-cache for the prompt
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        pkv_orig = out.past_key_values
        n_layers = _num_layers(pkv_orig)

        # Method 2: INT8 quantized KV
        int8_pairs = []
        for l in range(n_layers):
            k = _get_kv_layer(pkv_orig, l, 'key')
            v = _get_kv_layer(pkv_orig, l, 'value')
            int8_pairs.append((quantize_int8(k), quantize_int8(v)))
        int8_cache = build_dynamic_cache(int8_pairs, "cuda")

        # Generate from quantized KV: feed a dummy "continue" token
        # The KV already covers the full prompt, so we just need to generate
        # We feed the LAST token of the prompt as the "current" input
        last_token = inputs['input_ids'][:, -1:]
        attn_mask = torch.ones(1, seq_len, device="cuda", dtype=torch.long)
        try:
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=last_token,
                    past_key_values=int8_cache,
                    attention_mask=attn_mask,
                    max_new_tokens=64,
                    do_sample=False,
                )
            int8_answer = tokenizer.decode(gen_ids[0][1:], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"  INT8 gen failed: {e}")
            int8_answer = ""
        int8_f1 = compute_f1(int8_answer, gold)

        # Method 3: INT4 quantized KV
        int4_pairs = []
        for l in range(n_layers):
            k = _get_kv_layer(pkv_orig, l, 'key')
            v = _get_kv_layer(pkv_orig, l, 'value')
            int4_pairs.append((quantize_int4(k), quantize_int4(v)))
        int4_cache = build_dynamic_cache(int4_pairs, "cuda")

        try:
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=last_token,
                    past_key_values=int4_cache,
                    attention_mask=attn_mask,
                    max_new_tokens=64,
                    do_sample=False,
                )
            int4_answer = tokenizer.decode(gen_ids[0][1:], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"  INT4 gen failed: {e}")
            int4_answer = ""
        int4_f1 = compute_f1(int4_answer, gold)

        results.append({
            'idx': i, 'gold': gold,
            'full': {'answer': full_answer, 'f1': full_f1},
            'int8': {'answer': int8_answer, 'f1': int8_f1},
            'int4': {'answer': int4_answer, 'f1': int4_f1},
        })

        if (i + 1) % 5 == 0:
            avg = lambda k: np.mean([r[k]['f1'] for r in results])
            logger.info(f"  [{i+1}/{num_samples}] Full={avg('full'):.3f} INT8={avg('int8'):.3f} INT4={avg('int4'):.3f}")

            # Checkpoint
            ckpt = {'idx': i, 'results': results}
            with open(CKPT_DIR / 'quant_f1_ckpt.json', 'w') as f:
                json.dump(ckpt, f, default=str)

    del model; torch.cuda.empty_cache()

    summary = {
        'model': 'qwen2.5-3b',
        'num_samples': num_samples,
        'full_f1': float(np.mean([r['full']['f1'] for r in results])),
        'int8_f1': float(np.mean([r['int8']['f1'] for r in results])),
        'int4_f1': float(np.mean([r['int4']['f1'] for r in results])),
    }
    logger.info(f"\nFull: F1={summary['full_f1']:.3f}")
    logger.info(f"INT8: F1={summary['int8_f1']:.3f} (50% BW)")
    logger.info(f"INT4: F1={summary['int4_f1']:.3f} (25% BW)")
    save_results('quantization_f1_fixed', results, summary)
    return summary


# =========================================================================
# Experiment 2: Per-layer CKA (Qwen 3B vs 7B)
# =========================================================================
def run_per_layer_cka(num_samples=10):
    """Compute CKA between 3B and 7B at EACH layer, for both keys and values."""
    logger.info("\n" + "="*60)
    logger.info("Exp: Per-Layer CKA (3B vs 7B)")
    logger.info("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]
    prompts = [f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:" for s in samples]

    # Process through 3B
    logger.info("Loading 3B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

    # Collect per-layer KV from 3B
    kv_3b = []  # [{layer: {key: [T, D], value: [T, D]}} per sample]
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        pkv = out.past_key_values
        n = _num_layers(pkv)
        sample_kv = {}
        for l in range(n):
            k = _get_kv_layer(pkv, l, 'key')[0].mean(dim=0).cpu().float()
            v = _get_kv_layer(pkv, l, 'value')[0].mean(dim=0).cpu().float()
            sample_kv[l] = {'key': k, 'value': v}
        kv_3b.append(sample_kv)
    n_layers_3b = _num_layers(out.past_key_values)
    del model; torch.cuda.empty_cache()

    # Process through 7B
    logger.info("Loading 7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

    kv_7b = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        pkv = out.past_key_values
        n = _num_layers(pkv)
        sample_kv = {}
        for l in range(n):
            k = _get_kv_layer(pkv, l, 'key')[0].mean(dim=0).cpu().float()
            v = _get_kv_layer(pkv, l, 'value')[0].mean(dim=0).cpu().float()
            sample_kv[l] = {'key': k, 'value': v}
        kv_7b.append(sample_kv)
    n_layers_7b = _num_layers(out.past_key_values)
    del model; torch.cuda.empty_cache()

    # Compute per-layer CKA and cosine similarity
    # Align layers: 3B has 36 layers, 7B has 32 → compare layer i_3b with layer i_7b
    # Strategy: compare at relative depth (fraction of total layers)
    logger.info(f"3B layers: {n_layers_3b}, 7B layers: {n_layers_7b}")

    results = []
    for l3 in range(n_layers_3b):
        # Map to closest 7B layer by relative depth
        l7 = round(l3 * (n_layers_7b - 1) / (n_layers_3b - 1))

        cos_sims_key = []
        cos_sims_val = []
        cka_keys = []
        cka_vals = []

        for s_idx in range(num_samples):
            k3 = kv_3b[s_idx][l3]['key']
            v3 = kv_3b[s_idx][l3]['value']
            k7 = kv_7b[s_idx][l7]['key']
            v7 = kv_7b[s_idx][l7]['value']

            min_t = min(k3.shape[0], k7.shape[0])
            k3, k7 = k3[:min_t], k7[:min_t]
            v3, v7 = v3[:min_t], v7[:min_t]

            # Cosine similarity
            cos_k = torch.nn.functional.cosine_similarity(k3.unsqueeze(0), k7.unsqueeze(0), dim=-1).mean().item()
            cos_v = torch.nn.functional.cosine_similarity(v3.unsqueeze(0), v7.unsqueeze(0), dim=-1).mean().item()
            cos_sims_key.append(cos_k)
            cos_sims_val.append(cos_v)

            # Linear CKA
            def linear_cka(x, y):
                cross = torch.norm(x.T @ y, p='fro').item() ** 2
                self_x = torch.norm(x.T @ x, p='fro').item() ** 2
                self_y = torch.norm(y.T @ y, p='fro').item() ** 2
                return cross / (np.sqrt(self_x * self_y) + 1e-10)

            cka_keys.append(linear_cka(k3, k7))
            cka_vals.append(linear_cka(v3, v7))

        results.append({
            'layer_3b': l3,
            'layer_7b': l7,
            'relative_depth': l3 / (n_layers_3b - 1),
            'key_cos_sim': float(np.mean(cos_sims_key)),
            'val_cos_sim': float(np.mean(cos_sims_val)),
            'key_cka': float(np.mean(cka_keys)),
            'val_cka': float(np.mean(cka_vals)),
        })

        if l3 % 6 == 0:
            logger.info(f"  Layer 3B={l3}→7B={l7}: key_cos={np.mean(cos_sims_key):.3f} "
                       f"val_cos={np.mean(cos_sims_val):.3f} key_cka={np.mean(cka_keys):.3f}")

    summary = {
        'num_samples': num_samples,
        'n_layers_3b': n_layers_3b,
        'n_layers_7b': n_layers_7b,
        'mean_key_cos': float(np.mean([r['key_cos_sim'] for r in results])),
        'mean_val_cos': float(np.mean([r['val_cos_sim'] for r in results])),
        'mean_key_cka': float(np.mean([r['key_cka'] for r in results])),
        'mean_val_cka': float(np.mean([r['val_cka'] for r in results])),
    }
    logger.info(f"\n--- Per-Layer CKA Summary ---")
    logger.info(f"Mean key CKA: {summary['mean_key_cka']:.3f}")
    logger.info(f"Mean val CKA: {summary['mean_val_cka']:.3f}")
    logger.info(f"Mean key cos: {summary['mean_key_cos']:.3f}")
    logger.info(f"Mean val cos: {summary['mean_val_cos']:.3f}")

    save_results('per_layer_cka_3b_vs_7b', results, summary)
    return summary


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    start = time.time()

    run_quantization_f1_fixed(num_samples=20)
    run_per_layer_cka(num_samples=10)

    logger.info(f"\nALL DONE in {(time.time()-start)/60:.1f} minutes")
