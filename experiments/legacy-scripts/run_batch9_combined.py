#!/usr/bin/env python3
"""
Batch 9: Combined pipeline (Q2C selection + INT4 quantization) with manual_generate path.

Motivation: Batch 7 tested combined pipeline but with model.generate(). Now we know
manual_generate gives more accurate results, so re-test the combined pipeline.

Also test on 7B (BF16) and explore the full matrix:
- Selection methods: Q2C, SnapKV
- Retention levels: 25%, 50%, 75%
- Quantization: None, INT4, INT8
- Models: 3B (FP16), 7B (BF16)

This gives us the complete picture for the paper's main figure:
"Compression method × retention → F1" surface plot.
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch9.log')])
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
    interim_path = RESULTS_DIR / f'{exp_name}_INTERIM.json'
    summary = compute_summary(results)
    with open(interim_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)


def load_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        logger.info(f"[RESUME] {exp_name} from sample {ckpt['idx'] + 1}")
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


def compute_summary(results):
    summary = {'num_samples': len(results)}
    if not results:
        return summary
    all_keys = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, dict) and 'f1' in v:
                all_keys.add(k)
    for k in sorted(all_keys):
        vals = [r[k]['f1'] for r in results if k in r and isinstance(r[k], dict)]
        if vals:
            summary[f'{k}_f1'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
    return summary


def _get_kv_layer(pkv, layer_idx, kv_type):
    """Get key or value tensor from DynamicCache (transformers 5.x API)."""
    layer = pkv.layers[layer_idx]
    return layer.keys if kv_type == 'key' else layer.values


def quantize_tensor(t, bits):
    """Symmetric quantization to N bits."""
    if bits >= 16:
        return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    t_q = (t / scale).round().clamp(qmin, qmax)
    return t_q * scale


def quantize_kv_cache(pkv, bits):
    """Quantize all key and value tensors in the KV cache in-place."""
    for layer_idx in range(len(pkv.layers)):
        layer = pkv.layers[layer_idx]
        layer.keys.copy_(quantize_tensor(layer.keys, bits))
        layer.values.copy_(quantize_tensor(layer.values, bits))
    return pkv


def manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len,
                               attn_mask_prefix, max_new=64):
    """Token-by-token generation with pre-populated KV cache AND custom attention mask."""
    generated = [first_token_id]
    cur_len = seq_len
    full_mask = attn_mask_prefix.clone()

    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        new_token_mask = torch.ones(1, 1, device='cuda', dtype=full_mask.dtype)
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


def generate_combined(model, tokenizer, input_ids, seq_len, selected_positions, always_keep,
                       quant_bits=None, max_new=64):
    """Generate with selection + optional quantization.

    Pipeline:
    1. Forward pass with attention mask to get KV cache
    2. Optionally quantize the KV cache
    3. Manual generation with mask
    """
    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
    for p in always_keep:
        if p < seq_len:
            mask[0, p] = 1
    for p in selected_positions:
        if p < seq_len:
            mask[0, p] = 1

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    past_kv = out.past_key_values

    # Apply quantization to KV cache if requested
    if quant_bits is not None:
        past_kv = quantize_kv_cache(past_kv, quant_bits)

    return manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new)


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def run_combined_experiment(model, tokenizer, model_name, model_dtype, num_samples=50):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exp_name = f'combined_{model_name}'
    logger.info(f"\n{'='*60}\n{model_name} Combined Pipeline ({num_samples} samples)\n{'='*60}")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    # Test matrix
    selection_methods = ['q2c', 'snapkv']
    retentions = [0.25, 0.50, 0.75]
    quant_levels = [None, 8, 4]  # None=full precision, 8=INT8, 4=INT4

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        context = sample['context']
        question = sample['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        # Token ranges
        ctx_prefix = tokenizer("Context: ", add_special_tokens=True, return_tensors="pt")
        ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
        ctx_only = f"Context: {context}\nQuestion: "
        ctx_tokens = tokenizer(ctx_only, return_tensors="pt", max_length=512, truncation=True)
        context_end = min(ctx_tokens['input_ids'].shape[1], seq_len)
        question_start = context_end
        answer_suffix = tokenizer("\nAnswer:", add_special_tokens=False)['input_ids']
        question_end = seq_len - len(answer_suffix)
        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        if num_context < 5:
            continue

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_context}

        # === Baseline (full KV, full precision) ===
        with torch.no_grad():
            gen_full = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        ans_full = tokenizer.decode(gen_full[0][seq_len:], skip_special_tokens=True).strip()
        result['full'] = {'answer': ans_full, 'f1': compute_f1(ans_full, gold)}

        # === Quantization-only baselines ===
        for bits in [8, 4]:
            try:
                # Full positions, just quantize
                all_positions = set(range(seq_len))
                ans = generate_combined(model, tokenizer, input_ids, seq_len,
                                        all_positions, always_keep, quant_bits=bits)
                result[f'int{bits}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
            except Exception as e:
                result[f'int{bits}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        # === Compute attention scores ===
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)

        q2c_scores = torch.zeros(seq_len, device='cuda')
        snapkv_scores = torch.zeros(seq_len, device='cuda')
        window = min(32, seq_len)
        for layer_attn in out.attentions:
            snapkv_scores += layer_attn[0, :, -window:, :].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
        del out; torch.cuda.empty_cache()

        scores = {'q2c': q2c_scores, 'snapkv': snapkv_scores}
        ctx_tensor = torch.tensor(context_positions, device='cuda')

        # === Combined matrix ===
        for method_name in selection_methods:
            for retention in retentions:
                k = max(1, int(num_context * retention))
                ret_key = int(retention * 100)

                ctx_sc = scores[method_name][ctx_tensor]
                _, topk_idx = ctx_sc.topk(k)
                selected = set(context_positions[j] for j in topk_idx.cpu().numpy())

                for bits in quant_levels:
                    bits_name = f'int{bits}' if bits else 'fp'
                    key_name = f'{method_name}_{ret_key}_{bits_name}'

                    try:
                        ans = generate_combined(model, tokenizer, input_ids, seq_len,
                                                selected, always_keep, quant_bits=bits)
                        result[key_name] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                    except Exception as e:
                        result[key_name] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        # Compact log
        f_full = result['full']['f1']
        f_q50_fp = result.get('q2c_50_fp', {}).get('f1', -1)
        f_q50_i4 = result.get('q2c_50_int4', {}).get('f1', -1)
        f_s50_fp = result.get('snapkv_50_fp', {}).get('f1', -1)
        f_s50_i4 = result.get('snapkv_50_int4', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={f_full:.3f} "
                     f"q2c50_fp={f_q50_fp:.3f} q2c50_i4={f_q50_i4:.3f} "
                     f"snap50_fp={f_s50_fp:.3f} snap50_i4={f_s50_i4:.3f} ({elapsed:.1f}s)")

    summary = compute_summary(results)

    logger.info(f"\n--- {model_name} Combined Pipeline Summary (n={len(results)}) ---")
    for k in sorted(summary.keys()):
        if k.endswith('_f1'):
            logger.info(f"  {k}: {summary[k]:.4f}")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': model_name, 'dtype': model_dtype,
                   'generation': 'manual_generate_with_mask', 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()
    interim_path = RESULTS_DIR / f'{exp_name}_INTERIM.json'
    if interim_path.exists(): interim_path.unlink()

    return summary


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    np.random.seed(42)
    torch.manual_seed(42)

    # === 3B combined pipeline ===
    logger.info("\n" + "="*80 + "\nQwen2.5-3B Combined Pipeline\n" + "="*80)
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_3b.config.use_cache = True
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tok_3b.pad_token is None: tok_3b.pad_token = tok_3b.eos_token

    s3b = run_combined_experiment(model_3b, tok_3b, "qwen25_3b", "fp16", num_samples=50)
    del model_3b; torch.cuda.empty_cache(); gc.collect()

    # === 7B combined pipeline ===
    logger.info("\n" + "="*80 + "\nQwen2.5-7B Combined Pipeline\n" + "="*80)
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_7b.config.use_cache = True
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tok_7b.pad_token is None: tok_7b.pad_token = tok_7b.eos_token

    s7b = run_combined_experiment(model_7b, tok_7b, "qwen25_7b", "bf16", num_samples=50)
    del model_7b; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 9 COMPLETE in {elapsed:.1f} minutes")
