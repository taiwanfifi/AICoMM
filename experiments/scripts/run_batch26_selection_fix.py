#!/usr/bin/env python3
"""
Batch 26: Selection comparison for Mistral-7B and Qwen-14B with CORRECT attention masking.

Fixes the batch 25 bug where Q2C selection zeroed KV values instead of using attention masks.
Uses the batch 8 methodology (attention mask approach) which preserves RoPE positions correctly.

Tests: Q2C, SnapKV, H2O, Random at 25%, 50%, 75% retention on SQuAD v2.
Uses normalized F1 (consistent with batch 25 quantization results).
"""
import os, sys, json, time, logging, copy, gc, re, string
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch26.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_answer(s):
    """SQuAD-style answer normalization."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def compute_f1(pred, gold):
    """Token-F1 with normalization."""
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    pred_tokens = pred_n.split()
    gold_tokens = gold_n.split()
    if not gold_tokens: return (1.0, 1.0) if not pred_tokens else (0.0, 0.0)
    if not pred_tokens: return (0.0, 0.0)
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return (0.0, 0.0)
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    f1_norm = 2 * p * r / (p + r)
    # Also compute raw F1
    pred_raw = pred.lower().split()
    gold_raw = gold.lower().split()
    common_raw = set(pred_raw) & set(gold_raw)
    if not common_raw:
        f1_raw = 0.0
    else:
        p_r = len(common_raw) / len(pred_raw)
        r_r = len(common_raw) / len(gold_raw)
        f1_raw = 2 * p_r * r_r / (p_r + r_r)
    return f1_norm, f1_raw


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


def generate_with_selection(model, tokenizer, input_ids, seq_len, selected_positions, always_keep, max_new=64):
    """Generate answer using manual loop with attention mask for selected positions."""
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

    return manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new)


def compute_q2c_scores(attentions, question_start, question_end, seq_len):
    """Q2C: Query-to-Context attention from ALL layers (matches batch 8 methodology)."""
    scores = torch.zeros(seq_len, device='cuda')
    if question_end <= question_start:
        return scores
    for layer_attn in attentions:
        scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
    return scores


def compute_snapkv_scores(attentions, seq_len):
    """SnapKV: Observation window attention — use last 32 tokens as query window."""
    window = min(32, seq_len)
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        scores += layer_attn[0, :, -window:, :].sum(dim=(0, 1))
    return scores


def compute_h2o_scores(attentions, seq_len):
    """H2O: Heavy-Hitter Oracle — cumulative attention across all layers."""
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        scores += layer_attn[0].sum(dim=(0, 1))
    return scores


def load_squad(tokenizer, num_samples=50):
    """Load SQuAD v2 samples with same filtering as batch 25."""
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    samples = []
    for s in ds:
        if not s['answers']['text']: continue
        ctx = s['context']
        q = s['question']
        prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer briefly:"
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 400:
            samples.append({
                'prompt': prompt, 'gold': s['answers']['text'][0],
                'context': ctx, 'question': q,
                'tok_len': tok_len,
            })
        if len(samples) >= num_samples * 3: break
    np.random.seed(42)
    idx = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in idx]


def run_selection_experiment(model, tokenizer, model_name, num_samples=50):
    """Run Q2C/SnapKV/H2O/Random selection at 25%, 50%, 75% retention."""
    logger.info(f"\n{'='*60}\n{model_name} Selection Comparison ({num_samples} samples)\n{'='*60}")

    samples = load_squad(tokenizer, num_samples)
    retentions = [0.25, 0.50, 0.75]
    results = []

    for i, sample in enumerate(samples):
        t0 = time.time()
        gold = sample['gold']
        prompt = sample['prompt']

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        # Token ranges: find context vs question boundary
        ctx_prefix = tokenizer("Context: ", add_special_tokens=True, return_tensors="pt")
        ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
        ctx_only = f"Context: {sample['context']}\nQuestion: "
        ctx_tokens = tokenizer(ctx_only, return_tensors="pt", max_length=512, truncation=True)
        context_end = min(ctx_tokens['input_ids'].shape[1], seq_len)
        question_start = context_end
        answer_suffix = tokenizer("\nAnswer briefly:", add_special_tokens=False)['input_ids']
        question_end = seq_len - len(answer_suffix)
        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        if num_context < 5:
            continue

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_context}

        # === Baseline (full KV) ===
        with torch.no_grad():
            out_full = model(input_ids=input_ids, use_cache=True)
        first_tok = out_full.logits[:, -1, :].argmax(dim=-1).item()
        full_mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)
        ans_full = manual_generate_with_mask(model, tokenizer, out_full.past_key_values,
                                              first_tok, seq_len, full_mask)
        f1_norm, f1_raw = compute_f1(ans_full, gold)
        result['full'] = {'answer': ans_full, 'f1': f1_norm, 'f1_raw': f1_raw}
        del out_full; torch.cuda.empty_cache()

        # === Compute attention scores (output_attentions=True) ===
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)

        q2c_scores = compute_q2c_scores(out.attentions, question_start, question_end, seq_len)
        snapkv_scores = compute_snapkv_scores(out.attentions, seq_len)
        h2o_scores = compute_h2o_scores(out.attentions, seq_len)
        del out; torch.cuda.empty_cache()

        ctx_tensor = torch.tensor(context_positions, device='cuda')

        for retention in retentions:
            k = max(1, int(num_context * retention))
            ret_key = int(retention * 100)

            methods = {
                'q2c': q2c_scores,
                'snapkv': snapkv_scores,
                'h2o': h2o_scores,
            }

            for method_name, scores in methods.items():
                ctx_sc = scores[ctx_tensor]
                _, topk_idx = ctx_sc.topk(k)
                selected = set(context_positions[j] for j in topk_idx.cpu().numpy())

                try:
                    ans = generate_with_selection(model, tokenizer, input_ids, seq_len,
                                                  selected, always_keep)
                    f1_n, f1_r = compute_f1(ans, gold)
                    result[f'{method_name}_{ret_key}'] = {'answer': ans, 'f1': f1_n, 'f1_raw': f1_r}
                except Exception as e:
                    logger.warning(f"{method_name}_{ret_key} failed: {e}")
                    result[f'{method_name}_{ret_key}'] = {'answer': '', 'f1': 0.0, 'f1_raw': 0.0, 'error': str(e)}

            # Random selection
            if num_context >= k:
                random_indices = np.random.choice(num_context, size=k, replace=False)
                random_selected = set(context_positions[j] for j in random_indices)
                try:
                    ans = generate_with_selection(model, tokenizer, input_ids, seq_len,
                                                  random_selected, always_keep)
                    f1_n, f1_r = compute_f1(ans, gold)
                    result[f'random_{ret_key}'] = {'answer': ans, 'f1': f1_n, 'f1_raw': f1_r}
                except Exception as e:
                    result[f'random_{ret_key}'] = {'answer': '', 'f1': 0.0, 'f1_raw': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        f_full = result['full']['f1']
        f_q2c50 = result.get('q2c_50', {}).get('f1', -1)
        f_snap50 = result.get('snapkv_50', {}).get('f1', -1)
        f_h2o50 = result.get('h2o_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={f_full:.3f} "
                     f"q2c50={f_q2c50:.3f} snap50={f_snap50:.3f} "
                     f"h2o50={f_h2o50:.3f} ({elapsed:.1f}s)")

    # Compute summary
    summary = {'num_samples': len(results)}
    all_keys = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, dict) and 'f1' in v:
                all_keys.add(k)
    for k in sorted(all_keys):
        vals = [r[k]['f1'] for r in results if k in r and isinstance(r[k], dict)]
        vals_raw = [r[k].get('f1_raw', r[k]['f1']) for r in results if k in r and isinstance(r[k], dict)]
        if vals:
            summary[f'{k}_f1'] = float(np.mean(vals))
            summary[f'{k}_f1_std'] = float(np.std(vals))
            summary[f'{k}_f1_se'] = float(np.std(vals) / np.sqrt(len(vals)))
            summary[f'{k}_f1_raw'] = float(np.mean(vals_raw))

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY — {model_name} (n={len(results)})")
    logger.info(f"Baseline: F1={summary.get('full_f1', 0):.4f} (normalized)")
    logger.info(f"{'='*60}")
    for ret in [75, 50, 25]:
        line = f"  {ret}%: "
        for method in ['q2c', 'snapkv', 'h2o', 'random']:
            key = f'{method}_{ret}_f1'
            if key in summary:
                pct = summary[key] / summary['full_f1'] * 100 if summary.get('full_f1', 0) > 0 else 0
                line += f"{method}={summary[key]:.3f}({pct:.0f}%) "
        logger.info(line)

    return {'metadata': summary, 'model': model_name, 'normalized_f1': True,
            'methodology': 'attention_mask (batch 8 style)', 'results': results}


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    np.random.seed(42)
    torch.manual_seed(42)

    # === Mistral-7B Selection ===
    logger.info("\n" + "="*80 + "\nMistral-7B-Instruct-v0.3 Selection\n" + "="*80)
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    mistral_data = run_selection_experiment(model, tok, "Mistral-7B", num_samples=50)
    fpath = RESULTS_DIR / f'selection_mistral7b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fpath, 'w') as f:
        json.dump(mistral_data, f, indent=2, default=str)
    logger.info(f"[SAVED] {fpath}")

    del model, tok; torch.cuda.empty_cache(); gc.collect()

    # === Qwen-14B Selection ===
    logger.info("\n" + "="*80 + "\nQwen2.5-14B Selection\n" + "="*80)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B", trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    qwen14b_data = run_selection_experiment(model, tok, "Qwen-14B", num_samples=50)
    fpath = RESULTS_DIR / f'selection_qwen14b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fpath, 'w') as f:
        json.dump(qwen14b_data, f, indent=2, default=str)
    logger.info(f"[SAVED] {fpath}")

    del model, tok; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nBatch 26 COMPLETE in {elapsed:.1f} minutes")
