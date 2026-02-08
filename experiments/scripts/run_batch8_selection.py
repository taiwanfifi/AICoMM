#!/usr/bin/env python3
"""
Batch 8: Selection comparison with corrected generation path.

Motivation: Batch 5 used model.generate() with attention mask, which Topic 18
showed gives ~19% lower F1 than manual_generate_with_mask(). This re-runs the
Q2C vs H2O vs SnapKV vs Random comparison using the correct generation path.

Also: Tests the same methods on Qwen2.5-7B (BF16) to see if method rankings hold.

Experiments:
A. 3B selection comparison (manual_generate path) — 50 samples
   - Q2C at 25%, 50%, 75%
   - H2O at 25%, 50%, 75%
   - SnapKV at 25%, 50%, 75%
   - Random at 25%, 50%, 75%

B. 7B selection comparison (manual_generate path) — 50 samples
   - Same methods and retention levels
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch8.log')])
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


def compute_h2o_scores(attentions, seq_len):
    """H2O: Heavy-Hitter Oracle — positions that are heavily attended across ALL queries."""
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        # Sum attention received by each key from ALL query positions
        scores += layer_attn[0].sum(dim=(0, 1))  # Same as SnapKV for now
    return scores


def compute_q2c_scores(attentions, question_start, question_end, seq_len):
    """Q2C: Question-to-Context — positions attended by question tokens."""
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
        # SnapKV uses attention from a window of recent tokens
        scores += layer_attn[0, :, -window:, :].sum(dim=(0, 1))
    return scores


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def run_selection_experiment(model, tokenizer, model_name, model_dtype, num_samples=50):
    """Run selection comparison for a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exp_name = f'selection_{model_name}'
    logger.info(f"\n{'='*60}\n{model_name} Selection Comparison ({num_samples} samples)\n{'='*60}")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)
    retentions = [0.25, 0.50, 0.75]

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

        # === Baseline (full KV) ===
        with torch.no_grad():
            gen_full = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        ans_full = tokenizer.decode(gen_full[0][seq_len:], skip_special_tokens=True).strip()
        result['full'] = {'answer': ans_full, 'f1': compute_f1(ans_full, gold)}

        # === Compute attention scores ===
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
                    result[f'{method_name}_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                except Exception as e:
                    logger.warning(f"{method_name}_{ret_key} failed: {e}")
                    result[f'{method_name}_{ret_key}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

            # Random selection
            if num_context >= k:
                random_indices = np.random.choice(num_context, size=k, replace=False)
                random_selected = set(context_positions[j] for j in random_indices)
                try:
                    ans = generate_with_selection(model, tokenizer, input_ids, seq_len,
                                                  random_selected, always_keep)
                    result[f'random_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                except Exception as e:
                    result[f'random_{ret_key}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        f_full = result['full']['f1']
        f_q2c50 = result.get('q2c_50', {}).get('f1', -1)
        f_snap50 = result.get('snapkv_50', {}).get('f1', -1)
        f_h2o50 = result.get('h2o_50', {}).get('f1', -1)
        f_rand50 = result.get('random_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={f_full:.3f} "
                     f"q2c50={f_q2c50:.3f} snap50={f_snap50:.3f} "
                     f"h2o50={f_h2o50:.3f} rand50={f_rand50:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = compute_summary(results)

    logger.info(f"\n--- {model_name} Selection Summary (n={len(results)}) ---")
    for k in sorted(summary.keys()):
        if k == 'num_samples': continue
        if k.endswith('_f1'):
            logger.info(f"  {k}: {summary[k]:.4f}")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': model_name, 'dtype': model_dtype,
                   'generation': 'manual_generate_with_mask', 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    # Cleanup checkpoint
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

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # === Part A: Qwen2.5-3B selection (FP16) ===
    logger.info("\n" + "="*80 + "\nPART A: Qwen2.5-3B Selection Comparison\n" + "="*80)
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_3b.config.use_cache = True
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tok_3b.pad_token is None: tok_3b.pad_token = tok_3b.eos_token

    summary_3b = run_selection_experiment(model_3b, tok_3b, "qwen25_3b", "fp16", num_samples=50)
    del model_3b; torch.cuda.empty_cache(); gc.collect()

    # === Part B: Qwen2.5-7B selection (BF16) ===
    logger.info("\n" + "="*80 + "\nPART B: Qwen2.5-7B Selection Comparison\n" + "="*80)
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_7b.config.use_cache = True
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tok_7b.pad_token is None: tok_7b.pad_token = tok_7b.eos_token

    summary_7b = run_selection_experiment(model_7b, tok_7b, "qwen25_7b", "bf16", num_samples=50)
    del model_7b; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 8 COMPLETE in {elapsed:.1f} minutes")
    logger.info("3B summary:")
    for k, v in sorted(summary_3b.items()):
        if k.endswith('_f1'): logger.info(f"  {k}: {v:.4f}")
    logger.info("7B summary:")
    for k, v in sorted(summary_7b.items()):
        if k.endswith('_f1'): logger.info(f"  {k}: {v:.4f}")
