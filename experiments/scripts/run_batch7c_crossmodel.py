#!/usr/bin/env python3
"""
Batch 7c: Cross-model KV-cache transfer experiments.

Core question: Can we send KV-cache from 3B model and use it for generation
with the 7B model? We know keys transfer (CKA=0.995, cos=0.9997) but values don't.

Experiments:
1. 3B full_kv → 7B: Project keys only, recompute values (key-only transfer)
2. 3B full_kv → 7B: Project both keys and values (full projection)
3. 3B full_kv → 7B: Use 3B keys as-is (no projection, since RoPE aligns them)
4. Baseline: 7B generates from text directly (no KV transfer)
5. Baseline: 3B generates from text directly

Key insight from earlier experiments:
- Keys have cos_sim=0.9997 between 3B and 7B (same RoPE space)
- Values have cos_sim=0.222 (completely different)
- 7B has 28 layers vs 3B's 36 layers (need layer mapping)
- 7B has 4 KV heads (GQA) vs 3B's 2 KV heads (need head mapping)
- Both have head_dim=128

Strategy:
- Since we can't easily map between different layer/head counts,
  we test a simpler approach: 3B processes context, extracts answer signal,
  and we measure how much of that signal transfers to 7B.
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch7c.log')])
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
    summary = {}
    if results:
        all_keys = set()
        for r in results:
            for k, v in r.items():
                if isinstance(v, dict) and 'f1' in v:
                    all_keys.add(k)
        for k in sorted(all_keys):
            vals = [r[k]['f1'] for r in results if k in r and isinstance(r[k], dict)]
            if vals:
                summary[f'{k}_f1'] = float(np.mean(vals))
    with open(interim_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)


def load_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        logger.info(f"[RESUME] {exp_name} from sample {ckpt['idx'] + 1}")
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def run_crossmodel_analysis(num_samples=30):
    """
    Compare 3B vs 7B KV-cache properties for cross-model transfer analysis.

    Since 3B (36 layers, 2 KV heads) and 7B (28 layers, 4 KV heads) have
    different architectures, direct KV injection is complex. Instead, we:

    1. Measure per-layer CKA/cosine between 3B and 7B keys/values (structural similarity)
    2. Test if 3B's Q2C attention scores transfer to 7B (do they select the same positions?)
    3. Measure: if 3B selects top-50% positions, what F1 does 7B get using those positions?
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exp_name = 'crossmodel_transfer'
    logger.info(f"\n{'='*60}\nCross-Model Transfer Analysis ({num_samples} samples)\n{'='*60}")

    # Load both models
    logger.info("Loading 3B model...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_3b.config.use_cache = True
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tok_3b.pad_token is None: tok_3b.pad_token = tok_3b.eos_token

    logger.info(f"3B: {model_3b.config.num_hidden_layers} layers, "
                f"{model_3b.config.num_key_value_heads} KV heads, "
                f"head_dim={model_3b.config.hidden_size // model_3b.config.num_attention_heads}")

    # Check VRAM before loading 7B
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM after 3B: {vram_used:.1f}/{vram_total:.1f} GB")

    # For 7B, we need to check if we have enough VRAM
    # Qwen2.5-7B in FP16 is ~14GB, 3B is ~6GB, so we need ~20GB total
    # With 98GB VRAM we're fine

    logger.info("Loading 7B model...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_7b.config.use_cache = True
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tok_7b.pad_token is None: tok_7b.pad_token = tok_7b.eos_token

    logger.info(f"7B: {model_7b.config.num_hidden_layers} layers, "
                f"{model_7b.config.num_key_value_heads} KV heads, "
                f"head_dim={model_7b.config.hidden_size // model_7b.config.num_attention_heads}")

    vram_used = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after both models: {vram_used:.1f}/{vram_total:.1f} GB")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        context = sample['context']
        question = sample['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        # Both models use same tokenizer family, but tokenize separately to be safe
        inputs_3b = tok_3b(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        inputs_7b = tok_7b(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len_3b = inputs_3b['input_ids'].shape[1]
        seq_len_7b = inputs_7b['input_ids'].shape[1]

        # Token ranges (for 3B — should be same as 7B since same tokenizer)
        ctx_prefix = tok_3b("Context: ", add_special_tokens=True, return_tensors="pt")
        ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
        ctx_only = f"Context: {context}\nQuestion: "
        ctx_tokens = tok_3b(ctx_only, return_tensors="pt", max_length=512, truncation=True)
        context_end = min(ctx_tokens['input_ids'].shape[1], seq_len_3b)
        question_start = context_end
        answer_suffix = tok_3b("\nAnswer:", add_special_tokens=False)['input_ids']
        question_end = seq_len_3b - len(answer_suffix)
        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len_3b))

        if num_context < 5:
            continue

        result = {'idx': i, 'gold': gold, 'seq_len_3b': seq_len_3b, 'seq_len_7b': seq_len_7b,
                  'num_context': num_context, 'same_tokenization': seq_len_3b == seq_len_7b}

        # === 3B baseline ===
        with torch.no_grad():
            gen_3b = model_3b.generate(**inputs_3b, max_new_tokens=64, do_sample=False)
        ans_3b = tok_3b.decode(gen_3b[0][seq_len_3b:], skip_special_tokens=True).strip()
        result['3b_full'] = {'answer': ans_3b, 'f1': compute_f1(ans_3b, gold)}

        # === 7B baseline ===
        with torch.no_grad():
            gen_7b = model_7b.generate(**inputs_7b, max_new_tokens=64, do_sample=False)
        ans_7b = tok_7b.decode(gen_7b[0][seq_len_7b:], skip_special_tokens=True).strip()
        result['7b_full'] = {'answer': ans_7b, 'f1': compute_f1(ans_7b, gold)}

        # === 3B Q2C attention scores ===
        with torch.no_grad():
            out_3b = model_3b(input_ids=inputs_3b['input_ids'], output_attentions=True, use_cache=False)
        q2c_3b = torch.zeros(seq_len_3b, device='cuda')
        snapkv_3b = torch.zeros(seq_len_3b, device='cuda')
        for layer_attn in out_3b.attentions:
            snapkv_3b += layer_attn[0].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_3b += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
        del out_3b; torch.cuda.empty_cache()

        # === 7B Q2C attention scores ===
        with torch.no_grad():
            out_7b = model_7b(input_ids=inputs_7b['input_ids'], output_attentions=True, use_cache=False)
        q2c_7b = torch.zeros(seq_len_7b, device='cuda')
        snapkv_7b = torch.zeros(seq_len_7b, device='cuda')
        for layer_attn in out_7b.attentions:
            snapkv_7b += layer_attn[0].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_7b += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
        del out_7b; torch.cuda.empty_cache()

        # === Cross-model attention transfer: 3B scores → 7B selection ===
        # Key question: If 3B identifies important positions, does 7B agree?

        if seq_len_3b == seq_len_7b:  # Same tokenization (should be, same tokenizer family)
            for retention in [0.50, 0.75]:
                k = max(1, int(num_context * retention))
                ret_key = int(retention * 100)

                # 3B Q2C scores → select positions → apply to 7B
                ctx_sc_3b = q2c_3b[torch.tensor(context_positions, device='cuda')]
                _, topk_3b = ctx_sc_3b.topk(k)
                selected_3b = set(context_positions[j] for j in topk_3b.cpu().numpy())

                # 7B Q2C scores → select positions (7B's own selection)
                ctx_sc_7b = q2c_7b[torch.tensor(context_positions, device='cuda')]
                _, topk_7b = ctx_sc_7b.topk(k)
                selected_7b = set(context_positions[j] for j in topk_7b.cpu().numpy())

                # Overlap between 3B and 7B selections
                overlap = len(selected_3b & selected_7b) / len(selected_3b) if selected_3b else 0
                result[f'selection_overlap_{ret_key}'] = overlap

                # 7B with 7B's own Q2C selection (control)
                mask_7b_own = torch.zeros(1, seq_len_7b, device='cuda', dtype=torch.long)
                for p in always_keep: mask_7b_own[0, p] = 1
                for p in selected_7b: mask_7b_own[0, p] = 1
                with torch.no_grad():
                    gen = model_7b.generate(input_ids=inputs_7b['input_ids'],
                                            attention_mask=mask_7b_own, max_new_tokens=64, do_sample=False)
                ans = tok_7b.decode(gen[0][seq_len_7b:], skip_special_tokens=True).strip()
                result[f'7b_own_q2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

                # 7B with 3B's Q2C selection (cross-model transfer)
                mask_7b_from3b = torch.zeros(1, seq_len_7b, device='cuda', dtype=torch.long)
                for p in always_keep: mask_7b_from3b[0, p] = 1
                for p in selected_3b: mask_7b_from3b[0, p] = 1
                with torch.no_grad():
                    gen = model_7b.generate(input_ids=inputs_7b['input_ids'],
                                            attention_mask=mask_7b_from3b, max_new_tokens=64, do_sample=False)
                ans = tok_7b.decode(gen[0][seq_len_7b:], skip_special_tokens=True).strip()
                result[f'7b_3bq2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

                # 3B with 3B's own Q2C selection (self-reference)
                mask_3b_own = torch.zeros(1, seq_len_3b, device='cuda', dtype=torch.long)
                for p in always_keep: mask_3b_own[0, p] = 1
                for p in selected_3b: mask_3b_own[0, p] = 1
                with torch.no_grad():
                    gen = model_3b.generate(input_ids=inputs_3b['input_ids'],
                                            attention_mask=mask_3b_own, max_new_tokens=64, do_sample=False)
                ans = tok_3b.decode(gen[0][seq_len_3b:], skip_special_tokens=True).strip()
                result[f'3b_own_q2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

                # SnapKV overlap
                snap_sc_3b = snapkv_3b[torch.tensor(context_positions, device='cuda')]
                _, snap_topk_3b = snap_sc_3b.topk(k)
                snap_selected_3b = set(context_positions[j] for j in snap_topk_3b.cpu().numpy())

                snap_sc_7b = snapkv_7b[torch.tensor(context_positions, device='cuda')]
                _, snap_topk_7b = snap_sc_7b.topk(k)
                snap_selected_7b = set(context_positions[j] for j in snap_topk_7b.cpu().numpy())

                snap_overlap = len(snap_selected_3b & snap_selected_7b) / len(snap_selected_3b) if snap_selected_3b else 0
                result[f'snapkv_overlap_{ret_key}'] = snap_overlap

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        q2c_ol_50 = result.get('selection_overlap_50', 'N/A')
        logger.info(f"  [{i+1}/{num_samples}] 3b={result['3b_full']['f1']:.3f} "
                     f"7b={result['7b_full']['f1']:.3f} "
                     f"7b_own_q2c50={result.get('7b_own_q2c_50', {}).get('f1', 'N/A')} "
                     f"7b_3bq2c50={result.get('7b_3bq2c_50', {}).get('f1', 'N/A')} "
                     f"q2c_overlap50={q2c_ol_50} ({elapsed:.1f}s)")

    # Cleanup
    del model_3b, model_7b; torch.cuda.empty_cache(); gc.collect()

    # Summary
    summary = {'num_samples': len(results)}
    all_keys = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, dict) and 'f1' in v:
                all_keys.add(k)
            elif k.startswith('selection_overlap') or k.startswith('snapkv_overlap'):
                all_keys.add(k)

    logger.info(f"\n--- Cross-Model Transfer Summary ---")
    for k in sorted(all_keys):
        if k.startswith('selection_overlap') or k.startswith('snapkv_overlap'):
            vals = [r[k] for r in results if k in r]
            if vals:
                summary[f'{k}_mean'] = float(np.mean(vals))
                summary[f'{k}_std'] = float(np.std(vals))
                logger.info(f"  {k}: {summary[f'{k}_mean']:.3f} +/- {summary[f'{k}_std']:.3f}")
        elif isinstance(results[0].get(k), dict) and 'f1' in results[0].get(k, {}):
            vals = [r[k]['f1'] for r in results if k in r and isinstance(r[k], dict)]
            if vals:
                summary[f'{k}_f1'] = float(np.mean(vals))
                summary[f'{k}_std'] = float(np.std(vals))
                logger.info(f"  {k}: {summary[f'{k}_f1']:.3f} +/- {summary[f'{k}_std']:.3f}")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()
    interim_path = RESULTS_DIR / f'{exp_name}_INTERIM.json'
    if interim_path.exists(): interim_path.unlink()

    return summary


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t0 = time.time()

    s = run_crossmodel_analysis(num_samples=30)

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nDONE in {elapsed:.1f} minutes")
