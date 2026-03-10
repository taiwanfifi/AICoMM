#!/usr/bin/env python3
"""
Batch 7c v2: Cross-model KV-cache transfer experiments (FIXED).

Bug fix: 7B model needs bfloat16 (not float16) with eager attention on Blackwell GPU.
FP16 causes numerical overflow in attention computation for larger models.

Core question: Can a small model's Q2C selection transfer to a larger model?
If 3B identifies important context positions, does 7B agree and benefit?

Experiments per sample:
1. 3B baseline (full KV)
2. 7B baseline (full KV)
3. 3B Q2C selection at 50%/75% → measure F1
4. 7B Q2C selection at 50%/75% → measure F1 (7B's own selection)
5. 3B Q2C scores → 7B selection at 50%/75% → measure F1 (CROSS-MODEL TRANSFER)
6. Q2C overlap between 3B and 7B (do they agree on what's important?)
7. SnapKV overlap between 3B and 7B (task-agnostic comparison)

Uses manual_generate_with_mask for selection experiments (more accurate, per Topic 18 finding).
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch7c_v2.log')])
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
            elif k.startswith('selection_overlap') or k.startswith('snapkv_overlap'):
                all_keys.add(k)
    for k in sorted(all_keys):
        if k.startswith('selection_overlap') or k.startswith('snapkv_overlap'):
            vals = [r[k] for r in results if k in r]
            if vals:
                summary[f'{k}_mean'] = float(np.mean(vals))
                summary[f'{k}_std'] = float(np.std(vals))
        else:
            vals = [r[k]['f1'] for r in results if k in r and isinstance(r[k], dict)]
            if vals:
                summary[f'{k}_f1'] = float(np.mean(vals))
                summary[f'{k}_std'] = float(np.std(vals))
    return summary


def manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len,
                               attn_mask_prefix, max_new=64):
    """Token-by-token generation with pre-populated KV cache AND custom attention mask.

    This is the correct generation path for selection experiments (per Topic 18 finding).
    model.generate() handles attention masks suboptimally for KV selection, giving
    ~19% lower F1 than this manual loop.
    """
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
    """Generate answer using manual loop with attention mask for selected positions.

    Steps:
    1. Forward pass through full prompt to get KV cache + first token
    2. Build attention mask (1 for selected+always_keep, 0 for rest)
    3. Manual token-by-token generation with the mask
    """
    # Build attention mask for the prefix
    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
    for p in always_keep:
        if p < seq_len:
            mask[0, p] = 1
    for p in selected_positions:
        if p < seq_len:
            mask[0, p] = 1

    # Forward pass to get KV cache and first generated token
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    past_kv = out.past_key_values

    return manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new)


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def run_crossmodel_analysis(num_samples=50):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exp_name = 'crossmodel_v2'
    logger.info(f"\n{'='*60}\nCross-Model Transfer Analysis v2 ({num_samples} samples)\n{'='*60}")
    logger.info("FIX: 7B uses bfloat16 (FP16 overflows with eager on Blackwell)")
    logger.info("FIX: Using manual_generate_with_mask for selection (per Topic 18)")

    # Load 3B model (FP16 is fine for 3B)
    logger.info("Loading 3B model (fp16, eager)...")
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

    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM after 3B: {vram_used:.1f}/{vram_total:.1f} GB")

    # Load 7B model — MUST use bfloat16 for eager attention on Blackwell
    logger.info("Loading 7B model (bfloat16, eager)...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
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

    # Quick sanity check
    logger.info("Quick sanity check on 7B...")
    test_inputs = tok_7b("Context: Paris is the capital of France.\nQuestion: What is the capital?\nAnswer:",
                         return_tensors="pt").to("cuda")
    with torch.no_grad():
        test_gen = model_7b.generate(**test_inputs, max_new_tokens=10, do_sample=False)
    test_ans = tok_7b.decode(test_gen[0][test_inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    logger.info(f"7B sanity check: '{test_ans}' (verifying not garbage)")
    # FP16+eager produces "!" garbage; BF16 should produce actual text
    assert test_ans and test_ans not in ('!', '! Norm!', '!!'), f"7B still broken: {test_ans}"

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        context = sample['context']
        question = sample['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        inputs_3b = tok_3b(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        inputs_7b = tok_7b(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len_3b = inputs_3b['input_ids'].shape[1]
        seq_len_7b = inputs_7b['input_ids'].shape[1]

        # Token ranges
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

        # === BASELINES (full KV, model.generate) ===
        with torch.no_grad():
            gen_3b = model_3b.generate(**inputs_3b, max_new_tokens=64, do_sample=False)
        ans_3b = tok_3b.decode(gen_3b[0][seq_len_3b:], skip_special_tokens=True).strip()
        result['3b_full'] = {'answer': ans_3b, 'f1': compute_f1(ans_3b, gold)}

        with torch.no_grad():
            gen_7b = model_7b.generate(**inputs_7b, max_new_tokens=64, do_sample=False)
        ans_7b = tok_7b.decode(gen_7b[0][seq_len_7b:], skip_special_tokens=True).strip()
        result['7b_full'] = {'answer': ans_7b, 'f1': compute_f1(ans_7b, gold)}

        # === ATTENTION SCORES ===
        # 3B Q2C + SnapKV scores
        with torch.no_grad():
            out_3b = model_3b(input_ids=inputs_3b['input_ids'], output_attentions=True, use_cache=False)
        q2c_3b = torch.zeros(seq_len_3b, device='cuda')
        snapkv_3b = torch.zeros(seq_len_3b, device='cuda')
        for layer_attn in out_3b.attentions:
            snapkv_3b += layer_attn[0].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_3b += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
        del out_3b; torch.cuda.empty_cache()

        # 7B Q2C + SnapKV scores
        with torch.no_grad():
            out_7b = model_7b(input_ids=inputs_7b['input_ids'], output_attentions=True, use_cache=False)
        q2c_7b = torch.zeros(seq_len_7b, device='cuda')
        snapkv_7b = torch.zeros(seq_len_7b, device='cuda')
        for layer_attn in out_7b.attentions:
            snapkv_7b += layer_attn[0].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_7b += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
        del out_7b; torch.cuda.empty_cache()

        # === SELECTION + GENERATION ===
        if seq_len_3b == seq_len_7b:
            for retention in [0.50, 0.75]:
                k = max(1, int(num_context * retention))
                ret_key = int(retention * 100)

                # Compute selections
                ctx_sc_3b = q2c_3b[torch.tensor(context_positions, device='cuda')]
                _, topk_3b = ctx_sc_3b.topk(k)
                selected_3b = set(context_positions[j] for j in topk_3b.cpu().numpy())

                ctx_sc_7b = q2c_7b[torch.tensor(context_positions, device='cuda')]
                _, topk_7b = ctx_sc_7b.topk(k)
                selected_7b = set(context_positions[j] for j in topk_7b.cpu().numpy())

                # Q2C selection overlap
                overlap = len(selected_3b & selected_7b) / len(selected_3b) if selected_3b else 0
                result[f'q2c_overlap_{ret_key}'] = overlap

                # SnapKV overlap
                snap_sc_3b = snapkv_3b[torch.tensor(context_positions, device='cuda')]
                _, snap_topk_3b = snap_sc_3b.topk(k)
                snap_selected_3b = set(context_positions[j] for j in snap_topk_3b.cpu().numpy())
                snap_sc_7b = snapkv_7b[torch.tensor(context_positions, device='cuda')]
                _, snap_topk_7b = snap_sc_7b.topk(k)
                snap_selected_7b = set(context_positions[j] for j in snap_topk_7b.cpu().numpy())
                snap_overlap = len(snap_selected_3b & snap_selected_7b) / len(snap_selected_3b) if snap_selected_3b else 0
                result[f'snapkv_overlap_{ret_key}'] = snap_overlap

                # === F1 measurements with manual_generate (per Topic 18 fix) ===

                # 3B with 3B's own Q2C selection
                try:
                    ans = generate_with_selection(model_3b, tok_3b, inputs_3b['input_ids'],
                                                  seq_len_3b, selected_3b, always_keep)
                    result[f'3b_q2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                except Exception as e:
                    logger.warning(f"3b_q2c_{ret_key} failed: {e}")
                    result[f'3b_q2c_{ret_key}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

                # 7B with 7B's own Q2C selection
                try:
                    ans = generate_with_selection(model_7b, tok_7b, inputs_7b['input_ids'],
                                                  seq_len_7b, selected_7b, always_keep)
                    result[f'7b_q2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                except Exception as e:
                    logger.warning(f"7b_q2c_{ret_key} failed: {e}")
                    result[f'7b_q2c_{ret_key}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

                # 7B with 3B's Q2C selection (THE CROSS-MODEL TRANSFER)
                try:
                    ans = generate_with_selection(model_7b, tok_7b, inputs_7b['input_ids'],
                                                  seq_len_7b, selected_3b, always_keep)
                    result[f'7b_from3b_q2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                except Exception as e:
                    logger.warning(f"7b_from3b_q2c_{ret_key} failed: {e}")
                    result[f'7b_from3b_q2c_{ret_key}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

                # 3B with 7B's Q2C selection (reverse transfer — does 7B know better?)
                try:
                    ans = generate_with_selection(model_3b, tok_3b, inputs_3b['input_ids'],
                                                  seq_len_3b, selected_7b, always_keep)
                    result[f'3b_from7b_q2c_{ret_key}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}
                except Exception as e:
                    logger.warning(f"3b_from7b_q2c_{ret_key} failed: {e}")
                    result[f'3b_from7b_q2c_{ret_key}'] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        # Compact log line
        f3b = result['3b_full']['f1']
        f7b = result['7b_full']['f1']
        f7b_own = result.get('7b_q2c_50', {}).get('f1', -1)
        f7b_xfer = result.get('7b_from3b_q2c_50', {}).get('f1', -1)
        ol50 = result.get('q2c_overlap_50', -1)
        logger.info(f"  [{i+1}/{num_samples}] 3b={f3b:.3f} 7b={f7b:.3f} "
                     f"7b_q2c50={f7b_own:.3f} 7b_from3b50={f7b_xfer:.3f} "
                     f"overlap50={ol50:.3f} ({elapsed:.1f}s)")

    # Cleanup
    del model_3b, model_7b; torch.cuda.empty_cache(); gc.collect()

    # Final summary
    summary = compute_summary(results)

    logger.info(f"\n{'='*60}\nCross-Model Transfer Summary (n={len(results)})\n{'='*60}")
    for k in sorted(summary.keys()):
        if k == 'num_samples': continue
        logger.info(f"  {k}: {summary[k]:.4f}")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    # Cleanup checkpoint
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

    s = run_crossmodel_analysis(num_samples=50)

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nDONE in {elapsed:.1f} minutes")
