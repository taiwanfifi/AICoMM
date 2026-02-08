#!/usr/bin/env python3
"""
Batch 7: Comprehensive experiment suite with aggressive checkpointing.

ALL experiments save after EVERY sample (not every 5). Results are written
to results/ as JSON with timestamps. Checkpoints enable full resume.

Experiments:
1. Extreme quantization (re-run for JSON — batch 6 data lost with server)
2. Combined pipeline (re-run for JSON)
3. TriviaQA validation (second dataset)
4. Topic 18 verification: zeroing vs masking through SAME generation path
5. NaturalQuestions validation (third dataset, if time permits)
"""
import os, sys, json, time, logging, copy, math, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch7.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Core utilities
# ============================================================

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


def manual_generate(model, tokenizer, past_kv, first_token_id, seq_len, max_new=64):
    """Token-by-token generation with pre-populated KV cache."""
    generated = [first_token_id]
    cur_len = seq_len
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        attn_mask = torch.ones(1, cur_len + 1, device='cuda', dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       attention_mask=attn_mask, position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, attn_mask_prefix, max_new=64):
    """Token-by-token generation with pre-populated KV cache AND custom attention mask.
    This allows zeroed positions AND proper masking through the SAME generation path."""
    generated = [first_token_id]
    cur_len = seq_len
    # attn_mask_prefix is shape (1, seq_len) — the mask for the prefix
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        # Extend mask: prefix mask + 1 for each new token
        new_token_mask = torch.ones(1, 1, device='cuda', dtype=torch.long)
        if step == 0:
            full_mask = torch.cat([attn_mask_prefix, new_token_mask], dim=1)
        else:
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


def quantize_inplace(pkv, bits, group_size=32):
    """Generalized in-place quantization for any bit width."""
    if bits >= 8:
        max_val, min_val = 127, -128
    else:
        max_val = (1 << (bits - 1)) - 1
        min_val = -(1 << (bits - 1))
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            shape = t.shape
            if bits <= 4 and group_size > 0:
                flat = t.reshape(-1)
                pad = (group_size - flat.numel() % group_size) % group_size
                if pad > 0:
                    flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
                flat = flat.reshape(-1, group_size)
                scale = flat.abs().max(dim=1, keepdim=True).values / max_val
                scale = scale.clamp(min=1e-10)
                q = torch.clamp(torch.round(flat / scale), min_val, max_val)
                dq = (q * scale).reshape(-1)[:t.numel()].reshape(shape)
            else:
                scale = t.abs().max() / max_val
                if scale > 0:
                    dq = torch.clamp(torch.round(t / scale), min_val, max_val) * scale
                else:
                    dq = t
            tensor.copy_(dq.to(tensor.dtype))


def quantize_inplace_binary(pkv):
    """Binary (1-bit sign) quantization."""
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            scale = t.abs().mean()
            binary = torch.sign(t) * scale
            tensor.copy_(binary.to(tensor.dtype))


def save_results(name, results, metadata):
    """Save results with timestamp. Returns path."""
    path = RESULTS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump({'metadata': metadata, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] {name} -> {path}")
    return path


def save_checkpoint(exp_name, idx, results):
    """Save checkpoint after EVERY sample."""
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    with open(ckpt_path, 'w') as f:
        json.dump({'idx': idx, 'results': results}, f, default=str)
    # Also save intermediate results file (overwritten each time)
    interim_path = RESULTS_DIR / f'{exp_name}_INTERIM.json'
    summary = compute_summary(exp_name, results)
    with open(interim_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results, 'checkpoint_idx': idx}, f, indent=2, default=str)


def load_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        logger.info(f"[RESUME] {exp_name} from sample {ckpt['idx'] + 1}")
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


def compute_summary(exp_name, results):
    """Compute summary stats from results."""
    summary = {'exp_name': exp_name, 'num_samples': len(results), 'timestamp': str(datetime.now())}
    if not results:
        return summary
    # Collect all keys that have 'f1' sub-key
    all_keys = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, dict) and 'f1' in v:
                all_keys.add(k)
    for k in sorted(all_keys):
        vals = [r[k]['f1'] for r in results if k in r and isinstance(r[k], dict) and 'f1' in r[k]]
        if vals:
            summary[f'{k}_f1'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
    return summary


def cleanup_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        ckpt_path.unlink()
    interim_path = RESULTS_DIR / f'{exp_name}_INTERIM.json'
    if interim_path.exists():
        interim_path.unlink()


def load_model(model_name, need_attentions=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = dict(dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    if need_attentions:
        kwargs['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def load_triviaqa(num_samples):
    """Load TriviaQA with search context."""
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc', split='validation')
    samples = []
    for item in ds:
        if item['answer']['value'] and item['search_results']['search_context']:
            ctx = item['search_results']['search_context'][0]
            if len(ctx) > 50:
                samples.append({
                    'context': ctx[:2000],
                    'question': item['question'],
                    'answers': {'text': [item['answer']['value']]}
                })
                if len(samples) >= num_samples:
                    break
    logger.info(f"Loaded {len(samples)} TriviaQA samples")
    return samples


def get_token_ranges(tokenizer, context, question, input_ids, seq_len):
    """Get token position ranges for context and question."""
    ctx_prefix = tokenizer("Context: ", add_special_tokens=True, return_tensors="pt")
    ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
    ctx_only = f"Context: {context}\nQuestion: "
    ctx_tokens = tokenizer(ctx_only, return_tensors="pt", max_length=512, truncation=True)
    context_end = min(ctx_tokens['input_ids'].shape[1], seq_len)
    question_start = context_end
    answer_suffix = tokenizer("\nAnswer:", add_special_tokens=False)['input_ids']
    question_end = seq_len - len(answer_suffix)
    context_positions = list(range(ctx_prefix_len, context_end))
    always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))
    return context_positions, always_keep, question_start, question_end


def get_q2c_scores(attentions, seq_len, question_start, question_end):
    """Compute Q2C attention scores."""
    q2c = torch.zeros(seq_len, device='cuda')
    if question_end > question_start:
        for layer_attn in attentions:
            q2c += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
    return q2c


def get_snapkv_scores(attentions, seq_len):
    """Compute SnapKV (cumulative attention) scores."""
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        scores += layer_attn[0].sum(dim=(0, 1))
    return scores


def select_positions(scores, context_positions, num_context, retention):
    """Select top-k context positions by score."""
    k = max(1, int(num_context * retention))
    ctx_sc = scores[torch.tensor(context_positions, device='cuda')]
    _, topk = ctx_sc.topk(min(k, len(context_positions)))
    return set(context_positions[j] for j in topk.cpu().numpy())


# ============================================================
# Experiment 1: Extreme Quantization (re-run for JSON)
# ============================================================
def run_extreme_quant(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    exp_name = 'extreme_quant'
    logger.info(f"\n{'='*60}\nExp 1: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name)
    samples = load_squad(num_samples)
    bit_widths = [8, 4, 3, 2]

    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()

        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len,
                  'full': {'answer': full_ans, 'f1': compute_f1(full_ans, gold)}}

        for bits in bit_widths:
            pkv = copy.deepcopy(out.past_key_values)
            quantize_inplace(pkv, bits)
            ans = manual_generate(model, tokenizer, pkv, first_tok, seq_len)
            result[f'int{bits}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        pkv_bin = copy.deepcopy(out.past_key_values)
        quantize_inplace_binary(pkv_bin)
        ans_bin = manual_generate(model, tokenizer, pkv_bin, first_tok, seq_len)
        result['binary'] = {'answer': ans_bin, 'f1': compute_f1(ans_bin, gold)}

        result['time'] = time.time() - t0
        results.append(result)
        save_checkpoint(exp_name, i, results)

        parts = [f"[{i+1}/{num_samples}] full={result['full']['f1']:.3f}"]
        for b in bit_widths: parts.append(f"int{b}={result[f'int{b}']['f1']:.3f}")
        parts.append(f"bin={result['binary']['f1']:.3f}")
        logger.info(f"  {' '.join(parts)} ({result['time']:.1f}s)")

    del model; torch.cuda.empty_cache(); gc.collect()
    summary = compute_summary(exp_name, results)
    save_results(exp_name, results, summary)
    cleanup_checkpoint(exp_name)
    return summary


# ============================================================
# Experiment 2: Combined Pipeline (re-run for JSON)
# ============================================================
def run_combined_pipeline(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    exp_name = 'combined_pipeline'
    logger.info(f"\n{'='*60}\nExp 2: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
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

        ctx_pos, always_keep, q_start, q_end = get_token_ranges(
            tokenizer, sample['context'], sample['question'], input_ids, seq_len)
        num_ctx = len(ctx_pos)
        if num_ctx < 5: continue

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()

        # Get KV + attentions
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        q2c_scores = get_q2c_scores(out.attentions, seq_len, q_start, q_end)
        del out.attentions; torch.cuda.empty_cache()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_ctx,
                  'full': {'answer': full_ans, 'f1': compute_f1(full_ans, gold)}}

        # Q2C selection only (attention mask, via model.generate)
        for ret in [0.50, 0.75]:
            selected = select_positions(q2c_scores, ctx_pos, num_ctx, ret)
            mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
            for p in always_keep: mask[0, p] = 1
            for p in selected: mask[0, p] = 1
            with torch.no_grad():
                gen = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=64, do_sample=False)
            ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
            result[f'q2c_{int(ret*100)}_mask'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        # INT4 only (manual generate)
        pkv4 = copy.deepcopy(out.past_key_values)
        quantize_inplace(pkv4, 4)
        ans4 = manual_generate(model, tokenizer, pkv4, first_tok, seq_len)
        result['int4_only'] = {'answer': ans4, 'f1': compute_f1(ans4, gold)}

        # Combined: zero unselected + quantize (manual generate)
        for ret in [0.50, 0.75]:
            selected = select_positions(q2c_scores, ctx_pos, num_ctx, ret)
            unselected = set(ctx_pos) - selected

            for qbits in [4, 8]:
                pkv_c = copy.deepcopy(out.past_key_values)
                for layer in pkv_c.layers:
                    for attr in ['keys', 'values']:
                        tensor = getattr(layer, attr)
                        for pos in unselected:
                            tensor[:, :, pos, :] = 0
                quantize_inplace(pkv_c, qbits)
                ans = manual_generate(model, tokenizer, pkv_c, first_tok, seq_len)
                key = f'q2c{int(ret*100)}_int{qbits}_zero'
                result[key] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        result['time'] = time.time() - t0
        results.append(result)
        save_checkpoint(exp_name, i, results)

        logger.info(f"  [{i+1}/{num_samples}] full={result['full']['f1']:.3f} "
                     f"q2c50_mask={result['q2c_50_mask']['f1']:.3f} "
                     f"q2c50+int4_zero={result['q2c50_int4_zero']['f1']:.3f} "
                     f"q2c75+int4_zero={result['q2c75_int4_zero']['f1']:.3f} ({result['time']:.1f}s)")

        del out; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache(); gc.collect()
    summary = compute_summary(exp_name, results)
    save_results(exp_name, results, summary)
    cleanup_checkpoint(exp_name)
    return summary


# ============================================================
# Experiment 3: Topic 18 Verification (zeroing vs masking, SAME gen path)
# ============================================================
def run_topic18_verify(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """Controlled comparison: zeroing vs masking through the SAME generation path.

    This resolves whether Topic 18 (zeroed positions improve accuracy) is a real
    phenomenon or an artifact of different generation paths.

    All variants use manual_generate() with explicit attention mask:
    A) Full KV, manual gen (baseline)
    B) Attention mask only, manual gen (mask unselected, don't zero)
    C) Zero unselected, manual gen (zero, no mask — model sees zeros)
    D) Zero unselected + mask, manual gen (zero AND mask)
    E) Zero + INT4 quantize, manual gen
    F) Mask only + INT4 quantize, manual gen
    """
    exp_name = 'topic18_verify'
    logger.info(f"\n{'='*60}\nExp 3: {exp_name} — Topic 18 Verification ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
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

        ctx_pos, always_keep, q_start, q_end = get_token_ranges(
            tokenizer, sample['context'], sample['question'], input_ids, seq_len)
        num_ctx = len(ctx_pos)
        if num_ctx < 5: continue

        # Get KV + attentions
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        q2c_scores = get_q2c_scores(out.attentions, seq_len, q_start, q_end)
        del out.attentions; torch.cuda.empty_cache()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_ctx}

        # A) Full KV baseline (manual gen, no modifications)
        pkv_full = copy.deepcopy(out.past_key_values)
        full_mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)
        ans_full = manual_generate_with_mask(model, tokenizer, pkv_full, first_tok, seq_len, full_mask)
        result['full'] = {'answer': ans_full, 'f1': compute_f1(ans_full, gold)}

        for ret in [0.50, 0.75]:
            selected = select_positions(q2c_scores, ctx_pos, num_ctx, ret)
            unselected = set(ctx_pos) - selected
            ret_key = int(ret * 100)

            # Build masks
            keep_mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
            for p in always_keep: keep_mask[0, p] = 1
            for p in selected: keep_mask[0, p] = 1
            # full_ones = all positions visible
            full_ones = torch.ones(1, seq_len, device='cuda', dtype=torch.long)

            # B) Mask only (manual gen with mask, KV untouched)
            pkv_b = copy.deepcopy(out.past_key_values)
            ans_b = manual_generate_with_mask(model, tokenizer, pkv_b, first_tok, seq_len, keep_mask)
            result[f'mask_only_{ret_key}'] = {'answer': ans_b, 'f1': compute_f1(ans_b, gold)}

            # C) Zero only (manual gen, no mask — model sees zeros as real values)
            pkv_c = copy.deepcopy(out.past_key_values)
            for layer in pkv_c.layers:
                for attr in ['keys', 'values']:
                    tensor = getattr(layer, attr)
                    for pos in unselected:
                        tensor[:, :, pos, :] = 0
            ans_c = manual_generate_with_mask(model, tokenizer, pkv_c, first_tok, seq_len, full_ones)
            result[f'zero_only_{ret_key}'] = {'answer': ans_c, 'f1': compute_f1(ans_c, gold)}

            # D) Zero + Mask (zero KV AND mask those positions)
            pkv_d = copy.deepcopy(out.past_key_values)
            for layer in pkv_d.layers:
                for attr in ['keys', 'values']:
                    tensor = getattr(layer, attr)
                    for pos in unselected:
                        tensor[:, :, pos, :] = 0
            ans_d = manual_generate_with_mask(model, tokenizer, pkv_d, first_tok, seq_len, keep_mask)
            result[f'zero_mask_{ret_key}'] = {'answer': ans_d, 'f1': compute_f1(ans_d, gold)}

            # E) Zero + INT4 (zero, then quantize, no mask)
            pkv_e = copy.deepcopy(out.past_key_values)
            for layer in pkv_e.layers:
                for attr in ['keys', 'values']:
                    tensor = getattr(layer, attr)
                    for pos in unselected:
                        tensor[:, :, pos, :] = 0
            quantize_inplace(pkv_e, 4)
            ans_e = manual_generate_with_mask(model, tokenizer, pkv_e, first_tok, seq_len, full_ones)
            result[f'zero_int4_{ret_key}'] = {'answer': ans_e, 'f1': compute_f1(ans_e, gold)}

            # F) Mask + INT4 (mask only, quantize all KV, no zeroing)
            pkv_f = copy.deepcopy(out.past_key_values)
            quantize_inplace(pkv_f, 4)
            ans_f = manual_generate_with_mask(model, tokenizer, pkv_f, first_tok, seq_len, keep_mask)
            result[f'mask_int4_{ret_key}'] = {'answer': ans_f, 'f1': compute_f1(ans_f, gold)}

        result['time'] = time.time() - t0
        results.append(result)
        save_checkpoint(exp_name, i, results)

        logger.info(f"  [{i+1}/{num_samples}] full={result['full']['f1']:.3f} "
                     f"mask50={result['mask_only_50']['f1']:.3f} "
                     f"zero50={result['zero_only_50']['f1']:.3f} "
                     f"zero+mask50={result['zero_mask_50']['f1']:.3f} "
                     f"zero+int4_50={result['zero_int4_50']['f1']:.3f} "
                     f"mask+int4_50={result['mask_int4_50']['f1']:.3f} ({result['time']:.1f}s)")

        del out; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache(); gc.collect()
    summary = compute_summary(exp_name, results)

    # Print comparison table
    logger.info(f"\n--- Topic 18 Verification Summary ---")
    logger.info(f"  Full KV: {summary.get('full_f1', 0):.3f}")
    for ret in [50, 75]:
        logger.info(f"\n  === {ret}% retention ===")
        for method in ['mask_only', 'zero_only', 'zero_mask', 'zero_int4', 'mask_int4']:
            k = f'{method}_{ret}_f1'
            if k in summary:
                logger.info(f"  {method}: {summary[k]:.3f} +/- {summary.get(f'{method}_{ret}_std', 0):.3f}")

    save_results(exp_name, results, summary)
    cleanup_checkpoint(exp_name)
    return summary


# ============================================================
# Experiment 4: TriviaQA Validation (second dataset)
# ============================================================
def run_triviaqa(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    exp_name = 'triviaqa_validation'
    logger.info(f"\n{'='*60}\nExp 4: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_triviaqa(num_samples)
    if not samples:
        logger.error("No TriviaQA samples. Skipping.")
        del model; torch.cuda.empty_cache()
        return {}

    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        ctx_pos, always_keep, q_start, q_end = get_token_ranges(
            tokenizer, sample['context'], sample['question'], input_ids, seq_len)
        num_ctx = len(ctx_pos)

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()

        # Get KV + attentions
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        q2c_scores = get_q2c_scores(out.attentions, seq_len, q_start, q_end)
        snapkv_scores = get_snapkv_scores(out.attentions, seq_len)
        del out.attentions; torch.cuda.empty_cache()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_ctx,
                  'dataset': 'triviaqa',
                  'full': {'answer': full_ans, 'f1': compute_f1(full_ans, gold)}}

        # INT4 quantization
        pkv4 = copy.deepcopy(out.past_key_values)
        quantize_inplace(pkv4, 4)
        ans4 = manual_generate(model, tokenizer, pkv4, first_tok, seq_len)
        result['int4'] = {'answer': ans4, 'f1': compute_f1(ans4, gold)}

        # INT8 quantization
        pkv8 = copy.deepcopy(out.past_key_values)
        quantize_inplace(pkv8, 8)
        ans8 = manual_generate(model, tokenizer, pkv8, first_tok, seq_len)
        result['int8'] = {'answer': ans8, 'f1': compute_f1(ans8, gold)}

        # Selection at 50% and 75%
        if num_ctx >= 5:
            for method_name, scores in [('q2c', q2c_scores), ('snapkv', snapkv_scores)]:
                for ret in [0.50, 0.75]:
                    selected = select_positions(scores, ctx_pos, num_ctx, ret)
                    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
                    for p in always_keep: mask[0, p] = 1
                    for p in selected: mask[0, p] = 1
                    with torch.no_grad():
                        gen = model.generate(input_ids=input_ids, attention_mask=mask,
                                              max_new_tokens=64, do_sample=False)
                    ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
                    result[f'{method_name}_{int(ret*100)}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

            # Random baseline at 50%
            rng = np.random.RandomState(42 + i)
            random_scores = torch.tensor(rng.rand(seq_len), device='cuda', dtype=torch.float32)
            selected_rand = select_positions(random_scores, ctx_pos, num_ctx, 0.50)
            mask_rand = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
            for p in always_keep: mask_rand[0, p] = 1
            for p in selected_rand: mask_rand[0, p] = 1
            with torch.no_grad():
                gen = model.generate(input_ids=input_ids, attention_mask=mask_rand, max_new_tokens=64, do_sample=False)
            ans_rand = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
            result['random_50'] = {'answer': ans_rand, 'f1': compute_f1(ans_rand, gold)}

        result['time'] = time.time() - t0
        results.append(result)
        save_checkpoint(exp_name, i, results)

        logger.info(f"  [{i+1}/{num_samples}] full={result['full']['f1']:.3f} "
                     f"int4={result['int4']['f1']:.3f} "
                     f"q2c50={result.get('q2c_50', {}).get('f1', 'N/A')} "
                     f"snap50={result.get('snapkv_50', {}).get('f1', 'N/A')} ({result['time']:.1f}s)")

        del out; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache(); gc.collect()
    summary = compute_summary(exp_name, results)
    logger.info(f"\n--- TriviaQA Summary ---")
    for k in sorted(summary.keys()):
        if k.endswith('_f1'):
            logger.info(f"  {k}: {summary[k]:.3f}")
    save_results(exp_name, results, summary)
    cleanup_checkpoint(exp_name)
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start time: {datetime.now()}")
    total_start = time.time()

    # Exp 1: Extreme quantization (re-run for JSON)
    logger.info("\n" + "="*80 + "\nSTARTING EXP 1: Extreme Quantization\n" + "="*80)
    s1 = run_extreme_quant("Qwen/Qwen2.5-3B", 50)

    # Exp 2: Combined pipeline (re-run for JSON)
    logger.info("\n" + "="*80 + "\nSTARTING EXP 2: Combined Pipeline\n" + "="*80)
    s2 = run_combined_pipeline("Qwen/Qwen2.5-3B", 50)

    # Exp 3: Topic 18 verification
    logger.info("\n" + "="*80 + "\nSTARTING EXP 3: Topic 18 Verification\n" + "="*80)
    s3 = run_topic18_verify("Qwen/Qwen2.5-3B", 50)

    # Exp 4: TriviaQA validation
    logger.info("\n" + "="*80 + "\nSTARTING EXP 4: TriviaQA Validation\n" + "="*80)
    s4 = run_triviaqa("Qwen/Qwen2.5-3B", 50)

    total_min = (time.time() - total_start) / 60
    logger.info(f"\n{'='*80}\nALL BATCH 7 COMPLETE in {total_min:.1f} minutes\n{'='*80}")
    logger.info(f"Results in: {RESULTS_DIR}")
