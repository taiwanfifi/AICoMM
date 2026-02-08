#!/usr/bin/env python3
"""
Batch 6: Extreme quantization + combined pipelines + second dataset.

Experiments:
1. INT3/INT2/INT1 quantization to find information floor
2. Combined pipeline: Q2C selection + INT4/INT8 quantization
3. TriviaQA dataset validation (second dataset for robustness)
"""
import os, sys, json, time, logging, copy, math
from pathlib import Path
from datetime import datetime
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch6.log')])
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


def manual_generate(model, tokenizer, past_kv, first_token_id, seq_len, max_new=64):
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


def quantize_inplace(pkv, bits, group_size=32):
    """Generalized in-place quantization: supports any bit width from 1-8."""
    if bits >= 8:
        max_val = 127
        min_val = -128
    else:
        max_val = (1 << (bits - 1)) - 1  # e.g., bits=4 -> 7, bits=3 -> 3, bits=2 -> 1
        min_val = -(1 << (bits - 1))     # e.g., bits=4 -> -8, bits=3 -> -4, bits=2 -> -2

    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            shape = t.shape

            if bits <= 4 and group_size > 0:
                # Group-wise quantization for low bit widths
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
                # Per-tensor quantization for higher bit widths
                scale = t.abs().max() / max_val
                if scale > 0:
                    dq = torch.clamp(torch.round(t / scale), min_val, max_val) * scale
                else:
                    dq = t

            tensor.copy_(dq.to(tensor.dtype))


def quantize_inplace_binary(pkv):
    """Binary (1-bit sign) quantization: keep only the sign of each element."""
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            # Scale = mean absolute value, sign = direction
            scale = t.abs().mean()
            binary = torch.sign(t) * scale
            tensor.copy_(binary.to(tensor.dtype))


def save_results(name, results, metadata):
    path = RESULTS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump({'metadata': metadata, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] {name} -> {path}")


def save_checkpoint(ckpt_path, idx, results):
    with open(ckpt_path, 'w') as f:
        json.dump({'idx': idx, 'results': results}, f, default=str)


def load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


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
    """Load TriviaQA with context (rc subset)."""
    from datasets import load_dataset
    try:
        ds = load_dataset('trivia_qa', 'rc', split='validation')
        samples = []
        for item in ds:
            if item['answer']['value'] and item['search_results']['search_context']:
                ctx = item['search_results']['search_context'][0]
                if len(ctx) > 50:  # skip very short contexts
                    samples.append({
                        'context': ctx[:2000],  # truncate long contexts
                        'question': item['question'],
                        'answers': {'text': [item['answer']['value']]}
                    })
                    if len(samples) >= num_samples:
                        break
        logger.info(f"Loaded {len(samples)} TriviaQA samples")
        return samples
    except Exception as e:
        logger.warning(f"TriviaQA loading failed: {e}")
        return []


# ============================================================
# Exp 1: Extreme Quantization (find information floor)
# ============================================================
def run_extreme_quantization(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """Test INT8/INT4/INT3/INT2/binary quantization to find where accuracy degrades."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'extreme_quant_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_squad(num_samples)

    bit_widths = [8, 4, 3, 2]  # plus binary (1-bit sign)

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV + first token
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'full': {'answer': full_answer, 'f1': full_f1}}

        for bits in bit_widths:
            pkv = copy.deepcopy(out.past_key_values)
            quantize_inplace(pkv, bits)
            ans = manual_generate(model, tokenizer, pkv, first_tok, seq_len)
            f1 = compute_f1(ans, gold)
            result[f'int{bits}'] = {'answer': ans, 'f1': f1}

        # Binary (1-bit sign only)
        pkv_bin = copy.deepcopy(out.past_key_values)
        quantize_inplace_binary(pkv_bin)
        ans_bin = manual_generate(model, tokenizer, pkv_bin, first_tok, seq_len)
        result['binary'] = {'answer': ans_bin, 'f1': compute_f1(ans_bin, gold)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        parts = [f"[{i+1}/{num_samples}] full={full_f1:.3f}"]
        for b in bit_widths:
            parts.append(f"int{b}={result[f'int{b}']['f1']:.3f}")
        parts.append(f"bin={result['binary']['f1']:.3f}")
        logger.info(f"  {' '.join(parts)} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))
    logger.info(f"\n--- Extreme Quantization Summary ---")
    logger.info(f"  Full FP16: {summary['full_f1']:.3f}")
    for key in [f'int{b}' for b in bit_widths] + ['binary']:
        vals = [r[key]['f1'] for r in results]
        summary[f'{key}_f1'] = float(np.mean(vals))
        summary[f'{key}_std'] = float(np.std(vals))
        bits = key.replace('int', '').replace('binary', '1-sign')
        logger.info(f"  {key} ({bits} bits): {summary[f'{key}_f1']:.3f} +/- {summary[f'{key}_std']:.3f}")

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Exp 2: Combined Pipeline (Q2C + Quantization)
# ============================================================
def run_combined_pipeline(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """Test combined pipelines: Q2C selection + quantization.

    The trick: We can't easily combine attention-mask-based selection with
    manual_generate (which uses the full KV). Instead, we:
    1. Get attention scores to determine important positions
    2. Forward pass to get KV-cache
    3. For "combined": zero out unselected positions in KV-cache, then quantize, then generate

    This simulates: "send only selected positions, quantized"
    """
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'combined_pipeline_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_squad(num_samples)

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

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
        context_end = ctx_tokens['input_ids'].shape[1]
        question_start = context_end
        answer_suffix = tokenizer("\nAnswer:", add_special_tokens=False)['input_ids']
        question_end = seq_len - len(answer_suffix)
        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        if num_context < 5:
            continue

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get attention scores + KV cache
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        # Q2C scores
        q2c_scores = torch.zeros(seq_len, device='cuda')
        if question_end > question_start:
            for layer_attn in out.attentions:
                q2c_scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))
        del out.attentions
        torch.cuda.empty_cache()

        result = {
            'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_context,
            'full': {'answer': full_answer, 'f1': full_f1},
        }

        # Q2C selection only (via attention mask)
        for retention in [0.50, 0.75]:
            k = max(1, int(num_context * retention))
            ctx_sc = q2c_scores[torch.tensor(context_positions, device='cuda')]
            _, topk = ctx_sc.topk(k)
            selected = set(context_positions[j] for j in topk.cpu().numpy())
            mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
            for p in always_keep: mask[0, p] = 1
            for p in selected: mask[0, p] = 1
            with torch.no_grad():
                gen = model.generate(input_ids=input_ids, attention_mask=mask,
                                      max_new_tokens=64, do_sample=False)
            ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
            result[f'q2c_{int(retention*100)}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        # INT4 only
        pkv4 = copy.deepcopy(out.past_key_values)
        quantize_inplace(pkv4, 4)
        ans4 = manual_generate(model, tokenizer, pkv4, first_tok, seq_len)
        result['int4_only'] = {'answer': ans4, 'f1': compute_f1(ans4, gold)}

        # Combined: Q2C 50% selection + INT4 quantization
        # Strategy: zero out unselected context positions in KV, then quantize
        for retention in [0.50, 0.75]:
            k = max(1, int(num_context * retention))
            ctx_sc = q2c_scores[torch.tensor(context_positions, device='cuda')]
            _, topk = ctx_sc.topk(k)
            selected = set(context_positions[j] for j in topk.cpu().numpy())
            unselected = set(context_positions) - selected

            for qbits in [4, 8]:
                pkv_combined = copy.deepcopy(out.past_key_values)

                # Zero out unselected positions
                for layer in pkv_combined.layers:
                    for attr in ['keys', 'values']:
                        tensor = getattr(layer, attr)
                        for pos in unselected:
                            tensor[:, :, pos, :] = 0

                # Quantize
                quantize_inplace(pkv_combined, qbits)

                # Generate
                ans = manual_generate(model, tokenizer, pkv_combined, first_tok, seq_len)
                key = f'q2c{int(retention*100)}_int{qbits}'
                result[key] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        logger.info(f"  [{i+1}/{num_samples}] full={full_f1:.3f} "
                     f"q2c50={result['q2c_50']['f1']:.3f} "
                     f"q2c50+int4={result['q2c50_int4']['f1']:.3f} "
                     f"q2c50+int8={result['q2c50_int8']['f1']:.3f} "
                     f"q2c75+int4={result['q2c75_int4']['f1']:.3f} "
                     f"({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

        del out; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'num_samples': len(results)}
    logger.info(f"\n--- Combined Pipeline Summary ---")
    for key in ['full', 'q2c_50', 'q2c_75', 'int4_only', 'q2c50_int4', 'q2c50_int8', 'q2c75_int4', 'q2c75_int8']:
        vals = [r[key]['f1'] for r in results if key in r]
        if vals:
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
            logger.info(f"  {key}: {summary[f'{key}_f1']:.3f} +/- {summary[f'{key}_std']:.3f}")

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Exp 3: TriviaQA Validation
# ============================================================
def run_triviaqa_validation(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """Validate key findings on TriviaQA dataset."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'triviaqa_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_triviaqa(num_samples)

    if not samples:
        logger.error("No TriviaQA samples loaded. Skipping.")
        del model; torch.cuda.empty_cache()
        return {}

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

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
        context_positions = list(range(ctx_prefix_len, min(context_end, seq_len)))
        num_context = len(context_positions)
        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV + first token + attention
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        # Q2C scores
        q2c_scores = torch.zeros(seq_len, device='cuda')
        snapkv_scores = torch.zeros(seq_len, device='cuda')
        for layer_attn in out.attentions:
            snapkv_scores += layer_attn[0].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))

        del out.attentions
        torch.cuda.empty_cache()

        result = {
            'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_context,
            'full': {'answer': full_answer, 'f1': full_f1},
        }

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

        # Selection methods (at 50% retention)
        if num_context >= 5:
            for method_name, scores in [('q2c', q2c_scores), ('snapkv', snapkv_scores)]:
                k = max(1, int(num_context * 0.5))
                ctx_sc = scores[torch.tensor(context_positions, device='cuda')]
                _, topk = ctx_sc.topk(min(k, len(context_positions)))
                selected = set(context_positions[j] for j in topk.cpu().numpy())
                mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
                for p in always_keep: mask[0, p] = 1
                for p in selected: mask[0, p] = 1
                with torch.no_grad():
                    gen = model.generate(input_ids=input_ids, attention_mask=mask,
                                          max_new_tokens=64, do_sample=False)
                ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
                result[f'{method_name}_50'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        logger.info(f"  [{i+1}/{num_samples}] full={full_f1:.3f} "
                     f"int8={result['int8']['f1']:.3f} int4={result['int4']['f1']:.3f} "
                     f"q2c50={result.get('q2c_50', {}).get('f1', 'N/A')} "
                     f"snap50={result.get('snapkv_50', {}).get('f1', 'N/A')} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

        del out; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'dataset': 'triviaqa', 'num_samples': len(results)}
    logger.info(f"\n--- TriviaQA Validation Summary ---")
    for key in ['full', 'int8', 'int4', 'q2c_50', 'snapkv_50']:
        vals = [r[key]['f1'] for r in results if key in r]
        if vals:
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
            logger.info(f"  {key}: {summary[f'{key}_f1']:.3f} +/- {summary[f'{key}_std']:.3f}")

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    start = time.time()

    # Exp 1: Extreme quantization
    extreme = run_extreme_quantization("Qwen/Qwen2.5-3B", num_samples=50)

    # Exp 2: Combined pipeline
    combined = run_combined_pipeline("Qwen/Qwen2.5-3B", num_samples=50)

    # Exp 3: TriviaQA validation
    triviaqa = run_triviaqa_validation("Qwen/Qwen2.5-3B", num_samples=50)

    elapsed = (time.time() - start) / 60
    logger.info(f"\nALL BATCH 6 DONE in {elapsed:.1f} minutes")
