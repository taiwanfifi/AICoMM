#!/usr/bin/env python3
"""
Batch 4: Fixed quantization F1 + 7B baseline + selection experiments.

Key fixes from Batch 3:
- manual_generate() was using eos_token_id as dummy first input — WRONG
- Fix: Use model.generate() with past_key_values= the in-place modified cache
- Also: pass the original prompt's last token as continuation seed

Three experiments:
1. Quantization F1 (3B): INT8/INT4 in-place quantize, then model.generate()
2. 7B Baseline F1: Standard generation (no quantization)
3. Selection F1 (3B): SnapKV + Q2C + Random at 25%/50%/75% retention (attention mask)
"""
import os, sys, json, time, logging, copy
from pathlib import Path
from datetime import datetime
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch4.log')])
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


def quantize_inplace_int8(pkv):
    """In-place INT8 quantization of DynamicCache — per-tensor symmetric."""
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            scale = t.abs().max() / 127.0
            if scale > 0:
                quantized = torch.clamp(torch.round(t / scale), -128, 127) * scale
                tensor.copy_(quantized.to(tensor.dtype))


def quantize_inplace_int4(pkv, group_size=32):
    """In-place INT4 quantization of DynamicCache — per-group symmetric."""
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


def generate_with_kv(model, tokenizer, input_ids, past_kv, seq_len, max_new=64):
    """Generate using model.generate() with pre-computed KV-cache.

    The trick: we pass input_ids as JUST the last token of the prompt,
    and past_key_values as the in-place modified cache (which covers positions 0..seq_len-1).
    model.generate() then continues from position seq_len.
    """
    # Only pass the last token as input — the rest is already in past_kv
    last_token = input_ids[:, -1:]

    # Attention mask must cover all past positions + current token
    attn_mask = torch.ones(1, seq_len, device=input_ids.device, dtype=torch.long)

    with torch.no_grad():
        gen = model.generate(
            input_ids=last_token,
            past_key_values=past_kv,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=False,
        )
    # gen includes the last_token we passed, so skip it
    answer = tokenizer.decode(gen[0][1:], skip_special_tokens=True).strip()
    return answer


def generate_with_attn_mask(model, tokenizer, input_ids, attention_mask, max_new=64):
    """Generate with a custom attention mask (for selection experiments).

    input_ids: full prompt tokens
    attention_mask: 1 for kept positions, 0 for masked positions
    """
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
        )
    seq_len = input_ids.shape[1]
    answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
    return answer


def save_results(name, results, metadata):
    path = RESULTS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump({'metadata': metadata, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] {name} -> {path}")
    return path


def save_checkpoint(ckpt_path, idx, results):
    with open(ckpt_path, 'w') as f:
        json.dump({'idx': idx, 'results': results}, f, default=str)


def load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


# ============================================================
# Experiment 1: Quantization F1 (Fixed)
# ============================================================
def run_quantization_f1_v4(model_name="Qwen/Qwen2.5-3B", num_samples=30):
    """Quantization F1: in-place quantize + model.generate() with past_key_values."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'quant_f1_v4_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

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

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)
    if start_idx > 0:
        logger.info(f"Resuming from sample {start_idx}")

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Full KV baseline — standard generate
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV cache from forward pass
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        # Also get the first predicted token from original logits (for debugging)
        orig_first_token = out.logits[:, -1, :].argmax(dim=-1).item()
        orig_first_word = tokenizer.decode([orig_first_token])

        # INT8: deepcopy, quantize in-place, generate
        pkv_int8 = copy.deepcopy(out.past_key_values)
        quantize_inplace_int8(pkv_int8)
        int8_answer = generate_with_kv(model, tokenizer, inputs['input_ids'], pkv_int8, seq_len)
        int8_f1 = compute_f1(int8_answer, gold)

        # INT4: deepcopy, quantize in-place, generate
        pkv_int4 = copy.deepcopy(out.past_key_values)
        quantize_inplace_int4(pkv_int4)
        int4_answer = generate_with_kv(model, tokenizer, inputs['input_ids'], pkv_int4, seq_len)
        int4_f1 = compute_f1(int4_answer, gold)

        # Also test: unmodified KV passed through generate_with_kv (sanity check)
        pkv_orig = copy.deepcopy(out.past_key_values)
        orig_kv_answer = generate_with_kv(model, tokenizer, inputs['input_ids'], pkv_orig, seq_len)
        orig_kv_f1 = compute_f1(orig_kv_answer, gold)

        elapsed = time.time() - t0
        results.append({
            'idx': i, 'gold': gold, 'seq_len': seq_len, 'time': elapsed,
            'orig_first_token': orig_first_word,
            'full': {'answer': full_answer, 'f1': full_f1},
            'orig_kv': {'answer': orig_kv_answer, 'f1': orig_kv_f1},
            'int8': {'answer': int8_answer, 'f1': int8_f1},
            'int4': {'answer': int4_answer, 'f1': int4_f1},
        })

        logger.info(f"  [{i+1}/{num_samples}] seq={seq_len} full_f1={full_f1:.3f} orig_kv={orig_kv_f1:.3f} "
                     f"int8={int8_f1:.3f} int4={int4_f1:.3f} ({elapsed:.1f}s)")
        logger.info(f"    Gold: {gold[:80]}")
        logger.info(f"    Full: {full_answer[:80]}")
        logger.info(f"    OrigKV: {orig_kv_answer[:80]}")
        logger.info(f"    INT8: {int8_answer[:80]}")
        logger.info(f"    INT4: {int4_answer[:80]}")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    avg = lambda k: float(np.mean([r[k]['f1'] for r in results]))
    std = lambda k: float(np.std([r[k]['f1'] for r in results]))
    summary = {
        'model': model_name, 'num_samples': len(results),
        'full_f1': avg('full'), 'full_std': std('full'),
        'orig_kv_f1': avg('orig_kv'), 'orig_kv_std': std('orig_kv'),
        'int8_f1': avg('int8'), 'int8_std': std('int8'),
        'int4_f1': avg('int4'), 'int4_std': std('int4'),
    }
    logger.info(f"\n--- Quantization F1 Summary ---")
    logger.info(f"Full generate: {summary['full_f1']:.3f} +/- {summary['full_std']:.3f}")
    logger.info(f"Orig KV (sanity): {summary['orig_kv_f1']:.3f} +/- {summary['orig_kv_std']:.3f}")
    logger.info(f"INT8: {summary['int8_f1']:.3f} +/- {summary['int8_std']:.3f}")
    logger.info(f"INT4: {summary['int4_f1']:.3f} +/- {summary['int4_std']:.3f}")
    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Experiment 2: 7B Baseline
# ============================================================
def run_7b_baseline(num_samples=30):
    """Qwen2.5-7B baseline F1 on SQuAD."""
    exp_name = 'baseline_7b_v2'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        f1 = compute_f1(answer, gold)
        results.append({'idx': i, 'gold': gold, 'answer': answer, 'f1': f1, 'seq_len': seq_len})

        logger.info(f"  [{i+1}/{num_samples}] F1={f1:.3f} | Gold: {gold[:60]} | Pred: {answer[:60]}")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    summary = {
        'model': 'Qwen/Qwen2.5-7B', 'num_samples': len(results),
        'f1': float(np.mean([r['f1'] for r in results])),
        'f1_std': float(np.std([r['f1'] for r in results])),
    }
    logger.info(f"\n7B Baseline F1: {summary['f1']:.3f} +/- {summary['f1_std']:.3f}")
    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Experiment 3: Selection Methods (SnapKV, Q2C, Random)
# ============================================================
def compute_attention_scores(model, tokenizer, input_ids, question_start, question_end):
    """Compute attention scores and return SnapKV + Q2C importance scores for context positions."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=True)

    attentions = out.attentions  # tuple of (batch, heads, seq, seq) per layer
    num_layers = len(attentions)
    seq_len = input_ids.shape[1]

    # SnapKV: cumulative attention from ALL query positions to context positions
    # Sum attention across all layers, all heads, all query positions -> per-key importance
    snapkv_scores = torch.zeros(seq_len, device=input_ids.device)
    for layer_attn in attentions:
        # layer_attn: (1, heads, seq, seq) — [batch, heads, query_pos, key_pos]
        # Sum across heads and query positions
        snapkv_scores += layer_attn[0].sum(dim=(0, 1))  # sum heads, sum query -> (seq,)

    # Q2C: attention from QUESTION tokens to context positions only
    q2c_scores = torch.zeros(seq_len, device=input_ids.device)
    if question_end > question_start:
        for layer_attn in attentions:
            # Only query positions from question
            q2c_scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))

    return snapkv_scores, q2c_scores, out.past_key_values


def run_selection_f1(model_name="Qwen/Qwen2.5-3B", num_samples=30):
    """Selection F1: compare SnapKV, Q2C, Random at different retention levels using attention masks."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'selection_f1_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"  # Required for output_attentions
    )
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    retention_levels = [0.25, 0.50, 0.75]
    methods = ['snapkv', 'q2c', 'random']

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        t0 = time.time()
        gold = sample['answers']['text'][0]
        context = sample['context']
        question = sample['question']

        # Build prompt and find question boundaries
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Find where context ends and question starts in token space
        context_prefix = f"Context: {context}\nQuestion: "
        context_tokens = tokenizer(context_prefix, return_tensors="pt", max_length=512, truncation=True)
        context_end = context_tokens['input_ids'].shape[1]

        question_prefix = f"Context: {context}\nQuestion: {question}\nAnswer:"
        # question tokens are from context_end to seq_len-1 (last few tokens are "\nAnswer:")
        question_start = context_end
        # Find "Answer:" suffix
        answer_suffix = tokenizer("\nAnswer:", add_special_tokens=False)['input_ids']
        question_end = seq_len - len(answer_suffix)

        # Context positions are positions 1..context_end-1 (skip BOS)
        # These are the positions we will select from
        context_positions = list(range(1, context_end))
        num_context = len(context_positions)

        # Always-keep positions: BOS, question tokens, "Answer:" suffix
        always_keep = set(range(0, 1))  # BOS
        always_keep.update(range(question_start, seq_len))  # question + answer suffix

        # Full KV baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get attention scores
        snapkv_scores, q2c_scores, _ = compute_attention_scores(
            model, tokenizer, inputs['input_ids'], question_start, question_end
        )

        # Random scores (fixed seed per sample for reproducibility)
        rng = np.random.RandomState(42 + i)
        random_scores = torch.tensor(rng.rand(seq_len), device=inputs['input_ids'].device)

        result_entry = {
            'idx': i, 'gold': gold, 'seq_len': seq_len,
            'num_context': num_context,
            'full': {'answer': full_answer, 'f1': full_f1},
        }

        for retention in retention_levels:
            k = max(1, int(num_context * retention))

            for method in methods:
                if method == 'snapkv':
                    scores = snapkv_scores
                elif method == 'q2c':
                    scores = q2c_scores
                else:
                    scores = random_scores

                # Get top-k context positions by score
                context_scores = scores[context_positions]
                _, topk_idx = context_scores.topk(k)
                selected_context = set(context_positions[j] for j in topk_idx.cpu().numpy())

                # Build attention mask: 1 for selected_context + always_keep, 0 otherwise
                attn_mask = torch.zeros(1, seq_len, device=inputs['input_ids'].device, dtype=torch.long)
                for pos in always_keep:
                    attn_mask[0, pos] = 1
                for pos in selected_context:
                    attn_mask[0, pos] = 1

                answer = generate_with_attn_mask(
                    model, tokenizer, inputs['input_ids'], attn_mask
                )
                f1 = compute_f1(answer, gold)

                key = f'{method}_{int(retention*100)}'
                result_entry[key] = {'answer': answer, 'f1': f1}

        elapsed = time.time() - t0
        result_entry['time'] = elapsed
        results.append(result_entry)

        # Log summary for this sample
        log_parts = [f"[{i+1}/{num_samples}] full={full_f1:.3f}"]
        for ret in retention_levels:
            for method in methods:
                key = f'{method}_{int(ret*100)}'
                log_parts.append(f"{key}={result_entry[key]['f1']:.3f}")
        logger.info(f"  {' | '.join(log_parts)} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    # Summary
    summary = {'model': model_name, 'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))
    for ret in retention_levels:
        for method in methods:
            key = f'{method}_{int(ret*100)}'
            vals = [r[key]['f1'] for r in results]
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))

    logger.info(f"\n--- Selection F1 Summary ---")
    logger.info(f"Full: {summary['full_f1']:.3f}")
    for ret in retention_levels:
        line = f"  {int(ret*100)}%: "
        for method in methods:
            key = f'{method}_{int(ret*100)}'
            line += f"{method}={summary[f'{key}_f1']:.3f}±{summary[f'{key}_std']:.3f}  "
        logger.info(line)

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Experiment 4: Quick sanity check — verify generate_with_kv works
# ============================================================
def run_sanity_check(model_name="Qwen/Qwen2.5-3B"):
    """Quick 3-sample sanity check that generate_with_kv matches standard generate."""
    exp_name = 'sanity_check'
    logger.info(f"\n{'='*60}\nSanity Check: generate_with_kv\n{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    test_prompts = [
        "The capital of France is",
        "Context: The Eiffel Tower is 330 meters tall.\nQuestion: How tall is the Eiffel Tower?\nAnswer:",
        "Context: Python was created by Guido van Rossum in 1991.\nQuestion: Who created Python?\nAnswer:",
    ]

    all_pass = True
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Standard generate
        with torch.no_grad():
            gen1 = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        answer1 = tokenizer.decode(gen1[0][seq_len:], skip_special_tokens=True).strip()

        # generate_with_kv (unmodified KV)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        pkv = copy.deepcopy(out.past_key_values)
        answer2 = generate_with_kv(model, tokenizer, inputs['input_ids'], pkv, seq_len, max_new=32)

        match = answer1 == answer2
        if not match:
            all_pass = False
        logger.info(f"  Prompt: {prompt[:60]}...")
        logger.info(f"    Standard: {answer1[:80]}")
        logger.info(f"    WithKV:   {answer2[:80]}")
        logger.info(f"    Match: {'YES' if match else 'NO <<<'}")

        # Also test INT8 quantized
        pkv_q = copy.deepcopy(out.past_key_values)
        quantize_inplace_int8(pkv_q)
        answer_q = generate_with_kv(model, tokenizer, inputs['input_ids'], pkv_q, seq_len, max_new=32)
        logger.info(f"    INT8:     {answer_q[:80]}")

    del model; torch.cuda.empty_cache()
    logger.info(f"\nSanity check: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.version.cuda}")
    start = time.time()

    # Step 1: Sanity check first
    sanity_ok = run_sanity_check("Qwen/Qwen2.5-3B")

    if not sanity_ok:
        logger.warning("Sanity check FAILED — generate_with_kv doesn't match. Investigating...")
        logger.warning("Will still run experiments but results may need review.")

    # Step 2: Fixed quantization F1
    run_quantization_f1_v4("Qwen/Qwen2.5-3B", num_samples=30)

    # Step 3: Selection methods (SnapKV vs Q2C vs Random)
    run_selection_f1("Qwen/Qwen2.5-3B", num_samples=30)

    # Step 4: 7B baseline (uses most memory, run last)
    run_7b_baseline(num_samples=30)

    logger.info(f"\nALL DONE in {(time.time()-start)/60:.1f} minutes")
