#!/usr/bin/env python3
"""
Batch 20: Yi-1.5-6B-Chat Context-Length Scaling (needle-in-haystack)

KEY QUESTION: Does Yi INT4 remain lossless at longer contexts?
Qwen-7B (same 4 KV heads) collapses: 70.9% → 41.6% (512 → 4096)
If Yi stays at ~100%, the model-specific fragility hypothesis is CONFIRMED.

Design: Same as batch 18 — SQuAD samples padded with distractor text at
512, 1024, 2048, 4096 tokens. Same question/answer across all lengths.
30 samples per length. Uses ChatML template.
"""
import os, sys, json, time, logging, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/dev/shm/hf_7b'
os.environ['HF_DATASETS_CACHE'] = '/dev/shm/hf_7b/datasets'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch20.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = '01-ai/Yi-1.5-6B-Chat'
MODEL_SHORT = 'Yi-1.5-6B-Chat'
TARGET_LENGTHS = [512, 1024, 2048, 4096]
NUM_SAMPLES = 30


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


def load_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        logger.info(f"[RESUME] {exp_name} from sample {ckpt['idx'] + 1}")
        return ckpt['idx'] + 1, ckpt['results']
    return 0, []


def quantize_pertoken(t, bits):
    if bits >= 16: return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return (t / scale).round().clamp(qmin, qmax) * scale


def manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new=64):
    generated = [first_token_id]
    cur_len = seq_len
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        mask = torch.cat([mask, torch.ones(1, 1, device='cuda', dtype=torch.long)], dim=1)
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       attention_mask=mask, position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id: break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_quantized(model, tokenizer, input_ids, seq_len, bits, layer0_bits=None,
                       max_new=64, selection_mask=None):
    if layer0_bits is None:
        layer0_bits = bits

    if selection_mask is not None:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=selection_mask, use_cache=True)
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)

    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    for li in range(len(pkv.layers)):
        b = layer0_bits if li == 0 else bits
        if b < 16:
            layer = pkv.layers[li]
            layer.keys.copy_(quantize_pertoken(layer.keys, b))
            layer.values.copy_(quantize_pertoken(layer.values, b))

    mask = selection_mask.clone() if selection_mask is not None else torch.ones(1, seq_len, device='cuda', dtype=torch.long)
    return manual_generate_with_mask(model, tokenizer, pkv, first_token_id, seq_len, mask, max_new)


def prepare_scaling_samples(tokenizer, num_samples=30):
    """Prepare samples at multiple context lengths.

    Uses SQuAD samples with SHORT context padded with distractor text.
    Same question/answer at every length — only haystack size changes.
    Relevant context placed at END (needle at end of haystack).
    Uses ChatML template.
    """
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')

    # Select base samples with SHORT contexts (80-200 tokens)
    candidates = []
    for s in ds:
        if not s['answers']['text']: continue
        ctx_tokens = len(tokenizer.encode(s['context']))
        if 80 <= ctx_tokens <= 200:
            candidates.append(s)
        if len(candidates) >= num_samples * 3:
            break

    np.random.seed(42)
    base_indices = np.random.choice(len(candidates), min(num_samples, len(candidates)), replace=False)
    base_samples = [candidates[i] for i in base_indices]

    # Collect distractor contexts (other SQuAD passages)
    distractors = []
    for s in ds:
        if s['context'] not in [b['context'] for b in base_samples]:
            distractors.append(s['context'])
        if len(distractors) >= 500:
            break

    # Build samples for each target length
    all_samples = {}
    for target_len in TARGET_LENGTHS:
        length_samples = []
        for i, s in enumerate(base_samples):
            relevant_context = s['context']
            question = s['question']
            gold = s['answers']['text'][0]

            # Build ChatML prompt with placeholders to measure base length
            base_prompt = (
                f"<|im_start|>system\n"
                f"Answer the question using ONLY words from the context. "
                f"Give the shortest possible answer.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Context: PLACEHOLDER {relevant_context}\n"
                f"Question: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            base_len = len(tokenizer.encode(base_prompt))
            padding_needed = target_len - base_len

            if padding_needed <= 0:
                # Context already long enough, just truncate distractor
                full_context = relevant_context
            else:
                # Concatenate distractors until we have enough padding
                distractor_text = ""
                for d in distractors:
                    distractor_text += " " + d
                    d_tokens = len(tokenizer.encode(distractor_text))
                    if d_tokens >= padding_needed:
                        break

                # Truncate to exact length
                d_ids = tokenizer.encode(distractor_text)[:padding_needed]
                distractor_text = tokenizer.decode(d_ids, skip_special_tokens=True)
                # Needle at END of haystack
                full_context = f"{distractor_text.strip()} {relevant_context}"

            prompt = (
                f"<|im_start|>system\n"
                f"Answer the question using ONLY words from the context. "
                f"Give the shortest possible answer.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Context: {full_context}\n"
                f"Question: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            actual_len = len(tokenizer.encode(prompt))

            length_samples.append({
                'prompt': prompt,
                'gold': gold,
                'question': question[:100],
                'target_len': target_len,
                'actual_len': actual_len,
                'sample_idx': i,
            })

        all_samples[target_len] = length_samples
        lens = [s['actual_len'] for s in length_samples]
        logger.info(f"  Target {target_len}: {len(length_samples)} samples, "
                    f"actual mean={np.mean(lens):.0f}, range=[{min(lens)}, {max(lens)}]")

    return all_samples


def run_scaling(model, tokenizer, all_samples):
    """Run quantization experiments at each context length."""
    all_results = {}

    for target_len in TARGET_LENGTHS:
        samples = all_samples[target_len]
        exp_name = f'yi_scaling_{target_len}'
        start_idx, results = load_checkpoint(exp_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"Context length: {target_len} tokens ({len(samples)} samples)")
        logger.info(f"{'='*60}")

        for i, s in enumerate(samples):
            if i < start_idx: continue
            t0 = time.time()

            inputs = tokenizer(s['prompt'], return_tensors='pt',
                             max_length=target_len + 50, truncation=True).to('cuda')
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1]
            gold = s['gold']

            result = {'idx': i, 'gold': gold, 'seq_len': seq_len,
                      'target_len': target_len, 'sample_idx': s['sample_idx']}

            # 1. Full baseline
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
            result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

            # 2. INT4, INT8
            for bits in [4, 8]:
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits)
                result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

            # 3. Mixed-precision
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16)
            result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

            elapsed = time.time() - t0
            result['time'] = elapsed
            results.append(result)
            save_checkpoint(exp_name, i, results)

            fp16 = result['full']['f1']
            int4 = result['int4']['f1']
            int8 = result['int8']['f1']
            mixed = result['mixed_L0fp16_int4']['f1']
            logger.info(f"  [{i+1}/{len(samples)}] len={target_len} seq={seq_len} "
                        f"full={fp16:.3f} int8={int8:.3f} int4={int4:.3f} mixed={mixed:.3f} ({elapsed:.1f}s)")

        all_results[target_len] = results

        # Per-length summary
        fp16_f1 = float(np.mean([r['full']['f1'] for r in results]))
        int4_f1 = float(np.mean([r['int4']['f1'] for r in results]))
        int8_f1 = float(np.mean([r['int8']['f1'] for r in results]))
        mixed_f1 = float(np.mean([r['mixed_L0fp16_int4']['f1'] for r in results]))
        logger.info(f"\n  Length {target_len} summary:")
        logger.info(f"    Full: {fp16_f1:.4f}")
        logger.info(f"    INT8: {int8_f1:.4f} ({int8_f1/fp16_f1*100:.1f}%)")
        logger.info(f"    INT4: {int4_f1:.4f} ({int4_f1/fp16_f1*100:.1f}%)")
        logger.info(f"    Mixed: {mixed_f1:.4f} ({mixed_f1/fp16_f1*100:.1f}%)")

        # Clean checkpoint
        ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
        if ckpt_path.exists(): ckpt_path.unlink()

    return all_results


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Preparing scaling samples...")
    all_samples = prepare_scaling_samples(tokenizer, NUM_SAMPLES)

    logger.info(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()
    logger.info(f"Loaded: {model.config.num_hidden_layers} layers, "
                f"{model.config.num_key_value_heads} KV heads")

    all_results = run_scaling(model, tokenizer, all_samples)

    # Final cross-length summary
    logger.info(f"\n{'#'*80}")
    logger.info(f"CROSS-LENGTH SUMMARY: Yi-1.5-6B-Chat")
    logger.info(f"{'#'*80}")
    logger.info(f"{'Length':>8} {'Full':>8} {'INT8':>10} {'INT4':>10} {'Mixed':>10}")
    for tl in TARGET_LENGTHS:
        results = all_results[tl]
        fp16 = float(np.mean([r['full']['f1'] for r in results]))
        int8 = float(np.mean([r['int8']['f1'] for r in results]))
        int4 = float(np.mean([r['int4']['f1'] for r in results]))
        mixed = float(np.mean([r['mixed_L0fp16_int4']['f1'] for r in results]))
        logger.info(f"  {tl:>6} {fp16:>8.4f} {int8/fp16*100:>9.1f}% {int4/fp16*100:>9.1f}% {mixed/fp16*100:>9.1f}%")

    # Save combined results
    combined = {
        'metadata': {
            'model': MODEL_SHORT,
            'task': 'SQuAD-v2-needle-in-haystack-ChatML',
            'target_lengths': TARGET_LENGTHS,
            'num_samples_per_length': NUM_SAMPLES,
        },
        'results_by_length': {str(k): v for k, v in all_results.items()},
    }

    for tl in TARGET_LENGTHS:
        results = all_results[tl]
        fp16 = float(np.mean([r['full']['f1'] for r in results]))
        combined['metadata'][f'full_f1_{tl}'] = fp16
        for key in ['int4', 'int8', 'mixed_L0fp16_int4']:
            vals = [r[key]['f1'] for r in results]
            combined['metadata'][f'{key}_f1_{tl}'] = float(np.mean(vals))
            combined['metadata'][f'{key}_pct_{tl}'] = float(np.mean(vals)) / fp16 * 100 if fp16 > 0 else 0

    final_path = RESULTS_DIR / f'yi_scaling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nBatch 20 COMPLETE in {elapsed:.1f} minutes")

    del model; torch.cuda.empty_cache(); gc.collect()
