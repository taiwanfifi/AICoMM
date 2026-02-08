#!/usr/bin/env python3
"""
Batch 19c: Yi-1.5-6B-Chat — Selection methods only (Q2C bug fix)

Batch 19b had a boundary detection bug: "assistant" in system message
("helpful assistant") matched before <|im_start|>assistant marker,
causing empty question_positions and no Q2C results.

Fix: Search for "<|im_start|>assistant" instead of just "assistant".

This batch runs ONLY selection experiments (Q2C, SnapKV, H2O, Random)
since quantization results from 19b are already conclusive.
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch19c.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 1024
MODEL_NAME = '01-ai/Yi-1.5-6B-Chat'
MODEL_SHORT = 'Yi-1.5-6B-Chat'
NUM_SAMPLES = 50


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


def find_boundaries_chatml(tokenizer, input_ids, seq_len):
    """Find context/question/answer boundaries in ChatML-formatted prompt.

    Looks for 'Question:' to start question region, and '<|im_start|>assistant'
    (or the eos+im_start pattern) to end it.
    """
    q_start, a_start = None, None

    # Decode full text to find boundaries
    full_text = tokenizer.decode(input_ids[0])

    # Find "Question:" position in text
    q_text_pos = full_text.find("Question:")
    if q_text_pos == -1:
        q_text_pos = int(len(full_text) * 0.7)

    # Find the assistant marker - look for the LAST occurrence of the marker
    # before the end of the input (there may be "assistant" in system msg too)
    a_text_pos = full_text.rfind("<|im_start|>assistant")
    if a_text_pos == -1:
        a_text_pos = full_text.rfind("assistant\n")
    if a_text_pos == -1:
        a_text_pos = len(full_text)

    # Map text positions to token positions
    for i in range(seq_len):
        decoded = tokenizer.decode(input_ids[0][:i+1])
        if len(decoded) >= q_text_pos and q_start is None:
            q_start = i
        if len(decoded) >= a_text_pos and a_start is None and q_start is not None:
            a_start = i
            break

    if q_start is None:
        q_start = int(seq_len * 0.7)
    if a_start is None:
        a_start = seq_len

    context_positions = list(range(0, q_start))
    question_positions = list(range(q_start, a_start))

    return context_positions, question_positions


def compute_q2c_scores(model, tokenizer, input_ids, seq_len):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions

    cp, qp = find_boundaries_chatml(tokenizer, input_ids, seq_len)

    if not cp or not qp:
        return [], cp, qp

    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        for q_pos in qp:
            scores += layer_attn[0, :, q_pos, :].mean(dim=0)

    context_scores = [(pos, scores[pos].item()) for pos in cp]
    context_scores.sort(key=lambda x: x[1], reverse=True)
    return context_scores, cp, qp


def compute_h2o_scores(model, tokenizer, input_ids, seq_len):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions

    cp, qp = find_boundaries_chatml(tokenizer, input_ids, seq_len)

    if not cp or not qp:
        return [], cp, qp

    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        scores += layer_attn[0].sum(dim=(0, 1))

    context_scores = [(pos, scores[pos].item()) for pos in cp]
    context_scores.sort(key=lambda x: x[1], reverse=True)
    return context_scores, cp, qp


def make_selection_mask(seq_len, context_scores, context_positions, question_positions, retention):
    n_keep = int(len(context_positions) * retention)
    selected = set(pos for pos, _ in context_scores[:n_keep])
    always_keep = set(question_positions) | set(range(max(0, seq_len-5), seq_len))
    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
    for p in always_keep:
        if p < seq_len: mask[0, p] = 1
    for p in selected:
        if p < seq_len: mask[0, p] = 1
    return mask


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    candidates = [s for s in ds if s['answers']['text']]
    np.random.seed(42)
    indices = np.random.choice(len(candidates), min(num_samples, len(candidates)), replace=False)
    return [candidates[i] for i in indices]


def format_squad_chat(sample):
    context = sample['context']
    question = sample['question']
    prompt = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant. Answer the question using ONLY words from the context. "
        f"Give the shortest possible answer - just the exact words from the context, nothing else.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context: {context}\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def run_yi_selection(num_samples=50):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"\n{'#'*80}\nYi-1.5-6B-Chat Selection Methods (Q2C bug fix)\n{'#'*80}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded: {num_layers} layers, {model.config.num_key_value_heads} KV heads")

    samples = load_squad(num_samples)

    # Verify boundary detection on first sample
    prompt = format_squad_chat(samples[0])
    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to("cuda")
    cp, qp = find_boundaries_chatml(tokenizer, inputs['input_ids'], inputs['input_ids'].shape[1])
    logger.info(f"Boundary check: context_positions={len(cp)}, question_positions={len(qp)}")
    if qp:
        q_text = tokenizer.decode(inputs['input_ids'][0][qp[0]:qp[-1]+1])
        logger.info(f"Question region: '{q_text[:100]}'")

    results = []
    for i, sample in enumerate(samples):
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = format_squad_chat(sample)

        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # Full baseline
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        try:
            # Q2C
            cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
            if cs:
                for ret in [0.5, 0.25]:
                    sel_mask = make_selection_mask(seq_len, cs, cp, qp, ret)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask)
                    result[f'q2c_{int(ret*100)}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # SnapKV 50%
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
                snap_scores = torch.zeros(seq_len, device='cuda')
                obs_window = min(64, seq_len // 4)
                qw = list(range(max(0, seq_len - obs_window), seq_len))
                for la in out.attentions:
                    for qp_ in qw:
                        snap_scores += la[0, :, qp_, :].mean(dim=0)
                snap_ctx = [(p, snap_scores[p].item()) for p in cp]
                snap_ctx.sort(key=lambda x: x[1], reverse=True)
                snap_mask = make_selection_mask(seq_len, snap_ctx, cp, qp, 0.5)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                       selection_mask=snap_mask)
                result['snapkv_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # H2O 50%
                h2o_cs, _, _ = compute_h2o_scores(model, tokenizer, input_ids, seq_len)
                if h2o_cs:
                    h2o_mask = make_selection_mask(seq_len, h2o_cs, cp, qp, 0.5)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=h2o_mask)
                    result['h2o_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # Random 50%
                np.random.seed(42 + i)
                n_keep = int(len(cp) * 0.5)
                rand_sel = list(np.random.choice(cp, n_keep, replace=False))
                rand_mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
                for p in set(qp) | set(range(max(0, seq_len-5), seq_len)):
                    if p < seq_len: rand_mask[0, p] = 1
                for p in rand_sel:
                    if p < seq_len: rand_mask[0, p] = 1
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                       selection_mask=rand_mask)
                result['random_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}
            else:
                logger.warning(f"Sample {i}: Q2C returned empty scores (cp={len(cp)}, qp={len(qp)})")

        except Exception as e:
            logger.warning(f"Selection failed for sample {i}: {e}")
            import traceback; traceback.print_exc()

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        fp16 = result.get('full', {}).get('f1', -1)
        q2c50 = result.get('q2c_50', {}).get('f1', -1)
        snap50 = result.get('snapkv_50', {}).get('f1', -1)
        h2o50 = result.get('h2o_50', {}).get('f1', -1)
        rand50 = result.get('random_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={fp16:.3f} q2c={q2c50:.3f} snap={snap50:.3f} "
                    f"h2o={h2o50:.3f} rand={rand50:.3f} ({elapsed:.1f}s)")

    # Summary
    fp16_f1 = float(np.mean([r.get('full', {}).get('f1', 0) for r in results]))
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: {MODEL_SHORT} Selection Methods ({len(results)} samples)")
    logger.info(f"Baseline F1 = {fp16_f1:.4f}")

    all_keys = set()
    for r in results:
        for k in r:
            if isinstance(r[k], dict) and 'f1' in r[k]:
                all_keys.add(k)

    summary = {'model': MODEL_SHORT, 'full_f1': fp16_f1, 'num_samples': len(results)}
    for key in sorted(all_keys):
        vals = [r.get(key, {}).get('f1', 0) for r in results if key in r]
        if vals:
            f1 = float(np.mean(vals))
            summary[f'{key}_f1'] = f1
            pct = f1 / fp16_f1 * 100 if fp16_f1 > 0 else 0
            logger.info(f"  {key:25s}: F1={f1:.4f} ({pct:5.1f}%)")

    final_path = RESULTS_DIR / f'yi6b_selection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return summary


if __name__ == '__main__':
    logger.info(f"Start: {datetime.now()}")
    summary = run_yi_selection(NUM_SAMPLES)
    logger.info(f"Done: {datetime.now()}")
