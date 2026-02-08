#!/usr/bin/env python3
"""
Batch 13: Cross-Family Validation — Does Layer 0 bottleneck + Q2C dominance
hold for non-Qwen architectures?

Tests Microsoft Phi-3-mini (3.8B) — different architecture, open model.
Key experiments:
  1. Baseline F1 (SQuAD v2)
  2. Quantization sweep (INT4-INT8, per-token + per-channel)
  3. Layer-wise quantization sensitivity (Layer 0 bottleneck test)
  4. Q2C selection at 50% and 75%
  5. Mixed-precision (Layer 0 FP16 + rest INT4)

If Phi-3 fails or isn't available, falls back to Llama-3.2-3B-Instruct.
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch13.log')])
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


def quantize_perchannel(t, bits):
    if bits >= 16: return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
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


def generate_quantized(model, tokenizer, input_ids, seq_len, bits, mode='pertoken',
                       layer0_bits=None, max_new=64, selection_mask=None):
    """Generate with quantized KV, optionally with mixed precision and selection."""
    qfn = quantize_perchannel if mode == 'perchannel' else quantize_pertoken
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
            layer.keys.copy_(qfn(layer.keys, b))
            layer.values.copy_(qfn(layer.values, b))

    mask = selection_mask.clone() if selection_mask is not None else torch.ones(1, seq_len, device='cuda', dtype=torch.long)
    return manual_generate_with_mask(model, tokenizer, pkv, first_token_id, seq_len, mask, max_new)


def compute_q2c_scores(model, tokenizer, input_ids, seq_len):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    q_start, a_start = None, None
    decoded_so_far = ""
    for i, tok in enumerate(tokens):
        decoded_so_far = tokenizer.decode(input_ids[0][:i+1])
        if "Question:" in decoded_so_far and q_start is None:
            q_start = i
        if "Answer:" in decoded_so_far and a_start is None:
            a_start = i
            break

    if q_start is None or a_start is None:
        q_start = int(seq_len * 0.8)
        a_start = seq_len

    context_positions = list(range(0, q_start))
    question_positions = list(range(q_start, a_start))

    if not context_positions or not question_positions:
        return [], context_positions, question_positions

    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        for q_pos in question_positions:
            scores += layer_attn[0, :, q_pos, :].mean(dim=0)

    context_scores = [(pos, scores[pos].item()) for pos in context_positions]
    context_scores.sort(key=lambda x: x[1], reverse=True)
    return context_scores, context_positions, question_positions


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
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def try_load_model(model_ids, dtype):
    """Try loading models in order, return the first that works."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for model_id in model_ids:
        try:
            logger.info(f"Trying to load {model_id} ({dtype})...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map="cuda",
                trust_remote_code=True, attn_implementation='eager')
            model.config.use_cache = True
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Successfully loaded {model_id}")
            return model, tokenizer, model_id
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")
            continue

    raise RuntimeError(f"Could not load any model from: {model_ids}")


def run_crossfamily(model, tokenizer, model_name, num_layers, num_samples=50):
    exp_name = f'crossfamily_{model_name}'
    logger.info(f"\n{'='*60}\nCross-Family Validation: {model_name} ({num_layers} layers)\n{'='*60}")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    # Layer indices for probing (evenly spaced)
    probe_layers = sorted(set([0, num_layers//6, num_layers//3, num_layers//2,
                               2*num_layers//3, num_layers-1]))

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # 1. Full baseline
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 2. Quantization sweep
        for bits in [4, 6, 7, 8]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits, 'pertoken')
            result[f'ptk_int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        for bits in [4, 6]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits, 'perchannel')
            result[f'pch_int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 3. Mixed-precision (Layer 0 FP16 + rest INT4)
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, 'pertoken', layer0_bits=16)
        result['mixed_ptk_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 4. Layer-wise: only layer X at INT4
        for li in probe_layers:
            layer_bits = {li: 4}
            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=True)
            first_tok = out.logits[:, -1, :].argmax(dim=-1).item()
            pkv = out.past_key_values
            for layer_idx in range(len(pkv.layers)):
                if layer_idx in layer_bits:
                    layer = pkv.layers[layer_idx]
                    layer.keys.copy_(quantize_pertoken(layer.keys, 4))
                    layer.values.copy_(quantize_pertoken(layer.values, 4))
            mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)
            ans = manual_generate_with_mask(model, tokenizer, pkv, first_tok, seq_len, mask)
            result[f'only_L{li}_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 5. Q2C selection at 50% and 75%
        try:
            cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
            if cs:
                for retention in [0.5, 0.75]:
                    sel_mask = make_selection_mask(seq_len, cs, cp, qp, retention)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask)
                    pct = int(retention * 100)
                    result[f'q2c_{pct}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # SnapKV at 50%
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
                snap_scores = torch.zeros(seq_len, device='cuda')
                qw = list(range(max(0, seq_len-32), seq_len))
                for la in out.attentions:
                    for qp_ in qw:
                        snap_scores += la[0, :, qp_, :].mean(dim=0)
                snap_ctx = [(p, snap_scores[p].item()) for p in cp]
                snap_ctx.sort(key=lambda x: x[1], reverse=True)
                n_keep = int(len(cp) * 0.5)
                snap_sel = set(p for p, _ in snap_ctx[:n_keep])
                snap_mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
                for p in set(qp) | set(range(max(0, seq_len-5), seq_len)):
                    if p < seq_len: snap_mask[0, p] = 1
                for p in snap_sel:
                    if p < seq_len: snap_mask[0, p] = 1
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                       selection_mask=snap_mask)
                result['snapkv_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # Random at 50%
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
        except Exception as e:
            logger.warning(f"Selection failed for sample {i}: {e}")
            result['selection_error'] = str(e)

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        fp16 = result.get('full', {}).get('f1', -1)
        mixed = result.get('mixed_ptk_int4', {}).get('f1', -1)
        q2c = result.get('q2c_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={fp16:.3f} mixed={mixed:.3f} q2c50={q2c:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results), 'num_layers': num_layers, 'model': model_name}
    all_keys = set()
    for r in results:
        for k in r:
            if isinstance(r[k], dict) and 'f1' in r[k]:
                all_keys.add(k)

    fp16_f1 = float(np.mean([r.get('full', {}).get('f1', 0) for r in results]))
    summary['full_f1'] = fp16_f1

    for key in sorted(all_keys):
        vals = [r.get(key, {}).get('f1', 0) for r in results if key in r]
        if vals:
            f1 = float(np.mean(vals))
            summary[f'{key}_f1'] = f1
            summary[f'{key}_std'] = float(np.std(vals))
            pct = f1 / fp16_f1 * 100 if fp16_f1 > 0 else 0
            logger.info(f"  {key:25s}: F1={f1:.4f} ({pct:5.1f}%)")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    return summary


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    # Try models in order of preference (open, no auth required)
    model_candidates = [
        "microsoft/phi-3-mini-4k-instruct",    # 3.8B, Phi architecture
        "microsoft/Phi-3.5-mini-instruct",      # 3.8B, newer Phi
        "Qwen/Qwen2.5-1.5B",                   # 1.5B, same family but diff size
    ]

    model, tokenizer, model_name = try_load_model(model_candidates, torch.bfloat16)
    num_layers = model.config.num_hidden_layers
    logger.info(f"Model: {model_name}, {num_layers} layers, "
                f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    summary = run_crossfamily(model, tokenizer, model_name.replace('/', '_'), num_layers, num_samples=50)

    del model; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 13 COMPLETE in {elapsed:.1f} minutes")
