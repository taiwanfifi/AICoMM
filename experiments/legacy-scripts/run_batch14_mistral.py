#!/usr/bin/env python3
"""
Batch 14: Cross-Family Validation with Mistral-7B-Instruct-v0.3

Goal: Definitive test of whether Layer 0 bottleneck + Q2C dominance
holds for non-Qwen instruction-tuned models.

Mistral-7B-Instruct specs:
- 32 layers, GQA (8 KV heads, 32 Q heads), head_dim=128
- Sliding window attention (4096)
- Instruction-tuned → should have good extractive QA performance

Key experiments:
  1. Baseline F1 (SQuAD v2, 50 samples)
  2. Quantization sweep: INT4, INT6, INT7, INT8 (per-token + per-channel for INT4/6)
  3. Layer-wise sensitivity: only layer X at INT4 (6 probe layers)
  4. Layer-wise recovery: keep only layer X at FP16, rest INT4 (6 probe layers)
  5. Mixed-precision: Layer 0 FP16 + rest INT4
  6. Q2C, SnapKV, H2O, Random selection at 50%
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch14.log')])
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


def generate_layerwise_quant(model, tokenizer, input_ids, seq_len, layer_bits_map, max_new=64):
    """Generate with specific layers quantized to specific bits.
    layer_bits_map: dict of {layer_idx: bits}. All other layers stay FP16."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    first_tok = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    for layer_idx in range(len(pkv.layers)):
        if layer_idx in layer_bits_map:
            bits = layer_bits_map[layer_idx]
            if bits < 16:
                layer = pkv.layers[layer_idx]
                layer.keys.copy_(quantize_pertoken(layer.keys, bits))
                layer.values.copy_(quantize_pertoken(layer.values, bits))

    mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)
    return manual_generate_with_mask(model, tokenizer, pkv, first_tok, seq_len, mask, max_new)


def generate_except_layer_quant(model, tokenizer, input_ids, seq_len, keep_fp16_layer, bits=4, max_new=64):
    """Quantize ALL layers to `bits` EXCEPT `keep_fp16_layer` which stays FP16."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    first_tok = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    for layer_idx in range(len(pkv.layers)):
        if layer_idx != keep_fp16_layer:
            layer = pkv.layers[layer_idx]
            layer.keys.copy_(quantize_pertoken(layer.keys, bits))
            layer.values.copy_(quantize_pertoken(layer.values, bits))

    mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)
    return manual_generate_with_mask(model, tokenizer, pkv, first_tok, seq_len, mask, max_new)


def compute_q2c_scores(model, tokenizer, input_ids, seq_len):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions

    # Find question/context boundaries from decoded text
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


def compute_h2o_scores(attentions, seq_len):
    """H2O: Heavy Hitter Oracle — select positions with highest cumulative attention received."""
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        # Sum attention each position RECEIVES across all query positions and heads
        scores += layer_attn[0, :, :, :].sum(dim=(0, 1))  # [seq_len]
    return scores


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def format_prompt_mistral(tokenizer, context, question):
    """Format prompt for Mistral-Instruct using chat template."""
    messages = [
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def run_crossfamily(model, tokenizer, model_name, num_layers, num_samples=50):
    exp_name = f'crossfamily_{model_name}'
    logger.info(f"\n{'='*60}\nBatch 14 Cross-Family: {model_name} ({num_layers} layers)\n{'='*60}")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    # Layer indices for probing (evenly spaced)
    probe_layers = sorted(set([0, num_layers//6, num_layers//3, num_layers//2,
                               2*num_layers//3, num_layers-1]))
    logger.info(f"Probe layers: {probe_layers}")

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]

        # Use chat template for instruction-tuned models
        try:
            prompt = format_prompt_mistral(tokenizer, sample['context'], sample['question'])
        except Exception:
            prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # 1. Full baseline
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 2. Quantization sweep (per-token)
        for bits in [4, 6, 7, 8]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits, 'pertoken')
            result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 3. Per-channel INT4 and INT6
        for bits in [4, 6]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits, 'perchannel')
            result[f'pch_int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 4. Mixed-precision: Layer 0 FP16 + rest INT4 (per-token)
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, 'pertoken', layer0_bits=16)
        result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 5. Layer-wise: only layer X at INT4 (rest FP16)
        for li in probe_layers:
            ans = generate_layerwise_quant(model, tokenizer, input_ids, seq_len, {li: 4})
            result[f'only_L{li}_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 6. Layer-wise: keep ONLY layer X at FP16, everything else INT4
        for li in probe_layers:
            ans = generate_except_layer_quant(model, tokenizer, input_ids, seq_len, keep_fp16_layer=li)
            result[f'except_L{li}_fp16'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 7. Selection methods (Q2C, SnapKV, H2O, Random at 50%)
        try:
            cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
            if cs:
                # Q2C 50% and 75%
                for retention in [0.5, 0.75]:
                    sel_mask = make_selection_mask(seq_len, cs, cp, qp, retention)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask)
                    pct = int(retention * 100)
                    result[f'q2c_{pct}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # SnapKV at 50%: attention from last 32 tokens (observation window)
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
                snap_scores = torch.zeros(seq_len, device='cuda')
                qw = list(range(max(0, seq_len-32), seq_len))
                for la in out.attentions:
                    for qp_ in qw:
                        snap_scores += la[0, :, qp_, :].mean(dim=0)
                snap_ctx = [(p, snap_scores[p].item()) for p in cp]
                snap_ctx.sort(key=lambda x: x[1], reverse=True)
                snap_mask = make_selection_mask(seq_len, snap_ctx, cp, qp, 0.5)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                       selection_mask=snap_mask)
                result['snapkv_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # H2O at 50%: heavy hitter oracle
                h2o_scores = compute_h2o_scores(out.attentions, seq_len)
                h2o_ctx = [(p, h2o_scores[p].item()) for p in cp]
                h2o_ctx.sort(key=lambda x: x[1], reverse=True)
                h2o_mask = make_selection_mask(seq_len, h2o_ctx, cp, qp, 0.5)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                       selection_mask=h2o_mask)
                result['h2o_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

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
        int4 = result.get('int4', {}).get('f1', -1)
        mixed = result.get('mixed_L0fp16_int4', {}).get('f1', -1)
        q2c = result.get('q2c_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={fp16:.3f} int4={int4:.3f} mixed={mixed:.3f} q2c50={q2c:.3f} ({elapsed:.1f}s)")

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

    # Try Mistral-7B-Instruct (GQA, different architecture, instruction-tuned)
    model_candidates = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "tiiuae/falcon-7b-instruct",
    ]

    for model_id in model_candidates:
        try:
            logger.info(f"Trying {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="cuda",
                trust_remote_code=True, attn_implementation='eager')
            model.config.use_cache = True
            model.eval()
            model_name = model_id.replace('/', '_')
            logger.info(f"Loaded {model_id}")
            break
        except Exception as e:
            logger.warning(f"Failed {model_id}: {e}")
            continue
    else:
        raise RuntimeError("Could not load any cross-family model")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    logger.info(f"Model: {model_name}, {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")

    # Quick smoke test
    logger.info("Smoke test...")
    test_prompt = "What is 2+2? Answer:"
    test_ids = tokenizer(test_prompt, return_tensors="pt").to("cuda")['input_ids']
    with torch.no_grad():
        test_out = model.generate(test_ids, max_new_tokens=20)
    test_ans = tokenizer.decode(test_out[0][test_ids.shape[1]:], skip_special_tokens=True)
    logger.info(f"Smoke test: '{test_prompt}' -> '{test_ans[:100]}'")

    summary = run_crossfamily(model, tokenizer, model_name, num_layers, num_samples=50)

    del model; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 14 COMPLETE in {elapsed:.1f} minutes")
