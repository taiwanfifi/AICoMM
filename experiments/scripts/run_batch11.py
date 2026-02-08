#!/usr/bin/env python3
"""
Batch 11: Three experiments
  11a: Layer-wise quantization sensitivity — which layers tolerate INT4/INT6?
  11b: 7B TriviaQA (selection + quantization) — cross-dataset validation
  11c: INT6 anomaly investigation — run INT5/INT6/INT7 with FP32 accumulation

Motivation:
- 11a: Topic 11 (layer-heterogeneous compression) needs data on which layers
  need higher precision. Could enable mixed-precision protocol.
- 11b: We have 3B TriviaQA (batch 7) but not 7B. Need for cross-model comparison.
- 11c: INT6 anomaly (54% vs INT5=89%) is non-physical. Test hypothesis that
  it's a BF16 numerical issue.
"""
import os, sys, json, time, logging, copy, gc
from pathlib import Path
from datetime import datetime
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/dev/shm/hf_7b'  # Default to /dev/shm (more space)
os.environ['HF_DATASETS_CACHE'] = '/dev/shm/hf_7b/datasets'

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch11.log')])
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


def quantize_tensor(t, bits):
    if bits >= 16:
        return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    t_q = (t / scale).round().clamp(qmin, qmax)
    return t_q * scale


def manual_generate_with_mask(model, tokenizer, past_kv, first_token_id, seq_len, mask, max_new=64):
    """Token-by-token generation with attention mask."""
    generated = [first_token_id]
    cur_len = seq_len

    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        new_token_mask = torch.ones(1, 1, device='cuda', dtype=torch.long)
        mask = torch.cat([mask, new_token_mask], dim=1)

        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       attention_mask=mask, position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def manual_generate_quantized(model, tokenizer, input_ids, seq_len, quant_bits, max_new=64,
                               layer_bits=None):
    """Generate with quantized KV cache.

    Args:
        layer_bits: Optional dict mapping layer_idx -> bits. If provided, overrides quant_bits
                   for specific layers (for layer-wise sensitivity testing).
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    past_kv = out.past_key_values

    # Quantize KV cache in-place
    for layer_idx in range(len(past_kv.layers)):
        if layer_bits is not None:
            bits = layer_bits.get(layer_idx, 16)  # Default to FP16 for unspecified layers
        else:
            bits = quant_bits
        if bits < 16:
            layer = past_kv.layers[layer_idx]
            layer.keys.copy_(quantize_tensor(layer.keys, bits))
            layer.values.copy_(quantize_tensor(layer.values, bits))

    # Manual generation
    generated = [first_token_id]
    cur_len = seq_len
    full_mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)

    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        full_mask = torch.cat([full_mask, torch.ones(1, 1, device='cuda', dtype=torch.long)], dim=1)

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


def compute_q2c_scores(model, tokenizer, input_ids, seq_len):
    """Compute Q2C attention scores for selection."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions  # tuple of (batch, heads, seq, seq)

    # Find question boundaries
    text = tokenizer.decode(input_ids[0])
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Find "Question:" and "Answer:" positions
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
        # Fallback: use last 20% as question
        q_start = int(seq_len * 0.8)
        a_start = seq_len

    # Context positions (before question)
    context_positions = list(range(0, q_start))
    question_positions = list(range(q_start, a_start))

    if not context_positions or not question_positions:
        return list(range(seq_len)), context_positions, question_positions

    # Average attention from question tokens to context tokens across all layers and heads
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        # layer_attn: (1, heads, seq, seq)
        # Average over heads, sum over question positions attending to each context position
        for q_pos in question_positions:
            scores += layer_attn[0, :, q_pos, :].mean(dim=0)  # avg over heads

    # Normalize
    context_scores = [(pos, scores[pos].item()) for pos in context_positions]
    context_scores.sort(key=lambda x: x[1], reverse=True)

    return context_scores, context_positions, question_positions


def generate_with_selection(model, tokenizer, input_ids, seq_len, selected_positions, always_keep, max_new=64):
    """Generate using attention mask for selection."""
    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
    for p in always_keep:
        if p < seq_len: mask[0, p] = 1
    for p in selected_positions:
        if p < seq_len: mask[0, p] = 1

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


def load_triviaqa(num_samples):
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc', split='validation')
    samples = []
    for s in ds:
        if len(s['answer']['aliases']) > 0 and len(s.get('entity_pages', {}).get('wiki_context', [])) > 0:
            samples.append(s)
            if len(samples) >= num_samples:
                break
    return samples


# ============================================================
# EXPERIMENT 11a: Layer-wise Quantization Sensitivity
# ============================================================
def run_layerwise_quant(model, tokenizer, model_name, num_layers, num_samples=50):
    """Test quantizing ONLY specific layers to INT4 while keeping rest at FP16."""
    exp_name = f'layerwise_quant_{model_name}'
    logger.info(f"\n{'='*60}\n11a: Layer-wise Quantization Sensitivity ({model_name})\n{'='*60}")

    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    # Configurations:
    # 1. All layers INT4 (baseline from batch 10)
    # 2. Each individual layer at INT4 (rest FP16) — measures per-layer sensitivity
    # 3. Each individual layer at FP16 (rest INT4) — measures per-layer importance
    # 4. Layer groups: first 1/3, middle 1/3, last 1/3 at INT4

    third = num_layers // 3
    configs = {
        'all_fp16': {},  # no quantization
        'all_int4': {i: 4 for i in range(num_layers)},
        'first_third_int4': {i: 4 for i in range(third)},
        'middle_third_int4': {i: 4 for i in range(third, 2*third)},
        'last_third_int4': {i: 4 for i in range(2*third, num_layers)},
    }

    # Add per-layer configs: "only layer X at INT4"
    # Sample 6 evenly spaced layers to keep runtime manageable
    probe_layers = [0, num_layers//6, num_layers//3, num_layers//2, 2*num_layers//3, num_layers-1]
    for li in probe_layers:
        configs[f'only_layer{li}_int4'] = {li: 4}

    # Add per-layer configs: "everything INT4 EXCEPT layer X"
    for li in probe_layers:
        cfg = {i: 4 for i in range(num_layers)}
        del cfg[li]  # This layer stays FP16
        configs[f'except_layer{li}_fp16'] = cfg

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        for cfg_name, layer_bits in configs.items():
            try:
                if not layer_bits:  # all FP16
                    ans = manual_generate_quantized(model, tokenizer, input_ids, seq_len, 16)
                else:
                    ans = manual_generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                                    layer_bits=layer_bits)
                result[cfg_name] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}
            except Exception as e:
                logger.warning(f"{cfg_name} failed: {e}")
                result[cfg_name] = {'answer': '', 'f1': 0.0, 'error': str(e)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        fp16 = result.get('all_fp16', {}).get('f1', -1)
        int4 = result.get('all_int4', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] fp16={fp16:.3f} int4={int4:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results), 'num_layers': num_layers}
    for cfg_name in configs:
        vals = [r.get(cfg_name, {}).get('f1', 0) for r in results]
        summary[f'{cfg_name}_f1'] = float(np.mean(vals))
        summary[f'{cfg_name}_std'] = float(np.std(vals))

    logger.info(f"\n--- Layer-wise Quantization Summary ({model_name}) ---")
    fp16_f1 = summary['all_fp16_f1']
    for cfg_name in configs:
        f1 = summary[f'{cfg_name}_f1']
        pct = f1 / fp16_f1 * 100 if fp16_f1 > 0 else 0
        logger.info(f"  {cfg_name:30s}: F1={f1:.4f} ({pct:.1f}%)")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': model_name,
                   'configs': {k: str(v) for k, v in configs.items()},
                   'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    return summary


# ============================================================
# EXPERIMENT 11b: 7B TriviaQA (Selection + Quantization)
# ============================================================
def run_7b_triviaqa(model, tokenizer, num_samples=50):
    """7B TriviaQA with Q2C selection + quantization."""
    exp_name = 'triviaqa_7b'
    logger.info(f"\n{'='*60}\n11b: 7B TriviaQA Selection + Quantization\n{'='*60}")

    samples = load_triviaqa(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()

        # TriviaQA format
        gold = sample['answer']['aliases'][0]  # primary alias
        all_answers = sample['answer']['aliases']
        context = sample['entity_pages']['wiki_context'][0][:2000]  # truncate long contexts
        question = sample['question']

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'all_answers': all_answers, 'seq_len': seq_len}

        # Full baseline
        ans = manual_generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        best_f1 = max(compute_f1(ans, g) for g in all_answers)
        result['full'] = {'answer': ans[:200], 'f1': best_f1}

        # INT8 and INT4
        for bits in [8, 4]:
            ans = manual_generate_quantized(model, tokenizer, input_ids, seq_len, bits)
            best_f1 = max(compute_f1(ans, g) for g in all_answers)
            result[f'int{bits}'] = {'answer': ans[:200], 'f1': best_f1}

        # Q2C selection at 50% and 75%
        try:
            context_scores, context_positions, question_positions = compute_q2c_scores(
                model, tokenizer, input_ids, seq_len)
            always_keep = question_positions + list(range(max(0, seq_len-5), seq_len))

            for retention in [0.5, 0.75]:
                n_keep = int(len(context_positions) * retention)
                selected = [pos for pos, _ in context_scores[:n_keep]]
                ans = generate_with_selection(model, tokenizer, input_ids, seq_len,
                                             selected, always_keep)
                best_f1 = max(compute_f1(ans, g) for g in all_answers)
                pct = int(retention * 100)
                result[f'q2c_{pct}'] = {'answer': ans[:200], 'f1': best_f1}

            # SnapKV at 50%
            with torch.no_grad():
                out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
            attentions = out.attentions
            snap_scores = torch.zeros(seq_len, device='cuda')
            query_window = list(range(max(0, seq_len-32), seq_len))
            for layer_attn in attentions:
                for q_pos in query_window:
                    snap_scores += layer_attn[0, :, q_pos, :].mean(dim=0)

            snap_context = [(pos, snap_scores[pos].item()) for pos in context_positions]
            snap_context.sort(key=lambda x: x[1], reverse=True)
            n_keep = int(len(context_positions) * 0.5)
            selected = [pos for pos, _ in snap_context[:n_keep]]
            ans = generate_with_selection(model, tokenizer, input_ids, seq_len,
                                         selected, always_keep)
            best_f1 = max(compute_f1(ans, g) for g in all_answers)
            result['snapkv_50'] = {'answer': ans[:200], 'f1': best_f1}

            # Random at 50%
            np.random.seed(42 + i)
            n_keep = int(len(context_positions) * 0.5)
            random_sel = list(np.random.choice(context_positions, n_keep, replace=False))
            ans = generate_with_selection(model, tokenizer, input_ids, seq_len,
                                         random_sel, always_keep)
            best_f1 = max(compute_f1(ans, g) for g in all_answers)
            result['random_50'] = {'answer': ans[:200], 'f1': best_f1}

        except Exception as e:
            logger.warning(f"Selection failed for sample {i}: {e}")
            result['selection_error'] = str(e)

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        f_full = result.get('full', {}).get('f1', -1)
        f_q2c = result.get('q2c_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={f_full:.3f} q2c50={f_q2c:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results)}
    for key in ['full', 'int8', 'int4', 'q2c_50', 'q2c_75', 'snapkv_50', 'random_50']:
        vals = [r.get(key, {}).get('f1', 0) for r in results if key in r]
        if vals:
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))

    logger.info(f"\n--- 7B TriviaQA Summary ---")
    full = summary.get('full_f1', 0)
    for key in ['full', 'int8', 'int4', 'q2c_50', 'q2c_75', 'snapkv_50', 'random_50']:
        f1 = summary.get(f'{key}_f1', 0)
        pct = f1 / full * 100 if full > 0 else 0
        logger.info(f"  {key:15s}: F1={f1:.4f} ({pct:.1f}%)")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': 'qwen25_7b', 'dataset': 'triviaqa',
                   'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    return summary


# ============================================================
# EXPERIMENT 11c: INT6 Anomaly Investigation
# ============================================================
def run_int6_investigation(model, tokenizer, model_name, num_samples=50):
    """Investigate INT6 anomaly by testing with FP32 intermediate computation."""
    exp_name = f'int6_investigation_{model_name}'
    logger.info(f"\n{'='*60}\n11c: INT6 Anomaly Investigation ({model_name})\n{'='*60}")

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

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # Standard quantization at INT5, INT6, INT7
        for bits in [5, 6, 7]:
            ans = manual_generate_quantized(model, tokenizer, input_ids, seq_len, bits)
            result[f'int{bits}_standard'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # INT6 with FP32 quantization (do the quant math in FP32, then cast back)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
        past_kv = out.past_key_values

        for layer_idx in range(len(past_kv.layers)):
            layer = past_kv.layers[layer_idx]
            # Do quantization in FP32 to avoid BF16 rounding issues
            k_fp32 = layer.keys.float()
            v_fp32 = layer.values.float()
            k_q = quantize_tensor(k_fp32, 6).to(layer.keys.dtype)
            v_q = quantize_tensor(v_fp32, 6).to(layer.values.dtype)
            layer.keys.copy_(k_q)
            layer.values.copy_(v_q)

        # Generate with FP32-quantized cache
        generated = [first_token_id]
        cur_len = seq_len
        full_mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)

        for step in range(63):
            next_input = torch.tensor([[generated[-1]]], device='cuda')
            position_ids = torch.tensor([[cur_len]], device='cuda')
            full_mask = torch.cat([full_mask, torch.ones(1, 1, device='cuda', dtype=torch.long)], dim=1)
            with torch.no_grad():
                out2 = model(input_ids=next_input, past_key_values=past_kv,
                           attention_mask=full_mask, position_ids=position_ids, use_cache=True)
            past_kv = out2.past_key_values
            next_tok = out2.logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_tok)
            cur_len += 1
            if next_tok == tokenizer.eos_token_id: break

        ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
        result['int6_fp32quant'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # INT6 with per-channel quantization (instead of per-token)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
        past_kv = out.past_key_values

        for layer_idx in range(len(past_kv.layers)):
            layer = past_kv.layers[layer_idx]
            # Per-channel: amax over seq_len dimension (dim=-2) instead of head_dim (dim=-1)
            for tensor_name in ['keys', 'values']:
                t = getattr(layer, tensor_name)
                qmin, qmax = -32, 31  # INT6
                amax = t.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)  # per-channel
                scale = amax / qmax
                t_q = (t / scale).round().clamp(qmin, qmax) * scale
                getattr(layer, tensor_name).copy_(t_q)

        generated = [first_token_id]
        cur_len = seq_len
        full_mask = torch.ones(1, seq_len, device='cuda', dtype=torch.long)

        for step in range(63):
            next_input = torch.tensor([[generated[-1]]], device='cuda')
            position_ids = torch.tensor([[cur_len]], device='cuda')
            full_mask = torch.cat([full_mask, torch.ones(1, 1, device='cuda', dtype=torch.long)], dim=1)
            with torch.no_grad():
                out2 = model(input_ids=next_input, past_key_values=past_kv,
                           attention_mask=full_mask, position_ids=position_ids, use_cache=True)
            past_kv = out2.past_key_values
            next_tok = out2.logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_tok)
            cur_len += 1
            if next_tok == tokenizer.eos_token_id: break

        ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
        result['int6_perchannel'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        std6 = result.get('int6_standard', {}).get('f1', -1)
        fp32_6 = result.get('int6_fp32quant', {}).get('f1', -1)
        pch6 = result.get('int6_perchannel', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] int6_std={std6:.3f} int6_fp32={fp32_6:.3f} int6_pch={pch6:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results)}
    for key in ['int5_standard', 'int6_standard', 'int7_standard', 'int6_fp32quant', 'int6_perchannel']:
        vals = [r.get(key, {}).get('f1', 0) for r in results if key in r]
        if vals:
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))

    logger.info(f"\n--- INT6 Investigation Summary ({model_name}) ---")
    for key in ['int5_standard', 'int6_standard', 'int7_standard', 'int6_fp32quant', 'int6_perchannel']:
        f1 = summary.get(f'{key}_f1', 0)
        logger.info(f"  {key:25s}: F1={f1:.4f}")

    final_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'model': model_name,
                   'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    return summary


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    # === Load 7B model (BF16 for Blackwell) ===
    # Use /dev/shm cache for 7B (overlay disk is limited)
    os.environ['HF_HOME'] = '/dev/shm/hf_7b'
    logger.info("Loading Qwen2.5-7B (BF16, eager) from /dev/shm cache...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_7b.config.use_cache = True
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tok_7b.pad_token is None: tok_7b.pad_token = tok_7b.eos_token

    num_layers_7b = model_7b.config.num_hidden_layers
    logger.info(f"7B model: {num_layers_7b} layers, head_dim={model_7b.config.hidden_size // model_7b.config.num_attention_heads}")

    # 11c: INT6 anomaly investigation (7B only — 3B doesn't have this issue)
    s11c = run_int6_investigation(model_7b, tok_7b, "qwen25_7b", num_samples=50)

    # 11a: Layer-wise quantization sensitivity (7B)
    s11a_7b = run_layerwise_quant(model_7b, tok_7b, "qwen25_7b", num_layers_7b, num_samples=50)

    # 11b: 7B TriviaQA
    s11b = run_7b_triviaqa(model_7b, tok_7b, num_samples=50)

    del model_7b; torch.cuda.empty_cache(); gc.collect()

    # === Load 3B for layer-wise comparison ===
    os.environ['HF_HOME'] = '/workspace/.hf_home'  # 3B is on overlay
    logger.info("\nLoading Qwen2.5-3B (FP16, eager)...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model_3b.config.use_cache = True
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    if tok_3b.pad_token is None: tok_3b.pad_token = tok_3b.eos_token

    num_layers_3b = model_3b.config.num_hidden_layers

    # 11a: Layer-wise quantization sensitivity (3B)
    s11a_3b = run_layerwise_quant(model_3b, tok_3b, "qwen25_3b", num_layers_3b, num_samples=50)

    del model_3b; torch.cuda.empty_cache(); gc.collect()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 11 COMPLETE in {elapsed:.1f} minutes")
    logger.info(f"End: {datetime.now()}")
