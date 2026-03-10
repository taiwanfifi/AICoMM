#!/usr/bin/env python3
"""
Batch 19: Yi-1.5-6B-Chat — Cross-Family 4-KV-Head Validation

CRITICAL HYPOTHESIS TEST: Yi-1.5-6B-Chat has EXACTLY the same GQA config as Qwen2.5-7B:
  - 32 attention heads, 4 KV heads, head_dim=128, 32 layers
  - Different model family (01-AI vs Alibaba)

If Yi shows INT4 fragility similar to Qwen-7B (~77%), the KV head count hypothesis
is CONFIRMED cross-family. If Yi is robust like Mistral-7B (~99%), it's a Qwen-specific
artifact and the head count hypothesis is REFUTED.

Also includes layer-wise INT4 analysis to check if Layer 0 bottleneck appears.

Tests:
  1. Full baseline (BF16)
  2. INT4, INT8 (per-token quantization)
  3. Layer-wise INT4 (damage map — which layers are fragile?)
  4. only_L0_int4 (only Layer 0 at INT4, rest FP16)
  5. Mixed-precision (L0 FP16 + rest INT4)
  6. Q2C 50%, 25%
  7. SnapKV 50%
  8. H2O 50%
  9. Random 50%
  10. Combined: Q2C 50% + INT4, Q2C 50% + mixed
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch19.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
                       max_new=64, selection_mask=None, layer_only=None):
    """Generate with optional quantization.

    Args:
        layer_only: If set, only quantize THIS layer to `bits`, rest stay FP16.
                    Used for per-layer damage mapping.
    """
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
        if layer_only is not None:
            b = bits if li == layer_only else 16
        else:
            b = layer0_bits if li == 0 else bits

        if b < 16:
            layer = pkv.layers[li]
            layer.keys.copy_(quantize_pertoken(layer.keys, b))
            layer.values.copy_(quantize_pertoken(layer.values, b))

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


def compute_h2o_scores(model, tokenizer, input_ids, seq_len):
    """H2O: Heavy Hitter Oracle — sum all attention received by each position."""
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
        scores += layer_attn[0].sum(dim=(0, 1))

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
    candidates = [s for s in ds if s['answers']['text']]
    np.random.seed(42)
    indices = np.random.choice(len(candidates), min(num_samples, len(candidates)), replace=False)
    return [candidates[i] for i in indices]


def format_squad_prompt(sample):
    context = sample['context']
    question = sample['question']
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def run_yi_squad(num_samples=50):
    """Run full battery on Yi-1.5-6B-Chat with SQuAD v2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"\n{'#'*80}\nYi-1.5-6B-Chat on SQuAD v2 (Cross-Family 4-KV-Head Test)\n{'#'*80}")

    # Download and load model
    logger.info(f"Loading {MODEL_NAME} (will download if needed)...")
    t_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    logger.info(f"Loaded {MODEL_SHORT}: {num_layers} layers, {num_kv_heads} KV heads, "
                f"head_dim={head_dim}, load_time={time.time()-t_load:.1f}s")

    # Load SQuAD samples
    samples = load_squad(num_samples)
    logger.info(f"Loaded {len(samples)} SQuAD samples")

    exp_name = f'yi6b_squad'
    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = format_squad_prompt(sample)

        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len,
                  'question': sample['question'][:100]}

        # === QUANTIZATION EXPERIMENTS ===

        # 1. Full baseline (BF16)
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 2. INT8, INT4
        for bits in [8, 4]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits)
            result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 3. Layer-wise INT4 damage map (every 4th layer for speed, plus Layer 0)
        for li in [0, 4, 8, 12, 16, 20, 24, 28, 31]:
            if li >= num_layers: continue
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer_only=li)
            result[f'only_L{li}_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 4. Mixed-precision (L0 FP16 + rest INT4)
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16)
        result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # === SELECTION EXPERIMENTS ===
        try:
            # Q2C scores
            cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
            if cs:
                # Q2C 50%, 25%
                for ret in [0.5, 0.25]:
                    sel_mask = make_selection_mask(seq_len, cs, cp, qp, ret)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask)
                    result[f'q2c_{int(ret*100)}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # Combined: Q2C 50% + INT4
                sel_mask_50 = make_selection_mask(seq_len, cs, cp, qp, 0.5)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4,
                                       selection_mask=sel_mask_50)
                result['q2c_50_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # Combined: Q2C 50% + mixed (L0 FP16 + rest INT4)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16,
                                       selection_mask=sel_mask_50)
                result['q2c_50_mixed'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # SnapKV at 50%
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

                # H2O at 50%
                h2o_cs, _, _ = compute_h2o_scores(model, tokenizer, input_ids, seq_len)
                if h2o_cs:
                    h2o_mask = make_selection_mask(seq_len, h2o_cs, cp, qp, 0.5)
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
            import traceback; traceback.print_exc()
            result['selection_error'] = str(e)

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        fp16 = result.get('full', {}).get('f1', -1)
        int4 = result.get('int4', {}).get('f1', -1)
        int8 = result.get('int8', {}).get('f1', -1)
        mixed = result.get('mixed_L0fp16_int4', {}).get('f1', -1)
        q2c50 = result.get('q2c_50', {}).get('f1', -1)
        snap50 = result.get('snapkv_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] seq={seq_len} full={fp16:.3f} int8={int8:.3f} "
                    f"int4={int4:.3f} mixed={mixed:.3f} q2c50={q2c50:.3f} snap50={snap50:.3f} ({elapsed:.1f}s)")

    # === SUMMARY ===
    summary = {
        'num_samples': len(results),
        'num_layers': num_layers,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'model': MODEL_SHORT,
        'task': 'SQuAD-v2',
        'max_length': MAX_LENGTH,
        'avg_seq_len': float(np.mean([r['seq_len'] for r in results])),
    }

    fp16_f1 = float(np.mean([r.get('full', {}).get('f1', 0) for r in results]))
    summary['full_f1'] = fp16_f1
    summary['full_std'] = float(np.std([r.get('full', {}).get('f1', 0) for r in results]))

    all_keys = set()
    for r in results:
        for k in r:
            if isinstance(r[k], dict) and 'f1' in r[k]:
                all_keys.add(k)

    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: {MODEL_SHORT} on SQuAD v2 ({len(results)} samples)")
    logger.info(f"Baseline F1 = {fp16_f1:.4f}")
    logger.info(f"{'='*80}")

    for key in sorted(all_keys):
        vals = [r.get(key, {}).get('f1', 0) for r in results if key in r]
        if vals:
            f1 = float(np.mean(vals))
            summary[f'{key}_f1'] = f1
            summary[f'{key}_std'] = float(np.std(vals))
            pct = f1 / fp16_f1 * 100 if fp16_f1 > 0 else 0
            logger.info(f"  {key:25s}: F1={f1:.4f} ({pct:5.1f}%)")

    final_path = RESULTS_DIR / f'yi6b_squad_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    del model; torch.cuda.empty_cache(); gc.collect()
    return summary


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    summary = run_yi_squad(num_samples=NUM_SAMPLES)

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}")
    logger.info(f"Batch 19 COMPLETE in {elapsed:.1f} minutes")
    logger.info(f"Yi-6B SQuAD: full_f1={summary['full_f1']:.4f}")
    logger.info(f"INT4 = {summary.get('int4_f1', 0)/summary['full_f1']*100:.1f}% of baseline")
    logger.info(f"INT8 = {summary.get('int8_f1', 0)/summary['full_f1']*100:.1f}% of baseline")
    logger.info(f"Mixed = {summary.get('mixed_L0fp16_int4_f1', 0)/summary['full_f1']*100:.1f}% of baseline")
    logger.info(f"{'='*80}")
