#!/usr/bin/env python3
"""
Batch 17: HotpotQA Multi-Hop QA (distractor setting)

Tests compression on multi-hop reasoning with longer contexts (10 paragraphs, ~500-1000 tokens).
Key questions:
1. Does Q2C still dominate when answers require synthesizing multiple scattered evidence pieces?
2. Does INT4 fragility change with longer contexts? (Batch 15 showed improvement)
3. Does multi-hop structure affect selection methods differently?

17a: Qwen2.5-7B on HotpotQA distractor (50 samples, max_length=1024)
17b: Qwen2.5-3B on same samples for cross-size comparison
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch17.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 2048  # Longer than SQuAD (512) to test scaling


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

    # Sum attention received from ALL positions (H2O = cumulative attention)
    scores = torch.zeros(seq_len, device='cuda')
    for layer_attn in attentions:
        scores += layer_attn[0].sum(dim=(0, 1))  # Sum over heads and query positions

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


def load_hotpotqa(num_samples, tokenizer=None):
    """Load HotpotQA distractor setting (10 paragraphs per sample, multi-hop).

    Filter for samples where the FULL PROMPT fits within MAX_LENGTH tokens,
    ensuring the question is never truncated.
    """
    from datasets import load_dataset
    ds = load_dataset('hotpot_qa', 'distractor', split='validation')

    # Filter for answerable, non-yes/no questions
    candidates = []
    for s in ds:
        if s['answer'] and s['answer'].strip().lower() not in ('yes', 'no'):
            # Concatenate all context paragraphs
            context_text = ""
            for title, sents in zip(s['context']['title'], s['context']['sentences']):
                context_text += f"[{title}] " + " ".join(sents) + " "
            s['full_context'] = context_text.strip()

            # Build full prompt to check token length
            prompt = f"Context: {s['full_context']}\nQuestion: {s['question']}\nAnswer:"
            if tokenizer:
                tok_len = len(tokenizer.encode(prompt))
            else:
                # Rough estimate: ~4 chars per token
                tok_len = len(prompt) // 4

            s['est_tokens'] = tok_len

            # Only keep samples that fit within MAX_LENGTH (with some margin for generation)
            if tok_len <= MAX_LENGTH - 10:
                candidates.append(s)

            if len(candidates) >= num_samples * 5:
                break

    logger.info(f"Found {len(candidates)} HotpotQA samples fitting in {MAX_LENGTH} tokens")

    if len(candidates) < num_samples:
        logger.warning(f"Only {len(candidates)} samples fit! Will use all of them.")
        return candidates

    # Prefer longer samples (more context = better test) but ensure they all fit
    candidates.sort(key=lambda x: x['est_tokens'], reverse=True)
    selected = candidates[:num_samples]
    np.random.seed(42)
    np.random.shuffle(selected)
    return selected


def format_hotpotqa_prompt(sample):
    """Format HotpotQA sample as a prompt."""
    context = sample['full_context']
    question = sample['question']
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def run_hotpotqa(model_name, model_short, dtype, hf_home, num_samples=50, preloaded_samples=None):
    """Run HotpotQA experiments for a given model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ['HF_HOME'] = hf_home
    os.environ['HF_DATASETS_CACHE'] = os.path.join(hf_home, 'datasets')

    logger.info(f"\n{'#'*80}\n{model_short} on HotpotQA (multi-hop QA)\n{'#'*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded {model_short}: {num_layers} layers, "
                f"{model.config.num_key_value_heads} KV heads, "
                f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    exp_name = f'hotpotqa_{model_short}'
    if preloaded_samples is not None:
        samples = preloaded_samples
    else:
        samples = load_hotpotqa(num_samples, tokenizer=tokenizer)
    logger.info(f"Loaded {len(samples)} HotpotQA samples")

    # Log context length stats
    ctx_lens = [len(s['full_context']) for s in samples]
    logger.info(f"Context lengths: min={min(ctx_lens)}, max={max(ctx_lens)}, "
                f"mean={np.mean(ctx_lens):.0f}, median={np.median(ctx_lens):.0f}")

    start_idx, results = load_checkpoint(exp_name)

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()
        gold = sample['answer']
        prompt = format_hotpotqa_prompt(sample)

        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len,
                  'question': sample['question'][:100],
                  'answer_type': sample.get('type', 'unknown'),
                  'context_chars': len(sample['full_context'])}

        # 1. Full baseline
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 2. Quantization: INT4, INT8
        for bits in [4, 8]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits)
            result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 3. Mixed-precision (L0 FP16 + rest INT4)
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16)
        result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 4. Selection methods at 50% and 25%
        try:
            # Q2C
            cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
            if cs:
                for ret in [0.5, 0.25]:
                    sel_mask = make_selection_mask(seq_len, cs, cp, qp, ret)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask)
                    result[f'q2c_{int(ret*100)}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # Combined: Q2C 50% + INT4, Q2C 50% + mixed
                sel_mask_50 = make_selection_mask(seq_len, cs, cp, qp, 0.5)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4,
                                       selection_mask=sel_mask_50)
                result['q2c_50_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16,
                                       selection_mask=sel_mask_50)
                result['q2c_50_mixed'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # SnapKV at 50% (use wider observation window for longer context)
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
                snap_scores = torch.zeros(seq_len, device='cuda')
                obs_window = min(64, seq_len // 4)  # Wider window for longer contexts
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
            result['selection_error'] = str(e)

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        fp16 = result.get('full', {}).get('f1', -1)
        int4 = result.get('int4', {}).get('f1', -1)
        q2c50 = result.get('q2c_50', {}).get('f1', -1)
        snap50 = result.get('snapkv_50', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] seq={seq_len} full={fp16:.3f} int4={int4:.3f} "
                    f"q2c50={q2c50:.3f} snap50={snap50:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {
        'num_samples': len(results),
        'num_layers': num_layers,
        'model': model_short,
        'task': 'HotpotQA-distractor',
        'max_length': MAX_LENGTH,
        'avg_seq_len': float(np.mean([r['seq_len'] for r in results])),
        'avg_context_chars': float(np.mean([r.get('context_chars', 0) for r in results])),
    }

    fp16_f1 = float(np.mean([r.get('full', {}).get('f1', 0) for r in results]))
    summary['full_f1'] = fp16_f1
    summary['full_std'] = float(np.std([r.get('full', {}).get('f1', 0) for r in results]))

    all_keys = set()
    for r in results:
        for k in r:
            if isinstance(r[k], dict) and 'f1' in r[k]:
                all_keys.add(k)

    for key in sorted(all_keys):
        vals = [r.get(key, {}).get('f1', 0) for r in results if key in r]
        if vals:
            f1 = float(np.mean(vals))
            summary[f'{key}_f1'] = f1
            summary[f'{key}_std'] = float(np.std(vals))
            pct = f1 / fp16_f1 * 100 if fp16_f1 > 0 else 0
            logger.info(f"  {key:25s}: F1={f1:.4f} ({pct:5.1f}%)")

    final_path = RESULTS_DIR / f'hotpotqa_{model_short}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    del model; torch.cuda.empty_cache(); gc.collect()
    return summary


if __name__ == '__main__':
    from transformers import AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    # Pre-load samples using 7B tokenizer (same Qwen family, same tokenizer)
    logger.info("Pre-loading HotpotQA samples...")
    os.environ['HF_HOME'] = '/dev/shm/hf_7b'
    os.environ['HF_DATASETS_CACHE'] = '/dev/shm/hf_7b/datasets'
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    shared_samples = load_hotpotqa(50, tokenizer=tok)
    del tok
    logger.info(f"Pre-loaded {len(shared_samples)} shared samples")

    # 17a: Qwen2.5-7B on HotpotQA (BF16 for Blackwell)
    summary_7b = run_hotpotqa(
        model_name="Qwen/Qwen2.5-7B",
        model_short="Qwen2.5-7B",
        dtype=torch.bfloat16,
        hf_home='/dev/shm/hf_7b',
        num_samples=50,
        preloaded_samples=shared_samples
    )

    # 17b: Qwen2.5-3B on HotpotQA (FP16 is fine for 3B)
    summary_3b = run_hotpotqa(
        model_name="Qwen/Qwen2.5-3B",
        model_short="Qwen2.5-3B",
        dtype=torch.float16,
        hf_home='/workspace/.hf_home',
        num_samples=50,
        preloaded_samples=shared_samples
    )

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 17 COMPLETE in {elapsed:.1f} minutes")
    logger.info(f"7B HotpotQA: full_f1={summary_7b['full_f1']:.4f}, avg_seq={summary_7b['avg_seq_len']:.0f}")
    logger.info(f"3B HotpotQA: full_f1={summary_3b['full_f1']:.4f}, avg_seq={summary_3b['avg_seq_len']:.0f}")
