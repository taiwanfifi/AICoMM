#!/usr/bin/env python3
"""
Batch 18: Controlled Context-Length Scaling (Needle-in-Haystack)

Uses SQuAD samples padded with distractor text from other SQuAD passages
to create controlled context lengths at 512, 1024, 2048, 4096 tokens.

Same question/answer pair across all lengths — ONLY the haystack size changes.
This isolates the effect of context length on compression quality.

Key questions:
1. How does F1 vs context length scale for each method?
2. Does Q2C advantage grow monotonically with context length?
3. Does INT4 fragility change with context length?
4. At what length does each method break down?

Output: Clean curves for paper figure.
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch18.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Target context lengths to test
TARGET_LENGTHS = [512, 1024, 2048, 4096]
NUM_SAMPLES = 30  # Per length (total = 30 × 4 = 120 runs)


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


def save_checkpoint(exp_name, state):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    with open(ckpt_path, 'w') as f:
        json.dump(state, f, default=str)


def load_checkpoint(exp_name):
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists():
        return json.load(open(ckpt_path))
    return None


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


def prepare_scaling_samples(tokenizer, num_samples=30):
    """Create controlled-length samples by padding SQuAD questions with distractor text.

    For each base sample, we create versions at 512, 1024, 2048, 4096 tokens
    by inserting distractor paragraphs BEFORE the relevant context.
    """
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]

    # Get distractor texts (other SQuAD contexts)
    all_contexts = [s['context'] for s in answerable]

    # Select base samples with SHORT context (< 200 tokens) so we have room to pad
    base_samples = []
    for s in answerable:
        prompt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        tok_len = len(tokenizer.encode(prompt))
        if 80 < tok_len < 200:  # Short enough to pad significantly
            s['base_tokens'] = tok_len
            base_samples.append(s)
        if len(base_samples) >= num_samples * 2:
            break

    np.random.seed(42)
    np.random.shuffle(base_samples)
    base_samples = base_samples[:num_samples]

    logger.info(f"Selected {len(base_samples)} base samples (avg {np.mean([s['base_tokens'] for s in base_samples]):.0f} tokens)")

    # For each target length, create padded versions
    scaling_samples = {}
    for target_len in TARGET_LENGTHS:
        scaling_samples[target_len] = []
        for i, s in enumerate(base_samples):
            gold = s['answers']['text'][0]
            question = s['question']
            relevant_context = s['context']

            # Build distractor text from other contexts
            distractor_text = ""
            distractor_idx = (i * 7 + 13) % len(all_contexts)  # Deterministic but varied

            # Calculate how much padding we need
            base_prompt = f"Context: {relevant_context}\nQuestion: {question}\nAnswer:"
            base_len = len(tokenizer.encode(base_prompt))
            padding_needed = target_len - base_len

            if padding_needed > 0:
                while len(tokenizer.encode(distractor_text)) < padding_needed:
                    ctx = all_contexts[distractor_idx % len(all_contexts)]
                    distractor_text += f" {ctx}"
                    distractor_idx += 1

                # Truncate distractor to exact padding needed
                dist_tokens = tokenizer.encode(distractor_text)
                if len(dist_tokens) > padding_needed:
                    distractor_text = tokenizer.decode(dist_tokens[:padding_needed], skip_special_tokens=True)

            # Place relevant context AFTER distractor (needle at the end of haystack)
            if distractor_text.strip():
                full_context = f"{distractor_text.strip()} {relevant_context}"
            else:
                full_context = relevant_context

            prompt = f"Context: {full_context}\nQuestion: {question}\nAnswer:"
            actual_len = len(tokenizer.encode(prompt))

            scaling_samples[target_len].append({
                'idx': i,
                'gold': gold,
                'question': question,
                'prompt': prompt,
                'target_len': target_len,
                'actual_len': actual_len,
                'base_len': base_len,
            })

        actual_lens = [s['actual_len'] for s in scaling_samples[target_len]]
        logger.info(f"  Target {target_len}: actual avg={np.mean(actual_lens):.0f}, "
                    f"min={min(actual_lens)}, max={max(actual_lens)}")

    return scaling_samples


def run_scaling(model_name, model_short, dtype, hf_home):
    """Run context-length scaling experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ['HF_HOME'] = hf_home
    os.environ['HF_DATASETS_CACHE'] = os.path.join(hf_home, 'datasets')

    logger.info(f"\n{'#'*80}\n{model_short} Context-Length Scaling\n{'#'*80}")

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
                f"{model.config.num_key_value_heads} KV heads")

    # Prepare samples
    scaling_samples = prepare_scaling_samples(tokenizer, NUM_SAMPLES)

    exp_name = f'scaling_{model_short}'
    ckpt = load_checkpoint(exp_name)
    if ckpt:
        all_results = ckpt.get('results', {})
        logger.info(f"[RESUME] from checkpoint")
    else:
        all_results = {}

    for target_len in TARGET_LENGTHS:
        if str(target_len) in all_results and len(all_results[str(target_len)]) >= NUM_SAMPLES:
            logger.info(f"[SKIP] {target_len} already complete ({len(all_results[str(target_len)])} samples)")
            continue

        logger.info(f"\n{'='*60}\nTarget length: {target_len} tokens\n{'='*60}")
        results_for_len = all_results.get(str(target_len), [])
        start_idx = len(results_for_len)

        samples = scaling_samples[target_len]

        for i, sample in enumerate(samples):
            if i < start_idx: continue
            t0 = time.time()

            gold = sample['gold']
            prompt = sample['prompt']

            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=target_len + 50,  # Small margin
                             truncation=True).to("cuda")
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1]

            result = {
                'idx': sample['idx'],
                'gold': gold,
                'seq_len': seq_len,
                'target_len': target_len,
                'base_len': sample['base_len'],
            }

            # 1. Full baseline
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
            result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

            # 2. Quantization
            for bits in [4, 8]:
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits)
                result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

            # 3. Mixed-precision
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16)
            result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

            # 4. Selection methods at 50%
            try:
                # Q2C
                cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
                if cs:
                    sel_mask = make_selection_mask(seq_len, cs, cp, qp, 0.5)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask)
                    result['q2c_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                    # Q2C at 25%
                    sel_mask_25 = make_selection_mask(seq_len, cs, cp, qp, 0.25)
                    ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                           selection_mask=sel_mask_25)
                    result['q2c_25'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                    # Combined: Q2C 50% + mixed
                    sel_mask_50 = make_selection_mask(seq_len, cs, cp, qp, 0.5)
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

                    # Random at 50%
                    np.random.seed(42 + i + target_len)
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
                logger.warning(f"Selection failed for len={target_len} sample {i}: {e}")
                result['selection_error'] = str(e)

            elapsed = time.time() - t0
            result['time'] = elapsed
            results_for_len.append(result)

            # Save checkpoint after each sample
            all_results[str(target_len)] = results_for_len
            save_checkpoint(exp_name, {'results': all_results})

            fp16 = result.get('full', {}).get('f1', -1)
            int4 = result.get('int4', {}).get('f1', -1)
            q2c50 = result.get('q2c_50', {}).get('f1', -1)
            snap50 = result.get('snapkv_50', {}).get('f1', -1)
            logger.info(f"  [len={target_len}][{i+1}/{NUM_SAMPLES}] seq={seq_len} "
                        f"full={fp16:.3f} int4={int4:.3f} q2c50={q2c50:.3f} "
                        f"snap50={snap50:.3f} ({elapsed:.1f}s)")

    # Final summary
    summary = {
        'model': model_short,
        'num_samples_per_length': NUM_SAMPLES,
        'target_lengths': TARGET_LENGTHS,
    }

    logger.info(f"\n{'='*80}\nSUMMARY: {model_short} Context-Length Scaling\n{'='*80}")
    logger.info(f"{'Method':<25s} | " + " | ".join(f"{tl:>6d}" for tl in TARGET_LENGTHS))
    logger.info("-" * 80)

    all_methods = set()
    for tl in TARGET_LENGTHS:
        for r in all_results.get(str(tl), []):
            for k in r:
                if isinstance(r[k], dict) and 'f1' in r[k]:
                    all_methods.add(k)

    for method in sorted(all_methods):
        row = []
        for tl in TARGET_LENGTHS:
            vals = [r.get(method, {}).get('f1', 0) for r in all_results.get(str(tl), []) if method in r]
            if vals:
                mean_f1 = float(np.mean(vals))
                full_vals = [r.get('full', {}).get('f1', 0) for r in all_results.get(str(tl), [])]
                full_mean = float(np.mean(full_vals)) if full_vals else 1.0
                pct = mean_f1 / full_mean * 100 if full_mean > 0 else 0
                row.append(f"{pct:5.1f}%")
                summary[f'{method}_{tl}_f1'] = mean_f1
                summary[f'{method}_{tl}_pct'] = pct
            else:
                row.append("  N/A ")
        logger.info(f"{method:<25s} | " + " | ".join(row))

    # Save full results
    final_path = RESULTS_DIR / f'scaling_{model_short}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': all_results}, f, indent=2, default=str)
    logger.info(f"\n[SAVED] -> {final_path}")

    # Clean checkpoint
    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    del model; torch.cuda.empty_cache(); gc.collect()
    return summary


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    # Run 7B scaling (BF16 for Blackwell)
    summary = run_scaling(
        model_name="Qwen/Qwen2.5-7B",
        model_short="Qwen2.5-7B",
        dtype=torch.bfloat16,
        hf_home='/dev/shm/hf_7b',
    )

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 18 COMPLETE in {elapsed:.1f} minutes")
