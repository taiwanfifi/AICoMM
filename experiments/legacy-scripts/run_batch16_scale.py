#!/usr/bin/env python3
"""
Batch 16: Model Size Scaling (14B) + Non-Extractive Task (MMLU)

16a: Qwen2.5-14B on SQuAD v2 (50 samples)
  - Baseline, INT4/8, mixed-precision, Layer 0 sensitivity, Q2C/SnapKV/Random 50%
  - Key question: Does INT4 fragility increase? (3B=96%, 7B=77%, 14B=??)
  - Key question: Does Layer 0 bottleneck strengthen? (3B=87%, 7B=100%, 14B=??)

16b: Qwen2.5-7B on MMLU subset (50 questions from STEM)
  - Multiple-choice reasoning — fundamentally different from extractive QA
  - Tests: baseline accuracy, INT4/8, mixed-precision, Q2C 50%
  - Key question: Do compression findings generalize to non-extractive tasks?
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch16.log')])
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


def generate_layerwise_quant(model, tokenizer, input_ids, seq_len, layer_bits_map, max_new=64):
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


def load_mmlu_stem(num_samples):
    """Load MMLU STEM subset for reasoning validation."""
    from datasets import load_dataset
    # Use the standard MMLU dataset
    subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology',
                'college_chemistry', 'college_computer_science', 'college_mathematics',
                'college_physics', 'computer_security', 'electrical_engineering',
                'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
                'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
                'machine_learning']

    all_samples = []
    for subj in subjects:
        try:
            ds = load_dataset('cais/mmlu', subj, split='test')
            all_samples.extend(list(ds))
            if len(all_samples) >= num_samples * 2:
                break
        except Exception as e:
            logger.warning(f"Failed to load MMLU/{subj}: {e}")

    # Shuffle deterministically and select
    np.random.seed(42)
    np.random.shuffle(all_samples)
    return all_samples[:num_samples]


def run_14b_squad(num_samples=50):
    """16a: Qwen2.5-14B on SQuAD v2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"\n{'#'*80}\n16a: Qwen2.5-14B on SQuAD v2\n{'#'*80}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded Qwen2.5-14B: {num_layers} layers, "
                f"{model.config.num_key_value_heads} KV heads, "
                f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    exp_name = '14b_squad'
    samples = load_squad(num_samples)
    start_idx, results = load_checkpoint(exp_name)

    probe_layers = sorted(set([0, num_layers//6, num_layers//3, num_layers//2,
                               2*num_layers//3, num_layers-1]))
    logger.info(f"Probe layers: {probe_layers}")

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

        # 2. Quantization
        for bits in [4, 8]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits)
            result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 3. Mixed-precision
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16)
        result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 4. Layer-wise: only Layer 0 at INT4, and keep only L0 at FP16
        ans = generate_layerwise_quant(model, tokenizer, input_ids, seq_len, {0: 4})
        result['only_L0_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        ans = generate_except_layer_quant(model, tokenizer, input_ids, seq_len, keep_fp16_layer=0)
        result['except_L0_fp16'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # Test a mid-layer too
        mid_layer = num_layers // 2
        ans = generate_except_layer_quant(model, tokenizer, input_ids, seq_len, keep_fp16_layer=mid_layer)
        result[f'except_L{mid_layer}_fp16'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 5. Selection at 50%
        try:
            cs, cp, qp = compute_q2c_scores(model, tokenizer, input_ids, seq_len)
            if cs:
                sel_mask = make_selection_mask(seq_len, cs, cp, qp, 0.5)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                       selection_mask=sel_mask)
                result['q2c_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

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
                snap_mask = make_selection_mask(seq_len, snap_ctx, cp, qp, 0.5)
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
        int4 = result.get('int4', {}).get('f1', -1)
        mixed = result.get('mixed_L0fp16_int4', {}).get('f1', -1)
        only_l0 = result.get('only_L0_int4', {}).get('f1', -1)
        logger.info(f"  [{i+1}/{num_samples}] full={fp16:.3f} int4={int4:.3f} mixed={mixed:.3f} L0only={only_l0:.3f} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results), 'num_layers': num_layers, 'model': 'Qwen2.5-14B'}
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

    final_path = RESULTS_DIR / f'14b_squad_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump({'metadata': summary, 'results': results}, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    if ckpt_path.exists(): ckpt_path.unlink()

    del model; torch.cuda.empty_cache(); gc.collect()
    return summary


def run_7b_mmlu(num_samples=50):
    """16b: Qwen2.5-7B on MMLU STEM subset."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"\n{'#'*80}\n16b: Qwen2.5-7B on MMLU (reasoning task)\n{'#'*80}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    logger.info(f"Loaded Qwen2.5-7B: {num_layers} layers")

    exp_name = 'mmlu_7b'
    samples = load_mmlu_stem(num_samples)
    logger.info(f"Loaded {len(samples)} MMLU STEM samples")

    start_idx, results = load_checkpoint(exp_name)

    choices = ['A', 'B', 'C', 'D']

    for i, sample in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()

        question = sample['question']
        options = sample['choices']
        gold_idx = sample['answer']
        gold_letter = choices[gold_idx]

        prompt = f"Question: {question}\n"
        for j, opt in enumerate(options):
            prompt += f"{choices[j]}. {opt}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        result = {'idx': i, 'gold': gold_letter, 'seq_len': seq_len, 'question': question[:100]}

        # Generate with max_new=5 (just need the letter)
        for method_name, bits, layer0_bits in [
            ('full', 16, None),
            ('int8', 8, None),
            ('int4', 4, None),
            ('mixed_L0fp16_int4', 4, 16),
        ]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits,
                                    layer0_bits=layer0_bits, max_new=5)
            # Extract answer letter
            ans_clean = ans.strip().upper()
            correct = 1 if ans_clean and ans_clean[0] == gold_letter else 0
            result[method_name] = {'answer': ans[:50], 'correct': correct}

        # Q2C selection at 50% (for MMLU, "context" is the options, "question" is the stem)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
            attentions = out.attentions

            # Use last 10% of positions as "question" for Q2C scoring
            q_start = int(seq_len * 0.9)
            context_positions = list(range(0, q_start))
            question_positions = list(range(q_start, seq_len))

            scores = torch.zeros(seq_len, device='cuda')
            for layer_attn in attentions:
                for q_pos in question_positions:
                    scores += layer_attn[0, :, q_pos, :].mean(dim=0)

            context_scores = [(pos, scores[pos].item()) for pos in context_positions]
            context_scores.sort(key=lambda x: x[1], reverse=True)

            sel_mask = make_selection_mask(seq_len, context_scores, context_positions, question_positions, 0.5)
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16,
                                   selection_mask=sel_mask, max_new=5)
            ans_clean = ans.strip().upper()
            correct = 1 if ans_clean and ans_clean[0] == gold_letter else 0
            result['q2c_50'] = {'answer': ans[:50], 'correct': correct}
        except Exception as e:
            logger.warning(f"Selection failed for MMLU sample {i}: {e}")

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)
        save_checkpoint(exp_name, i, results)

        full_c = result.get('full', {}).get('correct', -1)
        int4_c = result.get('int4', {}).get('correct', -1)
        mixed_c = result.get('mixed_L0fp16_int4', {}).get('correct', -1)
        q2c_c = result.get('q2c_50', {}).get('correct', -1)
        logger.info(f"  [{i+1}/{num_samples}] gold={gold_letter} full={full_c} int4={int4_c} mixed={mixed_c} q2c={q2c_c} ({elapsed:.1f}s)")

    # Summary
    summary = {'num_samples': len(results), 'num_layers': num_layers, 'model': 'Qwen2.5-7B', 'task': 'MMLU-STEM'}

    fp16_acc = float(np.mean([r.get('full', {}).get('correct', 0) for r in results]))
    summary['full_acc'] = fp16_acc

    for key in ['full', 'int8', 'int4', 'mixed_L0fp16_int4', 'q2c_50']:
        vals = [r.get(key, {}).get('correct', 0) for r in results if key in r]
        if vals:
            acc = float(np.mean(vals))
            summary[f'{key}_acc'] = acc
            pct = acc / fp16_acc * 100 if fp16_acc > 0 else 0
            logger.info(f"  {key:25s}: Acc={acc:.4f} ({pct:5.1f}%)")

    final_path = RESULTS_DIR / f'mmlu_7b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
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

    # Run 14B SQuAD first (bigger model, main result)
    summary_14b = run_14b_squad(num_samples=50)

    # Then 7B MMLU (different task)
    summary_mmlu = run_7b_mmlu(num_samples=50)

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\n{'='*80}\nBatch 16 COMPLETE in {elapsed:.1f} minutes")
    logger.info(f"14B SQuAD: full={summary_14b['full_f1']:.4f}")
    logger.info(f"7B MMLU: full_acc={summary_mmlu['full_acc']:.4f}")
