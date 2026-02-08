#!/usr/bin/env python3
"""
Batch 5: Scaled experiments + new baselines + SVD F1 + second dataset.

Experiments:
1. Quantization + Selection F1 at 50 samples (3B) — scale up batch 4
2. SVD compression F1 at multiple ranks — completes Topic 06 comparison
3. H2O baseline — reviewer requirement
4. NaturalQuestions dataset — robustness check
5. Layer-heterogeneous compression — different quantization per layer (Topic 11)

Checkpointing: Every 5 samples, results saved to checkpoint.
"""
import os, sys, json, time, logging, copy
from pathlib import Path
from datetime import datetime
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch5.log')])
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


def manual_generate(model, tokenizer, past_kv, first_token_id, seq_len, max_new=64):
    """Generate tokens starting from first_token_id with pre-populated KV cache."""
    generated = [first_token_id]
    cur_len = seq_len
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        attn_mask = torch.ones(1, cur_len + 1, device='cuda', dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       attention_mask=attn_mask, position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def quantize_inplace_int8(pkv):
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            scale = t.abs().max() / 127.0
            if scale > 0:
                q = torch.clamp(torch.round(t / scale), -128, 127) * scale
                tensor.copy_(q.to(tensor.dtype))


def quantize_inplace_int4(pkv, group_size=32):
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            t = tensor.float()
            shape = t.shape
            flat = t.reshape(-1)
            pad = (group_size - flat.numel() % group_size) % group_size
            if pad > 0:
                flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
            flat = flat.reshape(-1, group_size)
            scale = flat.abs().max(dim=1, keepdim=True).values / 7.0
            scale = scale.clamp(min=1e-10)
            q = torch.clamp(torch.round(flat / scale), -8, 7)
            dq = (q * scale).reshape(-1)[:t.numel()].reshape(shape)
            tensor.copy_(dq.to(tensor.dtype))


def svd_compress_inplace(pkv, rank):
    """In-place SVD compression: keep top-`rank` singular values per layer KV tensor."""
    for layer in pkv.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            # tensor shape: (batch, heads, seq, head_dim)
            t = tensor.float()
            b, h, s, d = t.shape
            # Reshape to (b*h, s, d) for SVD
            t_2d = t.reshape(b * h, s, d)
            reconstructed = torch.zeros_like(t_2d)
            for i in range(b * h):
                try:
                    U, S, Vh = torch.linalg.svd(t_2d[i], full_matrices=False)
                    # Keep top-rank components
                    r = min(rank, S.shape[0])
                    reconstructed[i] = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
                except Exception:
                    reconstructed[i] = t_2d[i]  # fallback: no compression
            tensor.copy_(reconstructed.reshape(b, h, s, d).to(tensor.dtype))


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


def load_model(model_name, need_attentions=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = dict(dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    if need_attentions:
        kwargs['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_squad(num_samples):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in ds if len(s['answers']['text']) > 0]
    return answerable[:num_samples]


def load_nq(num_samples):
    """Load NaturalQuestions (validation split) — needs different format."""
    from datasets import load_dataset
    try:
        ds = load_dataset('google-research-datasets/natural_questions', 'default', split='validation')
        # NQ format is different — extract answerable samples
        samples = []
        for item in ds:
            # NQ has 'short_answers' nested structure
            if item.get('annotations') and len(item['annotations']['short_answers']) > 0:
                sa = item['annotations']['short_answers'][0]
                if sa.get('text'):
                    samples.append({
                        'context': item['document']['text'],
                        'question': item['question']['text'],
                        'answers': {'text': [sa['text']]}
                    })
                    if len(samples) >= num_samples:
                        break
        return samples
    except Exception as e:
        logger.warning(f"NQ loading failed: {e}. Trying TriviaQA instead.")
        try:
            ds = load_dataset('trivia_qa', 'rc', split='validation')
            samples = []
            for item in ds:
                if item['answer']['value']:
                    ctx = item['search_results']['search_context'][0] if item['search_results']['search_context'] else ''
                    if ctx:
                        samples.append({
                            'context': ctx[:2000],
                            'question': item['question'],
                            'answers': {'text': [item['answer']['value']]}
                        })
                        if len(samples) >= num_samples:
                            break
            return samples
        except Exception as e2:
            logger.warning(f"TriviaQA also failed: {e2}. Will skip second dataset.")
            return []


# ============================================================
# Exp 1: SVD F1 at multiple ranks
# ============================================================
def run_svd_f1(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """SVD compression F1 at ranks 8, 16, 32, 64."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'svd_f1_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_squad(num_samples)
    ranks = [8, 16, 32, 64]

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        t0 = time.time()
        gold = sample['answers']['text'][0]
        prompt = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        seq_len = inputs['input_ids'].shape[1]

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV + first token
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'full': {'answer': full_answer, 'f1': full_f1}}

        for rank in ranks:
            pkv = copy.deepcopy(out.past_key_values)
            svd_compress_inplace(pkv, rank)
            answer = manual_generate(model, tokenizer, pkv, first_tok, seq_len)
            f1 = compute_f1(answer, gold)
            result[f'svd_{rank}'] = {'answer': answer, 'f1': f1}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        parts = [f"[{i+1}/{num_samples}] full={full_f1:.3f}"]
        for r in ranks:
            parts.append(f"svd{r}={result[f'svd_{r}']['f1']:.3f}")
        logger.info(f"  {' '.join(parts)} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))
    for rank in ranks:
        k = f'svd_{rank}'
        vals = [r[k]['f1'] for r in results]
        summary[f'{k}_f1'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))

    logger.info(f"\n--- SVD F1 Summary ---")
    logger.info(f"  Full: {summary['full_f1']:.3f}")
    for r in ranks:
        logger.info(f"  SVD rank-{r}: {summary[f'svd_{r}_f1']:.3f} +/- {summary[f'svd_{r}_std']:.3f}")

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Exp 2: Scaled Selection + Quantization (50 samples)
# ============================================================
def run_selection_scaled(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """Scaled version of batch4 selection experiment with 50 samples."""
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'selection_scaled_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_squad(num_samples)

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
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        # Token ranges
        ctx_prefix = tokenizer("Context: ", add_special_tokens=True, return_tensors="pt")
        ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
        ctx_only = f"Context: {context}\nQuestion: "
        ctx_tokens = tokenizer(ctx_only, return_tensors="pt", max_length=512, truncation=True)
        context_end = ctx_tokens['input_ids'].shape[1]
        question_start = context_end
        answer_suffix = tokenizer("\nAnswer:", add_special_tokens=False)['input_ids']
        question_end = seq_len - len(answer_suffix)

        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        if num_context < 5:
            continue

        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Attention scores
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)

        attentions = out.attentions

        snapkv_scores = torch.zeros(seq_len, device='cuda')
        q2c_scores = torch.zeros(seq_len, device='cuda')
        for layer_attn in attentions:
            snapkv_scores += layer_attn[0].sum(dim=(0, 1))
            if question_end > question_start:
                q2c_scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))

        rng = np.random.RandomState(42 + i)
        random_scores = torch.tensor(rng.rand(seq_len), device='cuda', dtype=torch.float32)

        del attentions, out
        torch.cuda.empty_cache()

        result = {
            'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_context,
            'full': {'answer': full_answer, 'f1': full_f1},
        }

        for retention in retention_levels:
            k = max(1, int(num_context * retention))
            for method_name in methods:
                scores = {'snapkv': snapkv_scores, 'q2c': q2c_scores, 'random': random_scores}[method_name]
                ctx_scores = scores[torch.tensor(context_positions, device='cuda')]
                _, topk_idx = ctx_scores.topk(k)
                selected_ctx = set(context_positions[j] for j in topk_idx.cpu().numpy())

                attn_mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
                for pos in always_keep:
                    attn_mask[0, pos] = 1
                for pos in selected_ctx:
                    attn_mask[0, pos] = 1

                with torch.no_grad():
                    gen = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                          max_new_tokens=64, do_sample=False)
                answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
                f1 = compute_f1(answer, gold)
                key = f'{method_name}_{int(retention*100)}'
                result[key] = {'answer': answer, 'f1': f1}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        logger.info(f"  [{i+1}/{num_samples}] full={full_f1:.3f} "
                     f"q2c50={result['q2c_50']['f1']:.3f} snap50={result['snapkv_50']['f1']:.3f} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))
    summary['full_std'] = float(np.std([r['full']['f1'] for r in results]))

    logger.info(f"\n--- Selection F1 (Scaled) Summary ---")
    logger.info(f"  Full: {summary['full_f1']:.3f} +/- {summary['full_std']:.3f}")
    for ret in retention_levels:
        line = f"  {int(ret*100)}%: "
        for m in methods:
            key = f'{m}_{int(ret*100)}'
            vals = [r[key]['f1'] for r in results]
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
            line += f"{m}={summary[f'{key}_f1']:.3f}±{summary[f'{key}_std']:.3f}  "
        logger.info(line)

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Exp 3: H2O Baseline
# ============================================================
def run_h2o_baseline(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """H2O (Heavy Hitter Oracle) baseline — keeps tokens with highest cumulative attention
    from the most recent window. Different from SnapKV: H2O uses a sliding window +
    heavy hitters approach.

    H2O algorithm: Keep (1) recent tokens in a window, (2) heavy hitters (highest cumulative attention).
    For fair comparison with our methods, we implement simplified H2O:
    - heavy_ratio% of budget goes to heavy hitters (top cumulative attention in context)
    - window_ratio% goes to most recent context tokens
    """
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'h2o_baseline_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_squad(num_samples)

    retention_levels = [0.25, 0.50, 0.75]
    # H2O budget split: 50% heavy hitters, 50% recent window
    heavy_ratio = 0.5

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        t0 = time.time()
        gold = sample['answers']['text'][0]
        context = sample['context']
        question = sample['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        ctx_prefix = tokenizer("Context: ", add_special_tokens=True, return_tensors="pt")
        ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
        ctx_only = f"Context: {context}\nQuestion: "
        ctx_tokens = tokenizer(ctx_only, return_tensors="pt", max_length=512, truncation=True)
        context_end = ctx_tokens['input_ids'].shape[1]
        question_start = context_end

        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        if num_context < 5:
            continue

        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Attention scores for heavy hitters
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)

        # Cumulative attention (same as SnapKV but we split the budget)
        cum_attn = torch.zeros(seq_len, device='cuda')
        for layer_attn in out.attentions:
            cum_attn += layer_attn[0].sum(dim=(0, 1))

        del out
        torch.cuda.empty_cache()

        result = {
            'idx': i, 'gold': gold, 'seq_len': seq_len, 'num_context': num_context,
            'full': {'answer': full_answer, 'f1': full_f1},
        }

        for retention in retention_levels:
            budget = max(1, int(num_context * retention))
            n_heavy = max(1, int(budget * heavy_ratio))
            n_recent = budget - n_heavy

            # Heavy hitters: top cumulative attention in context
            ctx_scores = cum_attn[torch.tensor(context_positions, device='cuda')]
            _, heavy_idx = ctx_scores.topk(min(n_heavy, num_context))
            heavy_set = set(context_positions[j] for j in heavy_idx.cpu().numpy())

            # Recent window: last n_recent context positions
            recent_set = set(context_positions[-n_recent:]) if n_recent > 0 else set()

            selected = heavy_set | recent_set

            attn_mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
            for pos in always_keep:
                attn_mask[0, pos] = 1
            for pos in selected:
                attn_mask[0, pos] = 1

            with torch.no_grad():
                gen = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                      max_new_tokens=64, do_sample=False)
            answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
            f1 = compute_f1(answer, gold)
            result[f'h2o_{int(retention*100)}'] = {'answer': answer, 'f1': f1}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        logger.info(f"  [{i+1}/{num_samples}] full={full_f1:.3f} "
                     f"h2o25={result['h2o_25']['f1']:.3f} h2o50={result['h2o_50']['f1']:.3f} "
                     f"h2o75={result['h2o_75']['f1']:.3f} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))
    logger.info(f"\n--- H2O Baseline Summary ---")
    logger.info(f"  Full: {summary['full_f1']:.3f}")
    for ret in retention_levels:
        key = f'h2o_{int(ret*100)}'
        vals = [r[key]['f1'] for r in results]
        summary[f'{key}_f1'] = float(np.mean(vals))
        summary[f'{key}_std'] = float(np.std(vals))
        logger.info(f"  H2O {int(ret*100)}%: {summary[f'{key}_f1']:.3f} +/- {summary[f'{key}_std']:.3f}")

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


# ============================================================
# Exp 4: Quantization + SVD Combined (Pareto frontier data)
# ============================================================
def run_pareto_frontier(model_name="Qwen/Qwen2.5-3B", num_samples=50):
    """Generate data for Pareto frontier: compression ratio vs F1.

    Methods at various compression levels:
    - Selection: Q2C at 25/50/75%
    - Quantization: INT8 (2x), INT4 (4x)
    - SVD: rank 8/16/32/64
    - Combined: Q2C-50% + INT8, Q2C-50% + INT4
    """
    tag = model_name.split("/")[-1].replace(".", "").lower()
    exp_name = f'pareto_{tag}'
    logger.info(f"\n{'='*60}\nExp: {exp_name} ({num_samples} samples)\n{'='*60}")

    model, tokenizer = load_model(model_name, need_attentions=True)
    samples = load_squad(num_samples)

    ckpt_path = CKPT_DIR / f'{exp_name}_ckpt.json'
    start_idx, results = load_checkpoint(ckpt_path)

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        t0 = time.time()
        gold = sample['answers']['text'][0]
        context = sample['context']
        question = sample['question']
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        # Full baseline
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        full_answer = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
        full_f1 = compute_f1(full_answer, gold)

        # Get KV + first token + attention scores
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=True)
        first_tok = out.logits[:, -1, :].argmax(dim=-1).item()

        # Token ranges for Q2C
        ctx_prefix = tokenizer("Context: ", add_special_tokens=True, return_tensors="pt")
        ctx_prefix_len = ctx_prefix['input_ids'].shape[1]
        ctx_only = f"Context: {context}\nQuestion: "
        ctx_tokens = tokenizer(ctx_only, return_tensors="pt", max_length=512, truncation=True)
        context_end = ctx_tokens['input_ids'].shape[1]
        question_start = context_end
        answer_suffix = tokenizer("\nAnswer:", add_special_tokens=False)['input_ids']
        question_end = seq_len - len(answer_suffix)
        context_positions = list(range(ctx_prefix_len, context_end))
        num_context = len(context_positions)
        always_keep = set(range(ctx_prefix_len)) | set(range(question_start, seq_len))

        # Q2C scores
        q2c_scores = torch.zeros(seq_len, device='cuda')
        if question_end > question_start:
            for layer_attn in out.attentions:
                q2c_scores += layer_attn[0, :, question_start:question_end, :].sum(dim=(0, 1))

        del out.attentions
        torch.cuda.empty_cache()

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'full': {'answer': full_answer, 'f1': full_f1}}

        # --- Quantization only ---
        for qtype, qfn in [('int8', quantize_inplace_int8), ('int4', quantize_inplace_int4)]:
            pkv = copy.deepcopy(out.past_key_values)
            qfn(pkv)
            ans = manual_generate(model, tokenizer, pkv, first_tok, seq_len)
            result[qtype] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        # --- SVD only ---
        for rank in [8, 16, 32, 64]:
            pkv = copy.deepcopy(out.past_key_values)
            svd_compress_inplace(pkv, rank)
            ans = manual_generate(model, tokenizer, pkv, first_tok, seq_len)
            result[f'svd_{rank}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        # --- Q2C Selection only (using attention mask) ---
        if num_context >= 5:
            for retention in [0.25, 0.50, 0.75]:
                k = max(1, int(num_context * retention))
                ctx_sc = q2c_scores[torch.tensor(context_positions, device='cuda')]
                _, topk = ctx_sc.topk(k)
                selected = set(context_positions[j] for j in topk.cpu().numpy())
                mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
                for p in always_keep: mask[0, p] = 1
                for p in selected: mask[0, p] = 1
                with torch.no_grad():
                    gen = model.generate(input_ids=input_ids, attention_mask=mask,
                                          max_new_tokens=64, do_sample=False)
                ans = tokenizer.decode(gen[0][seq_len:], skip_special_tokens=True).strip()
                result[f'q2c_{int(retention*100)}'] = {'answer': ans, 'f1': compute_f1(ans, gold)}

        elapsed = time.time() - t0
        result['time'] = elapsed
        results.append(result)

        logger.info(f"  [{i+1}/{num_samples}] full={full_f1:.3f} int8={result['int8']['f1']:.3f} "
                     f"int4={result['int4']['f1']:.3f} svd32={result['svd_32']['f1']:.3f} "
                     f"q2c50={result.get('q2c_50', {}).get('f1', 'N/A')} ({elapsed:.1f}s)")

        if (i + 1) % 5 == 0:
            save_checkpoint(ckpt_path, i, results)

        del out; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()

    summary = {'model': model_name, 'num_samples': len(results)}
    summary['full_f1'] = float(np.mean([r['full']['f1'] for r in results]))

    logger.info(f"\n--- Pareto Frontier Summary ---")
    logger.info(f"  Full: {summary['full_f1']:.3f}")
    for key in ['int8', 'int4', 'svd_8', 'svd_16', 'svd_32', 'svd_64', 'q2c_25', 'q2c_50', 'q2c_75']:
        vals = [r[key]['f1'] for r in results if key in r]
        if vals:
            summary[f'{key}_f1'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
            logger.info(f"  {key}: {summary[f'{key}_f1']:.3f} +/- {summary[f'{key}_std']:.3f}")

    save_results(exp_name, results, summary)
    if ckpt_path.exists(): ckpt_path.unlink()
    return summary


if __name__ == '__main__':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch: {torch.__version__}")
    start = time.time()

    # Exp 1: SVD F1 (50 samples) — completes Topic 06
    svd = run_svd_f1("Qwen/Qwen2.5-3B", num_samples=50)

    # Exp 2: Scaled selection (50 samples) — strengthens Topic 01
    sel = run_selection_scaled("Qwen/Qwen2.5-3B", num_samples=50)

    # Exp 3: H2O baseline (50 samples) — reviewer requirement
    h2o = run_h2o_baseline("Qwen/Qwen2.5-3B", num_samples=50)

    # Exp 4: Pareto frontier data (50 samples) — key paper figure
    pareto = run_pareto_frontier("Qwen/Qwen2.5-3B", num_samples=50)

    elapsed = (time.time() - start) / 60
    logger.info(f"\nALL BATCH 5 DONE in {elapsed:.1f} minutes")
