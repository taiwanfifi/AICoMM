#!/usr/bin/env python3
"""
Batch 24: Yi-1.5-6B-Chat Multi-Task Validation

Yi-6B is our "INT4-robust" model (103% on SQuAD, 97.7%+ at all context lengths).
But we only have SQuAD data for it. For the paper's cross-architecture × cross-task matrix,
we need Yi on TriviaQA, HotpotQA, and MMLU.

Key question: Is Yi-6B INT4-robust across ALL tasks, or only on SQuAD?
If yes → strengthens "model-specific fragility" finding
If no → reveals task-model interaction effects

Also: Test delta encoding on Yi-6B to see if the CacheGen counter-finding is universal.
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
    handlers=[logging.StreamHandler(), logging.FileHandler('batch24.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = '01-ai/Yi-1.5-6B-Chat'
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


def quantize_pertoken(t, bits):
    if bits >= 16: return t
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return (t / scale).round().clamp(qmin, qmax) * scale


def anchor_delta_encode(t, group_size=10):
    B, H, S, D = t.shape
    deltas = torch.zeros_like(t)
    for start in range(0, S, group_size):
        end = min(start + group_size, S)
        anchor = t[:, :, start:start+1, :]
        deltas[:, :, start, :] = anchor.squeeze(2)
        if end > start + 1:
            deltas[:, :, start+1:end, :] = t[:, :, start+1:end, :] - anchor
    return deltas

def anchor_delta_decode(deltas, group_size=10):
    B, H, S, D = deltas.shape
    t = torch.zeros_like(deltas)
    for start in range(0, S, group_size):
        end = min(start + group_size, S)
        anchor = deltas[:, :, start:start+1, :]
        t[:, :, start, :] = anchor.squeeze(2)
        if end > start + 1:
            t[:, :, start+1:end, :] = anchor + deltas[:, :, start+1:end, :]
    return t


def manual_generate(model, tokenizer, past_kv, first_token_id, seq_len, max_new=64):
    generated = [first_token_id]
    cur_len = seq_len
    for step in range(max_new - 1):
        next_input = torch.tensor([[generated[-1]]], device='cuda')
        position_ids = torch.tensor([[cur_len]], device='cuda')
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past_kv,
                       position_ids=position_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        cur_len += 1
        if next_tok == tokenizer.eos_token_id: break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_with_compression(model, tokenizer, input_ids, seq_len,
                               method='none', bits=4, max_new=64):
    """Methods: none, quant, mixed_quant, anchor_delta_quant"""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)

    first_token_id = out.logits[:, -1, :].argmax(dim=-1).item()
    pkv = out.past_key_values

    for li in range(len(pkv.layers)):
        layer = pkv.layers[li]

        if method == 'none':
            pass
        elif method == 'quant':
            layer.keys.copy_(quantize_pertoken(layer.keys, bits))
            layer.values.copy_(quantize_pertoken(layer.values, bits))
        elif method == 'mixed_quant':
            if li > 0:
                layer.keys.copy_(quantize_pertoken(layer.keys, bits))
                layer.values.copy_(quantize_pertoken(layer.values, bits))
        elif method == 'anchor_delta_quant':
            deltas_k = anchor_delta_encode(layer.keys, 10)
            q_deltas_k = quantize_pertoken(deltas_k, bits)
            layer.keys.copy_(anchor_delta_decode(q_deltas_k, 10))
            deltas_v = anchor_delta_encode(layer.values, 10)
            q_deltas_v = quantize_pertoken(deltas_v, bits)
            layer.values.copy_(anchor_delta_decode(q_deltas_v, 10))

    return manual_generate(model, tokenizer, pkv, first_token_id, seq_len, max_new)


def format_yi_prompt(context, question):
    """Yi-1.5-6B-Chat uses ChatML format."""
    return f"<|im_start|>user\nContext: {context}\nQuestion: {question}\nAnswer briefly.<|im_end|>\n<|im_start|>assistant\n"


def prepare_squad(tokenizer, num_samples=30):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    samples = []
    for s in ds:
        if not s['answers']['text']: continue
        prompt = format_yi_prompt(s['context'], s['question'])
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 400:
            samples.append({'prompt': prompt, 'gold': s['answers']['text'][0],
                          'task': 'squad', 'tok_len': tok_len})
        if len(samples) >= num_samples * 3: break
    np.random.seed(42)
    idx = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in idx]


def prepare_triviaqa(tokenizer, num_samples=30):
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc', split='validation')
    samples = []
    for s in ds:
        if not s['answer']['aliases']: continue
        ctx = s['search_results']['search_context'][0][:1000] if s['search_results']['search_context'] else ""
        if not ctx: continue
        prompt = format_yi_prompt(ctx, s['question'])
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 400:
            samples.append({'prompt': prompt, 'gold': s['answer']['aliases'][0],
                          'task': 'triviaqa', 'tok_len': tok_len})
        if len(samples) >= num_samples * 3: break
    np.random.seed(42)
    idx = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in idx]


def prepare_hotpotqa(tokenizer, num_samples=30):
    from datasets import load_dataset
    ds = load_dataset('hotpot_qa', 'fullwiki', split='validation')
    samples = []
    for s in ds:
        sents = []
        for title, sent_list in zip(s['context']['title'], s['context']['sentences']):
            sents.append(f"{title}: {' '.join(sent_list)}")
        ctx = ' '.join(sents)[:1500]
        prompt = format_yi_prompt(ctx, s['question'])
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 500:
            samples.append({'prompt': prompt, 'gold': s['answer'],
                          'task': 'hotpotqa', 'tok_len': tok_len})
        if len(samples) >= num_samples * 3: break
    np.random.seed(42)
    idx = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in idx]


def prepare_mmlu(tokenizer, num_samples=30):
    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'all', split='test')
    samples = []
    for s in ds:
        choices = ['A', 'B', 'C', 'D']
        q = s['question']
        opts = '\n'.join([f"{c}. {s['choices'][i]}" for i, c in enumerate(choices)])
        prompt = f"<|im_start|>user\n{q}\n{opts}\nAnswer with just the letter.<|im_end|>\n<|im_start|>assistant\n"
        tok_len = len(tokenizer.encode(prompt))
        if tok_len <= 300:
            samples.append({'prompt': prompt, 'gold': choices[s['answer']],
                          'task': 'mmlu', 'tok_len': tok_len})
        if len(samples) >= num_samples * 3: break
    np.random.seed(42)
    idx = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in idx]


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

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

    methods = [
        ('none', 'none', 16),
        ('int8', 'quant', 8),
        ('int4', 'quant', 4),
        ('mixed_int4', 'mixed_quant', 4),
        ('anchor_delta_int4', 'anchor_delta_quant', 4),
        ('anchor_delta_int8', 'anchor_delta_quant', 8),
    ]

    all_task_results = {}

    for task_name, prepare_fn in [
        ('squad', prepare_squad),
        ('triviaqa', prepare_triviaqa),
        ('hotpotqa', prepare_hotpotqa),
        ('mmlu', prepare_mmlu),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK: {task_name}")
        logger.info(f"{'='*60}")

        samples = prepare_fn(tokenizer, NUM_SAMPLES)
        logger.info(f"Prepared {len(samples)} {task_name} samples")

        task_results = []
        for i, s in enumerate(samples):
            t0 = time.time()
            inputs = tokenizer(s['prompt'], return_tensors='pt', truncation=True, max_length=512).to('cuda')
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1]
            gold = s['gold']

            result = {'idx': i, 'gold': gold, 'seq_len': seq_len, 'task': task_name}

            for name, method, bits in methods:
                ans = generate_with_compression(model, tokenizer, input_ids, seq_len, method, bits)
                f1 = compute_f1(ans, gold)
                result[name] = {'answer': ans[:200], 'f1': f1}

            elapsed = time.time() - t0
            result['time'] = elapsed
            task_results.append(result)

            fp16 = result['none']['f1']
            i4 = result['int4']['f1']
            ad4 = result['anchor_delta_int4']['f1']
            logger.info(f"  [{i+1}/{len(samples)}] fp16={fp16:.3f} int4={i4:.3f} delta4={ad4:.3f} ({elapsed:.1f}s)")

        all_task_results[task_name] = task_results

        # Task summary
        baseline = float(np.mean([r['none']['f1'] for r in task_results]))
        for name, _, _ in methods:
            mean_f1 = float(np.mean([r[name]['f1'] for r in task_results]))
            pct = mean_f1 / baseline * 100 if baseline > 0 else 0
            logger.info(f"  {task_name} {name:25s}: {mean_f1:.4f} ({pct:.1f}%)")

    # Cross-task summary
    logger.info(f"\n{'#'*70}")
    logger.info(f"CROSS-TASK SUMMARY — Yi-1.5-6B-Chat")
    logger.info(f"{'#'*70}")

    for name, _, _ in methods:
        row = []
        for task_name in ['squad', 'triviaqa', 'hotpotqa', 'mmlu']:
            results = all_task_results[task_name]
            baseline = float(np.mean([r['none']['f1'] for r in results]))
            mean_f1 = float(np.mean([r[name]['f1'] for r in results]))
            pct = mean_f1 / baseline * 100 if baseline > 0 else 0
            row.append(f"{pct:.1f}%")
        logger.info(f"  {name:25s}: SQuAD={row[0]} TriviaQA={row[1]} HotpotQA={row[2]} MMLU={row[3]}")

    # Save
    combined = {
        'metadata': {
            'model': 'Yi-1.5-6B-Chat',
            'experiment': 'multi_task_validation',
            'tasks': list(all_task_results.keys()),
            'num_samples_per_task': NUM_SAMPLES,
        },
        'results': all_task_results,
    }
    final_path = RESULTS_DIR / f'yi_multitask_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"\n[SAVED] -> {final_path}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"Batch 24 COMPLETE in {elapsed:.1f} minutes")

    del model; torch.cuda.empty_cache(); gc.collect()
