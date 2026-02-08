#!/usr/bin/env python3
"""
Batch 21: Phi-3.5-mini-instruct (3.8B, 8 KV heads, head_dim=96)
7th model family. Microsoft architecture with DIFFERENT head_dim (96 vs 128).
Tests whether head_dim affects quantization robustness.

Full battery: baseline, INT4/8, layer-wise INT4, mixed-precision, selection methods.
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

# Monkey-patch DynamicCache for Phi-3.5 custom code compatibility (transformers 5.x)
from transformers import DynamicCache
if not hasattr(DynamicCache, 'get_usable_length'):
    def _get_usable_length(self, new_seq_length, layer_idx=None):
        if layer_idx is None:
            layer_idx = 0
        if not self.layers or layer_idx >= len(self.layers):
            return 0
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length

if not hasattr(DynamicCache, 'to_legacy_cache'):
    def _to_legacy_cache(self):
        legacy = []
        for layer in self.layers:
            legacy.append((layer.keys, layer.values))
        return tuple(legacy)
    DynamicCache.to_legacy_cache = _to_legacy_cache

if not hasattr(DynamicCache, 'from_legacy_cache'):
    @classmethod
    def _from_legacy_cache(cls, past_key_values):
        cache = cls()
        if past_key_values is not None:
            for key, value in past_key_values:
                cache.update(key, value, cache.get_seq_length() if cache.layers else 0)
        return cache
    DynamicCache.from_legacy_cache = _from_legacy_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch21.log')])
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'microsoft/Phi-3.5-mini-instruct'
MODEL_SHORT = 'Phi-3.5-mini-instruct'
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

    if selection_mask is not None:
        mask = selection_mask.clone()
        return manual_generate_with_mask(model, tokenizer, pkv, first_token_id, seq_len, mask, max_new)
    return manual_generate(model, tokenizer, pkv, first_token_id, seq_len, max_new)


def compute_q2c_scores(model, tokenizer, input_ids, seq_len, q_start, q_end):
    """Question-to-context attention scores for selection."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions  # tuple of (batch, heads, seq, seq)
    # Average attention from question tokens to context tokens across all layers/heads
    scores = torch.zeros(seq_len, device='cuda')
    n_layers = len(attentions)
    for layer_attn in attentions:
        # layer_attn: (1, n_heads, seq, seq)
        # Question tokens attending to all positions
        q_attn = layer_attn[0, :, q_start:q_end, :seq_len]  # (heads, q_len, seq)
        q_attn_avg = q_attn.mean(dim=(0, 1))  # (seq,)
        scores += q_attn_avg
    scores /= n_layers
    return scores.cpu().numpy()


def compute_snapkv_scores(model, tokenizer, input_ids, seq_len):
    """SnapKV: attention from last few tokens (recency-based)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions
    scores = torch.zeros(seq_len, device='cuda')
    n_layers = len(attentions)
    window = min(32, seq_len // 4)
    for layer_attn in attentions:
        recent_attn = layer_attn[0, :, -window:, :seq_len]
        scores += recent_attn.mean(dim=(0, 1))
    scores /= n_layers
    return scores.cpu().numpy()


def compute_h2o_scores(model, tokenizer, input_ids, seq_len):
    """H2O: cumulative attention scores."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attentions = out.attentions
    scores = torch.zeros(seq_len, device='cuda')
    n_layers = len(attentions)
    for layer_attn in attentions:
        cumul = layer_attn[0, :, :seq_len, :seq_len].sum(dim=1)  # (heads, seq)
        scores += cumul.mean(dim=0)
    scores /= n_layers
    return scores.cpu().numpy()


def build_selection_mask(scores, seq_len, retention, always_keep):
    """Build attention mask keeping top-scoring positions + always_keep positions."""
    n_keep = max(1, int(seq_len * retention))
    mask = torch.zeros(1, seq_len, device='cuda', dtype=torch.long)
    # Always keep special positions
    for pos in always_keep:
        if pos < seq_len:
            mask[0, pos] = 1
    # Select top positions from remaining
    selectable = np.ones(seq_len, dtype=bool)
    for pos in always_keep:
        if pos < seq_len:
            selectable[pos] = False
    selectable_indices = np.where(selectable)[0]
    selectable_scores = scores[selectable]
    n_select = max(0, n_keep - len(always_keep))
    if n_select > 0 and len(selectable_scores) > 0:
        top_idx = np.argsort(selectable_scores)[-n_select:]
        for idx in top_idx:
            mask[0, selectable_indices[idx]] = 1
    return mask


def find_boundaries_phi(tokenizer, input_ids, seq_len):
    """Find question and answer boundaries for Phi-3.5 instruct format."""
    full_text = tokenizer.decode(input_ids[0])

    # Phi-3.5 uses <|user|> ... <|end|>\n<|assistant|> format
    q_text_pos = full_text.find("Question:")
    if q_text_pos == -1:
        q_text_pos = full_text.find("<|user|>")

    a_text_pos = full_text.rfind("<|assistant|>")
    if a_text_pos == -1:
        a_text_pos = full_text.rfind("<|end|>")

    if q_text_pos == -1 or a_text_pos == -1:
        # Fallback: last 20% is question
        q_start = int(seq_len * 0.8)
        a_start = seq_len
        return q_start, a_start

    # Map text positions to token positions
    q_start = None
    a_start = None
    for i in range(seq_len):
        decoded = tokenizer.decode(input_ids[0][:i+1])
        if len(decoded) >= q_text_pos and q_start is None:
            q_start = i
        if len(decoded) >= a_text_pos and a_start is None and q_start is not None:
            a_start = i
            break

    if q_start is None: q_start = int(seq_len * 0.8)
    if a_start is None: a_start = seq_len

    return q_start, a_start


def format_phi_prompt(sample):
    """Format SQuAD sample for Phi-3.5-mini-instruct."""
    context = sample['context']
    question = sample['question']
    prompt = (
        f"<|user|>\n"
        f"Answer the question using ONLY words from the context. "
        f"Give the shortest possible answer - just the exact words from the context, nothing else.\n\n"
        f"Context: {context}\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def prepare_samples(tokenizer, num_samples=50):
    from datasets import load_dataset
    ds = load_dataset('rajpurkar/squad_v2', split='validation')
    samples = []
    for s in ds:
        if not s['answers']['text']: continue
        prompt = format_phi_prompt(s)
        tok_len = len(tokenizer.encode(prompt))
        if 100 <= tok_len <= 500:
            samples.append({
                'prompt': prompt,
                'gold': s['answers']['text'][0],
                'question': s['question'][:100],
                'tok_len': tok_len,
            })
        if len(samples) >= num_samples * 3:
            break
    np.random.seed(42)
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    return [samples[i] for i in indices]


def run_full_battery(model, tokenizer, samples):
    """Run full experiment battery."""
    num_layers = model.config.num_hidden_layers
    all_results = []
    exp_name = 'phi35_full'
    start_idx, all_results = load_checkpoint(exp_name)

    for i, s in enumerate(samples):
        if i < start_idx: continue
        t0 = time.time()

        inputs = tokenizer(s['prompt'], return_tensors='pt', truncation=True, max_length=512).to('cuda')
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]
        gold = s['gold']

        result = {'idx': i, 'gold': gold, 'seq_len': seq_len}

        # 1. Full baseline
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16)
        result['full'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 2. Quantization: INT4, INT8
        for bits in [4, 8]:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, bits)
            result[f'int{bits}'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 3. Mixed-precision: L0 FP16 + rest INT4
        ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer0_bits=16)
        result['mixed_L0fp16_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 4. Layer-wise INT4 (sample 6 layers evenly)
        test_layers = [0, num_layers//6, num_layers//3, num_layers//2, 2*num_layers//3, num_layers-1]
        for li in test_layers:
            ans = generate_quantized(model, tokenizer, input_ids, seq_len, 4, layer_only=li)
            result[f'only_L{li}_int4'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

        # 5. Selection methods (every 5th sample to save time)
        if i % 5 == 0:
            q_start, a_start = find_boundaries_phi(tokenizer, input_ids, seq_len)
            context_end = q_start
            question_positions = list(range(q_start, min(a_start, seq_len)))
            always_keep = question_positions + [0]  # Keep BOS + question

            try:
                # Q2C
                q2c_scores = compute_q2c_scores(model, tokenizer, input_ids, seq_len, q_start, a_start)
                mask = build_selection_mask(q2c_scores, seq_len, 0.5, always_keep)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16, selection_mask=mask)
                result['q2c_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # SnapKV
                snap_scores = compute_snapkv_scores(model, tokenizer, input_ids, seq_len)
                mask = build_selection_mask(snap_scores, seq_len, 0.5, always_keep)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16, selection_mask=mask)
                result['snapkv_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # H2O
                h2o_scores = compute_h2o_scores(model, tokenizer, input_ids, seq_len)
                mask = build_selection_mask(h2o_scores, seq_len, 0.5, always_keep)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16, selection_mask=mask)
                result['h2o_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}

                # Random
                rand_scores = np.random.rand(seq_len)
                mask = build_selection_mask(rand_scores, seq_len, 0.5, always_keep)
                ans = generate_quantized(model, tokenizer, input_ids, seq_len, 16, selection_mask=mask)
                result['random_50'] = {'answer': ans[:200], 'f1': compute_f1(ans, gold)}
            except Exception as e:
                logger.warning(f"  Selection failed for sample {i}: {e}")

        elapsed = time.time() - t0
        result['time'] = elapsed
        all_results.append(result)
        save_checkpoint(exp_name, i, all_results)

        fp16 = result['full']['f1']
        int4 = result['int4']['f1']
        int8 = result['int8']['f1']
        mixed = result['mixed_L0fp16_int4']['f1']
        logger.info(f"  [{i+1}/{len(samples)}] seq={seq_len} full={fp16:.3f} int8={int8:.3f} "
                    f"int4={int4:.3f} mixed={mixed:.3f} ({elapsed:.1f}s)")

    return all_results


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Start: {datetime.now()}")
    t_start = time.time()

    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation='eager')
    model.config.use_cache = True
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    logger.info(f"Loaded: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")

    logger.info("Preparing samples...")
    samples = prepare_samples(tokenizer, NUM_SAMPLES)
    logger.info(f"Prepared {len(samples)} samples")

    all_results = run_full_battery(model, tokenizer, samples)

    # Summary
    fp16_f1 = float(np.mean([r['full']['f1'] for r in all_results]))
    int4_f1 = float(np.mean([r['int4']['f1'] for r in all_results]))
    int8_f1 = float(np.mean([r['int8']['f1'] for r in all_results]))
    mixed_f1 = float(np.mean([r['mixed_L0fp16_int4']['f1'] for r in all_results]))

    logger.info(f"\n{'#'*60}")
    logger.info(f"SUMMARY: {MODEL_SHORT}")
    logger.info(f"{'#'*60}")
    logger.info(f"  Full: {fp16_f1:.4f}")
    logger.info(f"  INT8: {int8_f1:.4f} ({int8_f1/fp16_f1*100:.1f}%)")
    logger.info(f"  INT4: {int4_f1:.4f} ({int4_f1/fp16_f1*100:.1f}%)")
    logger.info(f"  Mixed: {mixed_f1:.4f} ({mixed_f1/fp16_f1*100:.1f}%)")

    # Layer-wise
    num_layers = model.config.num_hidden_layers
    test_layers = [0, num_layers//6, num_layers//3, num_layers//2, 2*num_layers//3, num_layers-1]
    logger.info(f"\n  Layer-wise INT4:")
    for li in test_layers:
        key = f'only_L{li}_int4'
        vals = [r[key]['f1'] for r in all_results if key in r]
        if vals:
            mean_f1 = float(np.mean(vals))
            logger.info(f"    Layer {li}: {mean_f1:.4f} ({mean_f1/fp16_f1*100:.1f}%)")

    # Selection (subset)
    for method in ['q2c_50', 'snapkv_50', 'h2o_50', 'random_50']:
        vals = [r[method]['f1'] for r in all_results if method in r]
        if vals:
            mean_f1 = float(np.mean(vals))
            logger.info(f"  {method}: {mean_f1:.4f} ({mean_f1/fp16_f1*100:.1f}%)")

    # Save
    combined = {
        'metadata': {
            'model': MODEL_SHORT,
            'task': 'SQuAD-v2',
            'num_samples': len(all_results),
            'full_f1': fp16_f1,
            'int4_f1': int4_f1, 'int4_pct': int4_f1/fp16_f1*100,
            'int8_f1': int8_f1, 'int8_pct': int8_f1/fp16_f1*100,
            'mixed_f1': mixed_f1, 'mixed_pct': mixed_f1/fp16_f1*100,
            'num_layers': num_layers,
            'num_kv_heads': num_kv_heads,
            'head_dim': head_dim,
        },
        'results': all_results,
    }
    final_path = RESULTS_DIR / f'phi35_squad_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {final_path}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nBatch 21 COMPLETE in {elapsed:.1f} minutes")

    del model; torch.cuda.empty_cache(); gc.collect()
