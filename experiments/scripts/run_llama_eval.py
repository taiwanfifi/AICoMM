#!/usr/bin/env python3
"""
Llama-3.1-8B evaluation on SQuAD v2 (100 samples).

Previously failed with 401 auth error (gated model).
Now has HF token access granted.

Runs the same evaluation as EXP 4 in run_reviewer_fixes.py,
but standalone for Llama-3.1-8B only.
"""

import sys
import os
import time
import json
import gc
from pathlib import Path
from datetime import datetime

os.environ['HF_HOME'] = '/workspace/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/hf_cache/datasets'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# HF token for gated model access
# Set HF_TOKEN env var before running (for gated model access)
# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

sys.path.insert(0, str(Path(__file__).parent))

from exp_utils import (
    setup_logging, load_model, free_model,
    load_squad_samples, format_qa_prompt,
    compute_f1, confidence_interval_95,
    save_results, make_timestamp,
    RESULTS_DIR,
)

import torch
import numpy as np

logger = setup_logging('llama_eval', 'llama_eval.log')
TIMESTAMP = make_timestamp()

MODEL_ID = 'meta-llama/Llama-3.1-8B'
N_SAMPLES = 100


def main():
    logger.info("=" * 70)
    logger.info(f"Llama-3.1-8B SQuAD Evaluation (n={N_SAMPLES})")
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    logger.info("=" * 70)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    samples = load_squad_samples(N_SAMPLES, seed=42)

    logger.info(f"Loading {MODEL_ID}...")
    model, tokenizer = load_model(MODEL_ID)

    per_sample = []
    start = time.time()

    for i, s in enumerate(samples):
        prompt = format_qa_prompt(s['context'], s['question'])
        gold = s['gold_answer']

        try:
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024,
                               truncation=True).to(device)

            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            ans = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:],
                                   skip_special_tokens=True).strip()
            f1 = compute_f1(ans, gold)
            per_sample.append({'sample_idx': i, 'f1': f1, 'answer': ans[:200]})

        except Exception as e:
            logger.error(f"Sample {i}: {e}")
            per_sample.append({'sample_idx': i, 'f1': 0.0, 'error': str(e)})

        if (i + 1) % 25 == 0:
            valid = [r['f1'] for r in per_sample if 'error' not in r]
            logger.info(f"  [{i+1}/{N_SAMPLES}] mean_f1={np.mean(valid):.3f}")

    elapsed = time.time() - start
    free_model(model)

    valid_f1s = [r['f1'] for r in per_sample if 'error' not in r]
    summary = {
        'model': MODEL_ID,
        'n_valid': len(valid_f1s),
        'n_errors': len(per_sample) - len(valid_f1s),
        'mean_f1': float(np.mean(valid_f1s)) if valid_f1s else 0.0,
        'std_f1': float(np.std(valid_f1s)) if valid_f1s else 0.0,
        'ci95_f1': float(confidence_interval_95(valid_f1s)) if valid_f1s else 0.0,
        'elapsed_minutes': elapsed / 60,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Llama-3.1-8B Results (n={summary['n_valid']}):")
    logger.info(f"  F1 = {summary['mean_f1']:.3f} +/- {summary['ci95_f1']:.3f}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"{'='*60}")

    output = {
        'metadata': {
            'experiment': 'llama_eval',
            'description': 'Llama-3.1-8B SQuAD v2 F1 evaluation (reviewer fix: gated model access)',
            'model': MODEL_ID,
            'n_samples': N_SAMPLES,
            'seed': 42,
            'timestamp': TIMESTAMP,
        },
        'summary': summary,
        'per_sample': per_sample,
    }
    save_results(output, f'exp_llama_eval_{TIMESTAMP}.json')
    logger.info("DONE")


if __name__ == '__main__':
    main()
