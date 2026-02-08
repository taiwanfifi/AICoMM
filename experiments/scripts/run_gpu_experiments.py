#!/usr/bin/env python3
"""
GPU Experiment Runner — runs all pending KV-cache experiments on the GPU server.
Includes checkpointing, model downloading, and result saving.

Usage:
    python3 run_gpu_experiments.py [--phase PHASE] [--exp EXP] [--model MODEL] [--samples N]

Examples:
    python3 run_gpu_experiments.py                          # Run all pending experiments
    python3 run_gpu_experiments.py --phase quick_wins        # Run quick feasibility experiments
    python3 run_gpu_experiments.py --exp exp04 --model qwen2.5-7b --samples 50
    python3 run_gpu_experiments.py --phase cross_model       # Cross-model CKA analysis
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Force CUDA device
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'experiment_run.log')
    ]
)
logger = logging.getLogger(__name__)

def _get_kv_layer(pkv, layer_idx, component='key'):
    """Extract key or value tensor from a layer of past_key_values.
    Handles DynamicCache (transformers 5.x with .layers), older DynamicCache
    (with .key_cache), and tuple formats.
    """
    # transformers 5.x: DynamicCache with .layers list of DynamicLayer
    if hasattr(pkv, 'layers'):
        layer = pkv.layers[layer_idx]
        return layer.keys if component == 'key' else layer.values
    # transformers 4.x: DynamicCache with .key_cache / .value_cache
    if hasattr(pkv, 'key_cache') and hasattr(pkv, 'value_cache'):
        return pkv.key_cache[layer_idx] if component == 'key' else pkv.value_cache[layer_idx]
    # Plain tuple format
    pair = pkv[layer_idx]
    return pair[0] if component == 'key' else pair[1]


def _get_kv_pairs(pkv):
    """Get list of (key, value) tensor pairs from past_key_values."""
    # transformers 5.x: DynamicCache with .layers
    if hasattr(pkv, 'layers'):
        return [(layer.keys, layer.values) for layer in pkv.layers]
    # transformers 4.x: DynamicCache with .key_cache / .value_cache
    if hasattr(pkv, 'key_cache') and hasattr(pkv, 'value_cache'):
        return list(zip(pkv.key_cache, pkv.value_cache))
    # Plain tuple
    return [(pkv[l][0], pkv[l][1]) for l in range(len(pkv))]


def _num_kv_layers(pkv):
    """Get the number of layers in past_key_values."""
    if hasattr(pkv, 'layers'):
        return len(pkv.layers)
    if hasattr(pkv, 'key_cache'):
        return len(pkv.key_cache)
    return len(pkv)


RESULTS_DIR = PROJECT_ROOT / 'results' / 'gpu_run'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu}, VRAM: {vram:.1f} GB")
        return gpu, vram
    else:
        logger.error("No CUDA GPU available!")
        sys.exit(1)


def save_checkpoint(exp_name, data, sample_idx):
    """Save checkpoint for resumability."""
    ckpt_path = CHECKPOINT_DIR / f'{exp_name}_checkpoint.json'
    checkpoint = {
        'exp_name': exp_name,
        'timestamp': datetime.now().isoformat(),
        'last_sample_idx': sample_idx,
        'results': data,
    }
    with open(ckpt_path, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)
    logger.info(f"  [CHECKPOINT] Saved at sample {sample_idx} → {ckpt_path}")


def load_checkpoint(exp_name):
    """Load checkpoint if exists."""
    ckpt_path = CHECKPOINT_DIR / f'{exp_name}_checkpoint.json'
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        logger.info(f"  [RESUME] Found checkpoint at sample {ckpt['last_sample_idx']}")
        return ckpt
    return None


def save_results(exp_name, results, metadata=None):
    """Save final results."""
    result_path = RESULTS_DIR / f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output = {
        'experiment': exp_name,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'results': results,
    }
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"[SAVED] {exp_name} → {result_path}")

    # Also clear checkpoint
    ckpt_path = CHECKPOINT_DIR / f'{exp_name}_checkpoint.json'
    if ckpt_path.exists():
        ckpt_path.unlink()

    return result_path


# =========================================================================
# Experiment: Cross-Model CKA Analysis (Topic 02 feasibility check)
# =========================================================================
def run_cross_model_cka(num_samples=20):
    """Compare KV-cache representations between Qwen2.5-3B and Qwen2.5-7B.

    Uses Centered Kernel Alignment (CKA) to measure representational similarity.
    This is the cheapest feasibility check for cross-model KV transfer (Topic 02).
    """
    exp_name = 'cross_model_cka'
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {exp_name}")
    logger.info(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # Check checkpoint
    ckpt = load_checkpoint(exp_name)
    start_idx = ckpt['last_sample_idx'] + 1 if ckpt else 0
    results = ckpt['results'] if ckpt else []

    # Load SQuAD samples
    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    # Load models sequentially (to manage VRAM)
    logger.info("Loading Qwen2.5-3B...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model_3b.eval()
    tok_3b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

    # Process all samples through 3B first
    kv_3b_list = []
    logger.info("Processing samples through 3B...")
    for i, sample in enumerate(samples):
        if i < start_idx:
            continue
        text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tok_3b(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            out = model_3b(**inputs, use_cache=True)
        # Extract KV for CKA: use last layer key, averaged over heads
        pkv = out.past_key_values
        # Get last layer key: shape [B, H, T, D]
        last_key = _get_kv_layer(pkv, -1, 'key')
        # Average over heads → [T, D]
        kv_repr = last_key[0].mean(dim=0).cpu().float()
        kv_3b_list.append(kv_repr)

        if (i + 1) % 5 == 0:
            logger.info(f"  3B: {i+1}/{num_samples} samples processed")

    # Free 3B model
    del model_3b
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Load 7B model
    logger.info("Loading Qwen2.5-7B...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model_7b.eval()
    tok_7b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

    # Process through 7B
    kv_7b_list = []
    logger.info("Processing samples through 7B...")
    for i, sample in enumerate(samples):
        text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        inputs = tok_7b(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            out = model_7b(**inputs, use_cache=True)
        pkv = out.past_key_values
        last_key = _get_kv_layer(pkv, -1, 'key')
        kv_repr = last_key[0].mean(dim=0).cpu().float()
        kv_7b_list.append(kv_repr)

        if (i + 1) % 5 == 0:
            logger.info(f"  7B: {i+1}/{num_samples} samples processed")

    del model_7b
    torch.cuda.empty_cache()

    # Compute CKA between 3B and 7B representations
    logger.info("Computing CKA similarity...")
    cka_scores = []
    linear_proj_errors = []

    for i in range(len(kv_3b_list)):
        kv3 = kv_3b_list[i]  # [T, D3]
        kv7 = kv_7b_list[i]  # [T, D7]

        # Align sequence lengths (use min)
        min_t = min(kv3.shape[0], kv7.shape[0])
        kv3 = kv3[:min_t]
        kv7 = kv7[:min_t]

        # Linear CKA
        K3 = kv3 @ kv3.T  # [T, T]
        K7 = kv7 @ kv7.T
        # CKA = HSIC(K3, K7) / sqrt(HSIC(K3, K3) * HSIC(K7, K7))
        # Using linear kernel: HSIC = ||K3^T K7||_F^2 / (n-1)^2
        # Simplified: CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        cross = torch.norm(kv3.T @ kv7, p='fro').item() ** 2
        self3 = torch.norm(kv3.T @ kv3, p='fro').item() ** 2
        self7 = torch.norm(kv7.T @ kv7, p='fro').item() ** 2
        cka = cross / (np.sqrt(self3 * self7) + 1e-10)
        cka_scores.append(cka)

        # Also test: can a linear projection fit?
        # Solve: W @ kv3.T ≈ kv7.T → W = kv7.T @ pinv(kv3.T)
        # Relative error = ||W @ kv3 - kv7|| / ||kv7||
        try:
            W = kv7.T @ torch.linalg.pinv(kv3.T)
            proj = (W @ kv3.T).T  # [T, D7]
            rel_err = torch.norm(proj - kv7).item() / (torch.norm(kv7).item() + 1e-10)
            linear_proj_errors.append(rel_err)
        except:
            linear_proj_errors.append(float('nan'))

        sample_result = {
            'sample_idx': i,
            'cka': cka,
            'linear_proj_error': linear_proj_errors[-1],
            'seq_len_3b': kv_3b_list[i].shape[0],
            'seq_len_7b': kv_7b_list[i].shape[0],
        }
        results.append(sample_result)
        save_checkpoint(exp_name, results, i)

    # Summary
    summary = {
        'mean_cka': float(np.mean(cka_scores)),
        'std_cka': float(np.std(cka_scores)),
        'min_cka': float(np.min(cka_scores)),
        'max_cka': float(np.max(cka_scores)),
        'mean_linear_proj_error': float(np.nanmean(linear_proj_errors)),
        'std_linear_proj_error': float(np.nanstd(linear_proj_errors)),
        'num_samples': len(cka_scores),
        'feasibility_assessment': 'promising' if np.mean(cka_scores) > 0.5 else 'needs_investigation' if np.mean(cka_scores) > 0.3 else 'unlikely',
    }

    logger.info(f"\n--- Cross-Model CKA Results ---")
    logger.info(f"Mean CKA: {summary['mean_cka']:.4f} ± {summary['std_cka']:.4f}")
    logger.info(f"Mean Linear Proj Error: {summary['mean_linear_proj_error']:.4f}")
    logger.info(f"Feasibility: {summary['feasibility_assessment']}")

    save_results(exp_name, results, metadata=summary)
    return summary


# =========================================================================
# Experiment: Quantization Baseline (Topic 06 quick win)
# =========================================================================
def run_quantization_baseline(model_name="qwen2.5-3b", num_samples=20):
    """Compare INT8/INT4 quantization vs SVD compression at matched bandwidth."""
    exp_name = f'quantization_vs_svd_{model_name.replace(".", "")}'
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {exp_name}")
    logger.info(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # Model mapping
    model_map = {
        'qwen2.5-3b': 'Qwen/Qwen2.5-3B',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B',
    }
    hf_name = model_map.get(model_name, model_name)

    logger.info(f"Loading {hf_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model.config.use_cache = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    results = []

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        answer = sample['answers']['text'][0]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")

        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        # Extract KV-cache
        pkv = out.past_key_values
        kv_pairs = _get_kv_pairs(pkv)

        # Compute sizes
        full_size = sum(k.numel() + v.numel() for k, v in kv_pairs) * 2  # FP16 = 2 bytes
        int8_size = sum(k.numel() + v.numel() for k, v in kv_pairs) * 1  # INT8 = 1 byte
        int4_size = sum(k.numel() + v.numel() for k, v in kv_pairs) // 2  # INT4 = 0.5 byte

        # INT8 quantization
        int8_errors = []
        for k, v in kv_pairs:
            for tensor in [k, v]:
                t = tensor.float()
                scale = t.abs().max() / 127.0
                quantized = torch.clamp(torch.round(t / scale), -128, 127)
                dequantized = quantized * scale
                rel_err = torch.norm(dequantized - t).item() / (torch.norm(t).item() + 1e-10)
                int8_errors.append(rel_err)

        # INT4 quantization (per-group, group_size=32)
        int4_errors = []
        group_size = 32
        for k, v in kv_pairs:
            for tensor in [k, v]:
                t = tensor.float().reshape(-1, group_size)
                scale = t.abs().max(dim=1, keepdim=True).values / 7.0
                quantized = torch.clamp(torch.round(t / (scale + 1e-10)), -8, 7)
                dequantized = quantized * scale
                dequantized = dequantized.reshape(tensor.shape)
                t_flat = tensor.float()
                rel_err = torch.norm(dequantized - t_flat).item() / (torch.norm(t_flat).item() + 1e-10)
                int4_errors.append(rel_err)

        # SVD compression at matched bandwidths
        svd_errors = {}
        for rank in [4, 8, 16, 32]:
            rank_errors = []
            svd_size = 0
            for k, v in kv_pairs:
                for tensor in [k, v]:
                    B, H, T, D = tensor.shape
                    eff_rank = min(rank, T, D)
                    mat = tensor.float()
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                    U_r = U[:, :, :, :eff_rank]
                    S_r = S[:, :, :eff_rank]
                    Vh_r = Vh[:, :, :eff_rank, :]
                    recon = (U_r * S_r.unsqueeze(2)) @ Vh_r
                    rel_err = torch.norm(recon - mat).item() / (torch.norm(mat).item() + 1e-10)
                    rank_errors.append(rel_err)
                    svd_size += (B * H * (T * eff_rank + eff_rank + eff_rank * D)) * 2
            svd_errors[rank] = {
                'mean_rel_error': float(np.mean(rank_errors)),
                'bandwidth_fraction': svd_size / full_size,
            }

        sample_result = {
            'sample_idx': i,
            'seq_len': inputs['input_ids'].shape[1],
            'full_size_bytes': full_size,
            'int8': {
                'size_bytes': int8_size,
                'bandwidth_fraction': int8_size / full_size,
                'mean_rel_error': float(np.mean(int8_errors)),
            },
            'int4': {
                'size_bytes': int4_size,
                'bandwidth_fraction': int4_size / full_size,
                'mean_rel_error': float(np.mean(int4_errors)),
            },
            'svd': svd_errors,
        }
        results.append(sample_result)

        if (i + 1) % 5 == 0:
            logger.info(f"  [{i+1}/{num_samples}] seq_len={inputs['input_ids'].shape[1]}, "
                       f"INT8_err={np.mean(int8_errors):.4f}, INT4_err={np.mean(int4_errors):.4f}")
            save_checkpoint(exp_name, results, i)

    # Summary
    summary = {
        'model': model_name,
        'num_samples': num_samples,
        'int8_mean_error': float(np.mean([r['int8']['mean_rel_error'] for r in results])),
        'int4_mean_error': float(np.mean([r['int4']['mean_rel_error'] for r in results])),
        'svd_rank4_error': float(np.mean([r['svd'][4]['mean_rel_error'] for r in results])),
        'svd_rank8_error': float(np.mean([r['svd'][8]['mean_rel_error'] for r in results])),
        'svd_rank16_error': float(np.mean([r['svd'][16]['mean_rel_error'] for r in results])),
        'svd_rank32_error': float(np.mean([r['svd'][32]['mean_rel_error'] for r in results])),
        'int8_bandwidth': 0.5,  # 50% of FP16
        'int4_bandwidth': 0.25,  # 25% of FP16
    }

    logger.info(f"\n--- Quantization vs SVD Results ---")
    logger.info(f"INT8 (50% BW): error={summary['int8_mean_error']:.4f}")
    logger.info(f"INT4 (25% BW): error={summary['int4_mean_error']:.4f}")
    for rank in [4, 8, 16, 32]:
        bw = np.mean([r['svd'][rank]['bandwidth_fraction'] for r in results])
        logger.info(f"SVD rank-{rank} ({bw*100:.1f}% BW): error={summary[f'svd_rank{rank}_error']:.4f}")

    del model
    torch.cuda.empty_cache()
    save_results(exp_name, results, metadata=summary)
    return summary


# =========================================================================
# Experiment: Layer-wise Probing (Topic 11)
# =========================================================================
def run_layer_probing(model_name="qwen2.5-3b", num_samples=50):
    """Probe each layer's KV-cache for task-relevant information."""
    exp_name = f'layer_probing_{model_name.replace(".", "")}'
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {exp_name}")
    logger.info(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model_map = {
        'qwen2.5-3b': ('Qwen/Qwen2.5-3B', 36),
        'qwen2.5-7b': ('Qwen/Qwen2.5-7B', 32),
    }
    hf_name, num_layers = model_map[model_name]

    logger.info(f"Loading {hf_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]
    samples = answerable[:num_samples]

    # Collect per-layer representations
    logger.info("Collecting KV representations per layer...")
    layer_representations = {l: [] for l in range(num_layers)}
    labels = []  # Binary: does this position contain the answer?

    for i, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        answer = sample['answers']['text'][0]
        answer_start = sample['answers']['answer_start'][0]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")

        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        pkv = out.past_key_values
        seq_len = inputs['input_ids'].shape[1]

        # Find answer token positions
        context_prefix = "Context: "
        context_start_char = len(context_prefix)
        answer_start_in_prompt = context_start_char + answer_start
        answer_end_in_prompt = answer_start_in_prompt + len(answer)

        # Get character-to-token mapping
        encoding = tokenizer(prompt, return_offsets_mapping=True, max_length=512, truncation=True)
        offsets = encoding['offset_mapping']
        answer_tokens = set()
        for tok_idx, (start, end) in enumerate(offsets):
            if start < answer_end_in_prompt and end > answer_start_in_prompt:
                answer_tokens.add(tok_idx)

        # Extract per-layer mean KV representation for answer vs non-answer positions
        for l in range(num_layers):
            key = _get_kv_layer(pkv, l, 'key')
            val = _get_kv_layer(pkv, l, 'value')

            # Mean over heads, concat K and V → [T, 2*D]
            k_mean = key[0].mean(dim=0).cpu().float()  # [T, D]
            v_mean = val[0].mean(dim=0).cpu().float()
            kv_repr = torch.cat([k_mean, v_mean], dim=-1)  # [T, 2*D]

            # Pool: mean of answer positions vs mean of non-answer positions
            answer_repr = kv_repr[list(answer_tokens)].mean(dim=0) if answer_tokens else torch.zeros(kv_repr.shape[1])
            non_answer_repr = kv_repr[[j for j in range(seq_len) if j not in answer_tokens]].mean(dim=0)

            layer_representations[l].append({
                'answer_repr': answer_repr.numpy(),
                'non_answer_repr': non_answer_repr.numpy(),
                'full_mean': kv_repr.mean(dim=0).numpy(),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{num_samples}] processed, {len(answer_tokens)} answer tokens")

    del model
    torch.cuda.empty_cache()

    # Probing: For each layer, train a classifier to distinguish answer vs non-answer representations
    logger.info("Training probes per layer...")
    probe_results = {}

    for l in range(num_layers):
        X = []
        y = []
        for rep in layer_representations[l]:
            X.append(rep['answer_repr'])
            y.append(1)
            X.append(rep['non_answer_repr'])
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        # Simple train/test split
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        probe_results[l] = {
            'probe_accuracy': float(acc),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }

        if l % 4 == 0 or l == num_layers - 1:
            logger.info(f"  Layer {l}: probe accuracy = {acc:.3f}")

    summary = {
        'model': model_name,
        'num_layers': num_layers,
        'num_samples': num_samples,
        'probe_accuracies': {l: probe_results[l]['probe_accuracy'] for l in range(num_layers)},
        'best_layer': max(probe_results, key=lambda l: probe_results[l]['probe_accuracy']),
        'worst_layer': min(probe_results, key=lambda l: probe_results[l]['probe_accuracy']),
    }

    logger.info(f"\n--- Layer Probing Results ---")
    logger.info(f"Best layer: {summary['best_layer']} (acc={probe_results[summary['best_layer']]['probe_accuracy']:.3f})")
    logger.info(f"Worst layer: {summary['worst_layer']} (acc={probe_results[summary['worst_layer']]['probe_accuracy']:.3f})")

    save_results(exp_name, probe_results, metadata=summary)
    return summary


# =========================================================================
# Main dispatcher
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='GPU Experiment Runner')
    parser.add_argument('--phase', default='all',
                       choices=['all', 'quick_wins', 'cross_model', 'quantization', 'layer_probing',
                                'remaining_exps'],
                       help='Which experiment phase to run')
    parser.add_argument('--model', default='qwen2.5-3b', help='Model to use')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples')
    args = parser.parse_args()

    gpu, vram = get_gpu_info()
    logger.info(f"Starting experiments on {gpu} ({vram:.0f}GB)")
    logger.info(f"Phase: {args.phase}, Model: {args.model}, Samples: {args.samples}")

    start_time = time.time()

    if args.phase in ('all', 'quick_wins', 'cross_model'):
        run_cross_model_cka(num_samples=args.samples)

    if args.phase in ('all', 'quick_wins', 'quantization'):
        run_quantization_baseline(model_name=args.model, num_samples=args.samples)

    if args.phase in ('all', 'quick_wins', 'layer_probing'):
        run_layer_probing(model_name=args.model, num_samples=args.samples)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
