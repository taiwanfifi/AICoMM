"""
Shared utilities for JSAC experiment scripts.

Provides:
  - Q2C scoring (last-layer and all-layer variants)
  - Cross-family tokenizer alignment
  - SQuAD v2 F1 evaluation (normalize_answer, compute_f1)
  - Model loading helpers (GPU with bfloat16, eager attention)
  - KV-cache extraction helpers (transformers 4.x / 5.x compat)
  - Statistical helpers (CI, paired t-test, Bonferroni)
  - Common constants (model configs, results directory)
"""

import os
import re
import gc
import json
import string
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

logger = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Model configurations for bandwidth calculations
MODEL_CONFIGS = {
    'Qwen2.5-3B': {'layers': 36, 'kv_heads': 2, 'head_dim': 128, 'attn_heads': 16},
    'Qwen2.5-7B': {'layers': 28, 'kv_heads': 4, 'head_dim': 128, 'attn_heads': 28},
    'Qwen2.5-14B': {'layers': 48, 'kv_heads': 8, 'head_dim': 128, 'attn_heads': 40},
    'Mistral-7B-v0.3': {'layers': 32, 'kv_heads': 8, 'head_dim': 128, 'attn_heads': 32},
    'Yi-6B-Chat': {'layers': 32, 'kv_heads': 4, 'head_dim': 128, 'attn_heads': 32},
}

# HuggingFace model names
HF_MODELS = {
    '3B': 'Qwen/Qwen2.5-3B',
    '7B': 'Qwen/Qwen2.5-7B',
    '14B': 'Qwen/Qwen2.5-14B',
    'Mistral-7B': 'mistralai/Mistral-7B-v0.3',
    'Yi-6B': '01-ai/Yi-6B-Chat',
}

# Default HF cache (override with HF_HOME env var)
DEFAULT_HF_HOME = '/dev/shm/hf_7b'


# =========================================================================
# Setup
# =========================================================================

def setup_logging(name: str, log_file: str | None = None):
    """Configure logging for experiment script."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
    )
    return logging.getLogger(name)


def setup_hf_cache():
    """Set HuggingFace cache directory if not already set."""
    if 'HF_HOME' not in os.environ:
        if Path(DEFAULT_HF_HOME).exists():
            os.environ['HF_HOME'] = DEFAULT_HF_HOME
            os.environ['HF_DATASETS_CACHE'] = str(Path(DEFAULT_HF_HOME) / 'datasets')


def log_gpu_info():
    """Log GPU information."""
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Torch: {torch.__version__}")


# =========================================================================
# Text evaluation (SQuAD v2 style)
# =========================================================================

def normalize_answer(s: str) -> str:
    """SQuAD v2 normalization: lowercase, remove articles/punct, collapse whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def compute_f1(pred: str, gold: str) -> float:
    """Normalized token-F1 (SQuAD v2 style)."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# =========================================================================
# Model loading
# =========================================================================

def load_model(model_name: str, device_map: str = "cuda"):
    """
    Load a causal LM with bfloat16 and eager attention.

    Returns: (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loaded {model_name} ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")
    return model, tokenizer


def free_model(model):
    """Delete model and free GPU memory."""
    del model
    torch.cuda.empty_cache()
    gc.collect()


# =========================================================================
# KV-cache helpers
# =========================================================================

def get_kv_layer(cache, layer_idx: int, component: str = 'key'):
    """Extract key or value tensor from cache. Handles transformers 4.x and 5.x."""
    if hasattr(cache, 'layers'):
        layer = cache.layers[layer_idx]
        return layer.keys if component == 'key' else layer.values
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx] if component == 'key' else cache.value_cache[layer_idx]
    pair = cache[layer_idx]
    return pair[0] if component == 'key' else pair[1]


def num_layers(cache) -> int:
    """Get number of layers from cache (handles both formats)."""
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    if hasattr(cache, 'key_cache'):
        return len(cache.key_cache)
    return len(cache)


# =========================================================================
# Q2C Scoring
# =========================================================================

def compute_q2c_last_layer(attentions, context_end_pos: int):
    """
    Q2C scoring using only the last layer.

    Args:
        attentions: tuple of attention tensors from model output
        context_end_pos: position where context ends in token space

    Returns:
        q2c_scores: numpy array of shape (context_end_pos,)
    """
    attn = attentions[-1][0]  # [heads, seq, seq]
    q2c = attn[:, context_end_pos:, :context_end_pos].sum(dim=(0, 1))
    return q2c.float().cpu().numpy()


def compute_q2c_all_layers(attentions, context_end_pos: int):
    """
    Q2C scoring averaged across ALL layers.

    Args:
        attentions: tuple of attention tensors from model output
        context_end_pos: position where context ends in token space

    Returns:
        q2c_scores: numpy array of shape (context_end_pos,)
    """
    scores = None
    for layer_attn in attentions:
        attn = layer_attn[0]  # [heads, seq, seq]
        layer_q2c = attn[:, context_end_pos:, :context_end_pos].sum(dim=(0, 1))
        if scores is None:
            scores = layer_q2c.float().cpu()
        else:
            scores += layer_q2c.float().cpu()
    scores = scores / len(attentions)
    return scores.numpy()


def prefill_and_score(model, tokenizer, prompt: str, context_end_pos: int,
                      q2c_mode: str = 'last_layer', max_length: int = 1024):
    """
    Run prefill and compute Q2C attention scores.

    Args:
        model: the language model
        tokenizer: the tokenizer
        prompt: the full prompt string
        context_end_pos: position where context ends
        q2c_mode: 'last_layer' or 'all_layers'
        max_length: max token length for truncation

    Returns:
        (kv_cache, q2c_scores, inputs, seq_len)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length,
                       truncation=True).to(device)

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)

    seq_len = out.attentions[-1].shape[-1]

    if q2c_mode == 'last_layer':
        q2c = compute_q2c_last_layer(out.attentions, context_end_pos)
    elif q2c_mode == 'all_layers':
        q2c = compute_q2c_all_layers(out.attentions, context_end_pos)
    else:
        raise ValueError(f"Unknown q2c_mode: {q2c_mode}")

    return out.past_key_values, q2c, inputs, seq_len


def select_positions(q2c_scores, retention_ratio: float):
    """Select top-k positions by Q2C score. Returns sorted indices."""
    n = len(q2c_scores)
    k = max(1, int(n * retention_ratio))
    indices = np.argsort(q2c_scores)[-k:]
    return np.sort(indices)


# =========================================================================
# Generation with selection mask
# =========================================================================

def generate_with_mask(model, tokenizer, input_ids, selected_positions,
                       context_len: int, seq_len: int, max_new: int = 64):
    """
    Generate with attention mask that excludes unselected context positions.
    Preserves RoPE encoding by masking rather than removing positions.
    """
    device = next(model.parameters()).device

    attn_mask = torch.zeros(1, seq_len, device=device, dtype=torch.long)
    for pos in selected_positions:
        attn_mask[0, pos] = 1
    attn_mask[0, context_len:seq_len] = 1

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
        )

    generated = tokenizer.decode(outputs[0][input_ids.shape[1]:],
                                 skip_special_tokens=True)
    return generated.strip()


# =========================================================================
# Cross-family tokenizer alignment
# =========================================================================

def get_token_char_spans(tokenizer, text: str):
    """
    Get character spans for each token.

    Returns list of (start_char, end_char) for each token in the encoding.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    return offsets


def align_positions_cross_tokenizer(
    source_tokenizer, target_tokenizer, text: str, source_positions: np.ndarray
):
    """
    Map selected positions from source tokenizer to target tokenizer
    via character-level alignment.

    Algorithm:
      1. Get character spans for both tokenizers
      2. For each selected source position, find its character span
      3. Find all target positions whose character spans overlap

    Args:
        source_tokenizer: edge model tokenizer
        target_tokenizer: cloud model tokenizer
        text: the raw text (same for both)
        source_positions: selected token positions in source tokenizer space

    Returns:
        target_positions: sorted numpy array of positions in target tokenizer space
    """
    source_spans = get_token_char_spans(source_tokenizer, text)
    target_spans = get_token_char_spans(target_tokenizer, text)

    target_selected = set()

    for src_pos in source_positions:
        if src_pos >= len(source_spans):
            continue
        src_start, src_end = source_spans[src_pos]
        if src_start == src_end:
            continue

        # Find all target tokens overlapping this character range
        for tgt_pos, (tgt_start, tgt_end) in enumerate(target_spans):
            if tgt_start == tgt_end:
                continue
            # Check overlap
            if tgt_start < src_end and tgt_end > src_start:
                target_selected.add(tgt_pos)

    return np.sort(np.array(list(target_selected), dtype=np.int64))


# =========================================================================
# Dataset loading
# =========================================================================

def load_squad_samples(num_samples: int, seed: int = SEED):
    """
    Load answerable SQuAD v2 samples with deterministic selection.

    Returns list of dicts with keys: context, question, gold_answer
    """
    from datasets import load_dataset

    dataset = load_dataset('rajpurkar/squad_v2', split='validation')
    answerable = [s for s in dataset if len(s['answers']['text']) > 0]

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(answerable), size=min(num_samples, len(answerable)),
                         replace=False)

    samples = []
    for i in indices:
        s = answerable[i]
        samples.append({
            'context': s['context'],
            'question': s['question'],
            'gold_answer': s['answers']['text'][0],
        })

    logger.info(f"Loaded {len(samples)} SQuAD v2 samples (seed={seed})")
    return samples


def format_qa_prompt(context: str, question: str) -> str:
    """Format a QA prompt for base models (NOT ChatML)."""
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def get_context_end_pos(tokenizer, context: str) -> int:
    """Get the token position where the context ends."""
    context_part = f"Context: {context}\n"
    context_ids = tokenizer.encode(context_part)
    return len(context_ids)


# =========================================================================
# Statistical helpers
# =========================================================================

def confidence_interval_95(data):
    """Compute 95% confidence interval using t-distribution."""
    from scipy import stats
    n = len(data)
    if n < 2:
        return 0.0
    se = stats.sem(data)
    ci = se * stats.t.ppf(0.975, n - 1)
    return ci


def paired_ttest(a, b):
    """Paired t-test. Returns (t_stat, p_value)."""
    from scipy import stats
    return stats.ttest_rel(a, b)


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction. Returns adjusted alpha threshold."""
    return alpha / len(p_values)


def overlap_percentage(set_a, set_b):
    """Compute overlap between two position sets as percentage of set_a."""
    if len(set_a) == 0:
        return 0.0
    return len(set(set_a) & set(set_b)) / len(set_a) * 100


# =========================================================================
# Result saving
# =========================================================================

def save_results(data: dict, filename: str):
    """Save results as JSON in the results directory."""
    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"[SAVED] -> {path}")
    return path


def make_timestamp() -> str:
    """Generate a timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================================
# Bandwidth calculation helpers
# =========================================================================

def kv_size_bytes(model_short: str, context_len: int, dtype_bytes: int = 2):
    """
    Calculate full KV-cache size in bytes.

    Formula: 2 * num_layers * num_kv_heads * context_len * head_dim * dtype_bytes
    """
    cfg = MODEL_CONFIGS.get(model_short)
    if cfg is None:
        logger.warning(f"Unknown model config: {model_short}, using Qwen2.5-7B defaults")
        cfg = MODEL_CONFIGS['Qwen2.5-7B']
    return 2 * cfg['layers'] * cfg['kv_heads'] * context_len * cfg['head_dim'] * dtype_bytes


def index_size_bytes(n_selected: int) -> int:
    """Size of position index payload: int32 per position."""
    return n_selected * 4


def tx_time_ms(payload_bytes: int, bandwidth_mbps: float) -> float:
    """Transmission time in milliseconds."""
    return (payload_bytes * 8) / (bandwidth_mbps * 1e6) * 1000
