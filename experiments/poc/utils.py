"""
Shared utilities for KV-cache PoC experiments.

Model: GPT-2 124M (default, fast) or TinyLlama-1.1B (upgrade option)
Hardware: M1-M4 Mac (CPU/MPS)

Note: These experiments require direct access to model internals
(KV-cache tensors, attention weights), so they MUST use a locally-loaded
HuggingFace model. API-based models (OpenAI, etc.) do not expose these.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPT-2 124M: fast, small (~500MB), 12 layers, 768 hidden, 12 heads
# TinyLlama 1.1B: bigger, slower, 22 layers, 2048 hidden, 32 heads (upgrade)
DEFAULT_MODEL = "gpt2"
UPGRADE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return best available device.

    MPS is disabled by default because it can cause mutex deadlocks
    with attention output on some macOS versions. Set env var
    USE_MPS=1 to enable.
    """
    if os.environ.get("USE_MPS") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
):
    """
    Load a causal LM and its tokenizer.

    Falls back to GPT-2 if the primary model cannot be loaded
    (e.g. network issues or insufficient memory).
    """
    if device is None:
        device = get_device()

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager",  # required for output_attentions
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} on {device}")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# KV-cache helpers
# ---------------------------------------------------------------------------

def extract_kv_cache(model_output) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract KV-cache from model output.

    Returns a list of (key, value) tuples, one per layer.
    Each tensor has shape (batch, num_heads, seq_len, head_dim).

    Handles both legacy tuple format and transformers 5.x DynamicCache.
    """
    past = model_output.past_key_values
    if past is None:
        raise ValueError("Model output has no past_key_values. "
                         "Pass use_cache=True during forward.")

    kv_list = []
    # transformers 5.x: DynamicCache with .layers attribute
    if hasattr(past, "layers"):
        for layer in past.layers:
            k = layer.keys.detach().cpu()
            v = layer.values.detach().cpu()
            kv_list.append((k, v))
    else:
        # Legacy: list/tuple of (key, value) tuples
        for layer_kv in past:
            k, v = layer_kv[0], layer_kv[1]
            kv_list.append((k.detach().cpu(), v.detach().cpu()))
    return kv_list


def extract_attentions(model_output) -> list[torch.Tensor]:
    """
    Extract attention weights from model output.

    Returns a list of tensors, one per layer.
    Each tensor has shape (batch, num_heads, seq_len, seq_len).
    """
    attns = model_output.attentions
    if attns is None:
        raise ValueError("Model output has no attentions. "
                         "Pass output_attentions=True when loading the model.")
    return [a.detach().cpu() for a in attns]


def kv_cache_size_bytes(kv_list: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Total size of KV-cache in bytes (FP32)."""
    total = 0
    for k, v in kv_list:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize(text: str, tokenizer, device: torch.device) -> dict:
    """Tokenize text and move tensors to device."""
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    past_key_values=None,
) -> float:
    """
    Compute perplexity of *text* under the model.

    If past_key_values is provided, treat it as prefix context
    (the KV-cache of a previous prefix).
    """
    inputs = tokenize(text, tokenizer, device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    logits = outputs.logits  # (batch, seq_len, vocab_size)

    # Shift so token i predicts token i+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return torch.exp(loss).item()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_json(data: dict, filename: str):
    """Save dict as JSON in results directory."""
    path = ensure_results_dir() / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")
    return path
