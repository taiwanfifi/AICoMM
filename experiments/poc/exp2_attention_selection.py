"""
Experiment 2: Attention-guided KV Selection

Purpose:
    Verify that attention weights can effectively guide KV selection
    (the core assumption behind SnapKV and our SSC attention filtering).

Hypothesis:
    "Transmitting only the highest-attention KV entries does not
     significantly degrade task quality."

What this does:
    1. Run inference on a prompt, extract full KV-cache and attention weights
    2. Use attention weights to select top-k% most important KV entries
    3. Zero out / remove the rest
    4. Continue generation with pruned KV-cache
    5. Plot k% vs. generation quality (perplexity) trade-off curve

Output:
    results/exp2_attention_selection.json  - numerical results
    results/exp2_tradeoff_curve.png        - k% vs perplexity plot
"""

import sys
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_model_and_tokenizer,
    tokenize,
    extract_kv_cache,
    extract_attentions,
    ensure_results_dir,
    save_json,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREFIX = (
    "A surveillance drone monitoring a forest region has detected unusual "
    "thermal signatures at coordinates 34.2N, 118.5W. The onboard sensor "
    "array indicates elevated temperatures consistent with combustion. "
    "Atmospheric readings show particulate matter concentration at 3x "
    "normal levels. Wind speed is 15 km/h from the southwest. "
    "The drone's visual camera confirms gray-white plumes rising from "
    "the canopy. Adjacent areas show no thermal anomalies."
)

CONTINUATION_PROMPT = " Based on this information, the recommended action is"

# Retention percentages to test
RETENTION_PERCENTAGES = [5, 10, 20, 30, 50, 70, 90, 100]


# ---------------------------------------------------------------------------
# KV selection logic
# ---------------------------------------------------------------------------

def compute_token_importance(attentions: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute per-token importance scores from attention weights.

    Strategy (inspired by SnapKV):
    - Use attention from the LAST few layers (where semantic decisions happen)
    - For each token position, aggregate attention it receives from later tokens
    - Higher score = this token's KV is more important to keep

    Returns: (seq_len,) tensor of importance scores
    """
    # Use last 4 layers (or all if fewer)
    num_observation_layers = min(4, len(attentions))
    observation_layers = attentions[-num_observation_layers:]

    seq_len = observation_layers[0].shape[-1]
    importance = torch.zeros(seq_len)

    for attn in observation_layers:
        # attn shape: (batch, num_heads, seq_len, seq_len)
        # Sum attention each token RECEIVES from all query positions
        # attn[0, :, q, k] = how much query q attends to key k
        received_attn = attn[0].sum(dim=0).sum(dim=0)  # (seq_len,)
        importance += received_attn.cpu()

    # Normalise
    importance = importance / importance.sum()
    return importance


def prune_kv_cache(
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    importance: torch.Tensor,
    retain_pct: float,
) -> tuple:
    """
    Prune KV-cache by zeroing out low-importance entries.

    Args:
        kv_cache: list of (K, V) per layer
        importance: (seq_len,) importance scores
        retain_pct: percentage of tokens to retain (0-100)

    Returns:
        (pruned_kv_cache, retained_indices, mask)
    """
    seq_len = importance.shape[0]
    num_retain = max(1, int(seq_len * retain_pct / 100))

    # Always keep the first token (BOS) and last few tokens
    num_always_keep = min(3, seq_len)

    # Get top-k indices by importance
    _, topk_indices = torch.topk(importance, min(num_retain, seq_len))
    keep_set = set(topk_indices.tolist())

    # Add always-keep tokens
    for i in range(num_always_keep):
        keep_set.add(i)
    for i in range(max(0, seq_len - 2), seq_len):
        keep_set.add(i)

    # Build mask
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for idx in keep_set:
        mask[idx] = True

    # Apply mask: zero out non-retained positions
    pruned_kv = []
    for k, v in kv_cache:
        k_pruned = k.clone()
        v_pruned = v.clone()
        # k shape: (batch, num_heads, seq_len, head_dim)
        k_pruned[:, :, ~mask, :] = 0.0
        v_pruned[:, :, ~mask, :] = 0.0
        pruned_kv.append((k_pruned, v_pruned))

    return pruned_kv, sorted(keep_set), mask


def list_to_past_kv(kv_list, device):
    """Convert list of (K, V) tuples to DynamicCache for transformers 5.x."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_list):
        # DynamicCache.update expects (key, value, layer_idx)
        cache.update(k.to(device), v.to(device), layer_idx)
    return cache


# ---------------------------------------------------------------------------
# Generation with KV-cache
# ---------------------------------------------------------------------------

def generate_with_kv(model, tokenizer, continuation_text, past_kv, device, max_new_tokens=30):
    """Generate text using a pre-filled KV-cache (greedy decoding)."""
    inputs = tokenize(continuation_text, tokenizer, device)
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()

    # Simple greedy decoding loop
    cache = past_kv
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids[:, -1:] if cache is not None and hasattr(cache, 'get_seq_length') and cache.get_seq_length() > 0 else generated_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        cache = outputs.past_key_values

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode only the new tokens
    new_tokens = generated_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compute_continuation_perplexity(model, tokenizer, continuation_text, past_kv, device):
    """Compute perplexity of continuation given a KV-cache prefix."""
    inputs = tokenize(continuation_text, tokenizer, device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return torch.exp(loss).item()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tradeoff(results: list[dict]):
    """Plot retention % vs perplexity trade-off curve."""
    results_dir = ensure_results_dir()

    pcts = [r["retain_pct"] for r in results]
    ppls = [r["perplexity"] for r in results]
    num_retained = [r["num_retained"] for r in results]

    baseline_ppl = ppls[-1]  # 100% retention

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("KV Retention %", fontsize=12)
    ax1.set_ylabel("Perplexity", color=color1, fontsize=12)
    ax1.plot(pcts, ppls, "b-o", markersize=6, linewidth=2, label="Perplexity")
    ax1.axhline(y=baseline_ppl, color="gray", linestyle="--", alpha=0.5,
                label=f"Baseline (100%): {baseline_ppl:.2f}")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=0)

    # Add perplexity delta annotations
    for i, (pct, ppl) in enumerate(zip(pcts, ppls)):
        delta = ppl - baseline_ppl
        if pct < 100:
            ax1.annotate(
                f"+{delta:.1f}",
                (pct, ppl),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="red" if delta > baseline_ppl * 0.5 else "orange",
            )

    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "Experiment 2: KV Retention % vs Generation Quality\n"
        "(Attention-guided selection, SnapKV-style)",
        fontsize=13,
    )

    plt.tight_layout()
    plot_path = results_dir / "exp2_tradeoff_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Experiment 2: Attention-guided KV Selection")
    print("=" * 60)

    model, tokenizer, device = load_model_and_tokenizer()

    # --- Step 1: Run prefix and extract KV + attention ---
    print(f"\nPrefix: {PREFIX[:60]}...")
    inputs = tokenize(PREFIX, tokenizer, device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prefix tokens: {seq_len}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=True)

    kv_full = extract_kv_cache(outputs)
    attentions = extract_attentions(outputs)

    # --- Step 2: Compute token importance ---
    importance = compute_token_importance(attentions)
    print(f"\nToken importance scores (top 10):")
    topk_vals, topk_idx = torch.topk(importance, min(10, seq_len))
    for val, idx in zip(topk_vals, topk_idx):
        token_text = tokenizer.decode([inputs["input_ids"][0, idx].item()])
        print(f"  pos {idx.item():>3}: {val.item():.4f}  token='{token_text}'")

    # --- Step 3: Test different retention levels ---
    print(f"\nTesting retention levels: {RETENTION_PERCENTAGES}")
    results = []

    for pct in RETENTION_PERCENTAGES:
        pruned_kv, retained_idx, mask = prune_kv_cache(kv_full, importance, pct)
        past_kv = list_to_past_kv(pruned_kv, device)

        # Compute perplexity on continuation
        ppl = compute_continuation_perplexity(
            model, tokenizer, CONTINUATION_PROMPT, past_kv, device,
        )

        # Generate text
        generated = generate_with_kv(
            model, tokenizer, CONTINUATION_PROMPT, past_kv, device,
        )

        num_retained = mask.sum().item()
        result = {
            "retain_pct": pct,
            "num_retained": num_retained,
            "num_total": seq_len,
            "actual_retain_pct": round(num_retained / seq_len * 100, 1),
            "perplexity": round(ppl, 2),
            "generated_text": generated[:200],
        }
        results.append(result)

        print(f"  {pct:>3}% retain ({num_retained:>3}/{seq_len} tokens): "
              f"PPL={ppl:.2f}  gen='{generated[:80]}...'")

    # --- Summary ---
    baseline_ppl = results[-1]["perplexity"]
    print(f"\n{'='*70}")
    print(f"{'Retain%':>8} | {'Tokens':>7} | {'PPL':>8} | {'PPL delta':>9} | {'Quality':>10}")
    print(f"{'-'*70}")
    for r in results:
        delta = r["perplexity"] - baseline_ppl
        quality = "GOOD" if delta < baseline_ppl * 0.1 else (
            "OK" if delta < baseline_ppl * 0.5 else "DEGRADED"
        )
        print(f"{r['retain_pct']:>7}% | {r['num_retained']:>7} | "
              f"{r['perplexity']:>8.2f} | {delta:>+8.2f} | {quality:>10}")
    print(f"{'='*70}")

    # --- Save ---
    save_json({
        "prefix": PREFIX,
        "continuation": CONTINUATION_PROMPT,
        "seq_len": seq_len,
        "retention_results": results,
        "baseline_perplexity": baseline_ppl,
    }, "exp2_attention_selection.json")

    # --- Plot ---
    plot_tradeoff(results)

    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
