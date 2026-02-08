"""
Experiment 1: KV-cache Structure Exploration

Purpose:
    Understand the actual structure, size, and statistical properties of
    KV-cache in a transformer model (TinyLlama-1.1B).

Hypothesis:
    "Different layers of KV-cache have different information densities,
     and exploitable sparse structures exist."

What this does:
    1. Load TinyLlama-1.1B and run inference on a text prompt
    2. Extract per-layer KV-cache tensors
    3. Measure shape, value distribution, sparsity per layer
    4. Run a second similar prompt and compute per-layer KV differences
    5. Visualise results

Output:
    results/exp1_kv_structure.json   - statistics table
    results/exp1_kv_heatmap.png      - layer-wise statistics heatmap
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from utils import (
    load_model_and_tokenizer,
    tokenize,
    extract_kv_cache,
    kv_cache_size_bytes,
    ensure_results_dir,
    save_json,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_A = (
    "A drone flying over a dense forest detects smoke rising from "
    "coordinates 34.2N 118.5W. The smoke appears gray and dense, "
    "suggesting a possible wildfire. Wind direction is northeast."
)

PROMPT_B = (
    "An aerial vehicle surveying a thick woodland observes plumes of "
    "smoke at location 34.2N 118.5W. The smoke is dark and concentrated, "
    "indicating a potential forest fire. Wind blows toward the northeast."
)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyse_layer(
    k: torch.Tensor, v: torch.Tensor, layer_idx: int,
) -> dict:
    """Compute statistics for a single layer's KV-cache."""
    k_flat = k.float().flatten()
    v_flat = v.float().flatten()

    sparsity_threshold = 1e-3

    stats = {
        "layer": layer_idx,
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "k_size_bytes": k.nelement() * k.element_size(),
        "v_size_bytes": v.nelement() * v.element_size(),
        # Key statistics
        "k_mean": k_flat.mean().item(),
        "k_std": k_flat.std().item(),
        "k_min": k_flat.min().item(),
        "k_max": k_flat.max().item(),
        "k_sparsity": (k_flat.abs() < sparsity_threshold).float().mean().item(),
        "k_l2_norm": k_flat.norm(2).item(),
        # Value statistics
        "v_mean": v_flat.mean().item(),
        "v_std": v_flat.std().item(),
        "v_min": v_flat.min().item(),
        "v_max": v_flat.max().item(),
        "v_sparsity": (v_flat.abs() < sparsity_threshold).float().mean().item(),
        "v_l2_norm": v_flat.norm(2).item(),
    }
    return stats


def compute_layer_diff(
    kv_a: list[tuple[torch.Tensor, torch.Tensor]],
    kv_b: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[dict]:
    """
    Compute per-layer difference statistics between two KV-caches.

    Note: This compares KV-caches from DIFFERENT inputs on the SAME model.
    We expect large differences (validating that naive delta is not useful
    for cross-input scenarios).
    """
    assert len(kv_a) == len(kv_b), "Layer count mismatch"

    diff_stats = []
    for layer_idx in range(len(kv_a)):
        k_a, v_a = kv_a[layer_idx]
        k_b, v_b = kv_b[layer_idx]

        # Pad to same seq_len (they might differ by a few tokens)
        min_seq = min(k_a.shape[2], k_b.shape[2])
        k_a = k_a[:, :, :min_seq, :]
        k_b = k_b[:, :, :min_seq, :]
        v_a = v_a[:, :, :min_seq, :]
        v_b = v_b[:, :, :min_seq, :]

        k_diff = (k_a - k_b).float()
        v_diff = (v_a - v_b).float()

        sparsity_threshold = 1e-3

        stats = {
            "layer": layer_idx,
            "k_diff_mean": k_diff.abs().mean().item(),
            "k_diff_std": k_diff.std().item(),
            "k_diff_max": k_diff.abs().max().item(),
            "k_diff_sparsity": (k_diff.abs() < sparsity_threshold).float().mean().item(),
            "k_diff_l2": k_diff.norm(2).item(),
            "v_diff_mean": v_diff.abs().mean().item(),
            "v_diff_std": v_diff.std().item(),
            "v_diff_max": v_diff.abs().max().item(),
            "v_diff_sparsity": (v_diff.abs() < sparsity_threshold).float().mean().item(),
            "v_diff_l2": v_diff.norm(2).item(),
            # Relative difference (normalised by original magnitude)
            "k_relative_diff": (
                k_diff.norm(2) / (k_a.float().norm(2) + 1e-8)
            ).item(),
            "v_relative_diff": (
                v_diff.norm(2) / (v_a.float().norm(2) + 1e-8)
            ).item(),
        }
        diff_stats.append(stats)

    return diff_stats


def plot_results(layer_stats: list[dict], diff_stats: list[dict]):
    """Generate visualisation of KV-cache properties."""
    results_dir = ensure_results_dir()
    num_layers = len(layer_stats)
    layers = list(range(num_layers))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 1: KV-cache Structure Analysis (TinyLlama-1.1B)", fontsize=14)

    # --- Plot 1: L2 norm per layer ---
    ax = axes[0, 0]
    k_norms = [s["k_l2_norm"] for s in layer_stats]
    v_norms = [s["v_l2_norm"] for s in layer_stats]
    ax.plot(layers, k_norms, "b-o", markersize=3, label="Key L2 norm")
    ax.plot(layers, v_norms, "r-s", markersize=3, label="Value L2 norm")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Per-layer KV L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Sparsity per layer ---
    ax = axes[0, 1]
    k_sparse = [s["k_sparsity"] * 100 for s in layer_stats]
    v_sparse = [s["v_sparsity"] * 100 for s in layer_stats]
    ax.bar([l - 0.2 for l in layers], k_sparse, 0.4, label="Key sparsity %", alpha=0.7)
    ax.bar([l + 0.2 for l in layers], v_sparse, 0.4, label="Value sparsity %", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Per-layer Sparsity (|x| < 0.001)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Cross-input diff (L2) ---
    ax = axes[1, 0]
    k_diff_l2 = [s["k_diff_l2"] for s in diff_stats]
    v_diff_l2 = [s["v_diff_l2"] for s in diff_stats]
    ax.plot(layers, k_diff_l2, "b-o", markersize=3, label="Key diff L2")
    ax.plot(layers, v_diff_l2, "r-s", markersize=3, label="Value diff L2")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm of Difference")
    ax.set_title("Cross-input KV Difference (Prompt A vs B)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Relative diff per layer ---
    ax = axes[1, 1]
    k_rel = [s["k_relative_diff"] * 100 for s in diff_stats]
    v_rel = [s["v_relative_diff"] * 100 for s in diff_stats]
    ax.plot(layers, k_rel, "b-o", markersize=3, label="Key relative diff %")
    ax.plot(layers, v_rel, "r-s", markersize=3, label="Value relative diff %")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relative Difference (%)")
    ax.set_title("Cross-input Relative KV Difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / "exp1_kv_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Experiment 1: KV-cache Structure Exploration")
    print("=" * 60)

    model, tokenizer, device = load_model_and_tokenizer()

    # --- Run prompt A ---
    print(f"\nPrompt A: {PROMPT_A[:60]}...")
    inputs_a = tokenize(PROMPT_A, tokenizer, device)
    with torch.no_grad():
        out_a = model(**inputs_a, use_cache=True, output_attentions=True)
    kv_a = extract_kv_cache(out_a)

    total_bytes = kv_cache_size_bytes(kv_a)
    num_tokens = inputs_a["input_ids"].shape[1]
    num_layers = len(kv_a)

    print(f"  Tokens: {num_tokens}")
    print(f"  Layers: {num_layers}")
    print(f"  Total KV-cache size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    print(f"  Per-token KV-cache: {total_bytes / num_tokens:,.0f} bytes")

    # --- Analyse each layer ---
    print("\nAnalysing per-layer statistics...")
    layer_stats = []
    for i, (k, v) in enumerate(kv_a):
        stats = analyse_layer(k, v, i)
        layer_stats.append(stats)

    # --- Run prompt B ---
    print(f"\nPrompt B: {PROMPT_B[:60]}...")
    inputs_b = tokenize(PROMPT_B, tokenizer, device)
    with torch.no_grad():
        out_b = model(**inputs_b, use_cache=True, output_attentions=True)
    kv_b = extract_kv_cache(out_b)

    # --- Cross-input diff ---
    print("\nComputing cross-input KV differences...")
    diff_stats = compute_layer_diff(kv_a, kv_b)

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print(f"{'Layer':>5} | {'K std':>8} | {'V std':>8} | "
          f"{'K sparse%':>9} | {'V sparse%':>9} | "
          f"{'K rel diff%':>11} | {'V rel diff%':>11}")
    print("-" * 80)
    for ls, ds in zip(layer_stats, diff_stats):
        print(f"{ls['layer']:>5} | {ls['k_std']:>8.4f} | {ls['v_std']:>8.4f} | "
              f"{ls['k_sparsity']*100:>8.2f}% | {ls['v_sparsity']*100:>8.2f}% | "
              f"{ds['k_relative_diff']*100:>10.2f}% | {ds['v_relative_diff']*100:>10.2f}%")
    print("=" * 80)

    # --- Save results ---
    results = {
        "model": str(model.config._name_or_path),
        "num_layers": num_layers,
        "num_tokens_a": num_tokens,
        "num_tokens_b": inputs_b["input_ids"].shape[1],
        "total_kv_bytes": total_bytes,
        "per_token_kv_bytes": total_bytes / num_tokens,
        "prompt_a": PROMPT_A,
        "prompt_b": PROMPT_B,
        "layer_stats": layer_stats,
        "diff_stats": diff_stats,
    }
    save_json(results, "exp1_kv_structure.json")

    # --- Plot ---
    plot_results(layer_stats, diff_stats)

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
