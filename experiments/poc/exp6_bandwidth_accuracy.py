"""
Experiment 6: Bandwidth-Accuracy Trade-off Curve

Purpose:
    Generate a comprehensive comparison of different methods across
    varying bandwidth budgets. This produces the key figure for the paper.

What this does:
    1. Test multiple methods at various retention levels (10%, 20%, ..., 100%)
    2. Methods compared:
       - Full KV (upper bound)
       - Text-only baseline (lower bound)
       - Attention-based selection (SnapKV style)
       - Task-aware selection (our method)
    3. Plot Pareto frontier curves

Output:
    results/exp6_bandwidth_accuracy.json
    results/exp6_pareto_curves.png  - Main figure for paper
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_model_and_tokenizer,
    tokenize,
    ensure_results_dir,
    save_json,
)

from task_qa import (
    QA_EXAMPLES,
    QAExample,
    format_qa_prompt,
    check_answer,
    compute_kv_importance_attention,
    compute_kv_importance_task_proxy,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETENTION_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def evaluate_full_kv(model, tokenizer, examples, device) -> dict:
    """Baseline: Use full KV-cache (100% transmission)."""
    correct = 0
    for example in examples:
        prompt = format_qa_prompt(example)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(prompt_ids, use_cache=True)
        kv = outputs.past_key_values

        generated = generate_with_cache(model, tokenizer, prompt_ids, kv, device)
        if check_answer(generated, example.answer):
            correct += 1

    return {
        "method": "Full KV",
        "retention_pct": 100,
        "accuracy": correct / len(examples) * 100,
    }


def evaluate_text_only(model, tokenizer, examples, device) -> dict:
    """
    Baseline: Text-only communication.

    Simulates sending only the question (not the context's KV).
    This represents the lower bound of what's achievable.
    """
    correct = 0
    for example in examples:
        # Only use question, not full context
        short_prompt = f"Question: {example.question}\n\nAnswer:"
        prompt_ids = tokenizer(short_prompt, return_tensors="pt")["input_ids"].to(device)

        generated = generate_with_cache(model, tokenizer, prompt_ids, None, device)
        if check_answer(generated, example.answer):
            correct += 1

    return {
        "method": "Text-only",
        "retention_pct": 0,
        "accuracy": correct / len(examples) * 100,
    }


def evaluate_with_selection(
    model,
    tokenizer,
    examples: list[QAExample],
    device: torch.device,
    retain_pct: float,
    selection_method: str,
) -> dict:
    """
    Evaluate using KV selection method.

    selection_method: "attention" or "task_aware"
    """
    from transformers.cache_utils import DynamicCache

    correct = 0
    for example in examples:
        prompt = format_qa_prompt(example)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        seq_len = prompt_ids.shape[1]

        # Get full KV
        with torch.no_grad():
            outputs = model(prompt_ids, use_cache=True, output_attentions=True)
        kv_full = outputs.past_key_values

        # Get importance scores
        if selection_method == "attention":
            importance = compute_kv_importance_attention(model, tokenizer, prompt, device)
        else:
            importance = compute_kv_importance_task_proxy(model, tokenizer, example, device)

        # Select positions
        num_retain = max(1, int(seq_len * retain_pct / 100))
        _, topk_indices = torch.topk(importance, min(num_retain, seq_len))
        keep_set = set(topk_indices.tolist())

        # Always keep first/last
        for i in range(min(3, seq_len)):
            keep_set.add(i)
        for i in range(max(0, seq_len - 2), seq_len):
            keep_set.add(i)

        # Create pruned cache
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for idx in keep_set:
            mask[idx] = True

        pruned_cache = DynamicCache()
        for layer_idx, layer in enumerate(kv_full.layers):
            k = layer.keys.clone()
            v = layer.values.clone()
            k[:, :, ~mask, :] = 0.0
            v[:, :, ~mask, :] = 0.0
            pruned_cache.update(k, v, layer_idx)

        # Generate
        generated = generate_with_cache(model, tokenizer, prompt_ids, pruned_cache, device)
        if check_answer(generated, example.answer):
            correct += 1

    method_name = "Attention-based" if selection_method == "attention" else "Task-aware (Ours)"
    return {
        "method": method_name,
        "retention_pct": retain_pct,
        "accuracy": correct / len(examples) * 100,
    }


def generate_with_cache(model, tokenizer, prompt_ids, cache, device, max_new_tokens=20):
    """Generate answer tokens using the given KV-cache."""
    generated_ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cache is not None and hasattr(cache, 'get_seq_length') and cache.get_seq_length() > 0:
                inp = generated_ids[:, -1:]
            else:
                inp = generated_ids
            outputs = model(input_ids=inp, past_key_values=cache, use_cache=True)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        cache = outputs.past_key_values

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(
        generated_ids[0, prompt_ids.shape[1]:],
        skip_special_tokens=True
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_all_evaluations(model, tokenizer, examples, device) -> dict:
    """Run all methods at all retention levels."""
    results = {
        "baselines": [],
        "attention_based": [],
        "task_aware": [],
    }

    print("\n--- Baselines ---")

    # Full KV baseline
    full_kv = evaluate_full_kv(model, tokenizer, examples, device)
    results["baselines"].append(full_kv)
    print(f"Full KV (100%): {full_kv['accuracy']:.1f}%")

    # Text-only baseline
    text_only = evaluate_text_only(model, tokenizer, examples, device)
    results["baselines"].append(text_only)
    print(f"Text-only (0%): {text_only['accuracy']:.1f}%")

    print("\n--- Attention-based Selection ---")
    for pct in RETENTION_LEVELS:
        result = evaluate_with_selection(
            model, tokenizer, examples, device, pct, "attention"
        )
        results["attention_based"].append(result)
        print(f"  {pct:>3}%: {result['accuracy']:.1f}%")

    print("\n--- Task-aware Selection (Ours) ---")
    for pct in RETENTION_LEVELS:
        result = evaluate_with_selection(
            model, tokenizer, examples, device, pct, "task_aware"
        )
        results["task_aware"].append(result)
        print(f"  {pct:>3}%: {result['accuracy']:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto_curves(results: dict):
    """Generate the main paper figure: Pareto curves for all methods."""
    results_dir = ensure_results_dir()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Baselines
    full_kv = results["baselines"][0]
    text_only = results["baselines"][1]

    # Attention-based
    attn_pcts = [r["retention_pct"] for r in results["attention_based"]]
    attn_accs = [r["accuracy"] for r in results["attention_based"]]

    # Task-aware
    task_pcts = [r["retention_pct"] for r in results["task_aware"]]
    task_accs = [r["accuracy"] for r in results["task_aware"]]

    # Plot
    ax.axhline(y=full_kv["accuracy"], color='gray', linestyle='--', linewidth=1.5,
               label=f'Full KV (upper bound): {full_kv["accuracy"]:.0f}%')
    ax.axhline(y=text_only["accuracy"], color='gray', linestyle=':', linewidth=1.5,
               label=f'Text-only (lower bound): {text_only["accuracy"]:.0f}%')

    ax.plot(attn_pcts, attn_accs, 'b-o', markersize=7, linewidth=2,
            label='Attention-based (SnapKV style)')
    ax.plot(task_pcts, task_accs, 'r-s', markersize=7, linewidth=2,
            label='Task-aware (TOKP, Ours)')

    # Highlight improvements
    for i, (ap, aa, tp, ta) in enumerate(zip(attn_pcts, attn_accs, task_pcts, task_accs)):
        if ta > aa and ap < 100:
            ax.annotate(
                '',
                xy=(tp, ta), xytext=(ap, aa),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
            )

    ax.set_xlabel("KV-cache Transmission %", fontsize=12)
    ax.set_ylabel("Task Accuracy %", fontsize=12)
    ax.set_title("Bandwidth-Accuracy Trade-off: Task-Oriented KV Protocol\n"
                 "(QA Task on GPT-2)", fontsize=14)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    # Add efficiency region annotation
    ax.fill_between(
        task_pcts, attn_accs, task_accs,
        where=[ta > aa for ta, aa in zip(task_accs, attn_accs)],
        alpha=0.2, color='green',
        label='TOKP advantage'
    )

    plt.tight_layout()
    plot_path = results_dir / "exp6_pareto_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()


def print_summary_table(results: dict):
    """Print a summary table comparing methods."""
    print("\n" + "=" * 80)
    print("SUMMARY: Bandwidth-Accuracy Trade-off")
    print("=" * 80)

    attn = {r["retention_pct"]: r["accuracy"] for r in results["attention_based"]}
    task = {r["retention_pct"]: r["accuracy"] for r in results["task_aware"]}

    print(f"{'Retention%':>10} | {'Attention':>10} | {'Task-aware':>10} | {'Delta':>8} | {'Winner':>12}")
    print("-" * 80)

    total_delta = 0
    count = 0

    for pct in RETENTION_LEVELS:
        a = attn.get(pct, 0)
        t = task.get(pct, 0)
        delta = t - a
        winner = "Task-aware" if delta > 0 else ("Tie" if delta == 0 else "Attention")

        print(f"{pct:>9}% | {a:>9.1f}% | {t:>9.1f}% | {delta:>+7.1f}% | {winner:>12}")

        if pct < 100:
            total_delta += delta
            count += 1

    print("=" * 80)
    avg_delta = total_delta / count if count > 0 else 0
    print(f"Average improvement (task-aware over attention, excl. 100%): {avg_delta:+.1f}%")

    # Area under curve comparison
    # Use np.trapz for older numpy, np.trapezoid for numpy 2.0+
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    attn_auc = trapz_func([attn[p] for p in RETENTION_LEVELS], RETENTION_LEVELS)
    task_auc = trapz_func([task[p] for p in RETENTION_LEVELS], RETENTION_LEVELS)
    print(f"Area under curve - Attention: {attn_auc:.0f}, Task-aware: {task_auc:.0f}")
    print(f"AUC improvement: {(task_auc - attn_auc) / attn_auc * 100:+.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 6: Bandwidth-Accuracy Trade-off Curve")
    print("=" * 70)

    model, tokenizer, device = load_model_and_tokenizer()

    examples = QA_EXAMPLES
    print(f"\nQA Examples: {len(examples)}")
    print(f"Retention levels: {RETENTION_LEVELS}")

    # --- Run all evaluations ---
    results = run_all_evaluations(model, tokenizer, examples, device)

    # --- Summary ---
    print_summary_table(results)

    # --- Save ---
    save_json({
        "retention_levels": RETENTION_LEVELS,
        "results": results,
        "num_examples": len(examples),
    }, "exp6_bandwidth_accuracy.json")

    # --- Plot ---
    plot_pareto_curves(results)

    print("\nExperiment 6 complete.")
    print("Main figure saved to: results/exp6_pareto_curves.png")


if __name__ == "__main__":
    main()
