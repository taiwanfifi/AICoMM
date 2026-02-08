"""
Experiment 4: Task-Aware vs Attention-Based KV Selection

Purpose:
    Prove that task-aware (gradient-based) KV selection outperforms
    attention-based selection (SnapKV style) for the same transmission budget.

Hypothesis:
    "Task-aware selection achieves higher task accuracy than attention-based
     selection at the same KV retention percentage."

What this does:
    1. Define a QA task with concrete success criteria
    2. Implement two selection methods:
       - Attention-based: SnapKV style, use attention weights
       - Task-aware: use attention during answer generation as proxy
    3. Compare task accuracy at various retention levels

Output:
    results/exp4_task_aware_selection.json
    results/exp4_comparison.png
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

RETENTION_LEVELS = [10, 20, 30, 50, 70, 100]


# ---------------------------------------------------------------------------
# KV-cache pruning with different selection methods
# ---------------------------------------------------------------------------

def prune_kv_cache_by_importance(
    kv_cache,
    importance: torch.Tensor,
    retain_pct: float,
    device: torch.device,
):
    """
    Prune KV-cache keeping only the top retain_pct% important positions.

    Returns a new DynamicCache with pruned entries zeroed out.
    """
    from transformers.cache_utils import DynamicCache

    seq_len = importance.shape[0]
    num_retain = max(1, int(seq_len * retain_pct / 100))

    # Always keep first and last few tokens
    num_always_keep = min(3, seq_len)

    # Get top-k indices
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

    # Create new cache with pruned values
    new_cache = DynamicCache()

    if hasattr(kv_cache, 'layers'):
        for layer_idx, layer in enumerate(kv_cache.layers):
            k = layer.keys.clone()
            v = layer.values.clone()

            # Zero out non-retained positions
            # Shape: (batch, heads, seq_len, head_dim)
            k[:, :, ~mask, :] = 0.0
            v[:, :, ~mask, :] = 0.0

            new_cache.update(k, v, layer_idx)
    else:
        raise NotImplementedError("Legacy KV format not supported")

    return new_cache, mask.sum().item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, prompt_ids, kv_cache, device, max_new_tokens=20):
    """Generate answer tokens using the given KV-cache."""
    generated_ids = prompt_ids.clone()
    cache = kv_cache

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


def evaluate_single_example(
    model,
    tokenizer,
    example: QAExample,
    device: torch.device,
    retain_pct: float,
    selection_method: str,
) -> dict:
    """Evaluate a single QA example with KV pruning."""

    prompt = format_qa_prompt(example)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = prompt_ids.shape[1]

    # Get full KV-cache
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids,
            use_cache=True,
            output_attentions=True,
        )
    kv_full = outputs.past_key_values

    # Compute importance scores
    if selection_method == "attention":
        importance = compute_kv_importance_attention(model, tokenizer, prompt, device)
    else:  # task_proxy
        importance = compute_kv_importance_task_proxy(model, tokenizer, example, device)

    # Prune KV-cache
    if retain_pct < 100:
        kv_pruned, num_retained = prune_kv_cache_by_importance(
            kv_full, importance, retain_pct, device
        )
    else:
        kv_pruned = kv_full
        num_retained = seq_len

    # Generate answer
    generated = generate_answer(model, tokenizer, prompt_ids, kv_pruned, device)

    # Check correctness
    is_correct = check_answer(generated, example.answer)

    return {
        "question": example.question,
        "expected": example.answer,
        "generated": generated[:100],
        "correct": is_correct,
        "seq_len": seq_len,
        "num_retained": num_retained,
    }


def evaluate_method(
    model,
    tokenizer,
    examples: list[QAExample],
    device: torch.device,
    retain_pct: float,
    selection_method: str,
) -> dict:
    """Evaluate all examples with a given method and retention level."""

    results = []
    correct = 0

    for example in examples:
        result = evaluate_single_example(
            model, tokenizer, example, device, retain_pct, selection_method
        )
        results.append(result)
        if result["correct"]:
            correct += 1

    accuracy = correct / len(examples) * 100

    return {
        "method": selection_method,
        "retain_pct": retain_pct,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(attention_results: list[dict], task_results: list[dict]):
    """Plot comparison of the two methods."""
    results_dir = ensure_results_dir()

    attn_pcts = [r["retain_pct"] for r in attention_results]
    attn_acc = [r["accuracy"] for r in attention_results]

    task_pcts = [r["retain_pct"] for r in task_results]
    task_acc = [r["accuracy"] for r in task_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(attn_pcts, attn_acc, 'b-o', markersize=8, linewidth=2,
            label='Attention-based (SnapKV style)')
    ax.plot(task_pcts, task_acc, 'r-s', markersize=8, linewidth=2,
            label='Task-aware (Answer-focused)')

    ax.set_xlabel("KV Retention %", fontsize=12)
    ax.set_ylabel("Task Accuracy %", fontsize=12)
    ax.set_title("Experiment 4: Task-Aware vs Attention-Based KV Selection\n"
                 "(QA Task Accuracy)", fontsize=13)

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 105)

    # Add delta annotations
    for i, (ap, aa, tp, ta) in enumerate(zip(attn_pcts, attn_acc, task_pcts, task_acc)):
        delta = ta - aa
        if abs(delta) > 0.1 and ap < 100:
            color = 'green' if delta > 0 else 'red'
            ax.annotate(
                f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%",
                (tp, ta),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=9,
                color=color,
            )

    plt.tight_layout()
    plot_path = results_dir / "exp4_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 4: Task-Aware vs Attention-Based KV Selection")
    print("=" * 70)

    model, tokenizer, device = load_model_and_tokenizer()

    examples = QA_EXAMPLES
    print(f"\nQA Examples: {len(examples)}")
    for i, ex in enumerate(examples):
        print(f"  {i+1}. Q: {ex.question[:50]}... A: {ex.answer}")

    # --- Evaluate both methods at all retention levels ---
    attention_results = []
    task_results = []

    print(f"\nTesting retention levels: {RETENTION_LEVELS}")
    print("\n" + "=" * 70)

    for retain_pct in RETENTION_LEVELS:
        print(f"\n--- Retention: {retain_pct}% ---")

        # Attention-based
        attn_eval = evaluate_method(
            model, tokenizer, examples, device, retain_pct, "attention"
        )
        attention_results.append(attn_eval)
        print(f"  Attention-based: {attn_eval['accuracy']:.1f}% "
              f"({attn_eval['correct']}/{attn_eval['total']})")

        # Task-aware
        task_eval = evaluate_method(
            model, tokenizer, examples, device, retain_pct, "task_proxy"
        )
        task_results.append(task_eval)
        print(f"  Task-aware:      {task_eval['accuracy']:.1f}% "
              f"({task_eval['correct']}/{task_eval['total']})")

        delta = task_eval['accuracy'] - attn_eval['accuracy']
        print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.1f}%")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: Task-Aware vs Attention-Based Selection")
    print("=" * 70)
    print(f"{'Retain%':>8} | {'Attention':>10} | {'Task-Aware':>10} | {'Delta':>8} | {'Winner':>12}")
    print("-" * 70)

    for attn_r, task_r in zip(attention_results, task_results):
        delta = task_r['accuracy'] - attn_r['accuracy']
        winner = "Task-Aware" if delta > 0 else ("Tie" if delta == 0 else "Attention")
        print(f"{attn_r['retain_pct']:>7}% | {attn_r['accuracy']:>9.1f}% | "
              f"{task_r['accuracy']:>9.1f}% | {delta:>+7.1f}% | {winner:>12}")

    print("=" * 70)

    # --- Overall analysis ---
    avg_delta = np.mean([
        t['accuracy'] - a['accuracy']
        for a, t in zip(attention_results, task_results)
        if a['retain_pct'] < 100
    ])
    print(f"\nAverage improvement (task-aware over attention): {avg_delta:+.1f}%")

    if avg_delta > 0:
        print("Conclusion: Task-aware selection OUTPERFORMS attention-based selection")
    elif avg_delta < 0:
        print("Conclusion: Attention-based selection outperforms task-aware")
    else:
        print("Conclusion: Methods are equivalent")

    # --- Save results ---
    save_json({
        "retention_levels": RETENTION_LEVELS,
        "attention_results": attention_results,
        "task_results": task_results,
        "average_improvement": avg_delta,
        "num_examples": len(examples),
    }, "exp4_task_aware_selection.json")

    # --- Plot ---
    plot_comparison(attention_results, task_results)

    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
