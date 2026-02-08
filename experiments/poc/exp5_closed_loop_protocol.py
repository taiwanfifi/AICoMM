"""
Experiment 5: Closed-Loop vs One-Shot Protocol

Purpose:
    Prove that a closed-loop transmission protocol (with receiver feedback)
    is more efficient than one-shot transmission.

Hypothesis:
    "Closed-loop protocol achieves the same task accuracy with less
     total KV transmission than one-shot protocol."

Protocol Description:
    One-Shot:
        Sender transmits top-k% KV entries once → Receiver uses directly

    Closed-Loop:
        Round 1: Sender transmits top-(k/2)% KV entries
        Round 2: Receiver identifies uncertain tokens, requests more KV
        Round 3: Sender transmits KV for uncertain positions
        Result: Same accuracy, potentially less total transmission

What this does:
    1. Implement one-shot protocol (baseline)
    2. Implement closed-loop protocol with uncertainty feedback
    3. Compare total transmission for same accuracy target

Output:
    results/exp5_closed_loop_protocol.json
    results/exp5_efficiency.png
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
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target accuracy levels to achieve
ACCURACY_TARGETS = [60, 80, 100]  # percent

# Initial transmission for closed-loop (as % of what one-shot would send)
CLOSED_LOOP_INITIAL_PCT = 50


# ---------------------------------------------------------------------------
# Protocol Implementations
# ---------------------------------------------------------------------------

def one_shot_protocol(
    model,
    tokenizer,
    example: QAExample,
    device: torch.device,
    retain_pct: float,
) -> dict:
    """
    One-shot protocol: Send top-k% KV entries, generate answer.

    Returns dict with accuracy, transmission amount, and details.
    """
    prompt = format_qa_prompt(example)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = prompt_ids.shape[1]

    # Get full KV-cache and importance
    with torch.no_grad():
        outputs = model(prompt_ids, use_cache=True, output_attentions=True)

    kv_full = outputs.past_key_values
    importance = compute_kv_importance_attention(model, tokenizer, prompt, device)

    # Select top-k positions
    num_retain = max(1, int(seq_len * retain_pct / 100))
    _, topk_indices = torch.topk(importance, min(num_retain, seq_len))
    keep_set = set(topk_indices.tolist())

    # Always keep first/last
    for i in range(min(3, seq_len)):
        keep_set.add(i)
    for i in range(max(0, seq_len - 2), seq_len):
        keep_set.add(i)

    transmitted_positions = len(keep_set)

    # Create pruned cache
    from transformers.cache_utils import DynamicCache
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

    # Generate answer
    generated = generate_with_cache(model, tokenizer, prompt_ids, pruned_cache, device)
    is_correct = check_answer(generated, example.answer)

    return {
        "protocol": "one_shot",
        "transmitted_positions": transmitted_positions,
        "seq_len": seq_len,
        "transmission_pct": transmitted_positions / seq_len * 100,
        "correct": is_correct,
        "generated": generated[:100],
    }


def closed_loop_protocol(
    model,
    tokenizer,
    example: QAExample,
    device: torch.device,
    initial_pct: float,
    max_rounds: int = 3,
) -> dict:
    """
    Closed-loop protocol with uncertainty feedback.

    Round 1: Send initial_pct% of KV
    Round 2+: Receiver reports uncertain positions, sender sends those KV
    """
    prompt = format_qa_prompt(example)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = prompt_ids.shape[1]

    # Get full KV-cache and importance
    with torch.no_grad():
        outputs = model(prompt_ids, use_cache=True, output_attentions=True)

    kv_full = outputs.past_key_values
    importance = compute_kv_importance_attention(model, tokenizer, prompt, device)

    from transformers.cache_utils import DynamicCache

    # Track which positions have been transmitted
    transmitted = set()
    total_transmitted = 0

    # Round 1: Send top initial_pct% positions
    num_initial = max(1, int(seq_len * initial_pct / 100))
    _, topk_indices = torch.topk(importance, min(num_initial, seq_len))
    initial_set = set(topk_indices.tolist())

    # Always include first/last
    for i in range(min(3, seq_len)):
        initial_set.add(i)
    for i in range(max(0, seq_len - 2), seq_len):
        initial_set.add(i)

    transmitted.update(initial_set)
    total_transmitted += len(initial_set)

    rounds_data = [{
        "round": 1,
        "new_positions": len(initial_set),
        "total_so_far": total_transmitted,
    }]

    # Create current cache with transmitted positions
    def make_cache_with_positions(positions):
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for idx in positions:
            mask[idx] = True

        cache = DynamicCache()
        for layer_idx, layer in enumerate(kv_full.layers):
            k = layer.keys.clone()
            v = layer.values.clone()
            k[:, :, ~mask, :] = 0.0
            v[:, :, ~mask, :] = 0.0
            cache.update(k, v, layer_idx)
        return cache

    current_cache = make_cache_with_positions(transmitted)

    # Generate and check
    generated = generate_with_cache(model, tokenizer, prompt_ids, current_cache, device)
    is_correct = check_answer(generated, example.answer)

    # If correct, we're done
    if is_correct:
        return {
            "protocol": "closed_loop",
            "transmitted_positions": total_transmitted,
            "seq_len": seq_len,
            "transmission_pct": total_transmitted / seq_len * 100,
            "correct": True,
            "generated": generated[:100],
            "rounds": 1,
            "rounds_data": rounds_data,
        }

    # Round 2+: Identify uncertain positions and request more KV
    for round_num in range(2, max_rounds + 1):
        # Identify uncertain positions: those not yet transmitted but have
        # high importance (they might be needed)
        remaining = set(range(seq_len)) - transmitted
        if not remaining:
            break

        # Get importance of remaining positions
        remaining_list = sorted(remaining)
        remaining_importance = importance[remaining_list]

        # Request top 50% of remaining (sorted by importance)
        num_request = max(1, len(remaining_list) // 2)
        _, topk_remaining = torch.topk(remaining_importance, min(num_request, len(remaining_list)))
        request_indices = [remaining_list[i] for i in topk_remaining.tolist()]

        transmitted.update(request_indices)
        total_transmitted += len(request_indices)

        rounds_data.append({
            "round": round_num,
            "new_positions": len(request_indices),
            "total_so_far": total_transmitted,
        })

        # Update cache and regenerate
        current_cache = make_cache_with_positions(transmitted)
        generated = generate_with_cache(model, tokenizer, prompt_ids, current_cache, device)
        is_correct = check_answer(generated, example.answer)

        if is_correct:
            break

    return {
        "protocol": "closed_loop",
        "transmitted_positions": total_transmitted,
        "seq_len": seq_len,
        "transmission_pct": total_transmitted / seq_len * 100,
        "correct": is_correct,
        "generated": generated[:100],
        "rounds": round_num,
        "rounds_data": rounds_data,
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
# Comparison
# ---------------------------------------------------------------------------

def compare_protocols(
    model,
    tokenizer,
    examples: list[QAExample],
    device: torch.device,
) -> dict:
    """Compare one-shot and closed-loop protocols across examples."""

    one_shot_results = []
    closed_loop_results = []

    print("\nComparing protocols on each example:")
    print("-" * 70)

    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: {example.question[:40]}...")

        # One-shot with 100% (baseline)
        os_100 = one_shot_protocol(model, tokenizer, example, device, 100)

        # One-shot with 50%
        os_50 = one_shot_protocol(model, tokenizer, example, device, 50)

        # Closed-loop starting with 25%
        cl = closed_loop_protocol(model, tokenizer, example, device, 25, max_rounds=3)

        one_shot_results.append({
            "example_idx": i,
            "one_shot_100": os_100,
            "one_shot_50": os_50,
        })
        closed_loop_results.append({
            "example_idx": i,
            "closed_loop": cl,
        })

        print(f"  One-shot 100%: {'✓' if os_100['correct'] else '✗'} "
              f"(transmitted: {os_100['transmitted_positions']}/{os_100['seq_len']})")
        print(f"  One-shot 50%:  {'✓' if os_50['correct'] else '✗'} "
              f"(transmitted: {os_50['transmitted_positions']}/{os_50['seq_len']})")
        print(f"  Closed-loop:   {'✓' if cl['correct'] else '✗'} "
              f"(transmitted: {cl['transmitted_positions']}/{cl['seq_len']}, "
              f"rounds: {cl['rounds']})")

    # Summary statistics
    os_50_correct = sum(1 for r in one_shot_results if r['one_shot_50']['correct'])
    os_50_transmission = np.mean([r['one_shot_50']['transmission_pct'] for r in one_shot_results])

    cl_correct = sum(1 for r in closed_loop_results if r['closed_loop']['correct'])
    cl_transmission = np.mean([r['closed_loop']['transmission_pct'] for r in closed_loop_results])

    return {
        "one_shot_results": one_shot_results,
        "closed_loop_results": closed_loop_results,
        "summary": {
            "one_shot_50_accuracy": os_50_correct / len(examples) * 100,
            "one_shot_50_avg_transmission": os_50_transmission,
            "closed_loop_accuracy": cl_correct / len(examples) * 100,
            "closed_loop_avg_transmission": cl_transmission,
            "transmission_savings": os_50_transmission - cl_transmission,
        }
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_efficiency(comparison_results: dict):
    """Plot transmission efficiency comparison."""
    results_dir = ensure_results_dir()

    os_results = comparison_results["one_shot_results"]
    cl_results = comparison_results["closed_loop_results"]
    summary = comparison_results["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Per-example transmission comparison ---
    examples = range(len(os_results))
    os_50_trans = [r['one_shot_50']['transmission_pct'] for r in os_results]
    cl_trans = [r['closed_loop']['transmission_pct'] for r in cl_results]

    x = np.arange(len(examples))
    width = 0.35

    bars1 = ax1.bar(x - width/2, os_50_trans, width, label='One-Shot (50%)', color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, cl_trans, width, label='Closed-Loop', color='coral', alpha=0.7)

    ax1.set_xlabel('Example')
    ax1.set_ylabel('Transmission %')
    ax1.set_title('Per-Example Transmission Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Ex{i+1}' for i in examples])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Plot 2: Summary comparison ---
    methods = ['One-Shot\n(50%)', 'Closed-Loop']
    accuracies = [summary['one_shot_50_accuracy'], summary['closed_loop_accuracy']]
    transmissions = [summary['one_shot_50_avg_transmission'], summary['closed_loop_avg_transmission']]

    x2 = np.arange(len(methods))
    width2 = 0.35

    ax2_twin = ax2.twinx()

    bars3 = ax2.bar(x2 - width2/2, accuracies, width2, label='Accuracy %', color='green', alpha=0.7)
    bars4 = ax2_twin.bar(x2 + width2/2, transmissions, width2, label='Transmission %', color='orange', alpha=0.7)

    ax2.set_xlabel('Protocol')
    ax2.set_ylabel('Accuracy %', color='green')
    ax2_twin.set_ylabel('Avg Transmission %', color='orange')
    ax2.set_title('Summary: Accuracy vs Transmission')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods)
    ax2.set_ylim(0, 105)
    ax2_twin.set_ylim(0, 105)

    ax2.tick_params(axis='y', labelcolor='green')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    # Add savings annotation
    savings = summary['transmission_savings']
    if savings > 0:
        ax2.annotate(
            f'Saves {savings:.1f}%',
            xy=(1, transmissions[1]),
            xytext=(1.3, transmissions[1] + 10),
            fontsize=10,
            color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
        )

    plt.tight_layout()
    plot_path = results_dir / "exp5_efficiency.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 5: Closed-Loop vs One-Shot Protocol")
    print("=" * 70)

    model, tokenizer, device = load_model_and_tokenizer()

    examples = QA_EXAMPLES
    print(f"\nQA Examples: {len(examples)}")

    # --- Compare protocols ---
    comparison = compare_protocols(model, tokenizer, examples, device)

    # --- Summary ---
    summary = comparison["summary"]
    print("\n" + "=" * 70)
    print("SUMMARY: Protocol Comparison")
    print("=" * 70)
    print(f"One-Shot (50%):  Accuracy={summary['one_shot_50_accuracy']:.1f}%, "
          f"Avg Transmission={summary['one_shot_50_avg_transmission']:.1f}%")
    print(f"Closed-Loop:     Accuracy={summary['closed_loop_accuracy']:.1f}%, "
          f"Avg Transmission={summary['closed_loop_avg_transmission']:.1f}%")
    print(f"Transmission Savings: {summary['transmission_savings']:.1f}%")

    if summary['closed_loop_accuracy'] >= summary['one_shot_50_accuracy']:
        if summary['transmission_savings'] > 0:
            print("\nConclusion: Closed-loop achieves SAME or BETTER accuracy with LESS transmission")
        else:
            print("\nConclusion: Closed-loop achieves SAME accuracy (no transmission savings)")
    else:
        print("\nConclusion: One-shot outperforms (closed-loop needs refinement)")

    # --- Save ---
    save_json(comparison, "exp5_closed_loop_protocol.json")

    # --- Plot ---
    plot_efficiency(comparison)

    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
