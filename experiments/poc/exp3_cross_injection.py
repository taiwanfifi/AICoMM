"""
Experiment 3: Cross-instance KV-cache Injection

Purpose:
    Verify that KV-cache from one model instance can be injected into
    another instance of the same architecture without quality loss.

Hypothesis:
    "Cross-instance KV-cache injection between identical models produces
     zero perplexity gap (PPL_injected ≈ PPL_native)."

What this does:
    1. Instance A processes a prompt prefix, extracts KV-cache
    2. Instance B starts from empty state, receives A's KV-cache
    3. B generates continuation using injected KV-cache
    4. Compare B's output vs B processing the prefix itself (baseline)
    5. Measure perplexity gap and text similarity

Output:
    results/exp3_cross_injection.json  - numerical results
    results/exp3_comparison.png        - visual comparison
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
    extract_kv_cache,
    kv_cache_size_bytes,
    ensure_results_dir,
    save_json,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Prefix that Instance A will process
PREFIX = (
    "The autonomous vehicle detected a pedestrian crossing the road "
    "at coordinates (25.033, 121.565). Current speed is 45 km/h. "
    "The pedestrian is moving from left to right at approximately "
    "5 km/h. Distance to pedestrian is 30 meters. Road conditions "
    "are wet due to recent rainfall. Visibility is moderate."
)

# Continuation text for evaluation
CONTINUATION = (
    " The recommended driving action based on this situation is to"
)

# Multiple prefixes to test robustness
TEST_PREFIXES = [
    # Short prefix
    "A fire has been detected in sector 7. Smoke density is high.",
    # Medium prefix
    (
        "The warehouse monitoring system reports the following anomalies: "
        "Temperature in zone B has risen from 22C to 45C over the past "
        "10 minutes. Humidity sensors show a sharp drop from 65% to 30%. "
        "Motion detectors in adjacent zones show no activity."
    ),
    # Long prefix
    (
        "Mission briefing for drone swarm operation Delta-7: The target "
        "area is a 5 square kilometer forest region in the northern sector. "
        "Three drones (D1, D2, D3) are deployed in triangular formation. "
        "D1 reports thermal anomaly at grid reference Alpha-3. D2 confirms "
        "visual contact with smoke plume at bearing 045 degrees. D3 is "
        "maintaining overwatch at altitude 200 meters. Weather conditions: "
        "wind speed 20 km/h from the west, temperature 32C, relative "
        "humidity 25%. The command center requires a coordinated assessment "
        "of the situation with confidence levels for fire detection."
    ),
]

TEST_CONTINUATIONS = [
    " The recommended response is",
    " Based on these readings, the system should",
    " The coordinated assessment from the drone swarm indicates",
]


# ---------------------------------------------------------------------------
# Helper: simple greedy generation (avoids model.generate API issues)
# ---------------------------------------------------------------------------

def simple_generate(model, tokenizer, input_ids, device, max_new_tokens=40, past_kv=None):
    """Simple greedy decoding loop."""
    generated_ids = input_ids.clone()
    cache = past_kv
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Use only last token if we have cache
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
    new_tokens = generated_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def run_injection_test(
    model, tokenizer, device,
    prefix: str,
    continuation: str,
    label: str = "",
) -> dict:
    """
    Run a single injection test:
    1. Instance A processes prefix → extract KV-cache
    2. Instance B uses injected KV-cache → generate continuation
    3. Instance B processes prefix natively → generate continuation (baseline)
    4. Compare results
    """
    print(f"\n--- Test: {label} ---")
    print(f"  Prefix: {prefix[:60]}...")

    # --- Instance A: process prefix, extract KV-cache ---
    prefix_inputs = tokenize(prefix, tokenizer, device)
    prefix_len = prefix_inputs["input_ids"].shape[1]
    print(f"  Prefix tokens: {prefix_len}")

    with torch.no_grad():
        prefix_output = model(**prefix_inputs, use_cache=True)

    kv_from_a = prefix_output.past_key_values
    # Calculate KV bytes (handle DynamicCache for transformers 5.x)
    if hasattr(kv_from_a, 'layers'):
        kv_bytes = sum(
            layer.keys.nelement() * layer.keys.element_size() +
            layer.values.nelement() * layer.values.element_size()
            for layer in kv_from_a.layers
        )
    else:
        kv_bytes = sum(
            k.nelement() * k.element_size() + v.nelement() * v.element_size()
            for k, v in kv_from_a
        )
    print(f"  KV-cache size: {kv_bytes:,} bytes ({kv_bytes/1024:.1f} KB)")

    # --- Instance B (injected): use A's KV-cache for continuation ---
    cont_inputs = tokenize(continuation, tokenizer, device)
    cont_ids = cont_inputs["input_ids"]

    with torch.no_grad():
        injected_output = model(
            input_ids=cont_ids,
            past_key_values=kv_from_a,
            use_cache=True,
        )

    # Compute perplexity for injected path
    injected_logits = injected_output.logits
    shift_logits = injected_logits[:, :-1, :].contiguous()
    shift_labels = cont_ids[:, 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss()
    injected_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    injected_ppl = torch.exp(injected_loss).item()

    # Generate with injected KV (simple greedy loop)
    injected_text = simple_generate(model, tokenizer, cont_ids, device, max_new_tokens=40, past_kv=kv_from_a)

    # --- Instance B (native): process prefix + continuation from scratch ---
    full_text = prefix + continuation
    full_inputs = tokenize(full_text, tokenizer, device)
    full_ids = full_inputs["input_ids"]

    with torch.no_grad():
        native_output = model(input_ids=full_ids, use_cache=True)

    # Perplexity on the continuation portion only
    native_logits = native_output.logits
    # We need logits for the continuation tokens only
    # The continuation starts at position prefix_len
    cont_start = prefix_len
    cont_logits = native_logits[:, cont_start - 1:-1, :].contiguous()
    cont_labels = full_ids[:, cont_start:].contiguous()

    if cont_logits.shape[1] > 0 and cont_labels.shape[1] > 0:
        min_len = min(cont_logits.shape[1], cont_labels.shape[1])
        native_loss = loss_fn(
            cont_logits[:, :min_len, :].reshape(-1, cont_logits.size(-1)),
            cont_labels[:, :min_len].reshape(-1),
        )
        native_ppl = torch.exp(native_loss).item()
    else:
        native_ppl = float("nan")

    # Generate natively (simple greedy loop)
    native_text = simple_generate(model, tokenizer, full_ids, device, max_new_tokens=40)

    # --- Compare ---
    ppl_gap = abs(injected_ppl - native_ppl)
    ppl_ratio = injected_ppl / native_ppl if native_ppl > 0 else float("inf")

    # Token-level match
    injected_tokens = tokenizer.encode(injected_text)
    native_tokens = tokenizer.encode(native_text)
    min_tok_len = min(len(injected_tokens), len(native_tokens))
    if min_tok_len > 0:
        token_match = sum(
            1 for a, b in zip(injected_tokens[:min_tok_len], native_tokens[:min_tok_len])
            if a == b
        ) / min_tok_len * 100
    else:
        token_match = 0.0

    result = {
        "label": label,
        "prefix_tokens": prefix_len,
        "kv_bytes": kv_bytes,
        "injected_ppl": round(injected_ppl, 4),
        "native_ppl": round(native_ppl, 4),
        "ppl_gap": round(ppl_gap, 4),
        "ppl_ratio": round(ppl_ratio, 4),
        "token_match_pct": round(token_match, 1),
        "injected_text": injected_text[:200],
        "native_text": native_text[:200],
    }

    print(f"  Injected PPL: {injected_ppl:.4f}")
    print(f"  Native PPL:   {native_ppl:.4f}")
    print(f"  PPL gap:      {ppl_gap:.4f}")
    print(f"  Token match:  {token_match:.1f}%")
    print(f"  Injected gen: '{injected_text[:80]}...'")
    print(f"  Native gen:   '{native_text[:80]}...'")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: list[dict]):
    """Plot comparison of injected vs native perplexity."""
    results_dir = ensure_results_dir()

    labels = [r["label"] for r in results]
    injected = [r["injected_ppl"] for r in results]
    native = [r["native_ppl"] for r in results]
    token_match = [r["token_match_pct"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- PPL comparison ---
    bars1 = ax1.bar(x - width / 2, native, width, label="Native (baseline)", color="steelblue")
    bars2 = ax1.bar(x + width / 2, injected, width, label="Injected KV", color="coral")
    ax1.set_xlabel("Test Case")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Perplexity: Native vs Injected KV-cache")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Add gap annotations
    for i, (n, inj) in enumerate(zip(native, injected)):
        gap = abs(inj - n)
        ax1.annotate(
            f"gap={gap:.2f}",
            (i, max(n, inj)),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    # --- Token match ---
    colors = ["green" if m > 80 else "orange" if m > 50 else "red" for m in token_match]
    ax2.bar(x, token_match, color=colors, alpha=0.7)
    ax2.set_xlabel("Test Case")
    ax2.set_ylabel("Token Match (%)")
    ax2.set_title("Generated Token Match: Injected vs Native")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color="green", linestyle="--", alpha=0.3, label="Perfect match")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = results_dir / "exp3_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Experiment 3: Cross-instance KV-cache Injection")
    print("=" * 60)

    model, tokenizer, device = load_model_and_tokenizer()

    results = []

    # --- Primary test ---
    r = run_injection_test(
        model, tokenizer, device,
        PREFIX, CONTINUATION,
        label="Primary (vehicle)",
    )
    results.append(r)

    # --- Additional tests ---
    for i, (prefix, cont) in enumerate(zip(TEST_PREFIXES, TEST_CONTINUATIONS)):
        label = f"Test {i+1} ({'short' if i==0 else 'medium' if i==1 else 'long'})"
        r = run_injection_test(model, tokenizer, device, prefix, cont, label=label)
        results.append(r)

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY: Cross-instance KV-cache Injection Results")
    print(f"{'='*80}")
    print(f"{'Test':>25} | {'Native PPL':>10} | {'Injected PPL':>12} | "
          f"{'Gap':>8} | {'Token Match':>11}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['label']:>25} | {r['native_ppl']:>10.4f} | {r['injected_ppl']:>12.4f} | "
              f"{r['ppl_gap']:>8.4f} | {r['token_match_pct']:>10.1f}%")
    print(f"{'='*80}")

    avg_gap = np.mean([r["ppl_gap"] for r in results])
    avg_match = np.mean([r["token_match_pct"] for r in results])
    print(f"\nAverage PPL gap:    {avg_gap:.4f}")
    print(f"Average token match: {avg_match:.1f}%")

    if avg_gap < 0.1:
        print("\nConclusion: KV-cache injection is LOSSLESS (gap < 0.1)")
    elif avg_gap < 1.0:
        print("\nConclusion: KV-cache injection has MINIMAL loss (gap < 1.0)")
    else:
        print(f"\nConclusion: KV-cache injection shows MEASURABLE gap ({avg_gap:.2f})")

    # --- Save ---
    save_json({
        "model": str(model.config._name_or_path),
        "results": results,
        "average_ppl_gap": round(avg_gap, 4),
        "average_token_match": round(avg_match, 1),
    }, "exp3_cross_injection.json")

    # --- Plot ---
    plot_comparison(results)

    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
