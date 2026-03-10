#!/usr/bin/env python3
"""
Generate attention heatmap visualization for JSAC paper.

Creates a qualitative figure showing attention alignment between
scout (7B) and cloud (14B) models on a single SQuAD example.

Output: figures/attention_heatmap.pdf

Can run standalone or after other experiments complete.
Needs: ~28GB VRAM (14B model), ~2 minutes.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from exp_utils import (
    setup_logging, setup_hf_cache, load_model, free_model,
    compute_q2c_last_layer, select_positions, HF_MODELS,
)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

logger = setup_logging(__name__)

# Use a clear, paper-ready SQuAD example
EXAMPLE = {
    'context': "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of the West Franks.",
    'question': "In what country is Normandy located?",
    'gold_answer': "France",
}


def get_token_texts(tokenizer, text, max_tokens=None):
    """Get individual token strings for visualization."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = []
    for (start, end) in encoding['offset_mapping']:
        token_text = text[start:end]
        tokens.append(token_text)
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def run_and_extract(model_name, context, question):
    """Run a model and extract Q2C scores."""
    model, tok = load_model(model_name)

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    context_prefix = f"Context: {context}\n"
    context_end = len(tok.encode(context_prefix))

    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)

    seq_len = out.attentions[-1].shape[-1]
    ce = min(context_end, seq_len - 1)
    q2c = compute_q2c_last_layer(out.attentions, ce)

    # Normalize Q2C to [0, 1]
    q2c_norm = (q2c - q2c.min()) / (q2c.max() - q2c.min() + 1e-8)

    # Get context token texts
    context_tokens = get_token_texts(tok, context_prefix)[:ce]

    del out
    free_model(model)
    del tok

    return q2c_norm, context_tokens, ce


def create_heatmap_figure(q2c_scout, q2c_cloud, tokens, retention=0.75,
                          output_path='figures/attention_heatmap.pdf'):
    """
    Create a paper-ready figure showing:
    1. Q2C score comparison (bar chart)
    2. Token selection overlap visualization
    """
    n_tokens = min(len(q2c_scout), len(q2c_cloud), len(tokens))
    q2c_scout = q2c_scout[:n_tokens]
    q2c_cloud = q2c_cloud[:n_tokens]
    tokens = tokens[:n_tokens]

    # Select positions
    k = max(1, int(n_tokens * retention))
    scout_pos = set(np.argsort(q2c_scout)[-k:])
    cloud_pos = set(np.argsort(q2c_cloud)[-k:])
    overlap = scout_pos & cloud_pos
    scout_only = scout_pos - cloud_pos
    cloud_only = cloud_pos - scout_pos

    overlap_pct = len(overlap) / k * 100

    # --- Figure setup ---
    fig, axes = plt.subplots(3, 1, figsize=(7, 4.5), height_ratios=[1, 1, 0.6],
                              gridspec_kw={'hspace': 0.4})

    # Colors
    C_SCOUT = '#2196F3'    # Blue
    C_CLOUD = '#FF5722'    # Orange-red
    C_OVERLAP = '#4CAF50'  # Green

    x = np.arange(n_tokens)

    # --- Panel 1: Scout (7B) Q2C scores ---
    ax1 = axes[0]
    colors1 = []
    for i in range(n_tokens):
        if i in overlap:
            colors1.append(C_OVERLAP)
        elif i in scout_only:
            colors1.append(C_SCOUT)
        else:
            colors1.append('#E0E0E0')
    ax1.bar(x, q2c_scout, color=colors1, width=1.0, edgecolor='none')
    ax1.set_ylabel('Q2C score', fontsize=8)
    ax1.set_title(f'Scout model (Qwen2.5-7B)', fontsize=9, fontweight='bold')
    ax1.set_xlim(-0.5, n_tokens - 0.5)
    ax1.set_xticks([])
    ax1.tick_params(labelsize=7)

    # --- Panel 2: Cloud (14B) Q2C scores ---
    ax2 = axes[1]
    colors2 = []
    for i in range(n_tokens):
        if i in overlap:
            colors2.append(C_OVERLAP)
        elif i in cloud_only:
            colors2.append(C_CLOUD)
        else:
            colors2.append('#E0E0E0')
    ax2.bar(x, q2c_cloud, color=colors2, width=1.0, edgecolor='none')
    ax2.set_ylabel('Q2C score', fontsize=8)
    ax2.set_title(f'Cloud model (Qwen2.5-14B)', fontsize=9, fontweight='bold')
    ax2.set_xlim(-0.5, n_tokens - 0.5)
    ax2.set_xticks([])
    ax2.tick_params(labelsize=7)

    # --- Panel 3: Token text with color coding ---
    ax3 = axes[2]
    ax3.set_xlim(-0.5, n_tokens - 0.5)
    ax3.set_ylim(-0.2, 1.2)
    ax3.axis('off')
    ax3.set_title(f'Token selection at {int(retention*100)}% retention '
                  f'(overlap: {overlap_pct:.0f}%)',
                  fontsize=9, fontweight='bold')

    # Draw tokens as colored boxes
    for i in range(n_tokens):
        if i in overlap:
            color = C_OVERLAP
            alpha = 0.7
        elif i in scout_only:
            color = C_SCOUT
            alpha = 0.5
        elif i in cloud_only:
            color = C_CLOUD
            alpha = 0.5
        else:
            color = '#F5F5F5'
            alpha = 0.3

        rect = mpatches.FancyBboxPatch(
            (i - 0.45, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=alpha,
            edgecolor='#999999', linewidth=0.3,
        )
        ax3.add_patch(rect)

        # Add token text (truncated)
        token_text = tokens[i].strip()
        if len(token_text) > 4:
            token_text = token_text[:3] + '..'
        if token_text:
            ax3.text(i, 0.5, token_text, ha='center', va='center',
                     fontsize=4.5, fontfamily='monospace',
                     color='black' if alpha > 0.4 else '#888888')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_OVERLAP, alpha=0.7, label=f'Both ({len(overlap)})'),
        mpatches.Patch(facecolor=C_SCOUT, alpha=0.5, label=f'Scout only ({len(scout_only)})'),
        mpatches.Patch(facecolor=C_CLOUD, alpha=0.5, label=f'Cloud only ({len(cloud_only)})'),
        mpatches.Patch(facecolor='#F5F5F5', alpha=0.3, edgecolor='#999',
                       label=f'Unselected ({n_tokens - k*2 + len(overlap)})'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()

    return {
        'n_tokens': n_tokens,
        'k_selected': k,
        'overlap': len(overlap),
        'overlap_pct': overlap_pct,
        'scout_only': len(scout_only),
        'cloud_only': len(cloud_only),
    }


def main():
    setup_hf_cache()

    logger.info("=" * 70)
    logger.info("Attention Heatmap Visualization")
    logger.info("=" * 70)

    context = EXAMPLE['context']
    question = EXAMPLE['question']

    # Extract Q2C from both models
    logger.info("\nExtracting scout (7B) attention...")
    q2c_scout, tokens_scout, ce_scout = run_and_extract(
        HF_MODELS['7B'], context, question
    )

    logger.info("Extracting cloud (14B) attention...")
    q2c_cloud, tokens_cloud, ce_cloud = run_and_extract(
        HF_MODELS['14B'], context, question
    )

    # Use the shorter token list (tokenizers should be identical for same-family)
    n = min(len(q2c_scout), len(q2c_cloud))
    tokens = tokens_scout[:n]

    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'papers' / 'jsac' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / 'attention_heatmap.pdf')

    # Generate figure at primary operating point
    stats = create_heatmap_figure(
        q2c_scout[:n], q2c_cloud[:n], tokens,
        retention=0.75, output_path=output_path
    )

    logger.info(f"\nStats: {stats}")
    logger.info(f"Example: '{question}' → '{EXAMPLE['gold_answer']}'")
    logger.info(f"Context: {len(tokens)} tokens, {stats['overlap_pct']:.0f}% overlap at 75% retention")


if __name__ == '__main__':
    main()
