#!/usr/bin/env python3
"""Generate publication-ready figures for the merged JSAC paper.

Reads experiment JSON results and produces PDF+PNG figures in papers/jsac/figures/.
All figures use IEEE journal style (Times New Roman, 9pt body, 300 DPI).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

COLORS = {
    '3B_7B': '#2196F3',
    '3B_14B': '#FF9800',
    '7B_14B': '#4CAF50',
    'q2c': '#2196F3',
    'snapkv': '#FF9800',
    'h2o': '#F44336',
    'bf16': '#4CAF50',
    'int8': '#2196F3',
    'int4': '#F44336',
    'mixed': '#FF9800',
    'adaptive': '#4CAF50',
    'scout': '#E91E63',
    'static_int8': '#2196F3',
    'local': '#9E9E9E',
    'equal': '#F44336',
    'model_aware': '#4CAF50',
}


def load_json(pattern):
    """Load first matching JSON file."""
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        print(f"WARNING: No files matching {pattern}")
        return None
    with open(files[0]) as f:
        return json.load(f)


def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(FIG_DIR / f"{name}.pdf", format='pdf')
    fig.savefig(FIG_DIR / f"{name}.png", format='png')
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


# ============================================================
# Figure 1: System Architecture (placeholder - needs manual design)
# ============================================================
def fig1_system_architecture():
    """System architecture diagram - placeholder for manual creation."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.text(0.5, 0.5, 'System Architecture\n(Create manually in drawing tool)',
            ha='center', va='center', fontsize=14, style='italic', color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    save_fig(fig, 'fig1_system_architecture')


# ============================================================
# Figure 2: Scout Overlap Bar Chart (n=200)
# ============================================================
def fig2_scout_overlap():
    """Scout overlap for 3 Qwen pairs at 75/50/25% retention (n=200)."""
    data = load_json("exp_scout_n200_*.json")
    if data is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    pair_json_keys = ['3B->7B', '3B->14B', '7B->14B']
    pair_labels = ['3B$\\to$7B', '3B$\\to$14B', '7B$\\to$14B']
    pair_colors = [COLORS['3B_7B'], COLORS['3B_14B'], COLORS['7B_14B']]
    ret_keys = ['75%', '50%', '25%']
    ret_labels = ['75%', '50%', '25%']

    summaries = data.get('summaries', {})

    # Left: Overlap bars
    x = np.arange(len(ret_keys))
    width = 0.22

    for i, (pk, pl, pc) in enumerate(zip(pair_json_keys, pair_labels, pair_colors)):
        pair_s = summaries.get(pk, {})
        ret_results = pair_s.get('retention_results', {})
        overlaps = [ret_results.get(rk, {}).get('overlap_pct_mean', 0) for rk in ret_keys]
        ax1.bar(x + (i - 1) * width, overlaps, width, label=pl,
                color=pc, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel("Retention Level")
    ax1.set_ylabel("Position Overlap (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ret_labels)
    ax1.set_ylim(0, 100)
    ax1.legend(title="Model Pair", loc='upper right', fontsize=7)
    ax1.set_title("(a) Cross-Model Position Overlap")
    ax1.grid(axis='y', alpha=0.3)

    # Right: F1 comparison (cloud own vs scout)
    for i, (pk, pl, pc) in enumerate(zip(pair_json_keys, pair_labels, pair_colors)):
        pair_s = summaries.get(pk, {})
        ret_results = pair_s.get('retention_results', {})
        cloud_own = [ret_results.get(rk, {}).get('cloud_own_f1_mean', 0) for rk in ret_keys]
        scout_f1 = [ret_results.get(rk, {}).get('scout_f1_mean', 0) for rk in ret_keys]

        offset = (i - 1) * 0.28
        w = 0.12
        ax2.bar(x + offset - w/2, cloud_own, w, color=pc,
                edgecolor='black', linewidth=0.5, alpha=0.4)
        ax2.bar(x + offset + w/2, scout_f1, w, color=pc,
                edgecolor='black', linewidth=0.5)

    own_patch = mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black',
                                linewidth=0.5, label='Cloud Own')
    scout_patch = mpatches.Patch(facecolor='gray', edgecolor='black',
                                  linewidth=0.5, label='Scout')
    ax2.legend(handles=[own_patch, scout_patch], loc='upper right', fontsize=7)
    ax2.set_xlabel("Retention Level")
    ax2.set_ylabel("Token F1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ret_labels)
    ax2.set_ylim(0, 0.85)
    ax2.set_title("(b) F1: Cloud Own vs.\ Scout")
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig2_scout_overlap_f1')


# ============================================================
# Figure 3: Selection Method Comparison (Q2C vs SnapKV vs H2O)
# ============================================================
def fig3_selection_methods():
    """Bar chart comparing Q2C, SnapKV, H2O on Qwen-7B (n=200)."""
    data = load_json("exp_paper_a_unified_qwen_7b_*.json")
    if data is None:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ret_pcts = ['75%', '50%', '25%']
    methods = ['q2c', 'snapkv', 'h2o']
    method_labels = ['Q2C', 'SnapKV', 'H2O']
    method_colors = [COLORS['q2c'], COLORS['snapkv'], COLORS['h2o']]

    summary = data.get('summary', {})

    x = np.arange(len(ret_pcts))
    width = 0.22

    for i, (m, ml, mc) in enumerate(zip(methods, method_labels, method_colors)):
        vals = [summary.get(f'sel_{rp}_{m}_mean', 0) for rp in ret_pcts]
        ax.bar(x + (i - 1) * width, vals, width, label=ml,
               color=mc, edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Retention Level")
    ax.set_ylabel("Token F1")
    ax.set_xticks(x)
    ax.set_xticklabels(ret_pcts)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig3_selection_methods')


# ============================================================
# Figure 4: Perplexity Bar Chart
# ============================================================
def fig4_perplexity():
    """Perplexity comparison across quantization levels for 3 models."""
    ppl_qwen = load_json("exp_perplexity_20260210_*.json")
    ppl_mistral = load_json("exp_perplexity_mistral_*.json")

    if ppl_qwen is None:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Data from experiment results
    models = ['Qwen-7B', 'Qwen-14B', 'Mistral-7B']
    quants = ['BF16', 'INT8', 'INT4', 'Mixed']
    qcolors = [COLORS['bf16'], COLORS['int8'], COLORS['int4'], COLORS['mixed']]

    # Hardcoded from verified experiment JSONs
    ppl_data = {
        'Qwen-7B':    [8.63,  8.85,  80.27, 8.97],
        'Qwen-14B':   [5.73,  5.73,  5.87,  5.83],
        'Mistral-7B': [6.49,  6.49,  6.51,  6.51],
    }

    x = np.arange(len(models))
    width = 0.18

    for i, (q, qc) in enumerate(zip(quants, qcolors)):
        vals = [ppl_data[m][i] for m in models]
        # Cap for display (Qwen-7B INT4 = 80.27)
        display_vals = [min(v, 15) for v in vals]
        bars = ax.bar(x + (i - 1.5) * width, display_vals, width, label=q,
                      color=qc, edgecolor='black', linewidth=0.5)
        # Annotate capped bar
        for j, (dv, rv) in enumerate(zip(display_vals, vals)):
            if rv > 15:
                ax.text(x[j] + (i - 1.5) * width, dv + 0.3, f'{rv:.1f}',
                        ha='center', va='bottom', fontsize=6, fontweight='bold',
                        color=COLORS['int4'])

    ax.set_xlabel("Model")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0, 16)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig4_perplexity')


# ============================================================
# Figure 5: Attention Entropy per Model
# ============================================================
def fig5_attention_entropy():
    """Attention entropy comparison for 3B/7B/14B."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # From experiment JSONs
    models = ['Qwen-3B', 'Qwen-7B', 'Qwen-14B']
    overall = [2.61, 2.56, 2.13]
    last_layer = [3.31, 3.01, 3.67]
    q2c_dist = [4.21, 5.49, 4.65]

    x = np.arange(len(models))
    width = 0.22

    ax.bar(x - width, overall, width, label='Overall', color='#2196F3',
           edgecolor='black', linewidth=0.5)
    ax.bar(x, last_layer, width, label='Last Layer', color='#FF9800',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width, q2c_dist, width, label='Q2C Dist.', color='#4CAF50',
           edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Entropy (nats)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig5_attention_entropy')


# ============================================================
# Figure 6: Protocol Deadline Compliance
# ============================================================
def fig6_protocol_deadline():
    """Protocol deadline compliance with real 5G traces."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # From protocol trace experiment
    deadlines = [1, 3, 5]
    policies = {
        'Static INT8': [58, 75, 88],
        'Adaptive':    [75, 88, 100],
        'Scout':       [100, 100, 100],
    }
    colors = [COLORS['static_int8'], COLORS['adaptive'], COLORS['scout']]
    markers = ['s', 'D', '*']

    for (pol, vals), c, m in zip(policies.items(), colors, markers):
        ms = 8 if m == '*' else 5
        ax.plot(deadlines, vals, '-' + m, color=c, label=pol, markersize=ms,
                linewidth=1.5)

    ax.set_xlabel("Deadline (seconds)")
    ax.set_ylabel("Deadline Compliance (%)")
    ax.set_xticks(deadlines)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7)
    ax.grid(axis='both', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig6_protocol_deadline')


# ============================================================
# Figure 7: Multi-Agent Scaling
# ============================================================
def fig7_multiagent():
    """Multi-agent allocation comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # From batch30 / protocol experiments
    bws = [50, 100, 200]
    bw_labels = ['50', '100', '200']

    # N=4 deadline compliance
    equal_4 = [0, 100, 100]
    model_aware_4 = [100, 100, 100]

    ax1.bar(np.arange(3) - 0.15, equal_4, 0.3, label='Equal',
            color=COLORS['equal'], edgecolor='black', linewidth=0.5)
    ax1.bar(np.arange(3) + 0.15, model_aware_4, 0.3, label='Model-aware',
            color=COLORS['model_aware'], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Total Bandwidth (Mbps)")
    ax1.set_ylabel("Deadline Compliance (%)")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(bw_labels)
    ax1.set_ylim(0, 120)
    ax1.legend(fontsize=7)
    ax1.set_title("(a) $N = 4$ Agents")
    ax1.grid(axis='y', alpha=0.3)

    # N=8 deadline compliance
    equal_8 = [0, 0, 100]
    model_aware_8 = [0, 100, 100]

    ax2.bar(np.arange(3) - 0.15, equal_8, 0.3, label='Equal',
            color=COLORS['equal'], edgecolor='black', linewidth=0.5)
    ax2.bar(np.arange(3) + 0.15, model_aware_8, 0.3, label='Model-aware',
            color=COLORS['model_aware'], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel("Total Bandwidth (Mbps)")
    ax2.set_ylabel("Deadline Compliance (%)")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(bw_labels)
    ax2.set_ylim(0, 120)
    ax2.legend(fontsize=7)
    ax2.set_title("(b) $N = 8$ Agents")
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig7_multiagent_scaling')


# ============================================================
# Figure 8: Long Context Overlap Stability
# ============================================================
def fig8_long_context():
    """Overlap stability at 1K and 2K context lengths."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ctx_lengths = [1024, 2048]
    ret_75 = [83.3, 82.7]
    ret_50 = [69.4, 68.2]
    ret_25 = [55.6, 53.9]

    x = np.arange(len(ctx_lengths))
    width = 0.22

    ax.bar(x - width, ret_75, width, label='75% Ret.', color='#4CAF50',
           edgecolor='black', linewidth=0.5)
    ax.bar(x, ret_50, width, label='50% Ret.', color='#FF9800',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width, ret_25, width, label='25% Ret.', color='#F44336',
           edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Position Overlap (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(['1024', '2048'])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig8_long_context_overlap')


# ============================================================
# Figure 9: Quality-Bandwidth Operating Points
# ============================================================
def fig9_operating_points():
    """Pareto frontier: quality vs bandwidth for operating modes."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Qwen-7B operating points (n=200 data)
    modes = {
        'Full BF16':  (9700, 100.0),
        'INT8':       (4700, 99.4),
        'Mixed INT4': (2600, 93.6),
        'INT4':       (2300, 68.7),
        'Scout':      (0.336, 99.4),  # 7B->14B at 75%
    }

    for name, (bw, q) in modes.items():
        if name == 'Scout':
            ax.scatter(bw, q, marker='*', s=150, color=COLORS['scout'],
                      edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(name, (bw, q), textcoords="offset points",
                       xytext=(10, -5), fontsize=7)
        else:
            color = {'Full BF16': COLORS['bf16'], 'INT8': COLORS['int8'],
                     'INT4': COLORS['int4'], 'Mixed INT4': COLORS['mixed']}[name]
            ax.scatter(bw, q, marker='o', s=60, color=color,
                      edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(name, (bw, q), textcoords="offset points",
                       xytext=(10, 5), fontsize=7)

    ax.set_xscale('log')
    ax.set_xlabel("Payload Size (KB)")
    ax.set_ylabel("Quality (% of Full KV)")
    ax.set_ylim(60, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    # Add region label
    ax.axvspan(0.1, 1, alpha=0.1, color=COLORS['scout'])
    ax.text(0.3, 65, 'Scout\nRegion', fontsize=7, color=COLORS['scout'],
            ha='center', style='italic')

    ax.grid(axis='both', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig9_operating_points')


# ============================================================
# Figure 10: Cross-Family Overlap
# ============================================================
def fig10_cross_family():
    """Cross-family overlap comparison: same-family vs cross-family."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    retentions = ['75%', '50%', '25%']
    x = np.arange(len(retentions))
    width = 0.3

    # Same-family: 7B->14B (Qwen)
    same_family = [83.7, 69.5, 54.8]
    # Cross-family: Qwen-7B -> Mistral-7B
    cross_family = [73.4, 58.6, 41.4]

    ax.bar(x - width/2, same_family, width, label='Qwen 7B$\\to$14B',
           color=COLORS['7B_14B'], edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, cross_family, width, label='Qwen$\\to$Mistral',
           color='#9C27B0', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Retention Level")
    ax.set_ylabel("Position Overlap (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(retentions)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig10_cross_family_overlap')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating JSAC paper figures...")
    print()

    print("[1/10] System architecture (placeholder)")
    fig1_system_architecture()

    print("[2/10] Scout overlap (n=200)")
    fig2_scout_overlap()

    print("[3/10] Selection method comparison")
    fig3_selection_methods()

    print("[4/10] Perplexity comparison")
    fig4_perplexity()

    print("[5/10] Attention entropy")
    fig5_attention_entropy()

    print("[6/10] Protocol deadline compliance")
    fig6_protocol_deadline()

    print("[7/10] Multi-agent scaling")
    fig7_multiagent()

    print("[8/10] Long context overlap")
    fig8_long_context()

    print("[9/10] Quality-bandwidth operating points")
    fig9_operating_points()

    print("[10/10] Cross-family overlap")
    fig10_cross_family()

    print()
    print(f"All figures saved to {FIG_DIR}")
