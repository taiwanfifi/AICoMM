#!/usr/bin/env python3
"""Generate publication-ready figures for Paper B (Scout protocol)."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BATCH30_DIR = RESULTS_DIR
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "papers" / "paper-B" / "figures"
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

# Load data
def load_batch28():
    pairs = {}
    for f in RESULTS_DIR.glob("batch28_scout_*.json"):
        with open(f) as fh:
            data = json.load(fh)
        edge = data["metadata"]["edge_model"].split("/")[-1]
        cloud = data["metadata"]["cloud_model"].split("/")[-1]
        key = f"{edge}→{cloud}"
        pairs[key] = data
    return pairs

def load_batch30():
    f = list(BATCH30_DIR.glob("batch30_*.json"))[0]
    with open(f) as fh:
        return json.load(fh)


# ============================================================
# Figure 1: Scout Overlap & F1 Grouped Bar Chart
# ============================================================
def fig1_scout_overlap_f1(batch28):
    """Three model pairs, overlap bars + F1 comparison."""
    pair_order = ["Qwen2.5-3B→Qwen2.5-7B", "Qwen2.5-3B→Qwen2.5-14B", "Qwen2.5-7B→Qwen2.5-14B"]
    pair_labels = ["3B→7B", "3B→14B", "7B→14B"]
    retentions = ["75%", "50%", "25%"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # Left: Overlap bars
    x = np.arange(len(retentions))
    width = 0.22
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for i, (pair, label) in enumerate(zip(pair_order, pair_labels)):
        data = batch28[pair]["metadata"]["retention_results"]
        overlaps = [data[r]["overlap_pct"] for r in retentions]
        ax1.bar(x + (i - 1) * width, overlaps, width, label=label, color=colors[i], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel("Retention Level")
    ax1.set_ylabel("Position Overlap (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(retentions)
    ax1.set_ylim(0, 100)
    ax1.legend(title="Model Pair", loc='upper right')
    ax1.set_title("(a) Cross-Model Position Overlap")
    ax1.grid(axis='y', alpha=0.3)

    # Right: F1 comparison (cloud own vs scout)
    x = np.arange(len(retentions))
    width = 0.12

    for i, (pair, label) in enumerate(zip(pair_order, pair_labels)):
        data = batch28[pair]["metadata"]["retention_results"]
        cloud_own = [data[r]["cloud_own_f1"] for r in retentions]
        scout = [data[r]["scout_f1"] for r in retentions]

        offset = (i - 1) * 0.28
        ax2.bar(x + offset - width/2, cloud_own, width, color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.4)
        ax2.bar(x + offset + width/2, scout, width, color=colors[i], edgecolor='black', linewidth=0.5)

    # Legend
    own_patch = mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', linewidth=0.5, label='Cloud Own')
    scout_patch = mpatches.Patch(facecolor='gray', edgecolor='black', linewidth=0.5, label='Scout')
    handles = [own_patch, scout_patch]
    for c, l in zip(colors, pair_labels):
        handles.append(mpatches.Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=l))
    ax2.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)

    ax2.set_xlabel("Retention Level")
    ax2.set_ylabel("Token-F1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(retentions)
    ax2.set_ylim(0, 0.85)
    ax2.set_title("(b) F1: Cloud Own vs. Scout Selection")
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f"fig1_scout_overlap_f1.{ext}")
    plt.close()
    print("  Fig 1: Scout overlap & F1 comparison")


# ============================================================
# Figure 2: Quality-Bandwidth Operating Points
# ============================================================
def fig2_operating_points(batch30):
    """Quality vs bandwidth fraction for 3 models + scout region."""
    pareto = batch30["pareto_frontiers"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    colors = {'Qwen-7B': '#2196F3', 'Mistral-7B': '#FF9800', 'Qwen-14B': '#4CAF50'}
    markers = {'Qwen-7B': 'o', 'Mistral-7B': 's', 'Qwen-14B': '^'}

    for model in ['Qwen-7B', 'Mistral-7B', 'Qwen-14B']:
        points = pareto[model]
        bws = [p["bw_frac"] for p in points]
        qs = [p["quality"] for p in points]
        ax.scatter(bws, qs, color=colors[model], marker=markers[model], s=60,
                   edgecolors='black', linewidth=0.5, zorder=5, label=model)

        # Add full BF16 point at 1.0
        ax.scatter([1.0], [100.0 if model != 'Mistral-7B' else 100.0],
                   color=colors[model], marker=markers[model], s=60,
                   edgecolors='black', linewidth=0.5, zorder=5, alpha=0.3)

    # Scout region (near zero bandwidth, ~81-110% quality)
    ax.axhspan(81, 110, xmin=0, xmax=0.05, alpha=0.15, color='red')
    ax.annotate('Scout\nregion', xy=(0.02, 95), fontsize=7, ha='center',
                color='red', fontweight='bold')

    ax.set_xlabel("Bandwidth Fraction (vs Full BF16)")
    ax.set_ylabel("Task Quality (% of baseline)")
    ax.set_xlim(-0.02, 1.1)
    ax.set_ylim(90, 112)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title("Quality-Bandwidth Operating Points")
    ax.grid(alpha=0.3)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f"fig2_operating_points.{ext}")
    plt.close()
    print("  Fig 2: Quality-bandwidth operating points")


# ============================================================
# Figure 3: Deadline Compliance vs Quality (3 models)
# ============================================================
def fig3_deadline_quality(batch30):
    """Deadline compliance vs avg quality for 3 models, 5s deadline."""
    proto = batch30["protocol_simulation"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

    models = ['Qwen-7B', 'Mistral-7B', 'Qwen-14B']
    policies = ['static_int8', 'static_int4', 'adaptive', 'scout', 'no_transfer']
    policy_labels = ['INT8', 'INT4', 'Adaptive', 'Scout', 'Local']
    policy_colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9E9E9E']
    policy_markers = ['o', 's', 'D', '*', 'v']
    deadlines = [1000, 3000, 5000]
    deadline_labels = ['1s', '3s', '5s']

    for mi, (model, ax) in enumerate(zip(models, axes)):
        for pi, (policy, plabel) in enumerate(zip(policies, policy_labels)):
            qs = []
            dls = []
            for d in deadlines:
                key = f"{model}_deadline{d}"
                if key in proto:
                    entry = proto[key][policy]
                    qs.append(entry["avg_quality"])
                    dls.append(entry["deadline_success_rate"] * 100)

            ax.plot(dls, qs, marker=policy_markers[pi], color=policy_colors[pi],
                    label=plabel if mi == 0 else None, markersize=5, linewidth=1)

        ax.set_xlabel("Deadline Compliance (%)")
        if mi == 0:
            ax.set_ylabel("Avg Quality (%)")
        ax.set_title(model, fontsize=9)
        ax.set_xlim(-5, 108)
        ax.set_ylim(55, 112)
        ax.grid(alpha=0.3)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    axes[0].legend(loc='lower left', fontsize=6.5)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f"fig3_deadline_quality.{ext}")
    plt.close()
    print("  Fig 3: Deadline compliance vs quality")


# ============================================================
# Figure 4: Multi-Agent Scaling
# ============================================================
def fig4_multiagent_scaling(batch30):
    """Total quality vs N agents for 3 allocation policies at 100 Mbps."""
    ma = batch30["multi_agent"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

    agents = [2, 4, 8]
    bws = [50, 100, 200]
    policies = ['equal', 'model_aware', 'quality_max']
    policy_labels = ['Equal', 'Model-Aware', 'Quality-Max']
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    markers = ['o', 's', 'D']

    # Left: Quality per agent at 100 Mbps
    for pi, (policy, plabel) in enumerate(zip(policies, policy_labels)):
        qs = []
        for n in agents:
            key = f"multiagent_{n}agents_100mbps"
            total = ma[key][policy]["avg_total_quality"]
            qs.append(total / n)
        ax1.plot(agents, qs, marker=markers[pi], color=colors[pi], label=plabel,
                 markersize=6, linewidth=1.5)

    ax1.set_xlabel("Number of Agents")
    ax1.set_ylabel("Avg Quality per Agent (%)")
    ax1.set_xticks(agents)
    ax1.set_ylim(95, 108)
    ax1.legend(loc='lower left', fontsize=7)
    ax1.set_title("(a) Quality per Agent (100 Mbps)")
    ax1.grid(alpha=0.3)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    # Right: Deadline compliance across bandwidths for 4 agents
    x = np.arange(len(bws))
    width = 0.22
    for pi, (policy, plabel) in enumerate(zip(policies, policy_labels)):
        dls = []
        for bw in bws:
            key = f"multiagent_4agents_{bw}mbps"
            dl = ma[key][policy]["avg_all_meet_deadline"] * 100
            dls.append(dl)
        ax2.bar(x + (pi - 1) * width, dls, width, label=plabel, color=colors[pi],
                edgecolor='black', linewidth=0.5)

    ax2.set_xlabel("Total Bandwidth (Mbps)")
    ax2.set_ylabel("All-Meet-Deadline (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(['50', '100', '200'])
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper left', fontsize=7)
    ax2.set_title("(b) Deadline Compliance (4 Agents)")
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f"fig4_multiagent_scaling.{ext}")
    plt.close()
    print("  Fig 4: Multi-agent scaling")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Loading data...")
    batch28 = load_batch28()
    batch30 = load_batch30()

    print(f"Batch 28: {len(batch28)} model pairs")
    print(f"Batch 30: protocol + multi-agent data")
    print()

    print("Generating figures...")
    fig1_scout_overlap_f1(batch28)
    fig2_operating_points(batch30)
    fig3_deadline_quality(batch30)
    fig4_multiagent_scaling(batch30)

    print(f"\nAll figures saved to {FIG_DIR}")
