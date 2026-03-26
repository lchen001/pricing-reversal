"""
Generate ablation figure: Kendall's tau before vs after removing thinking token costs.
"""

import json
import os
from itertools import combinations
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- Style: match LaTeX paper ---
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors: muted academic palette
C_ORIG = '#C44E52'   # muted red
C_ABLAT = '#4C72B0'  # muted blue

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "consolidated")
MODEL_INFO = os.path.join(BASE, "constant", "model_info.json")
EXP_CONFIG = os.path.join(BASE, "constant", "experiment_config.json")

with open(MODEL_INFO) as f:
    model_info = json.load(f)
with open(EXP_CONFIG) as f:
    exp_config = json.load(f)

pricing = {}
for m in model_info["models"]:
    pricing[m["model_name"]] = (m["input_price_per_MTok"], m["output_price_per_MTok"])

models = [(m["model_name"], m["short_name"]) for m in exp_config["models"]]
datasets = [(d["file_prefix"], d["dataset_name"]) for d in exp_config["datasets"]]

listed_prices = {}
for model_name, short_name in models:
    ip, op = pricing[model_name]
    listed_prices[model_name] = ip + op

model_names = [m[0] for m in models]

# Compute costs
results = {}
for file_prefix, dataset_name in datasets:
    for model_name, short_name in models:
        fpath = os.path.join(DATA_DIR, f"{file_prefix}-{model_name}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        ip, op = pricing[model_name]
        total_original = 0.0
        total_ablation = 0.0
        for rec in data["records"]:
            pt = rec["prompt_tokens"]
            ct = rec["completion_tokens"]
            tt = rec.get("thinking_tokens", 0)
            gen_tokens = ct - tt
            total_original += (pt * ip + ct * op) / 1e6
            total_ablation += (pt * ip + gen_tokens * op) / 1e6
        results[(dataset_name, model_name)] = {
            "original_cost": total_original,
            "ablation_cost": total_ablation,
        }

# Compute per-task metrics
task_names = []
tau_orig_list = []
tau_ablat_list = []
rev_orig_list = []
rev_ablat_list = []

for file_prefix, dataset_name in datasets:
    lp = [listed_prices[m] for m in model_names]
    orig_costs = []
    ablat_costs = []
    valid = True
    for m in model_names:
        key = (dataset_name, m)
        if key not in results:
            valid = False
            break
        orig_costs.append(results[key]["original_cost"])
        ablat_costs.append(results[key]["ablation_cost"])
    if not valid:
        continue

    tau_orig, _ = kendalltau(lp, orig_costs)
    tau_ablat, _ = kendalltau(lp, ablat_costs)

    n_rev_orig = 0
    n_rev_ablat = 0
    for i, j in combinations(range(len(model_names)), 2):
        lp_order = (lp[i] < lp[j])
        if lp_order != (orig_costs[i] < orig_costs[j]):
            n_rev_orig += 1
        if lp_order != (ablat_costs[i] < ablat_costs[j]):
            n_rev_ablat += 1

    pretty = {
        "aime": "AIME", "arc-agi-v1": "ARC-AGI", "arenahard": "ArenaHard",
        "gpqa": "GPQA", "hle": "HLE", "livecodebench": "LiveCode",
        "livemathbench": "LiveMath", "mmlupro": "MMLUPro", "simpleqa": "SimpleQA",
    }
    task_names.append(pretty.get(dataset_name, dataset_name))
    tau_orig_list.append(tau_orig)
    tau_ablat_list.append(tau_ablat)
    rev_orig_list.append(n_rev_orig)
    rev_ablat_list.append(n_rev_ablat)

# --- Figure: two-panel ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.5))

x = np.arange(len(task_names))
width = 0.33

# Panel (a): Kendall's tau
bars1 = ax1.bar(x - width/2, tau_orig_list, width,
                label='With thinking tokens',
                color=C_ORIG, alpha=0.85, edgecolor='white', linewidth=0.6)
bars2 = ax1.bar(x + width/2, tau_ablat_list, width,
                label='Without thinking tokens',
                color=C_ABLAT, alpha=0.85, edgecolor='white', linewidth=0.6)
ax1.set_ylabel(r"Kendall's $\tau$")
ax1.set_xticks(x)
ax1.set_xticklabels(task_names, rotation=40, ha='right')
ax1.set_ylim(0, 1.12)
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='none')
ax1.set_title(r'\textbf{(a) Ranking correlation}')
ax1.axhline(y=1.0, color='#aaaaaa', linestyle='--', alpha=0.4, linewidth=0.7)

# Average annotations
avg_orig = np.mean(tau_orig_list)
avg_ablat = np.mean(tau_ablat_list)
ax1.axhline(y=avg_orig, color=C_ORIG, linestyle=':', alpha=0.5, linewidth=1.0)
ax1.axhline(y=avg_ablat, color=C_ABLAT, linestyle=':', alpha=0.5, linewidth=1.0)
ax1.text(len(task_names) - 0.3, avg_orig - 0.04, f'avg $= {avg_orig:.2f}$',
         fontsize=8, color=C_ORIG, ha='right', style='italic')
ax1.text(len(task_names) - 0.3, avg_ablat + 0.02, f'avg $= {avg_ablat:.2f}$',
         fontsize=8, color=C_ABLAT, ha='right', style='italic')

# Improvement arrows on each bar pair
for i in range(len(task_names)):
    delta = tau_ablat_list[i] - tau_orig_list[i]
    ax1.annotate('', xy=(x[i] + width/2, tau_ablat_list[i] - 0.005),
                 xytext=(x[i] - width/2, tau_orig_list[i] + 0.005),
                 arrowprops=dict(arrowstyle='->', color='#555555',
                                 lw=0.7, connectionstyle='arc3,rad=0.25'))

# Panel (b): Pairwise reversals
bars3 = ax2.bar(x - width/2, rev_orig_list, width,
                label='With thinking tokens',
                color=C_ORIG, alpha=0.85, edgecolor='white', linewidth=0.6)
bars4 = ax2.bar(x + width/2, rev_ablat_list, width,
                label='Without thinking tokens',
                color=C_ABLAT, alpha=0.85, edgecolor='white', linewidth=0.6)
ax2.set_ylabel(r'Pairwise ranking reversals')
ax2.set_xticks(x)
ax2.set_xticklabels(task_names, rotation=40, ha='right')
ax2.legend(loc='upper left', framealpha=0.9, edgecolor='none')
ax2.set_title(r'\textbf{(b) Ranking reversals}')

# Average annotations
avg_rev_orig = np.mean(rev_orig_list)
avg_rev_ablat = np.mean(rev_ablat_list)
ax2.axhline(y=avg_rev_orig, color=C_ORIG, linestyle=':', alpha=0.5, linewidth=1.0)
ax2.axhline(y=avg_rev_ablat, color=C_ABLAT, linestyle=':', alpha=0.5, linewidth=1.0)
ax2.text(len(task_names) - 0.3, avg_rev_orig + 0.25, f'avg $= {avg_rev_orig:.1f}$',
         fontsize=8, color=C_ORIG, ha='right', style='italic')
ax2.text(len(task_names) - 0.3, avg_rev_ablat + 0.25, f'avg $= {avg_rev_ablat:.1f}$',
         fontsize=8, color=C_ABLAT, ha='right', style='italic')

# Subtle grid
for ax in (ax1, ax2):
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

plt.tight_layout(w_pad=2.5)
out_path = os.path.join(BASE, "figure", "ablation_thinking_tokens.pdf")
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Saved to {out_path}")
