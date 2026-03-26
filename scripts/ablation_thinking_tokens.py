"""
Ablation: What happens to cost rankings when thinking token cost is set to 0?

For each task, compute:
1. Listed price ranking (from model_info.json)
2. Original actual cost ranking (with thinking tokens)
3. Ablation actual cost ranking (thinking tokens cost = 0)

Then compare rankings using Kendall's tau and count ranking inversions.
"""

import json
import os
from itertools import combinations
from scipy.stats import kendalltau

# --- Config ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "consolidated")
MODEL_INFO = os.path.join(BASE, "constant", "model_info.json")
EXP_CONFIG = os.path.join(BASE, "constant", "experiment_config.json")

# Load configs
with open(MODEL_INFO) as f:
    model_info = json.load(f)
with open(EXP_CONFIG) as f:
    exp_config = json.load(f)

# Build pricing lookup: model_name -> (input_price_per_MTok, output_price_per_MTok)
pricing = {}
for m in model_info["models"]:
    pricing[m["model_name"]] = (m["input_price_per_MTok"], m["output_price_per_MTok"])

# Models and datasets from experiment config
models = [(m["model_name"], m["short_name"]) for m in exp_config["models"]]
datasets = [(d["file_prefix"], d["dataset_name"]) for d in exp_config["datasets"]]

# Listed price for each model = input_price + output_price (per MTok)
listed_prices = {}
for model_name, short_name in models:
    ip, op = pricing[model_name]
    listed_prices[model_name] = ip + op

print("=" * 80)
print("ABLATION: Removing Thinking Token Costs")
print("=" * 80)
print()

# --- Compute costs per (task, model) ---
results = {}

for file_prefix, dataset_name in datasets:
    for model_name, short_name in models:
        fpath = os.path.join(DATA_DIR, f"{file_prefix}-{model_name}.json")
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            data = json.load(f)

        ip, op = pricing[model_name]

        total_original = 0.0
        total_ablation = 0.0
        total_thinking = 0
        total_completion = 0
        total_prompt = 0

        for rec in data["records"]:
            pt = rec["prompt_tokens"]
            ct = rec["completion_tokens"]
            tt = rec.get("thinking_tokens", 0)
            gen_tokens = ct - tt  # non-thinking output tokens

            original_cost = (pt * ip + ct * op) / 1e6
            ablation_cost = (pt * ip + gen_tokens * op) / 1e6

            total_original += original_cost
            total_ablation += ablation_cost
            total_thinking += tt
            total_completion += ct
            total_prompt += pt

        results[(dataset_name, model_name)] = {
            "original_cost": total_original,
            "ablation_cost": total_ablation,
            "thinking_tokens": total_thinking,
            "completion_tokens": total_completion,
            "prompt_tokens": total_prompt,
            "thinking_pct": total_thinking / total_completion * 100 if total_completion > 0 else 0,
        }

# --- Ranking analysis per task ---
print(f"{'Task':<16} {'tau(list,orig)':>16} {'tau(list,ablat)':>16} {'d_tau':>8} {'Rev_orig':>10} {'Rev_ablat':>10} {'dRev':>8}")
print("-" * 90)

model_names = [m[0] for m in models]
all_tau_orig = []
all_tau_ablat = []
all_rev_orig = []
all_rev_ablat = []

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
    n_pairs = 0
    for i, j in combinations(range(len(model_names)), 2):
        n_pairs += 1
        lp_order = (lp[i] < lp[j])
        orig_order = (orig_costs[i] < orig_costs[j])
        ablat_order = (ablat_costs[i] < ablat_costs[j])
        if lp_order != orig_order:
            n_rev_orig += 1
        if lp_order != ablat_order:
            n_rev_ablat += 1

    all_tau_orig.append(tau_orig)
    all_tau_ablat.append(tau_ablat)
    all_rev_orig.append(n_rev_orig)
    all_rev_ablat.append(n_rev_ablat)

    print(f"{dataset_name:<16} {tau_orig:>16.3f} {tau_ablat:>16.3f} {tau_ablat - tau_orig:>+8.3f} {n_rev_orig:>10} {n_rev_ablat:>10} {n_rev_ablat - n_rev_orig:>+8}")

avg_tau_orig = sum(all_tau_orig) / len(all_tau_orig)
avg_tau_ablat = sum(all_tau_ablat) / len(all_tau_ablat)
avg_rev_orig = sum(all_rev_orig) / len(all_rev_orig)
avg_rev_ablat = sum(all_rev_ablat) / len(all_rev_ablat)

print("-" * 90)
print(f"{'AVERAGE':<16} {avg_tau_orig:>16.3f} {avg_tau_ablat:>16.3f} {avg_tau_ablat - avg_tau_orig:>+8.3f} {avg_rev_orig:>10.1f} {avg_rev_ablat:>10.1f} {avg_rev_ablat - avg_rev_orig:>+8.1f}")
print()

# --- Per-model thinking token stats ---
print("=" * 80)
print("THINKING TOKEN STATISTICS (aggregated across all tasks)")
print("=" * 80)
print(f"{'Model':<30} {'Think Tok':>12} {'Total Tok':>12} {'Think%':>10} {'Orig Cost':>12} {'Ablat Cost':>12} {'Cost Red%':>10}")
print("-" * 100)

for model_name, short_name in models:
    total_think = 0
    total_comp = 0
    total_orig = 0.0
    total_ablat = 0.0
    for _, dn in datasets:
        key = (dn, model_name)
        if key in results:
            total_think += results[key]["thinking_tokens"]
            total_comp += results[key]["completion_tokens"]
            total_orig += results[key]["original_cost"]
            total_ablat += results[key]["ablation_cost"]
    think_pct = total_think / total_comp * 100 if total_comp > 0 else 0
    cost_reduction = (total_orig - total_ablat) / total_orig * 100 if total_orig > 0 else 0
    print(f"{short_name:<30} {total_think:>12,} {total_comp:>12,} {think_pct:>9.1f}% ${total_orig:>11.2f} ${total_ablat:>11.2f} {cost_reduction:>+9.1f}%")

print()

# --- Per-task detailed cost table ---
print("=" * 80)
print("PER-TASK COST COMPARISON: Original vs Ablation")
print("=" * 80)
for file_prefix, dataset_name in datasets:
    print(f"\n--- {dataset_name} ---")
    print(f"  {'Model':<25} {'Listed$':>10} {'OrigCost':>12} {'AblatCost':>12} {'Think%':>8} {'R(L)':>6} {'R(O)':>6} {'R(A)':>6}")

    model_data = []
    for model_name, short_name in models:
        key = (dataset_name, model_name)
        if key not in results:
            continue
        r = results[key]
        model_data.append((model_name, short_name, listed_prices[model_name],
                          r["original_cost"], r["ablation_cost"], r["thinking_pct"]))

    lp_sorted = sorted(range(len(model_data)), key=lambda i: model_data[i][2])
    orig_sorted = sorted(range(len(model_data)), key=lambda i: model_data[i][3])
    ablat_sorted = sorted(range(len(model_data)), key=lambda i: model_data[i][4])

    lp_ranks = [0] * len(model_data)
    orig_ranks = [0] * len(model_data)
    ablat_ranks = [0] * len(model_data)
    for rank, idx in enumerate(lp_sorted):
        lp_ranks[idx] = rank + 1
    for rank, idx in enumerate(orig_sorted):
        orig_ranks[idx] = rank + 1
    for rank, idx in enumerate(ablat_sorted):
        ablat_ranks[idx] = rank + 1

    for i, (mn, sn, lp, oc, ac, tp) in enumerate(model_data):
        print(f"  {sn:<25} ${lp:>9.2f} ${oc:>11.2f} ${ac:>11.2f} {tp:>7.1f}% {lp_ranks[i]:>6} {orig_ranks[i]:>6} {ablat_ranks[i]:>6}")
