"""Quantify prevalence, severity, and task-dependence of pricing reversal."""
import json, os
from itertools import combinations

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, 'constant/model_info.json')) as f:
    model_info = json.load(f)
with open(os.path.join(BASE, 'constant/experiment_config.json')) as f:
    exp_config = json.load(f)

pricing = {}
for m in model_info['models']:
    pricing[m['model_name']] = (m['input_price_per_MTok'], m['output_price_per_MTok'])

models = [(m['model_name'], m['short_name']) for m in exp_config['models']]
datasets = [(d['file_prefix'], d['dataset_name']) for d in exp_config['datasets']]

listed_prices = {}
for mn, sn in models:
    ip, op = pricing[mn]
    listed_prices[mn] = ip + op

model_names = [m[0] for m in models]
short_names = {m[0]: m[1] for m in models}

costs = {}
for fp, dn in datasets:
    for mn, sn in models:
        fpath = os.path.join(BASE, f'data/consolidated/{fp}-{mn}.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        costs[(dn, mn)] = sum(r['cost'] for r in data['records'])

# === Prevalence ===
total_comparisons = 0
total_reversals = 0
reversals_per_task = {}

for fp, dn in datasets:
    rev = 0
    pairs = 0
    for mi, mj in combinations(model_names, 2):
        if (dn, mi) not in costs or (dn, mj) not in costs:
            continue
        pairs += 1
        if (listed_prices[mi] < listed_prices[mj]) != (costs[(dn, mi)] < costs[(dn, mj)]):
            rev += 1
    reversals_per_task[dn] = (rev, pairs)
    total_comparisons += pairs
    total_reversals += rev

print("=== PREVALENCE ===")
print(f"Total pairs per task: {len(list(combinations(model_names, 2)))}")
print(f"Total comparisons (pairs x tasks): {total_comparisons}")
print(f"Total reversals: {total_reversals}")
print(f"Overall reversal rate: {total_reversals/total_comparisons*100:.1f}%")
print()
for dn, (rev, pairs) in reversals_per_task.items():
    print(f"  {dn:<16} {rev}/{pairs} = {rev/pairs*100:.1f}%")

# === Severity ===
print("\n=== TOP 15 MOST EXTREME REVERSALS ===")
extreme = []
for fp, dn in datasets:
    for mi, mj in combinations(model_names, 2):
        if (dn, mi) not in costs or (dn, mj) not in costs:
            continue
        lp_i, lp_j = listed_prices[mi], listed_prices[mj]
        ac_i, ac_j = costs[(dn, mi)], costs[(dn, mj)]
        if (lp_i < lp_j) != (ac_i < ac_j):
            if lp_i < lp_j:
                price_ratio = lp_j / lp_i
                cost_ratio = ac_i / ac_j
                extreme.append((dn, short_names[mi], short_names[mj], price_ratio, cost_ratio))
            else:
                price_ratio = lp_i / lp_j
                cost_ratio = ac_j / ac_i
                extreme.append((dn, short_names[mj], short_names[mi], price_ratio, cost_ratio))

extreme.sort(key=lambda x: x[4], reverse=True)
for dn, cheap, expensive, pr, cr in extreme[:15]:
    print(f"  {dn:<14} {cheap:<20} vs {expensive:<20} price {pr:.1f}x cheaper, actual {cr:.1f}x MORE expensive")

# === Task dependence ===
print("\n=== CHEAPEST MODEL PER TASK ===")
cheapest_set = set()
for fp, dn in datasets:
    best = min(model_names, key=lambda m: costs.get((dn, m), float('inf')))
    cheapest_set.add(short_names[best])
    print(f"  {dn:<16} cheapest: {short_names[best]:<20} cost=${costs[(dn, best)]:.2f}")

print(f"\nDistinct cheapest models across 9 tasks: {len(cheapest_set)}")
print(f"Models: {sorted(cheapest_set)}")

# === Per-model: how often is it cheapest vs how price ranks it? ===
print("\n=== MODEL RANKING SUMMARY ===")
lp_rank = sorted(model_names, key=lambda m: listed_prices[m])
print(f"Listed price ranking: {[short_names[m] for m in lp_rank]}")
for mn in model_names:
    cheapest_count = sum(1 for fp, dn in datasets
                        if min(model_names, key=lambda m: costs.get((dn, m), float('inf'))) == mn)
    print(f"  {short_names[mn]:<20} cheapest on {cheapest_count}/9 tasks, listed price rank: {lp_rank.index(mn)+1}")
