"""Per-query thinking token variance analysis for §6.1."""
import json, os, math
import statistics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, 'constant/experiment_config.json')) as f:
    exp_config = json.load(f)

models = [(m['model_name'], m['short_name']) for m in exp_config['models']]
datasets = [(d['file_prefix'], d['dataset_name']) for d in exp_config['datasets']]

print("=== PER-QUERY THINKING TOKEN VARIANCE ===\n")
print(f"{'Model':<20} {'Task':<16} {'Mean':>8} {'Std':>8} {'CV':>6} {'Min':>8} {'Max':>8} {'Max/Min':>8}")
print("-" * 96)

all_cvs = []
all_ranges = []

for mn, sn in models:
    for fp, dn in datasets:
        fpath = os.path.join(BASE, f'data/consolidated/{fp}-{mn}.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        
        tt_per_query = [r.get('thinking_tokens', 0) for r in data['records']]
        
        if not tt_per_query or max(tt_per_query) == 0:
            continue
        
        mean_tt = statistics.mean(tt_per_query)
        std_tt = statistics.stdev(tt_per_query) if len(tt_per_query) > 1 else 0
        cv = std_tt / mean_tt if mean_tt > 0 else 0
        min_tt = min(tt_per_query)
        max_tt = max(tt_per_query)
        ratio = max_tt / min_tt if min_tt > 0 else float('inf')
        
        all_cvs.append((sn, dn, cv))
        all_ranges.append((sn, dn, min_tt, max_tt, ratio))
        
        print(f"{sn:<20} {dn:<16} {mean_tt:>8.0f} {std_tt:>8.0f} {cv:>6.2f} {min_tt:>8} {max_tt:>8} {ratio:>8.1f}")

print("\n=== SUMMARY STATISTICS ===")
cvs = [x[2] for x in all_cvs]
print(f"Average CV across all (model, task): {statistics.mean(cvs):.2f}")
print(f"Median CV: {statistics.median(cvs):.2f}")
print(f"Min CV: {min(cvs):.2f}")
print(f"Max CV: {max(cvs):.2f}")

print("\n=== TOP 10 HIGHEST CV (model, task) ===")
all_cvs.sort(key=lambda x: x[2], reverse=True)
for sn, dn, cv in all_cvs[:10]:
    print(f"  {sn:<20} {dn:<16} CV={cv:.2f}")

print("\n=== TOP 10 LARGEST MAX/MIN RATIO ===")
all_ranges.sort(key=lambda x: x[4], reverse=True)
for sn, dn, mn, mx, ratio in all_ranges[:10]:
    print(f"  {sn:<20} {dn:<16} min={mn:>6} max={mx:>8} ratio={ratio:.1f}x")

# Per-query COST variance (not just tokens)
print("\n\n=== PER-QUERY COST VARIANCE ===\n")
print(f"{'Model':<20} {'Task':<16} {'Mean$':>10} {'Std$':>10} {'CV':>6} {'Min$':>10} {'Max$':>10}")
print("-" * 90)

cost_cvs = []
for mn, sn in models:
    for fp, dn in datasets:
        fpath = os.path.join(BASE, f'data/consolidated/{fp}-{mn}.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        
        costs = [r['cost'] for r in data['records']]
        
        if not costs:
            continue
        
        mean_c = statistics.mean(costs)
        std_c = statistics.stdev(costs) if len(costs) > 1 else 0
        cv = std_c / mean_c if mean_c > 0 else 0
        
        cost_cvs.append((sn, dn, cv, mean_c, std_c))
        
        print(f"{sn:<20} {dn:<16} {mean_c:>10.6f} {std_c:>10.6f} {cv:>6.2f} {min(costs):>10.6f} {max(costs):>10.6f}")

print("\n=== COST CV SUMMARY ===")
ccvs = [x[2] for x in cost_cvs]
print(f"Average cost CV: {statistics.mean(ccvs):.2f}")
print(f"Median cost CV: {statistics.median(ccvs):.2f}")

print("\n=== TOP 10 HIGHEST COST CV ===")
cost_cvs.sort(key=lambda x: x[2], reverse=True)
for sn, dn, cv, mc, sc in cost_cvs[:10]:
    print(f"  {sn:<20} {dn:<16} CV={cv:.2f}  mean=${mc:.6f}")
