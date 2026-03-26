"""
Generate the repeated trial figure for Section 6.
Three panels (one per model), raw thinking token counts for ALL AIME queries.
Each query shows a vertical range bar + scatter of 6 runs.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 9,
})

models = {
    'gpt-5.2-high': {
        'orig': 'aime-hybrid-gpt-5.2-high.json',
        'label': 'GPT-5.2',
        'color': '#1f77b4',
    },
    'gpt-5-mini': {
        'orig': 'aime-hybrid-gpt-5-mini.json',
        'label': 'GPT-5 Mini',
        'color': '#ff7f0e',
    },
    'gemini-3-flash-preview': {
        'orig': 'aime-hybrid-gemini-3-flash-preview.json',
        'label': 'Gemini 3 Flash',
        'color': '#2ca02c',
    },
}

data_dir = 'data'

# Collect data — match by index field, not sequential position
all_data = {}

for model_key, info in models.items():
    with open(os.path.join(data_dir, 'consolidated', info['orig'])) as f:
        orig_records = json.load(f)['records']
    orig_by_idx = {r['index']: r for r in orig_records}

    runs_by_idx = []
    for ri in range(5):
        fp = os.path.join(data_dir, 'repeated_trial', 'aime', model_key, f'run{ri}.json')
        if os.path.exists(fp):
            with open(fp) as f:
                recs = json.load(f)['records']
            runs_by_idx.append({r['index']: r for r in recs})

    if not runs_by_idx:
        print(f'{model_key}: no data')
        continue

    # Find indices present in original and at least one run
    all_run_indices = set()
    for run in runs_by_idx:
        all_run_indices |= set(run.keys())
    common_indices = sorted(set(orig_by_idx.keys()) & all_run_indices)

    query_data = []
    for idx in common_indices:
        orig_tt = orig_by_idx[idx]['thinking_tokens']
        trial_tts = [run[idx]['thinking_tokens'] for run in runs_by_idx if idx in run]
        all_tts = [orig_tt] + trial_tts
        query_data.append((idx, all_tts, orig_tt))

    all_data[model_key] = query_data
    print(f'{model_key}: {len(query_data)} queries')

# Plot: 3 panels side by side — all queries, sorted by index
model_keys = list(all_data.keys())

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=False)

for mi, (model_key, ax) in enumerate(zip(model_keys, axes)):
    info = models[model_key]
    query_data = all_data[model_key]
    n_q = len(query_data)

    for xi, (q_idx, raw, orig_tt) in enumerate(query_data):
        x = xi + 1

        # Vertical line (min to max)
        mn, mx = min(raw), max(raw)
        ax.plot([x, x], [mn, mx], color=info['color'], linewidth=1.0, alpha=0.5, zorder=1)

        # Repeated trial points (circles) — small for 60 queries
        trial_vals = raw[1:]
        rng = np.random.default_rng(42 + mi * 100 + xi)
        jitter = rng.uniform(-0.2, 0.2, len(trial_vals))
        ax.scatter([x + j for j in jitter], trial_vals,
                   color=info['color'], s=6, alpha=0.6, zorder=2, edgecolors='none')

        # Original point (star)
        ax.scatter([x], [orig_tt], color=info['color'], s=20, marker='*',
                   zorder=3, edgecolors='none')

    ax.set_title(info['label'], fontsize=10)
    ax.set_xlabel('Query Index')
    # Show sparse x-ticks for readability
    tick_step = 10
    ticks = list(range(tick_step, n_q + 1, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlim(0, n_q + 1)
    ax.tick_params(axis='x', labelsize=7)
    # Format Y axis in thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

axes[0].set_ylabel('Thinking Tokens')

# Add a shared legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='gray', linestyle='None', markersize=8, label='Original run'),
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=4, label='Repeated trial'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=8,
           bbox_to_anchor=(0.5, 1.05), frameon=False)

plt.tight_layout()

# Save
for out_dir in ['figure']:
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'repeated_trial_variance.pdf'),
                bbox_inches='tight', dpi=300)
    print(f'Saved to {out_dir}/repeated_trial_variance.pdf')

plt.close()

# Also print summary stats for the paper
print('\n=== Summary for paper text ===')
for model_key in model_keys:
    info = models[model_key]
    query_data = all_data[model_key]
    cvs = []
    ratios = []
    for _, raw, _ in query_data:
        mn, mx = min(raw), max(raw)
        if mn > 0:
            ratios.append(mx / mn)
            cvs.append(np.std(raw) / np.mean(raw))
    print(f'{info["label"]}: n={len(query_data)}, CV mean={np.mean(cvs):.3f}, max={np.max(cvs):.3f}, '
          f'Max/Min mean={np.mean(ratios):.2f}x, max={np.max(ratios):.2f}x')

# Overall
all_cvs = []
all_ratios = []
for model_key in model_keys:
    for _, raw, _ in all_data[model_key]:
        mn, mx = min(raw), max(raw)
        if mn > 0:
            all_ratios.append(mx / mn)
            all_cvs.append(np.std(raw) / np.mean(raw))
print(f'\nOverall: CV mean={np.mean(all_cvs):.3f}, median={np.median(all_cvs):.3f}')
print(f'Overall: Max/Min mean={np.mean(all_ratios):.2f}x, max={np.max(all_ratios):.2f}x')
print(f'Queries with CV > 0.3: {sum(1 for c in all_cvs if c > 0.3)}/{len(all_cvs)} '
      f'({100*sum(1 for c in all_cvs if c > 0.3)/len(all_cvs):.0f}%)')
print(f'Total queries: {len(all_cvs)}')
