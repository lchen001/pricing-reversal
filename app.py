"""
Price Reversal Explorer — Interactive data browser for the paper
"The Price Reversal Phenomenon: When Cheaper Reasoning Models End Up Costing More"

Run:  streamlit run app.py
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "consolidated")
RT_DIR = os.path.join(BASE, "data", "repeated_trial")
MODEL_INFO_PATH = os.path.join(BASE, "constant", "model_info.json")
EXP_CONFIG_PATH = os.path.join(BASE, "constant", "experiment_config.json")


@st.cache_data
def load_all():
    """Load all consolidated data + configs once (text fields stripped to save memory)."""
    with open(MODEL_INFO_PATH) as f:
        model_info = json.load(f)
    with open(EXP_CONFIG_PATH) as f:
        exp_config = json.load(f)

    # Build pricing / short-name lookups
    pricing = {}
    short_names = {}
    for m in model_info["models"]:
        pricing[m["model_name"]] = (
            m["input_price_per_MTok"],
            m["output_price_per_MTok"],
        )
        short_names[m["model_name"]] = m["short_name"]

    # Dataset config
    ds_config = {d["dataset_name"]: d for d in exp_config["datasets"]}
    model_list = [m["model_name"] for m in exp_config["models"]]
    all_model_names = [m["model_name"] for m in model_info["models"]]

    # Text fields to strip from in-memory data (loaded on demand from disk)
    _TEXT_FIELDS = {"origin_query", "prompt", "raw_output", "prediction", "ground_truth", "extra_fields"}

    # Load consolidated data into a dict keyed by (dataset, model)
    # Store file paths for on-demand text loading
    data = {}
    file_paths = {}
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        with open(fpath) as f:
            d = json.load(f)
        key = (d["dataset_name"], d["model_name"])
        file_paths[key] = fpath
        # Keep only a short preview per record; strip large text fields
        for rec in d["records"]:
            rec["_preview"] = (rec.get("origin_query", rec.get("prompt", "")) or "")[:80]
            for field in _TEXT_FIELDS:
                rec.pop(field, None)
        data[key] = d

    return data, pricing, short_names, ds_config, model_list, all_model_names, file_paths


def load_full_records(file_path):
    """Load a single JSON file from disk with all text fields (not cached globally)."""
    with open(file_path) as f:
        d = json.load(f)
    return {r["index"]: r for r in d["records"]}


@st.cache_data
def load_repeated_trials():
    """Load AIME repeated trial data."""
    rt = {}
    for model_dir in sorted(glob.glob(os.path.join(RT_DIR, "aime", "*"))):
        model = os.path.basename(model_dir)
        runs = []
        for ri in range(5):
            fp = os.path.join(model_dir, f"run{ri}.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    runs.append(json.load(f))
        if runs:
            rt[model] = runs
    return rt


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def listed_cost(pricing, model, prompt_tok, completion_tok, thinking_tok=0):
    """Compute listed (nominal) cost excluding hidden thinking tokens."""
    inp, out = pricing[model]
    visible_completion = completion_tok - (thinking_tok or 0)
    return prompt_tok / 1e6 * inp + visible_completion / 1e6 * out


def sn(short_names, model):
    return short_names.get(model, model)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Price Reversal Explorer",
    page_icon="💰",
    layout="wide",
)

data, pricing, short_names, ds_config, model_list, all_model_names, file_paths = load_all()

datasets = sorted(set(k[0] for k in data.keys()))
models_in_data = sorted(set(k[1] for k in data.keys()))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

logo_path = os.path.join(BASE, "asset", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=120)
st.sidebar.title("Price Reversal Explorer")
page = st.sidebar.radio(
    "Navigate",
    [
        "🔄 Pricing Reversal",
        "📊 Cost Breakdown",
        "🔍 Per-Query Deep Dive",
        "⚔️ Query-Level Comparison",
        "🎲 Repeated Trial Variance",
    ],
)

# ===================================================================
# PAGE 1: Pricing Reversal Explorer
# ===================================================================
if page == "🔄 Pricing Reversal":
    st.title("Pricing Reversal Explorer")
    st.markdown(
        "Compare two models: which one has a lower **listed price** vs lower **actual cost**? "
        "A *pricing reversal* occurs when the cheaper-listed model actually costs more."
    )

    # Weight slider: composite listed price = w * input_price + (1-w) * output_price
    w = st.slider(
        "Input price weight (listed price = w × input \\$/MTok + (1−w) × output \\$/MTok)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="price_weight",
    )

    def composite_price(model):
        inp, out = pricing[model]
        return w * inp + (1 - w) * out

    # Initialize defaults in session state (avoids index reset on rerun)
    if "reversal_model_a" not in st.session_state:
        st.session_state["reversal_model_a"] = (
            "gemini-3-flash-preview"
            if "gemini-3-flash-preview" in models_in_data
            else models_in_data[0]
        )
    if "reversal_model_b" not in st.session_state:
        default_b = "gpt-5.2-high" if "gpt-5.2-high" in models_in_data else models_in_data[0]
        if default_b == st.session_state["reversal_model_a"] and len(models_in_data) > 1:
            default_b = [m for m in models_in_data if m != st.session_state["reversal_model_a"]][0]
        st.session_state["reversal_model_b"] = default_b

    col1, col2 = st.columns(2)
    with col1:
        model_a = st.selectbox(
            "Model A",
            models_in_data,
            format_func=lambda x: sn(short_names, x),
            key="reversal_model_a",
        )
    with col2:
        avail_b = [m for m in models_in_data if m != model_a]
        # Ensure stored value is still valid after model_a changes
        if st.session_state.get("reversal_model_b") not in avail_b:
            st.session_state["reversal_model_b"] = avail_b[0] if avail_b else None
        model_b = st.selectbox(
            "Model B",
            avail_b,
            format_func=lambda x: sn(short_names, x),
            key="reversal_model_b",
        )

    price_a = composite_price(model_a)
    price_b = composite_price(model_b)
    sn_a = sn(short_names, model_a)
    sn_b = sn(short_names, model_b)

    # --- Listed price side-by-side metrics ---
    pcol1, pcol2, pcol3 = st.columns([2, 2, 3])
    cheaper_listed = sn_a if price_a < price_b else sn_b if price_b < price_a else "Tie"
    with pcol1:
        st.metric(f"{sn_a} listed price", f"${price_a:.2f}/MTok")
    with pcol2:
        st.metric(f"{sn_b} listed price", f"${price_b:.2f}/MTok")
    with pcol3:
        if price_a != price_b:
            if price_a < price_b:
                ratio_text = f"{sn_a} costs {price_b/price_a:.1f}× less"
            else:
                ratio_text = f"{sn_b} costs {price_a/price_b:.1f}× less"
            st.metric("Listed price verdict", ratio_text)
        else:
            st.metric("Listed price verdict", "Same price")

    # --- Compute per-dataset comparison ---
    rows = []
    for ds in datasets:
        da = data.get((ds, model_a))
        db = data.get((ds, model_b))
        if da is None or db is None:
            continue

        actual_a = da["cost"]
        actual_b = db["cost"]
        n_queries = len(da["records"])

        reversal = (price_a < price_b) != (actual_a < actual_b)
        cost_ratio = actual_a / actual_b if actual_b > 0 else float("inf")
        rows.append(
            {
                "Dataset": ds,
                f"Actual {sn_a} ($)": actual_a,
                f"Actual {sn_b} ($)": actual_b,
                f"Avg cost/query {sn_a} ($)": actual_a / max(n_queries, 1),
                f"Avg cost/query {sn_b} ($)": actual_b / max(n_queries, 1),
                "Cost Ratio (A/B)": cost_ratio,
                "Reversal": reversal,
                "Listed cheaper": sn_a if price_a < price_b else sn_b,
                "Actually cheaper": sn_a if actual_a < actual_b else sn_b,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        n_reversal = df["Reversal"].sum()
        st.metric(
            "Pricing Reversals",
            f"{n_reversal} / {len(df)} datasets",
            delta=f"{n_reversal/len(df)*100:.0f}% reversal rate"
            if n_reversal > 0
            else "No reversals",
            delta_color="inverse",
        )

        # ==============================================================
        # CHART 1: Actual Cost side-by-side (original bar chart)
        # ==============================================================
        st.subheader("Actual Cost per Dataset")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df["Dataset"],
                y=df[f"Actual {sn_a} ($)"],
                name=sn_a,
                marker_color="#1f77b4",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["Dataset"],
                y=df[f"Actual {sn_b} ($)"],
                name=sn_b,
                marker_color="#ff7f0e",
            )
        )
        for _, row in df[df["Reversal"]].iterrows():
            fig.add_annotation(
                x=row["Dataset"],
                y=max(row[f"Actual {sn_a} ($)"], row[f"Actual {sn_b} ($)"]),
                text="⚠️ reversal",
                showarrow=False,
                yshift=12,
                font=dict(size=12),
            )
        fig.update_layout(
            barmode="group",
            height=400,
            margin=dict(t=40, b=0),
            yaxis_title="Actual Cost ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ==============================================================
        # CHART 2: Cost Ratio vs Listed Price Ratio
        # This is the key chart — shows the gap between listed price
        # ranking and actual cost ranking at a glance.
        # ==============================================================
        st.subheader("Listed Price Ratio vs Actual Cost Ratio")
        st.caption(
            f"Each bar = actual cost ratio ({sn_a} / {sn_b}) per dataset.  "
            f"Dashed line = listed price ratio.  "
            f"Red bars = pricing reversal (actual ranking contradicts listed price)."
        )

        price_ratio = price_a / price_b if price_b > 0 else 1.0
        bar_colors = [
            "#d62728" if row["Reversal"] else "#2ca02c"
            for _, row in df.iterrows()
        ]

        fig_ratio = go.Figure()
        fig_ratio.add_trace(
            go.Bar(
                x=df["Dataset"],
                y=df["Cost Ratio (A/B)"],
                marker_color=bar_colors,
                text=[f"{v:.2f}×" for v in df["Cost Ratio (A/B)"]],
                textposition="outside",
                name="Actual cost ratio",
            )
        )
        # Reference line: listed price ratio
        fig_ratio.add_hline(
            y=price_ratio,
            line_dash="dash",
            line_color="#555",
            line_width=2,
            annotation_text=f"Listed price ratio = {price_ratio:.2f}×",
            annotation_position="top left",
            annotation_font_size=13,
        )
        # Reference line: equal cost
        fig_ratio.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#aaa",
            line_width=1,
            annotation_text="Equal cost (1.0×)",
            annotation_position="bottom right",
            annotation_font_size=11,
        )
        fig_ratio.update_layout(
            height=420,
            margin=dict(t=50, b=0),
            yaxis_title=f"Cost Ratio ({sn_a} / {sn_b})",
            showlegend=False,
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

        # ==============================================================
        # Detail table
        # ==============================================================
        st.subheader("Detail Table")
        styled = df.style.apply(
            lambda row: [
                "background-color: #ffcccc" if row["Reversal"] else "" for _ in row
            ],
            axis=1,
        )
        st.dataframe(
            styled,
            hide_index=True,
            column_config={
                "Reversal": st.column_config.CheckboxColumn("Reversal?"),
                "Cost Ratio (A/B)": st.column_config.NumberColumn(
                    "Cost Ratio (A/B)", format="%.2f×"
                ),
            },
        )

# ===================================================================
# PAGE 2: Cost Breakdown
# ===================================================================
elif page == "📊 Cost Breakdown":
    st.title("Cost & Token Breakdown")
    st.markdown(
        "See how prompt, thinking, and generation tokens contribute to total cost per model."
    )

    ds_choice = st.selectbox("Dataset", datasets)

    rows = []
    for model in models_in_data:
        d = data.get((ds_choice, model))
        if d is None:
            continue
        inp_p, out_p = pricing.get(model, (0, 0))
        total_prompt = sum(r["prompt_tokens"] for r in d["records"])
        total_thinking = sum(r.get("thinking_tokens", 0) or 0 for r in d["records"])
        total_completion = sum(r["completion_tokens"] for r in d["records"])
        total_generation = total_completion - total_thinking

        prompt_cost = total_prompt / 1e6 * inp_p
        thinking_cost = total_thinking / 1e6 * out_p
        generation_cost = total_generation / 1e6 * out_p

        rows.append(
            {
                "Model": sn(short_names, model),
                "Prompt Tokens": total_prompt,
                "Thinking Tokens": total_thinking,
                "Generation Tokens": total_generation,
                "Prompt Cost ($)": prompt_cost,
                "Thinking Cost ($)": thinking_cost,
                "Generation Cost ($)": generation_cost,
                "Total Cost ($)": d["cost"],
                "Accuracy": d.get("performance", 0),
                "Thinking %": total_thinking / max(total_completion, 1) * 100,
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values("Total Cost ($)", ascending=False)

        # Stacked bar: cost breakdown
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df["Prompt Cost ($)"],
                name="Prompt",
                marker_color="#636efa",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df["Thinking Cost ($)"],
                name="Thinking",
                marker_color="#ef553b",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df["Generation Cost ($)"],
                name="Generation",
                marker_color="#00cc96",
            )
        )
        fig.update_layout(
            barmode="stack",
            yaxis_title="Cost ($)",
            height=450,
            margin=dict(t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics row
        col1, col2, col3 = st.columns(3)
        max_think = df.loc[df["Thinking Tokens"].idxmax()]
        min_think = df.loc[df["Thinking Tokens"].idxmin()]
        ratio = (
            max_think["Thinking Tokens"] / max(min_think["Thinking Tokens"], 1)
        )
        col1.metric("Max Thinking Tokens", f'{int(max_think["Thinking Tokens"]):,}', max_think["Model"])
        col2.metric("Min Thinking Tokens", f'{int(min_think["Thinking Tokens"]):,}', min_think["Model"])
        col3.metric("Thinking Token Ratio", f"{ratio:.0f}x")

        st.dataframe(
            df.style.format(
                {
                    "Prompt Cost ($)": "${:.4f}",
                    "Thinking Cost ($)": "${:.4f}",
                    "Generation Cost ($)": "${:.4f}",
                    "Total Cost ($)": "${:.4f}",
                    "Accuracy": "{:.1%}",
                    "Thinking %": "{:.1f}%",
                }
            ),
            hide_index=True,
        )

# ===================================================================
# PAGE 3: Per-Query Deep Dive
# ===================================================================
elif page == "🔍 Per-Query Deep Dive":
    st.title("Per-Query Deep Dive")
    st.markdown(
        "Pick a dataset and query to see how every model behaves — "
        "token usage, cost, and whether it got the answer right."
    )

    ds_choice = st.selectbox("Dataset", datasets)

    # Get available queries (use the first model that has this dataset)
    sample_key = next((k for k in data if k[0] == ds_choice), None)
    if sample_key is None:
        st.warning("No data for this dataset.")
        st.stop()

    records = data[sample_key]["records"]
    query_indices = sorted(set(r["index"] for r in records))

    # Build a preview map
    query_preview = {}
    for r in records:
        text = r.get("_preview", "")[:80]
        query_preview[r["index"]] = f"Q{r['index']}: {text}..."

    qi = st.selectbox(
        "Query",
        query_indices,
        format_func=lambda x: query_preview.get(x, f"Q{x}"),
    )

    rows = []
    for model in models_in_data:
        d = data.get((ds_choice, model))
        if d is None:
            continue
        rec = next((r for r in d["records"] if r["index"] == qi), None)
        if rec is None:
            continue
        rows.append(
            {
                "Model": sn(short_names, model),
                "model_name": model,
                "Prompt Tokens": rec["prompt_tokens"],
                "Thinking Tokens": rec.get("thinking_tokens", 0) or 0,
                "Completion Tokens": rec["completion_tokens"],
                "Cost ($)": rec["cost"],
                "Score": rec["score"],
                "Prediction": "",
                "Ground Truth": "",
            }
        )

    if rows:
        df = pd.DataFrame(rows)

        # Bar chart: thinking tokens per model
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Thinking Tokens", "Cost ($)"],
        )
        colors = [
            "#ef553b" if s == 0 else "#00cc96"
            for s in df["Score"]
        ]
        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df["Thinking Tokens"],
                marker_color=colors,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df["Cost ($)"],
                marker_color=colors,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        # Add legend manually
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#00cc96", name="Correct"))
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#ef553b", name="Incorrect"))
        fig.update_layout(height=400, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Detail table
        st.dataframe(
            df.drop(columns=["model_name"]).style.format(
                {"Cost ($)": "${:.6f}", "Score": "{:.0f}"}
            ),
            hide_index=True,
        )

        # Show the query text & model responses (load full text on demand)
        st.subheader("Model Responses")
        full_records_cache = {}
        for _, row in df.iterrows():
            mname = row["Model"]
            mkey = row["model_name"]
            fpath = file_paths.get((ds_choice, mkey))
            if fpath is None:
                continue
            if mkey not in full_records_cache:
                full_records_cache[mkey] = load_full_records(fpath)
            full_rec = full_records_cache[mkey].get(qi)
            if full_rec is None:
                continue
            rec = next((r for r in data[(ds_choice, mkey)]["records"] if r["index"] == qi), None)
            with st.expander(f"{mname}  —  tokens: {rec.get('prompt_tokens',0):,} prompt / {rec.get('thinking_tokens',0):,} thinking / {rec.get('completion_tokens',0):,} completion  |  cost: ${rec.get('cost',0):.6f}  |  score: {rec.get('score','N/A')}"):
                raw = full_rec.get("raw_output", full_rec.get("prediction", ""))
                st.text(raw if raw else "(no response recorded)")

        # Show full query text from the first model
        first_model_key = df.iloc[0]["model_name"]
        if first_model_key in full_records_cache:
            full_sample = full_records_cache[first_model_key].get(qi, {})
            with st.expander("Show full query text"):
                st.text(full_sample.get("origin_query", full_sample.get("prompt", "")))

# ===================================================================
# PAGE 4: Query-Level Comparison
# ===================================================================
elif page == "⚔️ Query-Level Comparison":
    st.title("Query-Level Model Comparison")
    st.markdown(
        "Pick two models and a dataset to compare them **query by query**. "
        "See exactly which queries cause pricing reversals and how thinking tokens diverge."
    )

    ds_choice = st.selectbox("Dataset", datasets, key="qlc_ds")

    available = [m for m in models_in_data if (ds_choice, m) in data]
    col1, col2 = st.columns(2)
    with col1:
        model_a = st.selectbox(
            "Model A",
            available,
            index=available.index("gemini-3-flash-preview")
            if "gemini-3-flash-preview" in available
            else 0,
            format_func=lambda x: sn(short_names, x),
            key="qlc_a",
        )
    with col2:
        avail_b = [m for m in available if m != model_a]
        model_b = st.selectbox(
            "Model B",
            avail_b,
            index=avail_b.index("gpt-5.2-high")
            if "gpt-5.2-high" in avail_b
            else 0,
            format_func=lambda x: sn(short_names, x),
            key="qlc_b",
        )

    da = data[(ds_choice, model_a)]
    db = data[(ds_choice, model_b)]

    recs_a = {r["index"]: r for r in da["records"]}
    recs_b = {r["index"]: r for r in db["records"]}
    common_idx = sorted(set(recs_a.keys()) & set(recs_b.keys()))

    if not common_idx:
        st.warning("No overlapping queries between these two models.")
        st.stop()

    sn_a = sn(short_names, model_a)
    sn_b = sn(short_names, model_b)
    inp_a, out_a = pricing.get(model_a, (0, 0))
    inp_b, out_b = pricing.get(model_b, (0, 0))

    rows = []
    for idx in common_idx:
        ra, rb = recs_a[idx], recs_b[idx]
        cost_a = ra["cost"]
        cost_b = rb["cost"]
        tt_a = ra.get("thinking_tokens", 0) or 0
        tt_b = rb.get("thinking_tokens", 0) or 0
        listed_a = ra["prompt_tokens"] / 1e6 * inp_a + (ra["completion_tokens"] - tt_a) / 1e6 * out_a
        listed_b = rb["prompt_tokens"] / 1e6 * inp_b + (rb["completion_tokens"] - tt_b) / 1e6 * out_b
        reversal = (listed_a < listed_b) != (cost_a < cost_b)
        rows.append(
            {
                "Query": idx,
                f"Cost {sn_a} ($)": cost_a,
                f"Cost {sn_b} ($)": cost_b,
                "Cost Diff ($)": cost_a - cost_b,
                f"Thinking {sn_a}": tt_a,
                f"Thinking {sn_b}": tt_b,
                "Thinking Diff": tt_a - tt_b,
                f"Score {sn_a}": ra["score"],
                f"Score {sn_b}": rb["score"],
                "Reversal": reversal,
                "Preview": ra.get("_preview", "")[:60],
            }
        )

    df = pd.DataFrame(rows)
    n_rev = df["Reversal"].sum()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queries", len(df))
    col2.metric("Reversals", f"{n_rev} ({n_rev/len(df)*100:.0f}%)")
    avg_diff = df["Cost Diff ($)"].mean()
    col3.metric(
        f"Avg Cost Diff",
        f"${avg_diff:+.5f}",
        help=f"Positive = {sn_a} costs more",
    )
    a_wins = (df[f"Cost {sn_a} ($)"] < df[f"Cost {sn_b} ($)"]).sum()
    col4.metric(f"{sn_a} cheaper on", f"{a_wins}/{len(df)} queries")

    # Scatter: cost A vs cost B
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Per-Query Cost: {sn_a} vs {sn_b}",
            f"Per-Query Thinking Tokens: {sn_a} vs {sn_b}",
        ],
    )

    colors = ["#ef553b" if r else "#636efa" for r in df["Reversal"]]

    fig.add_trace(
        go.Scatter(
            x=df[f"Cost {sn_a} ($)"],
            y=df[f"Cost {sn_b} ($)"],
            mode="markers",
            marker=dict(size=7, color=colors, opacity=0.7),
            text=df["Preview"],
            customdata=df["Query"],
            hovertemplate=(
                "Q%{customdata}<br>"
                f"{sn_a}: $%{{x:.5f}}<br>"
                f"{sn_b}: $%{{y:.5f}}<br>"
                "%{text}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df[f"Thinking {sn_a}"],
            y=df[f"Thinking {sn_b}"],
            mode="markers",
            marker=dict(size=7, color=colors, opacity=0.7),
            text=df["Preview"],
            customdata=df["Query"],
            hovertemplate=(
                "Q%{customdata}<br>"
                f"{sn_a}: %{{x:,}} tokens<br>"
                f"{sn_b}: %{{y:,}} tokens<br>"
                "%{text}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Add diagonal reference lines
    for col_idx, (xcol, ycol) in enumerate(
        [
            (f"Cost {sn_a} ($)", f"Cost {sn_b} ($)"),
            (f"Thinking {sn_a}", f"Thinking {sn_b}"),
        ],
        start=1,
    ):
        max_val = max(df[xcol].max(), df[ycol].max()) * 1.05
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_xaxes(title_text=sn_a, row=1, col=1)
    fig.update_yaxes(title_text=sn_b, row=1, col=1)
    fig.update_xaxes(title_text=sn_a, row=1, col=2)
    fig.update_yaxes(title_text=sn_b, row=1, col=2)

    # Manual legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#636efa", size=8), name="No reversal"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#ef553b", size=8), name="Reversal"))

    fig.update_layout(height=450, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Points above the diagonal: Model B costs more (or uses more thinking tokens). "
        "Red = pricing reversal (listed-price ordering disagrees with actual-cost ordering)."
    )

    # Sortable table
    st.subheader("Per-Query Detail")
    sort_col = st.selectbox(
        "Sort by",
        ["Cost Diff ($)", "Thinking Diff", "Query", f"Cost {sn_a} ($)", f"Cost {sn_b} ($)"],
        key="qlc_sort",
    )
    ascending = st.checkbox("Ascending", value=False, key="qlc_asc")
    df_show = df.sort_values(sort_col, ascending=ascending)
    st.dataframe(
        df_show.style.apply(
            lambda row: ["background-color: #ffcccc" if row["Reversal"] else "" for _ in row],
            axis=1,
        ).format(
            {
                f"Cost {sn_a} ($)": "${:.6f}",
                f"Cost {sn_b} ($)": "${:.6f}",
                "Cost Diff ($)": "${:+.6f}",
                f"Score {sn_a}": "{:.0f}",
                f"Score {sn_b}": "{:.0f}",
            }
        ),
        hide_index=True,
        height=500,
    )

    # --- Query drill-down ---
    st.subheader("Query Drill-Down")
    query_options = df_show["Query"].tolist()
    if query_options:
        chosen_q = st.selectbox("Select a query to inspect", query_options, key="qlc_drill")
        row = df[df["Query"] == chosen_q].iloc[0]

        # Load full text on demand for drill-down
        full_a = load_full_records(file_paths[(ds_choice, model_a)]).get(chosen_q, {})
        full_b = load_full_records(file_paths[(ds_choice, model_b)]).get(chosen_q, {})

        sample_full = full_a or full_b
        if sample_full:
            with st.expander("Show full query text", expanded=False):
                st.text(sample_full.get("origin_query", sample_full.get("prompt", "")))

        # side-by-side responses
        rec_a = recs_a.get(chosen_q)
        rec_b = recs_b.get(chosen_q)
        col_left, col_right = st.columns(2)
        for col, label, rec, full_rec in [(col_left, sn_a, rec_a, full_a), (col_right, sn_b, rec_b, full_b)]:
            with col:
                st.markdown(f"**{label}**")
                if rec:
                    st.markdown(
                        f"Prompt: **{rec.get('prompt_tokens',0):,}** &nbsp;|&nbsp; "
                        f"Thinking: **{rec.get('thinking_tokens',0):,}** &nbsp;|&nbsp; "
                        f"Completion: **{rec.get('completion_tokens',0):,}** &nbsp;|&nbsp; "
                        f"Cost: **${rec.get('cost',0):.6f}** &nbsp;|&nbsp; "
                        f"Score: **{rec.get('score','N/A')}**"
                    )
                    raw = full_rec.get("raw_output", full_rec.get("prediction", "")) if full_rec else ""
                    st.text_area("Response", value=raw if raw else "(no response)", height=300, key=f"qlc_resp_{label}", disabled=True)
                else:
                    st.info("No data")

# ===================================================================
# PAGE 5: Repeated Trial Variance
# ===================================================================
elif page == "🎲 Repeated Trial Variance":
    st.title("Repeated Trial Variance (AIME)")
    st.markdown(
        "The same query sent to the same model multiple times produces "
        "wildly different thinking token counts. This is the core source of cost unpredictability."
    )

    rt = load_repeated_trials()
    if not rt:
        st.warning("No repeated trial data found.")
        st.stop()

    model_choice = st.selectbox(
        "Model",
        list(rt.keys()),
        format_func=lambda x: sn(short_names, x),
    )

    runs = rt[model_choice]

    # Build per-query thinking token matrix
    # Align by index
    all_indices = set()
    for run in runs:
        for rec in run["records"]:
            all_indices.add(rec["index"])
    all_indices = sorted(all_indices)

    matrix = []  # rows = queries, cols = runs
    for idx in all_indices:
        row = []
        for run in runs:
            rec = next((r for r in run["records"] if r["index"] == idx), None)
            if rec:
                row.append(rec.get("thinking_tokens", 0) or 0)
            else:
                row.append(None)
        matrix.append(row)

    df_rt = pd.DataFrame(
        matrix,
        columns=[f"Run {i}" for i in range(len(runs))],
        index=[f"Q{idx}" for idx in all_indices],
    )

    # Compute stats
    means = df_rt.mean(axis=1)
    stds = df_rt.std(axis=1)
    cvs = stds / means.replace(0, np.nan)

    st.metric(
        "Avg Coefficient of Variation",
        f"{cvs.mean():.2f}",
        help="CV = std / mean across runs. Higher = more unpredictable.",
    )

    # Range plot: min-max per query with individual dots
    fig = go.Figure()

    for i, idx_label in enumerate(df_rt.index):
        vals = df_rt.loc[idx_label].dropna().values
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=[i] * len(vals),
                y=vals,
                mode="markers",
                marker=dict(size=5, color="#1f77b4", opacity=0.6),
                showlegend=False,
                hovertemplate=f"{idx_label}<br>Thinking tokens: %{{y:,}}<extra></extra>",
            )
        )
        fig.add_shape(
            type="line",
            x0=i,
            x1=i,
            y0=min(vals),
            y1=max(vals),
            line=dict(color="#1f77b4", width=2),
        )

    fig.update_layout(
        xaxis=dict(
            tickvals=list(range(len(df_rt.index))),
            ticktext=list(df_rt.index),
            tickangle=90,
            title="Query",
        ),
        yaxis_title="Thinking Tokens",
        height=500,
        margin=dict(t=10, b=100),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show the raw data
    with st.expander("Show raw data"):
        df_display = df_rt.copy()
        df_display["Mean"] = means.round(0)
        df_display["Std"] = stds.round(0)
        df_display["CV"] = cvs.round(3)
        st.dataframe(df_display)
