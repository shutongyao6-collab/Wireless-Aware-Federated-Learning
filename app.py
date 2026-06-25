"""
Wireless-Aware Federated Learning · Experimental Analysis Dashboard
Each section corresponds to one requirement in the project description.
All results are based on real simulation outputs with mean ± std error bars.

Run:
streamlit run app.py
"""

import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Federated Learning Experimental Dashboard",
    layout="wide"
)

RESULTS_DIR = "results"

METHOD_ORDER = [
    "baseline",
    "only_compression",
    "only_scheduling",
    "joint_optimization"
]

METHOD_NAMES = {
    "baseline": "Baseline",
    "only_compression": "Compression Only",
    "only_scheduling": "Scheduling Only",
    "joint_optimization": "Joint Optimization"
}

NAME_ORDER = [METHOD_NAMES[m] for m in METHOD_ORDER]

COLORS = {
    "Baseline": "#1f77b4",
    "Compression Only": "#2ca02c",
    "Scheduling Only": "#ff7f0e",
    "Joint Optimization": "#d62728"
}


@st.cache_data
def load_curves():
    rows = []

    for m in METHOD_ORDER:
        path = os.path.join(RESULTS_DIR, f"{m}.json")

        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            hist = json.load(f)

        for r in hist:
            rows.append({
                "Method": METHOD_NAMES[m],
                "Round": r["round"],
                "Accuracy": r["accuracy"],
                "Accuracy Std": r.get("accuracy_std", 0.0),
                "Cumulative Time (s)": r["global_time"],
                "Cumulative Communication (MB)": r.get("total_transmitted_bytes", 0) / (1024 * 1024),
                "Communication Std": r.get("total_transmitted_bytes_std", 0) / (1024 * 1024),
                "Cumulative Energy": r.get("total_energy", 0.0),
                "Energy Std": r.get("total_energy_std", 0.0),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["Method"] = pd.Categorical(
            df["Method"],
            categories=NAME_ORDER,
            ordered=True
        )

    return df


@st.cache_data
def load_robustness():
    path = os.path.join(RESULTS_DIR, "drop_rate_robustness.json")

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.DataFrame(json.load(open(path))).rename(
        columns={
            "Drop_Rate_Scale": "Drop Rate Scale",
            "Final_Accuracy": "Final Accuracy",
            "Final_Accuracy_std": "Final Accuracy Std"
        }
    )

    if "Final Accuracy Std" not in df.columns:
        df["Final Accuracy Std"] = 0.0

    df["Method"] = df["Method"].map(METHOD_NAMES)
    df["Method"] = pd.Categorical(
        df["Method"],
        categories=NAME_ORDER,
        ordered=True
    )

    return df


@st.cache_data
def load_dp():
    path = os.path.join(RESULTS_DIR, "dp_privacy.json")

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.DataFrame(json.load(open(path)))

    df["DP Noise σ"] = df["dp_sigma"].apply(
        lambda v: "No DP" if v == 0 else str(v)
    )

    return df


@st.cache_data
def load_distribution():
    path = os.path.join(RESULTS_DIR, "client_data_distribution.json")

    if not os.path.exists(path):
        return pd.DataFrame()

    d = json.load(open(path))
    items = sorted(d.items(), key=lambda kv: int(kv[0]))

    return pd.DataFrame({
        "Client": [f"C{k}" for k, _ in items],
        "Number of Samples": [v for _, v in items]
    })


df = load_curves()
robust = load_robustness()
dp = load_dp()
dist = load_distribution()


st.title("📡 Wireless-Aware Federated Learning · Experimental Analysis Dashboard")

st.caption(
    "The four methods form a 2×2 ablation study: scheduling enabled/disabled × "
    "compression enabled/disabled. The main comparison is controlled and based on "
    "multi-seed mean ± std results. Differential privacy and robustness are evaluated "
    "as independent single-variable experiments."
)

if df.empty:
    st.warning("No training results found. Please run `python main.py` first.")
    st.stop()


final = df.sort_values("Round").groupby("Method", observed=True).tail(1).copy()

final["Acc/MB"] = final.apply(
    lambda r: r["Accuracy"] / r["Cumulative Communication (MB)"]
    if r["Cumulative Communication (MB)"] > 0 else 0,
    axis=1
)

final = final.sort_values("Method")


# ===== Summary =====
st.subheader("📌 Key Findings Summary")

try:
    base = final[final["Method"] == "Baseline"].iloc[0]
    joint = final[final["Method"] == "Joint Optimization"].iloc[0]

    ratio = (
        base["Cumulative Communication (MB)"] / joint["Cumulative Communication (MB)"]
        if joint["Cumulative Communication (MB)"] > 0 else float("nan")
    )

    gap = (base["Accuracy"] - joint["Accuracy"]) * 100

    st.markdown(
        f"- **Accuracy:** Joint Optimization achieves "
        f"**{joint['Accuracy']:.4f} ± {joint['Accuracy Std']:.4f}**, while the Baseline achieves "
        f"**{base['Accuracy']:.4f} ± {base['Accuracy Std']:.4f}** "
        f"(difference: approximately {gap:.1f} percentage points).\n"
        f"- **Communication cost:** Joint Optimization uses "
        f"**{joint['Cumulative Communication (MB)']:.2f} MB**, compared with "
        f"**{base['Cumulative Communication (MB)']:.2f} MB** for the Baseline, "
        f"which is approximately **1/{ratio:.1f}** of the Baseline communication cost.\n"
        f"- **Conclusion:** With comparable accuracy, the proposed joint optimization "
        f"reduces communication cost to around **1/{ratio:.0f}** of the Baseline. "
        f"Acc/MB improves from **{base['Acc/MB']:.3f}** to "
        f"**{joint['Acc/MB']:.3f}**."
    )

except (IndexError, KeyError):
    st.info("Baseline or Joint Optimization results are missing.")


# ===== 1. Summary table =====
st.subheader("1. Final Performance Summary of All Methods (mean ± std)")

tbl = final[
    [
        "Method",
        "Accuracy",
        "Accuracy Std",
        "Cumulative Time (s)",
        "Cumulative Communication (MB)",
        "Cumulative Energy",
        "Acc/MB"
    ]
].rename(
    columns={
        "Accuracy": "Final Accuracy",
        "Accuracy Std": "Final Accuracy Std"
    }
)

st.dataframe(
    tbl.style.format({
        "Final Accuracy": "{:.4f}",
        "Final Accuracy Std": "{:.4f}",
        "Cumulative Time (s)": "{:.1f}",
        "Cumulative Communication (MB)": "{:.2f}",
        "Cumulative Energy": "{:.1f}",
        "Acc/MB": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ===== 2. Pareto trade-off =====
st.subheader("2. Accuracy–Communication Pareto Trade-off [Communication Efficiency / Compression]")

st.caption(
    "Upper-left is preferable: lower communication cost with higher accuracy. "
    "Compression is applied to transmitted model updates, so compression-based methods "
    "slightly reduce accuracy but substantially decrease communication cost."
)

fig_p = px.scatter(
    final,
    x="Cumulative Communication (MB)",
    y="Accuracy",
    color="Method",
    text="Method",
    error_x="Communication Std",
    error_y="Accuracy Std",
    color_discrete_map=COLORS,
    category_orders={"Method": NAME_ORDER}
)

fig_p.update_traces(
    textposition="top center",
    marker=dict(size=14)
)

fig_p.update_layout(
    height=420,
    showlegend=False
)

st.plotly_chart(fig_p, use_container_width=True)


# ===== 3. Convergence and latency =====
c1, c2 = st.columns(2)

with c1:
    st.subheader("3.1 Accuracy Convergence [Accuracy Preservation]")

    fig = px.line(
        df,
        x="Round",
        y="Accuracy",
        color="Method",
        markers=True,
        error_y="Accuracy Std",
        color_discrete_map=COLORS,
        category_orders={"Method": NAME_ORDER}
    )

    fig.update_layout(height=380)

    st.plotly_chart(fig, use_container_width=True)


with c2:
    st.subheader("3.2 Accuracy–Latency Frontier [Latency]")

    fig = px.line(
        df,
        x="Cumulative Time (s)",
        y="Accuracy",
        color="Method",
        markers=True,
        line_shape="hv",
        color_discrete_map=COLORS,
        category_orders={"Method": NAME_ORDER}
    )

    fig.update_layout(height=380)

    st.plotly_chart(fig, use_container_width=True)


# ===== 4. Efficiency and energy =====
c3, c4 = st.columns(2)

with c3:
    st.subheader("4.1 Communication Efficiency: Acc/MB [Communication Efficiency]")

    fig = px.bar(
        final,
        x="Method",
        y="Acc/MB",
        color="Method",
        text_auto=".3f",
        color_discrete_map=COLORS,
        category_orders={"Method": NAME_ORDER}
    )

    fig.update_layout(
        height=380,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


with c4:
    st.subheader("4.2 Cumulative Energy Consumption [Energy Efficiency]")

    st.caption(
        "Energy proxy = sum of computation time and communication time across clients. "
        "Lower values indicate better energy efficiency."
    )

    fig = px.line(
        df,
        x="Round",
        y="Cumulative Energy",
        color="Method",
        markers=True,
        error_y="Energy Std",
        color_discrete_map=COLORS,
        category_orders={"Method": NAME_ORDER}
    )

    fig.update_layout(height=380)

    st.plotly_chart(fig, use_container_width=True)


# ===== 5. Robustness and privacy =====
c5, c6 = st.columns(2)

with c5:
    st.subheader("5. Packet Loss Robustness [Link Failure / Reliability]")

    if robust.empty:
        st.info("Please run `python run_drop_rate_exp.py` first.")
    else:
        fig = px.line(
            robust,
            x="Drop Rate Scale",
            y="Final Accuracy",
            color="Method",
            markers=True,
            error_y="Final Accuracy Std",
            color_discrete_map=COLORS,
            category_orders={"Method": NAME_ORDER}
        )

        fig.update_layout(height=380)

        st.plotly_chart(fig, use_container_width=True)


with c6:
    st.subheader("6. Differential Privacy: Privacy–Accuracy Trade-off [DP]")

    st.caption(
        "Joint Optimization is fixed, and only the DP noise level σ is varied. "
        "A larger σ indicates stronger privacy protection but usually causes a larger accuracy penalty."
    )

    if dp.empty:
        st.info("Please run `python run_dp_exp.py` first.")
    else:
        fig = px.bar(
            dp,
            x="DP Noise σ",
            y="accuracy",
            error_y="accuracy_std",
            text_auto=".3f",
            color="accuracy",
            color_continuous_scale="Blues",
            labels={"accuracy": "Final Accuracy"}
        )

        fig.update_layout(
            height=380,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


# ===== 7. Non-IID distribution =====
st.subheader("7. Client Data Distribution [Resource/Data Imbalance and Non-IID Setting]")

if not dist.empty:
    fig = px.bar(
        dist,
        x="Client",
        y="Number of Samples",
        color="Number of Samples",
        color_continuous_scale="Blues"
    )

    fig.update_layout(height=340)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Client data distribution file not found.")