import os
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FL-PPSN Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #0d9488;
    }
    .metric-label { font-size: 12px; color: #94a3b8; margin: 0; }
    .metric-value { font-size: 26px; font-weight: bold; color: #f1f5f9; margin: 0; }
    .metric-delta { font-size: 12px; color: #34d399; }
    h1 { color: #f1f5f9 !important; }
    h2, h3 { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Circle-icons-graph.svg/120px-Circle-icons-graph.svg.png", width=60)
st.sidebar.title("FL-PPSN")
st.sidebar.markdown("**Federated Learning based Privacy-Preserving Social Networks**")
st.sidebar.markdown("---")

log_path = st.sidebar.text_input("Log file path", value="experiments/log_run.csv")
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (live training)", value=False)
if auto_refresh:
    import time
    refresh_sec = st.sidebar.slider("Refresh every (seconds)", 5, 60, 15)

# ─────────────────────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_logs(path):
    """Load training logs from CSV file."""
    if os.path.exists(path):
        logger.info(f"Loading logs from {path}")
        df = pd.read_csv(path)
        return df
    else:
        logger.warning(f"Log file not found: {path}")
    return None


df = load_logs(log_path)

# ─────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────

st.title("🔐 Federated Learning – Privacy-Preserving Social Networks")
st.markdown("Real-time training metrics, privacy accounting & fairness analysis")
st.markdown("---")

if df is None or df.empty:
    st.warning("⚠️ No training data found yet.")
    st.markdown(f"""
    **To generate data, run:**
    ```bash
    # Quick test (no dataset download needed)
    python train_federated.py --synthetic --rounds 20

    # Full run with Facebook SNAP dataset
    python train_federated.py --rounds 50 --clients 5
    ```
    Then refresh this page.
    """)

    # Show demo with synthetic data
    st.markdown("### 📊 Demo Preview (Synthetic Data)")
    rounds = list(range(1, 21))
    demo_df = pd.DataFrame({
        "round":           rounds,
        "global_acc":      [0.3 + 0.03 * r + np.random.normal(0, 0.01) for r in rounds],
        "global_f1":       [0.25 + 0.025 * r + np.random.normal(0, 0.01) for r in rounds],
        "worst_client_f1": [0.2 + 0.02 * r + np.random.normal(0, 0.01) for r in rounds],
        "cumulative_eps":  [0.8 * r for r in rounds],
        "comms_mb":        [0.5 + 0.01 * r for r in rounds],
        "fairness_gap":    [0.15 - 0.005 * r + np.random.normal(0, 0.01) for r in rounds],
    })
    df = demo_df
    st.info("Showing demo data. Run training to see real results.")

# ─────────────────────────────────────────────────────────────
#  KPI Cards
# ─────────────────────────────────────────────────────────────

latest = df.iloc[-1]
n_rounds = len(df)

col1, col2, col3, col4, col5 = st.columns(5)

def kpi(col, label, value, delta=None, color="#0d9488"):
    delta_html = f'<p class="metric-delta">▲ {delta}</p>' if delta else ""
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

kpi(col1, "Best Val Accuracy",  f"{df['global_acc'].max():.4f}",  color="#0d9488")
kpi(col2, "Latest Test Acc",    f"{latest.get('global_test_acc', latest['global_acc']):.4f}", color="#3b82f6")
kpi(col3, "Worst Client F1",   f"{df['worst_client_f1'].iloc[-1]:.4f}", color="#f59e0b")
kpi(col4, "Cumulative ε",      f"{latest['cumulative_eps']:.3f}", color="#8b5cf6")
kpi(col5, "Total Comms (MB)",  f"{df['comms_mb'].sum():.2f}", color="#ef4444")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
#  Charts — Row 1
# ─────────────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Accuracy & F1 over Rounds")
    chart_df = df.set_index("round")[
        [c for c in ["global_acc", "global_f1"] if c in df.columns]
    ]
    st.line_chart(chart_df, use_container_width=True)

with col_b:
    st.subheader("⚖️ Fairness: Mean vs Worst Client")
    fair_cols = [c for c in ["global_acc", "worst_client_f1"] if c in df.columns]
    st.line_chart(df.set_index("round")[fair_cols], use_container_width=True)
    if "fairness_gap" in df.columns:
        st.caption(f"Current fairness gap: **{df['fairness_gap'].iloc[-1]:.4f}** "
                   f"(lower = more fair) | Best: **{df['fairness_gap'].min():.4f}**")

# ─────────────────────────────────────────────────────────────
#  Charts — Row 2
# ─────────────────────────────────────────────────────────────

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🔐 Privacy Budget (ε) over Rounds")
    if "cumulative_eps" in df.columns:
        eps_df = df.set_index("round")[["cumulative_eps"]]
        if "mean_eps" in df.columns:
            eps_df["per_round_eps"] = df.set_index("round")["mean_eps"]
        st.line_chart(eps_df, use_container_width=True)
        st.caption(f"Final cumulative ε: **{df['cumulative_eps'].iloc[-1]:.4f}** "
                   f"(lower = stronger privacy)")
    else:
        st.info("Privacy tracking not found in log. Enable --secure-na during training.")

with col_d:
    st.subheader("📡 Communication Cost (MB/round)")
    if "comms_mb" in df.columns:
        comm_df = df.set_index("round")[["comms_mb"]]
        comm_df["cumulative_mb"] = comm_df["comms_mb"].cumsum()
        st.line_chart(comm_df, use_container_width=True)
        st.caption(f"Total communication: **{df['comms_mb'].sum():.2f} MB** "
                   f"over {n_rounds} rounds")
    else:
        st.info("Communication cost not tracked.")

# ─────────────────────────────────────────────────────────────
#  Metric selector
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Custom Metric Explorer")

numeric_cols = [c for c in df.columns if c != "round" and df[c].dtype in [float, np.float64, np.float32]]
if numeric_cols:
    selected = st.multiselect("Select metrics to plot", numeric_cols,
                               default=numeric_cols[:min(3, len(numeric_cols))])
    if selected:
        st.line_chart(df.set_index("round")[selected], use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  Recent rounds table
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Training Log (Latest 5 Rounds)")
display_cols = [c for c in [
    "round", "global_acc", "global_f1", "worst_client_f1",
    "fairness_gap", "cumulative_eps", "comms_mb", "round_time_s"
] if c in df.columns]

styled = df[display_cols].tail(5).style.format({
    c: "{:.4f}" for c in display_cols if c not in ["round"]
}).highlight_max(
    subset=[c for c in ["global_acc", "global_f1"] if c in display_cols],
    color="#0d9488",
).highlight_min(
    subset=[c for c in ["fairness_gap", "cumulative_eps"] if c in display_cols],
    color="#1e3a5f",
)
st.dataframe(styled, use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  Ablation results (if available)
# ─────────────────────────────────────────────────────────────

abl_path = "experiments/ablation_results.csv"
if os.path.exists(abl_path):
    st.markdown("---")
    st.subheader("🧪 Ablation Study Results")
    abl_df = pd.read_csv(abl_path)
    st.dataframe(abl_df.style.highlight_max(
        subset=[c for c in ["best_val_acc", "best_test_acc"] if c in abl_df.columns],
        color="#0d9488"
    ).highlight_min(
        subset=[c for c in ["final_eps", "fairness_gap", "total_comms_mb"] if c in abl_df.columns],
        color="#1e3a5f"
    ), use_container_width=True)

    # Bar chart of accuracy
    if "config" in abl_df.columns and "best_val_acc" in abl_df.columns:
        st.bar_chart(abl_df.set_index("config")["best_val_acc"])

# ─────────────────────────────────────────────────────────────
#  Upload alternative log
# ─────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("📤 Upload a different log file"):
    uploaded = st.file_uploader("Upload log_run.csv", type=["csv"])
    if uploaded:
        df2 = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df2)} rows. Available columns: {df2.columns.tolist()}")
        metric = st.selectbox("Plot metric", [c for c in df2.columns if c != "round"])
        if metric:
            st.line_chart(df2.set_index("round")[metric])

# ─────────────────────────────────────────────────────────────
#  Auto-refresh
# ─────────────────────────────────────────────────────────────

if auto_refresh:
    import time
    st.sidebar.success(f"Auto-refreshing every {refresh_sec}s...")
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()

# Footer
st.markdown("---")
st.caption("FL-PPSN | Federated Learning based Privacy-Preserving Social Networks | "
           "SecureSA + SecureNA + Adaptive LDP + Fairness-Aware Aggregation")
