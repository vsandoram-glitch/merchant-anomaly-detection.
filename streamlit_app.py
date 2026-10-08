
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Merchant Sales Anomaly Monitor",
    page_icon="📈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

DAILY_FILE = ROOT / "daily_sales_prepared.csv"
BUSINESS_FILE = ROOT / "business_daily_prepared.csv"
RISK_FILE = ROOT / "merchant_risk.csv"

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    [data-testid="stMetric"] {
        border: 0;
        padding: 0.1rem 0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
    }
    .thin-rule {
        border-top: 1px solid rgba(120,120,120,.25);
        margin: .75rem 0 1rem 0;
    }
    .muted {
        color: #6f7d8c;
        font-size: .93rem;
    }
    .status-line {
        color: #5b6775;
        font-size: .95rem;
        margin-bottom: .75rem;
    }
    .note-box {
        border-left: 4px solid #2c6ca3;
        padding: .55rem .75rem;
        background: rgba(44,108,163,.06);
        margin: .5rem 0 1rem 0;
        font-size: .92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    missing = [p.name for p in [DAILY_FILE, BUSINESS_FILE, RISK_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dashboard file(s): " + ", ".join(missing)
        )

    daily = pd.read_csv(DAILY_FILE, parse_dates=["Date"])
    business = pd.read_csv(BUSINESS_FILE, parse_dates=["Date"])
    risk = pd.read_csv(RISK_FILE)

    # Defensive typing
    daily["High_Confidence_Day"] = (
        daily["High_Confidence_Day"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(daily["High_Confidence_Day"])
        .astype(bool)
    )

    return daily, business, risk


try:
    daily, business_daily, merchant_risk = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ============================================================
# CONSTANTS / EVALUATOR RESULTS
# ============================================================

EVALUATOR_RESULTS = pd.DataFrame(
    [
        [
            "Isolation Forest — Original",
            "Original",
            0.943,
            0.909,
            0.667,
            0.769,
            0.989,
            0.828,
        ],
        [
            "Isolation Forest — New",
            "New",
            0.981,
            1.000,
            0.867,
            0.929,
            1.000,
            0.934,
        ],
        [
            "Gradient Boosting",
            "New",
            0.962,
            0.923,
            0.800,
            0.857,
            0.989,
            0.895,
        ],
        [
            "Logistic Regression",
            "New",
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
        ],
        [
            "Random Forest",
            "New",
            0.981,
            1.000,
            0.867,
            0.929,
            1.000,
            0.934,
        ],
    ],
    columns=[
        "Model",
        "Dataset",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced acc.",
    ],
)

# ============================================================
# PAGE HEADER
# ============================================================

st.title("Merchant Sales Anomaly Monitor")

st.markdown(
    """
    <div class="muted">
    Captured sales only. Merchant classification covers the full 90-day period.
    Red markers identify dates flagged by the separate historical-deviation rule.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TOP FILTERS
# ============================================================

flagged_merchants = (
    merchant_risk.loc[merchant_risk["Prediction"].eq(1), "Merchant"]
    .sort_values()
    .tolist()
)
all_merchants = sorted(merchant_risk["Merchant"].dropna().unique().tolist())

f1, f2, f3, f4 = st.columns([1.0, 1.05, 1.2, 1.6])

with f1:
    dataset_choice = st.selectbox(
        "Dataset",
        ["Original portfolio"],
        index=0,
    )

with f2:
    model_choice = st.selectbox(
        "Merchant model",
        ["Isolation Forest"],
        index=0,
    )

with f3:
    merchant_filter = st.selectbox(
        "Merchant filter",
        ["All merchants", "Flagged only"],
        index=0,
    )

available_merchants = (
    flagged_merchants if merchant_filter == "Flagged only" else all_merchants
)

with f4:
    selected_merchant = st.selectbox(
        "Merchant",
        available_merchants,
        index=0,
    )

st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

# ============================================================
# PORTFOLIO KPI ROW
# ============================================================

merchant_count = int(merchant_risk["Merchant_ID"].nunique())
flagged_count = int(merchant_risk["Prediction"].sum())
flagged_days = int(daily["High_Confidence_Day"].sum())
captured_sales = float(daily["Daily_Sales"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Merchants", f"{merchant_count:,}")
k2.metric("Flagged by model", f"{flagged_count:,}")
k3.metric("Flagged days", f"{flagged_days:,}")
k4.metric("Captured sales USD", f"{captured_sales:,.1f}")

st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

# ============================================================
# SELECTED MERCHANT SUMMARY
# ============================================================

m_risk = merchant_risk.loc[
    merchant_risk["Merchant"].eq(selected_merchant)
].iloc[0]

m_daily = daily.loc[
    daily["Merchant"].eq(selected_merchant)
].sort_values("Date").copy()

m_anom = m_daily.loc[
    m_daily["High_Confidence_Day"].eq(True)
].copy()

classification = "Anomalous" if int(m_risk["Prediction"]) == 1 else "Normal"
day_count = int(m_risk["High_Confidence_Day_Count"])
risk_score = float(m_risk["Anomaly_Score"])

st.subheader(selected_merchant)

st.markdown(
    f"""
    <div class="status-line">
    <b>Isolation Forest:</b> {classification} classification.
    {day_count} flagged day(s).
    Portfolio anomaly score <b>{risk_score:.4f}</b>
    (higher values indicate a more unusual merchant profile).
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MERCHANT SALES CHART
# ============================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=m_daily["Date"],
        y=m_daily["Daily_Sales"],
        mode="lines+markers",
        name="Daily sales",
        line=dict(width=2.2),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual USD %{y:,.2f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=m_daily["Date"],
        y=m_daily["Expected_Sales"],
        mode="lines",
        name="Expected sales",
        line=dict(width=1.8, dash="dash"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Expected USD %{y:,.2f}<extra></extra>",
    )
)

if not m_anom.empty:
    fig.add_trace(
        go.Scatter(
            x=m_anom["Date"],
            y=m_anom["Daily_Sales"],
            mode="markers",
            name="Flagged day",
            marker=dict(size=10, symbol="circle"),
            customdata=m_anom[["Expected_Sales", "Robust_Z"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Actual USD %{y:,.2f}<br>"
                "Expected USD %{customdata[0]:,.2f}<br>"
                "Robust z %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )

fig.update_layout(
    height=470,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title=None,
    yaxis_title="USD",
    legend_title=None,
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div class="muted">
    Blue: actual daily sales. Dashed line: expected sales from prior observations.
    Red markers: flagged day. Hover over a point for date and amount.
    Early dates may have insufficient history.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# BUSINESS DRILL-DOWN CONTROLS
# ============================================================

date_values = m_daily["Date"].dt.date.tolist()
business_values = sorted(
    business_daily.loc[
        business_daily["Merchant"].eq(selected_merchant),
        "Business",
    ]
    .dropna()
    .unique()
    .tolist()
)

d1, d2, _ = st.columns([1.1, 1.1, 5])

with d1:
    # Prefer latest anomaly date, otherwise latest date
    if not m_anom.empty:
        default_date = m_anom["Date"].max().date()
    else:
        default_date = m_daily["Date"].max().date()

    default_date_idx = (
        date_values.index(default_date)
        if default_date in date_values
        else len(date_values) - 1
    )

    drill_date = st.selectbox(
        "Business drill-down date",
        date_values,
        index=default_date_idx,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
    )

with d2:
    drill_business = st.selectbox(
        "Business category",
        business_values,
        index=0,
    )

drill_date_ts = pd.Timestamp(drill_date)

day_row = m_daily.loc[m_daily["Date"].eq(drill_date_ts)]

if not day_row.empty:
    row = day_row.iloc[0]
    actual_day = float(row["Daily_Sales"])
    expected_day = row["Expected_Sales"]
    robust_z = row["Robust_Z"]
    is_flag = bool(row["High_Confidence_Day"])

    status_text = "Flagged day" if is_flag else "No day flag"

    exp_text = (
        f"USD {float(expected_day):,.2f}"
        if pd.notna(expected_day)
        else "insufficient history"
    )
    rz_text = (
        f"{float(robust_z):.2f}"
        if pd.notna(robust_z)
        else "not available"
    )

    st.markdown(
        f"""
        <div class="muted">
        {drill_date.strftime('%Y-%m-%d')}: <b>{status_text}</b>.
        Actual USD {actual_day:,.2f}, expected {exp_text}, robust z {rz_text}.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# BUSINESS CATEGORY TABLE
# ============================================================

bd_date = business_daily.loc[
    business_daily["Merchant"].eq(selected_merchant)
    & business_daily["Date"].eq(drill_date_ts)
].copy()

bd_date["Difference_USD"] = (
    bd_date["Business_Sales"] - bd_date["Expected_Business_Sales"]
)

display_business = bd_date[
    [
        "Business",
        "Business_Sales",
        "Expected_Business_Sales",
        "Difference_USD",
    ]
].copy()

display_business.columns = [
    "Business category",
    "Actual USD",
    "Expected USD",
    "Difference USD",
]

st.dataframe(
    display_business,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Actual USD": st.column_config.NumberColumn(format="$%.2f"),
        "Expected USD": st.column_config.NumberColumn(format="$%.2f"),
        "Difference USD": st.column_config.NumberColumn(format="$%.2f"),
    },
)

# ============================================================
# SELECTED BUSINESS CATEGORY TIME SERIES
# ============================================================

b_series = business_daily.loc[
    business_daily["Merchant"].eq(selected_merchant)
    & business_daily["Business"].eq(drill_business)
].sort_values("Date").copy()

fig_b = go.Figure()

fig_b.add_trace(
    go.Scatter(
        x=b_series["Date"],
        y=b_series["Business_Sales"],
        mode="lines+markers",
        name=f"{drill_business} actual",
        line=dict(width=2.1),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual USD %{y:,.2f}<extra></extra>",
    )
)

fig_b.add_trace(
    go.Scatter(
        x=b_series["Date"],
        y=b_series["Expected_Business_Sales"],
        mode="lines",
        name=f"{drill_business} expected",
        line=dict(width=1.6, dash="dash"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Expected USD %{y:,.2f}<extra></extra>",
    )
)

fig_b.update_layout(
    height=370,
    margin=dict(l=10, r=10, t=15, b=10),
    xaxis_title=None,
    yaxis_title="USD",
    legend_title=None,
    hovermode="x unified",
)

st.plotly_chart(fig_b, use_container_width=True)

st.markdown(
    """
    <div class="muted">
    Category expectations are estimated separately, so their sum may differ from
    the merchant expected value. Deviations show contributions to investigate,
    not proven causes.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MERCHANT CLASSIFICATIONS
# ============================================================

st.subheader("Merchant classifications")

classification_table = merchant_risk[
    [
        "Merchant",
        "Prediction",
        "Anomaly_Score",
        "High_Confidence_Day_Count",
    ]
].copy()

classification_table["Classification"] = np.where(
    classification_table["Prediction"].eq(1),
    "Anomalous",
    "Normal",
)

classification_table = classification_table[
    [
        "Merchant",
        "Classification",
        "Anomaly_Score",
        "High_Confidence_Day_Count",
    ]
]

classification_table.columns = [
    "Merchant",
    "Classification",
    "IF score",
    "Flagged days",
]

predictions_csv = (
    classification_table[["Merchant", "Classification"]]
    .assign(
        Prediction=lambda x: np.where(
            x["Classification"].eq("Anomalous"), 1, 0
        )
    )[["Merchant", "Prediction"]]
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "Download predictions CSV",
    data=predictions_csv,
    file_name="predictions_original_isolation_forest.csv",
    mime="text/csv",
)

st.dataframe(
    classification_table,
    hide_index=True,
    use_container_width=True,
    height=330,
    column_config={
        "IF score": st.column_config.NumberColumn(format="%.4f"),
    },
)

# ============================================================
# RECORDED EVALUATOR RESULTS
# ============================================================

st.subheader("Recorded evaluator results")

st.markdown(
    """
    <div class="note-box">
    Results below are the recorded external Model Evaluator results from the assignment.
    Hidden merchant labels are not exposed by the evaluator.
    </div>
    """,
    unsafe_allow_html=True,
)

evaluator_display = EVALUATOR_RESULTS.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "Balanced acc.",
]:
    evaluator_display[col] = evaluator_display[col].map(lambda x: f"{x:.2%}")

st.dataframe(
    evaluator_display,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Best supervised result on the supplied hidden evaluation dataset: "
    "Logistic Regression (100% across the recorded classification metrics)."
)


