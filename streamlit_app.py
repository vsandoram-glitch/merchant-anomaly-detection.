
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

# All model outputs are read from the batch pipeline, never fabricated here.
# Put this app in the directory with the CSV outputs or set OUTPUT_DIR.
import os
OUTPUT_DIR = Path(os.environ.get("MERCHANT_OUTPUT_DIR", str(ROOT))).resolve()
DATASETS = {
    "Original portfolio": ("daily_sales_prepared.csv", "business_daily_prepared.csv"),
    "New portfolio": ("daily_sales_new_prepared.csv", "business_daily_new_prepared.csv"),
}
MODEL_FILES = {
    "Isolation Forest": "isolation_forest",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Logistic Regression": "logistic_regression",
}

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

@st.cache_data(show_spinner=False)
def load_dataset(dataset):
    daily_filename, business_filename = DATASETS[dataset]
    required = [OUTPUT_DIR / daily_filename, OUTPUT_DIR / business_filename]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing preprocessed file(s): " + ", ".join(missing))
    daily = pd.read_csv(required[0], parse_dates=["Date"])
    business = pd.read_csv(required[1], parse_dates=["Date"])
    if "High_Confidence_Day" not in daily.columns:
        daily = daily.rename(columns={"High_Confidence_Anomaly_Day": "High_Confidence_Day"})
    parsed = daily["High_Confidence_Day"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError("Unexpected values in High_Confidence_Day")
    daily["High_Confidence_Day"] = parsed.astype(bool)
    return daily, business


@st.cache_data(show_spinner=False)
def load_predictions(dataset, model):
    prefix = "original" if dataset == "Original portfolio" else "new"
    prediction_path = OUTPUT_DIR / f"predictions_{prefix}_{MODEL_FILES[model]}.csv"
    # Backward compatible with the original portfolio risk output.
    if not prediction_path.exists() and dataset == "Original portfolio" and model == "Isolation Forest":
        risk_path = OUTPUT_DIR / "merchant_risk.csv"
        if not risk_path.exists():
            raise FileNotFoundError(str(prediction_path))
        risk = pd.read_csv(risk_path)
        return risk
    if not prediction_path.exists():
        raise FileNotFoundError(str(prediction_path))
    risk = pd.read_csv(prediction_path)
    needed = {"Merchant", "Prediction"}
    if not needed.issubset(risk.columns):
        raise ValueError(f"{prediction_path.name} must contain Merchant, Prediction")
    if risk["Merchant"].isna().any() or risk["Merchant"].duplicated().any():
        raise ValueError(f"Duplicate or blank merchants in {prediction_path.name}")
    risk["Prediction"] = pd.to_numeric(risk["Prediction"], errors="raise")
    if not risk["Prediction"].isin([0,1]).all():
        raise ValueError("Predictions must be 0 or 1")
    # IF scores, when available, describe IF only: do not attach them to other models.
    if model == "Isolation Forest" and dataset == "Original portfolio":
        base = OUTPUT_DIR / "merchant_risk.csv"
        if base.exists():
            scores = pd.read_csv(base)
            score_cols = [c for c in ("Merchant", "Anomaly_Score", "Merchant_ID") if c in scores]
            risk = risk.merge(scores[score_cols], on="Merchant", how="left", validate="one_to_one")
    return risk


def available_models(dataset):
    prefix = "original" if dataset == "Original portfolio" else "new"
    choices = []
    for model, suffix in MODEL_FILES.items():
        exists = (OUTPUT_DIR / f"predictions_{prefix}_{suffix}.csv").exists()
        fallback = dataset == "Original portfolio" and model == "Isolation Forest" and (OUTPUT_DIR / "merchant_risk.csv").exists()
        if exists or fallback:
            choices.append(model)
    return choices

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

f1, f2, f3, f4 = st.columns([1.0, 1.1, 1.2, 1.6])

with f1:
    dataset_choice = st.selectbox("Dataset", list(DATASETS), key="dataset_choice")

models = available_models(dataset_choice)
if not models:
    st.error("No prediction files available for this dataset. Run the batch pipeline first.")
    st.stop()

with f2:
    model_choice = st.selectbox("Merchant model", models, key="model_choice")

try:
    daily, business_daily = load_dataset(dataset_choice)
    merchant_risk = load_predictions(dataset_choice, model_choice)
except Exception as exc:
    st.error(f"Unable to load the selected dataset/model: {exc}")
    st.stop()

merchant_names = set(daily["Merchant"].dropna().unique())
risk_names = set(merchant_risk["Merchant"].dropna().unique())
if merchant_names != risk_names:
    st.error(f"Dataset/prediction merchant mismatch: {len(merchant_names-risk_names)} missing predictions, {len(risk_names-merchant_names)} unexpected predictions.")
    st.stop()

merchant_risk = merchant_risk.copy()
merchant_risk["Prediction"] = pd.to_numeric(merchant_risk["Prediction"], errors="raise").astype(int)
if not merchant_risk["Prediction"].isin([0, 1]).all() or merchant_risk["Merchant"].duplicated().any():
    st.error("Prediction file contains duplicate merchants or invalid labels.")
    st.stop()

flagged_merchants = sorted(merchant_risk.loc[merchant_risk["Prediction"].eq(1), "Merchant"].tolist())
all_merchants = sorted(risk_names)
with f3:
    merchant_filter = st.selectbox("Merchant filter", ["All merchants", "Flagged only"], key="merchant_filter")

available_merchants = flagged_merchants if merchant_filter == "Flagged only" else all_merchants
if not available_merchants:
    st.info("No merchants flagged by this model. Switch the merchant filter to All merchants.")
    st.stop()
with f4:
    selected_merchant = st.selectbox("Merchant", available_merchants, key="selected_merchant")

st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

# ============================================================
# PORTFOLIO KPI ROW
# ============================================================

merchant_count = int(merchant_risk["Merchant"].nunique())
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
day_count = int(m_anom.shape[0])
risk_score = float(m_risk["Anomaly_Score"]) if "Anomaly_Score" in m_risk and pd.notna(m_risk["Anomaly_Score"]) else None

st.subheader(selected_merchant)

score_note = (f" Isolation Forest anomaly score {risk_score:.4f} (larger means more unusual)." if risk_score is not None and model_choice == "Isolation Forest" else "")
st.markdown(f"**{model_choice}: {classification}.** {day_count} historical-deviation day(s).{score_note}")
st.caption("Merchant classification is model-dependent; the historical-deviation markers are calculated separately and remain unchanged when the merchant model changes.")

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
            marker=dict(size=10, symbol="circle", color="#d62828"),
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

classification_table = merchant_risk[["Merchant", "Prediction"]].copy()
classification_table["Classification"] = np.where(classification_table["Prediction"].eq(1), "Anomalous", "Normal")
days_by_merchant = daily.groupby("Merchant")["High_Confidence_Day"].sum().rename("Flagged days")
classification_table = classification_table.join(days_by_merchant, on="Merchant")
show_cols = ["Merchant", "Classification", "Flagged days"]
if model_choice == "Isolation Forest" and "Anomaly_Score" in merchant_risk.columns:
    classification_table["IF score"] = merchant_risk["Anomaly_Score"].to_numpy()
    show_cols.insert(2, "IF score")
classification_table = classification_table.sort_values(["Prediction", "Merchant"], ascending=[False, True])

prefix = "original" if dataset_choice == "Original portfolio" else "new"
file_name = f"predictions_{prefix}_{MODEL_FILES[model_choice]}.csv"
st.download_button(
    "Download predictions CSV",
    data=classification_table[["Merchant", "Prediction"]].to_csv(index=False).encode("utf-8"),
    file_name=file_name,
    mime="text/csv",
)
st.dataframe(classification_table[show_cols], hide_index=True, use_container_width=True, height=330)

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


