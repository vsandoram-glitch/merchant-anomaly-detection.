"""
Big Data Analytics - Merchant Anomaly Detection Tool
Single runnable Python file for the assignment.

Modes of use
------------
1) Batch mode:
   python merchant_anomaly_detection_final.py --mode batch --data-dir . --output-dir outputs

   Required original files in --data-dir:
   - Transactions.csv
   - merchant.csv
   - business.csv
   - status.csv

   Optional new-dataset files in --data-dir:
   - Transactions_New.csv
   - merchant_New.csv
   - business_New.csv
   - status_New.csv

2) Streamlit dashboard:
   streamlit run merchant_anomaly_detection_final.py -- --mode app

The script performs the full workflow: data ingestion, preparation, captured-only
sales aggregation, feature engineering, Isolation Forest modelling, supervised
model comparison, evaluator-ready prediction exports, and a single-page UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
FINAL_CONTAMINATION = 0.10
ROBUST_Z_THRESHOLD = 4.0
MIN_RELATIVE_FACTOR = 1.5

FEATURE_COLUMNS = [
    "Sales_CV",
    "Transaction_CV",
    "Peak_Positive_Deviation",
    "Peak_Negative_Deviation",
    "P95_Absolute_Deviation",
    "Mean_Absolute_Deviation",
    "High_Confidence_Day_Count",
    "Max_Robust_Z",
    "Largest_Daily_Log_Change",
    "Period_Shift",
]

ORIGINAL_FILES = {
    "transactions": "Transactions.csv",
    "merchant": "merchant.csv",
    "business": "business.csv",
    "status": "status.csv",
}

NEW_FILES = {
    "transactions": "Transactions_New.csv",
    "merchant": "merchant_New.csv",
    "business": "business_New.csv",
    "status": "status_New.csv",
}


def clean_status_code(series: pd.Series) -> pd.Series:
    """Normalise status codes such as 1, '1', '01', '1.0' to two-character strings."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def locate_files(data_dir: Path, mapping: Dict[str, str]) -> Dict[str, Path]:
    paths = {key: data_dir / filename for key, filename in mapping.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    return paths


def read_source_files(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(paths["transactions"], parse_dates=["Date"], dtype={"Status_Code": "string"})
    merchant = pd.read_csv(paths["merchant"])
    business = pd.read_csv(paths["business"])
    status = pd.read_csv(paths["status"], dtype={"Status_Code": "string"})

    tx["Status_Code"] = clean_status_code(tx["Status_Code"])
    status["Status_Code"] = clean_status_code(status["Status_Code"])
    return tx, merchant, business, status


def prepare_daily_data(
    transactions: pd.DataFrame,
    merchant: pd.DataFrame,
    business: pd.DataFrame,
    status: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge references, filter captured sales and create daily merchant/business data."""
    df = (
        transactions
        .merge(merchant, on="Merchant_ID", how="left", validate="many_to_one")
        .merge(business, on="Business_ID", how="left", validate="many_to_one")
        .merge(status, on="Status_Code", how="left", validate="many_to_one")
    )

    captured = df[df["Status"].eq("Captured")].copy()

    calendar = pd.date_range(transactions["Date"].min(), transactions["Date"].max(), freq="D", name="Date")

    daily_agg = (
        captured.groupby(["Merchant_ID", "Date"], as_index=False)
        .agg(Daily_Sales=("Amount", "sum"), Transaction_Count=("Transaction_ID", "size"))
    )

    merchant_calendar = pd.MultiIndex.from_product(
        [merchant["Merchant_ID"].unique(), calendar], names=["Merchant_ID", "Date"]
    ).to_frame(index=False)

    daily = merchant_calendar.merge(daily_agg, on=["Merchant_ID", "Date"], how="left")
    daily[["Daily_Sales", "Transaction_Count"]] = daily[["Daily_Sales", "Transaction_Count"]].fillna(0)
    daily = daily.merge(merchant[["Merchant_ID", "Merchant"]], on="Merchant_ID", how="left")
    daily = daily.sort_values(["Merchant_ID", "Date"]).reset_index(drop=True)

    business_agg = (
        captured.groupby(["Merchant_ID", "Business_ID", "Date"], as_index=False)
        .agg(Business_Sales=("Amount", "sum"), Business_Transactions=("Transaction_ID", "size"))
    )

    business_calendar = pd.MultiIndex.from_product(
        [merchant["Merchant_ID"].unique(), business["Business_ID"].unique(), calendar],
        names=["Merchant_ID", "Business_ID", "Date"],
    ).to_frame(index=False)

    business_daily = business_calendar.merge(
        business_agg, on=["Merchant_ID", "Business_ID", "Date"], how="left"
    )
    business_daily[["Business_Sales", "Business_Transactions"]] = business_daily[
        ["Business_Sales", "Business_Transactions"]
    ].fillna(0)
    business_daily = (
        business_daily
        .merge(merchant[["Merchant_ID", "Merchant"]], on="Merchant_ID", how="left")
        .merge(business[["Business_ID", "Business"]], on="Business_ID", how="left")
        .sort_values(["Merchant_ID", "Business_ID", "Date"])
        .reset_index(drop=True)
    )

    return df, daily, business_daily


def add_time_series_features(daily: pd.DataFrame, business_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.copy()
    daily["Weekday"] = daily["Date"].dt.dayofweek
    g = daily.groupby("Merchant_ID", sort=False)

    daily["Prior_14_Median"] = g["Daily_Sales"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=7).median()
    )
    daily["Weekday_Median"] = daily.groupby(["Merchant_ID", "Weekday"])["Daily_Sales"].transform(
        lambda s: s.shift(1).expanding(min_periods=4).median()
    )
    daily["Expected_Sales"] = daily["Weekday_Median"].fillna(daily["Prior_14_Median"])
    daily["Log_Deviation"] = np.log((daily["Daily_Sales"] + 1) / (daily["Expected_Sales"] + 1))
    daily["Abs_Log_Deviation"] = daily["Log_Deviation"].abs()
    daily["Candidate_Extreme_Day"] = daily["Abs_Log_Deviation"].ge(np.log(MIN_RELATIVE_FACTOR))
    daily["Daily_Log_Change"] = daily.groupby("Merchant_ID")["Daily_Sales"].transform(
        lambda s: np.log1p(s).diff().abs()
    )

    parts = []
    for _, grp in daily.groupby("Merchant_ID", sort=False):
        grp = grp.copy()
        shifted = grp["Log_Deviation"].shift(1)
        grp["Residual_Median"] = shifted.rolling(28, min_periods=14).median()
        grp["Residual_MAD"] = shifted.rolling(28, min_periods=14).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        grp["Robust_Z"] = 0.6745 * (grp["Log_Deviation"] - grp["Residual_Median"]) / grp[
            "Residual_MAD"
        ].replace(0, np.nan)
        parts.append(grp)
    daily = pd.concat(parts, ignore_index=True).sort_values(["Merchant_ID", "Date"]).reset_index(drop=True)
    daily["High_Confidence_Day"] = (
        daily["Robust_Z"].abs().ge(ROBUST_Z_THRESHOLD)
        & daily["Abs_Log_Deviation"].ge(np.log(MIN_RELATIVE_FACTOR))
    )

    business_daily = business_daily.copy()
    business_daily["Weekday"] = business_daily["Date"].dt.dayofweek
    business_daily["Prior_14_Median"] = business_daily.groupby(["Merchant_ID", "Business_ID"])[
        "Business_Sales"
    ].transform(lambda s: s.shift(1).rolling(14, min_periods=7).median())
    business_daily["Weekday_Median"] = business_daily.groupby(["Merchant_ID", "Business_ID", "Weekday"])[
        "Business_Sales"
    ].transform(lambda s: s.shift(1).expanding(min_periods=4).median())
    business_daily["Expected_Business_Sales"] = business_daily["Weekday_Median"].fillna(
        business_daily["Prior_14_Median"]
    )
    business_daily["Business_Deviation"] = np.log(
        (business_daily["Business_Sales"] + 1) / (business_daily["Expected_Business_Sales"] + 1)
    )
    business_daily["Abs_Business_Deviation"] = business_daily["Business_Deviation"].abs()
    return daily, business_daily


def build_merchant_features(daily: pd.DataFrame) -> pd.DataFrame:
    merchant_features = daily.groupby("Merchant_ID").agg(
        Mean_Sales=("Daily_Sales", "mean"),
        SD_Sales=("Daily_Sales", "std"),
        Mean_Transactions=("Transaction_Count", "mean"),
        SD_Transactions=("Transaction_Count", "std"),
        Peak_Positive_Deviation=("Log_Deviation", lambda x: x.clip(lower=0).max()),
        Peak_Negative_Deviation=("Log_Deviation", lambda x: (-x.clip(upper=0)).max()),
        P95_Absolute_Deviation=("Abs_Log_Deviation", lambda x: x.quantile(0.95)),
        Mean_Absolute_Deviation=("Abs_Log_Deviation", "mean"),
        High_Confidence_Day_Count=("High_Confidence_Day", "sum"),
        Max_Robust_Z=("Robust_Z", lambda x: x.abs().max()),
    )

    merchant_features["Sales_CV"] = merchant_features["SD_Sales"] / (merchant_features["Mean_Sales"] + 1)
    merchant_features["Transaction_CV"] = merchant_features["SD_Transactions"] / (
        merchant_features["Mean_Transactions"] + 1
    )
    merchant_features["Largest_Daily_Log_Change"] = daily.groupby("Merchant_ID")["Daily_Log_Change"].max()

    ordered = daily.sort_values(["Merchant_ID", "Date"])
    first_30 = ordered.groupby("Merchant_ID").head(30).groupby("Merchant_ID")["Daily_Sales"].mean()
    last_30 = ordered.groupby("Merchant_ID").tail(30).groupby("Merchant_ID")["Daily_Sales"].mean()
    merchant_features["Period_Shift"] = np.log((last_30 + 1) / (first_30 + 1))
    merchant_features["Max_Robust_Z"] = merchant_features["Max_Robust_Z"].clip(upper=20)
    return merchant_features


def prepare_X(features: pd.DataFrame, medians: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    X = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    if medians is None:
        medians = X.median()
    X = X.fillna(medians).fillna(0)
    return X, medians


def save_submission(
    merchant_lookup: pd.DataFrame,
    feature_index: Iterable[int],
    predictions: np.ndarray,
    filename: Path,
) -> pd.DataFrame:
    pred_series = pd.Series(predictions, index=pd.Index(feature_index, name="Merchant_ID"), name="Prediction")
    output = merchant_lookup[["Merchant_ID", "Merchant"]].copy()
    output["Prediction"] = output["Merchant_ID"].map(pred_series).astype(int)
    output = output[["Merchant", "Prediction"]].sort_values("Merchant").reset_index(drop=True)

    assert output["Merchant"].notna().all()
    assert output["Merchant"].nunique() == len(output)
    assert set(output["Prediction"].unique()).issubset({0, 1})

    filename.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(filename, index=False)
    print(f"Saved {filename} | rows={len(output)} | anomalies={int(output['Prediction'].sum())}")
    return output


def train_original(data_dir: Path, output_dir: Path):
    paths = locate_files(data_dir, ORIGINAL_FILES)
    transactions, merchant, business, status = read_source_files(paths)
    merged, daily, business_daily = prepare_daily_data(transactions, merchant, business, status)
    daily, business_daily = add_time_series_features(daily, business_daily)
    merchant_features = build_merchant_features(daily)
    X, medians = prepare_X(merchant_features)

    iso_model = IsolationForest(
        n_estimators=500,
        contamination=FINAL_CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    iso_pred = (iso_model.fit_predict(X) == -1).astype(int)
    iso_score = -iso_model.score_samples(X)

    merchant_risk = merchant_features.assign(Prediction=iso_pred, Anomaly_Score=iso_score).reset_index()
    merchant_risk = merchant_risk.merge(merchant[["Merchant_ID", "Merchant"]], on="Merchant_ID", how="left")
    merchant_risk = merchant_risk.sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)
    merchant_risk["Risk_Rank"] = np.arange(1, len(merchant_risk) + 1)
    merchant_risk["Risk_Label"] = np.where(merchant_risk["Prediction"].eq(1), "Flagged", "Normal")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "daily_sales_prepared.csv", index=False)
    business_daily.to_csv(output_dir / "business_daily_prepared.csv", index=False)
    merchant_risk.to_csv(output_dir / "merchant_risk.csv", index=False)
    save_submission(merchant, merchant_features.index, iso_pred, output_dir / "predictions_original_isolation_forest.csv")

    return {
        "merchant": merchant,
        "daily": daily,
        "business_daily": business_daily,
        "features": merchant_features,
        "X": X,
        "medians": medians,
        "iso_model": iso_model,
        "iso_pred": iso_pred,
        "merchant_risk": merchant_risk,
    }


def train_new_if_available(data_dir: Path, output_dir: Path, original_state: dict):
    if not all((data_dir / f).exists() for f in NEW_FILES.values()):
        print("New data files not found. Skipping Deliverable 2 prediction exports.")
        return None

    paths = locate_files(data_dir, NEW_FILES)
    transactions, merchant_new, business_new, status_new = read_source_files(paths)
    _, daily_new, business_daily_new = prepare_daily_data(transactions, merchant_new, business_new, status_new)
    daily_new, business_daily_new = add_time_series_features(daily_new, business_daily_new)
    features_new = build_merchant_features(daily_new)
    X_new, _ = prepare_X(features_new, original_state["medians"])

    X_train = original_state["X"]
    y_train = original_state["iso_pred"]
    iso_model = original_state["iso_model"]

    merchant_new_ordered = merchant_new.set_index("Merchant_ID").loc[features_new.index].reset_index()

    pred_if = (iso_model.predict(X_new) == -1).astype(int)
    save_submission(merchant_new_ordered, features_new.index, pred_if, output_dir / "predictions_new_isolation_forest.csv")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_new)
    save_submission(merchant_new_ordered, features_new.index, pred_lr, output_dir / "predictions_new_logistic_regression.csv")

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_new)
    save_submission(merchant_new_ordered, features_new.index, pred_rf, output_dir / "predictions_new_random_forest.csv")

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    gb.fit(X_train, y_train, sample_weight=weights)
    pred_gb = gb.predict(X_new)
    save_submission(merchant_new_ordered, features_new.index, pred_gb, output_dir / "predictions_new_gradient_boosting.csv")

    daily_new.to_csv(output_dir / "daily_sales_new_prepared.csv", index=False)
    business_daily_new.to_csv(output_dir / "business_daily_new_prepared.csv", index=False)
    return {
        "daily_new": daily_new,
        "business_daily_new": business_daily_new,
        "features_new": features_new,
        "X_new": X_new,
        "merchant_new": merchant_new_ordered,
        "predictions": {"isolation_forest": pred_if, "logistic_regression": pred_lr, "random_forest": pred_rf, "gradient_boosting": pred_gb},
    }


def run_batch(data_dir: Path, output_dir: Path) -> None:
    state = train_original(data_dir, output_dir)
    train_new_if_available(data_dir, output_dir, state)
    print("\nBatch run completed.")
    print(f"Outputs written to: {output_dir.resolve()}")


def render_streamlit_app(data_dir: Path, output_dir: Path) -> None:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.set_page_config(
        page_title="Merchant Sales Anomaly Monitor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---------- Light presentation styling ----------
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            background: rgba(127,127,127,0.06);
            border: 1px solid rgba(127,127,127,0.16);
            padding: 14px 16px;
            border-radius: 12px;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600;
        }
        .portfolio-note {
            padding: 0.7rem 0.9rem;
            border-radius: 10px;
            background: rgba(127,127,127,0.07);
            border-left: 4px solid rgba(127,127,127,0.45);
            margin-bottom: 1rem;
        }
        .small-note {
            font-size: 0.88rem;
            opacity: 0.78;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Load prepared hosted-demo outputs ----------
    required_outputs = [
        output_dir / "daily_sales_prepared.csv",
        output_dir / "business_daily_prepared.csv",
        output_dir / "merchant_risk.csv",
    ]

    if not all(path.exists() for path in required_outputs):
        st.error(
            "Prepared dashboard files are missing. Ensure "
            "daily_sales_prepared.csv, business_daily_prepared.csv and "
            "merchant_risk.csv are in the repository root."
        )
        st.stop()

    daily = pd.read_csv(output_dir / "daily_sales_prepared.csv", parse_dates=["Date"])
    business_daily = pd.read_csv(
        output_dir / "business_daily_prepared.csv",
        parse_dates=["Date"],
    )
    merchant_risk = pd.read_csv(output_dir / "merchant_risk.csv")

    # ---------- Core portfolio objects ----------
    flagged_sorted = (
        merchant_risk.loc[merchant_risk["Prediction"].eq(1)]
        .sort_values("Risk_Rank")
        .copy()
    )

    total_merchants = merchant_risk["Merchant_ID"].nunique()
    flagged_count = int(merchant_risk["Prediction"].sum())
    flagged_rate = flagged_count / total_merchants if total_merchants else 0
    high_conf_days = int(daily["High_Confidence_Day"].sum())
    total_sales = float(daily["Daily_Sales"].sum())
    start_date = daily["Date"].min()
    end_date = daily["Date"].max()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Dashboard controls")

        status_filter = st.radio(
            "Merchant status",
            ["Flagged only", "All merchants"],
            index=0,
        )

        st.divider()
        st.subheader("Portfolio")
        st.caption(f"**{total_merchants}** merchants")
        st.caption(
            f"{start_date.strftime('%d %b %Y')} – "
            f"{end_date.strftime('%d %b %Y')}"
        )
        st.caption("Captured transactions only")

        st.divider()
        with st.expander("About this monitor"):
            st.markdown(
                """
                **Merchant screening:** Isolation Forest  
                **Anomaly-day localisation:** robust historical deviation  
                **Business drill-down:** actual vs expected category sales  
                **Hosted mode:** prepared analytical outputs
                """
            )

    # ---------- Page header ----------
    st.title("Merchant Sales Anomaly Monitor")
    st.caption(
        "Portfolio-level monitoring of unusual merchant sales behaviour, "
        "with day-level localisation and business-category explanation."
    )

    st.markdown(
        f"""
        <div class="portfolio-note">
        Monitoring window: <b>{start_date.strftime('%d %b %Y')}</b> to
        <b>{end_date.strftime('%d %b %Y')}</b> ·
        Sales definition: <b>Captured transactions only</b> ·
        Flagged rate: <b>{flagged_rate:.1%}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- KPI row ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Merchants", f"{total_merchants:,}")
    c2.metric("Flagged merchants", f"{flagged_count:,}", f"{flagged_rate:.1%} of portfolio")
    c3.metric("High-confidence days", f"{high_conf_days:,}")
    c4.metric("Captured sales", f"${total_sales / 1_000_000:,.1f}M")

    st.divider()

    # ---------- Merchant selector ----------
    if status_filter == "Flagged only":
        choices = flagged_sorted["Merchant"].tolist()
    else:
        choices = merchant_risk["Merchant"].sort_values().tolist()

    if not choices:
        st.warning("No merchants are available for the selected filter.")
        st.stop()

    default_merchant = (
        flagged_sorted.iloc[0]["Merchant"]
        if not flagged_sorted.empty
        else choices[0]
    )
    default_index = choices.index(default_merchant) if default_merchant in choices else 0

    selected = st.selectbox(
        "Select merchant",
        choices,
        index=default_index,
        help="Choose a merchant to inspect its daily sales behaviour and anomaly drivers.",
    )

    m_daily = daily[daily["Merchant"].eq(selected)].copy()
    m_risk = merchant_risk[merchant_risk["Merchant"].eq(selected)].iloc[0]
    anomalies = m_daily[m_daily["High_Confidence_Day"].eq(True)].copy()

    # ---------- Main tabs ----------
    tab1, tab2, tab3 = st.tabs(
        ["Merchant investigation", "Portfolio view", "Model evaluation"]
    )

    # ============================================================
    # TAB 1 — Merchant investigation
    # ============================================================
    with tab1:
        st.subheader(f"Merchant overview — {selected}")

        status_text = "FLAGGED" if int(m_risk["Prediction"]) == 1 else "NORMAL"

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk status", status_text)
        r2.metric("Risk rank", f"{int(m_risk['Risk_Rank'])} of {total_merchants}")
        r3.metric("Anomaly score", f"{float(m_risk['Anomaly_Score']):.3f}")
        r4.metric("High-confidence days", int(m_risk["High_Confidence_Day_Count"]))

        if int(m_risk["Prediction"]) == 1:
            st.warning(
                f"{selected} is flagged for review and ranks "
                f"#{int(m_risk['Risk_Rank'])} in the portfolio risk ordering."
            )
        else:
            st.success(
                f"{selected} is not currently flagged by the merchant-level anomaly model."
            )

        # ---------- Daily sales chart ----------
        st.markdown("### Daily sales vs expected behaviour")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=m_daily["Date"],
                y=m_daily["Daily_Sales"],
                mode="lines",
                name="Daily sales",
                line=dict(width=2.4),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Daily sales: $%{y:,.0f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=m_daily["Date"],
                y=m_daily["Expected_Sales"],
                mode="lines",
                name="Expected sales",
                line=dict(width=1.6, dash="dash"),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Expected sales: $%{y:,.0f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomalies["Date"],
                y=anomalies["Daily_Sales"],
                mode="markers",
                name="High-confidence anomaly",
                marker=dict(size=13, symbol="diamond"),
                customdata=(
                    anomalies[["Expected_Sales", "Robust_Z", "Log_Deviation"]].to_numpy()
                    if not anomalies.empty
                    else None
                ),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Actual sales: $%{y:,.0f}<br>"
                    "Expected sales: $%{customdata[0]:,.0f}<br>"
                    "Robust Z: %{customdata[1]:.2f}<br>"
                    "Log deviation: %{customdata[2]:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis_title="Date",
            yaxis_title="Sales (USD)",
            legend_title=None,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Anomaly-day selector ----------
        st.markdown("### Anomaly-day investigation")

        if anomalies.empty:
            st.info(
                "No high-confidence anomaly day was identified for this merchant. "
                "The latest available date is shown for business-category context."
            )
            chosen_date = m_daily["Date"].max()
        else:
            anomaly_date_options = anomalies["Date"].dt.date.tolist()
            chosen_date = pd.to_datetime(
                st.selectbox(
                    "Select anomaly date",
                    anomaly_date_options,
                    index=0,
                )
            )

        selected_day = m_daily[m_daily["Date"].eq(chosen_date)]
        if not selected_day.empty:
            day = selected_day.iloc[0]
            actual = float(day["Daily_Sales"])
            expected = day["Expected_Sales"]
            gap = actual - expected if pd.notna(expected) else np.nan
            direction = "above" if pd.notna(gap) and gap > 0 else "below"

            q1, q2, q3 = st.columns(3)
            q1.metric("Actual sales", f"${actual:,.0f}")
            q2.metric(
                "Expected sales",
                f"${expected:,.0f}" if pd.notna(expected) else "Warm-up period",
            )
            q3.metric(
                "Variance to expected",
                f"${gap:,.0f}" if pd.notna(gap) else "N/A",
            )

        # ---------- Business-category drill-down ----------
        bd = business_daily[
            business_daily["Merchant"].eq(selected)
            & business_daily["Date"].eq(chosen_date)
        ].copy()

        bd["Dollar_Gap"] = bd["Business_Sales"] - bd["Expected_Business_Sales"]
        bd["Absolute_Dollar_Gap"] = bd["Dollar_Gap"].abs()
        bd = bd.sort_values("Absolute_Dollar_Gap", ascending=False)

        if not bd.empty:
            top_driver = bd.iloc[0]
            gap_text = (
                f"${abs(float(top_driver['Dollar_Gap'])):,.0f}"
                if pd.notna(top_driver["Dollar_Gap"])
                else "an unavailable amount"
            )
            driver_direction = (
                "above"
                if pd.notna(top_driver["Dollar_Gap"]) and top_driver["Dollar_Gap"] > 0
                else "below"
            )

            if "direction" in locals() and pd.notna(gap):
                st.markdown(
                    f"""
                    <div class="portfolio-note">
                    <b>Alert interpretation:</b> On {chosen_date.strftime('%d %b %Y')},
                    total merchant sales were <b>{direction} expected behaviour</b>.
                    The largest business-category contribution came from
                    <b>{top_driver['Business']}</b>, which was approximately
                    <b>{gap_text} {driver_direction} its own expected sales</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        d1, d2 = st.columns([1.08, 1])

        with d1:
            st.markdown("#### Business-category detail")

            display_bd = bd[
                [
                    "Business",
                    "Business_Sales",
                    "Expected_Business_Sales",
                    "Dollar_Gap",
                    "Business_Deviation",
                ]
            ].copy()

            display_bd.columns = [
                "Business category",
                "Actual sales",
                "Expected sales",
                "Dollar gap",
                "Log deviation",
            ]

            st.dataframe(
                display_bd,
                hide_index=True,
                use_container_width=True,
                height=300,
                column_config={
                    "Actual sales": st.column_config.NumberColumn(format="$%0.0f"),
                    "Expected sales": st.column_config.NumberColumn(format="$%0.0f"),
                    "Dollar gap": st.column_config.NumberColumn(format="$%0.0f"),
                    "Log deviation": st.column_config.NumberColumn(format="%.3f"),
                },
            )

        with d2:
            st.markdown("#### Category variance to expected")
            if not bd.empty:
                fig2 = px.bar(
                    bd,
                    x="Business",
                    y="Dollar_Gap",
                    labels={
                        "Business": "Business category",
                        "Dollar_Gap": "Actual − expected sales (USD)",
                    },
                )
                fig2.update_layout(
                    height=335,
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # TAB 2 — Portfolio view
    # ============================================================
    with tab2:
        st.subheader("Flagged merchant portfolio")
        st.caption(
            "Portfolio prioritisation based on the merchant-level Isolation Forest anomaly score."
        )

        p1, p2 = st.columns([1.15, 1])

        with p1:
            flagged_display = flagged_sorted[
                [
                    "Risk_Rank",
                    "Merchant",
                    "Anomaly_Score",
                    "High_Confidence_Day_Count",
                    "Sales_CV",
                    "Period_Shift",
                ]
            ].copy()

            flagged_display.columns = [
                "Risk rank",
                "Merchant",
                "Anomaly score",
                "High-confidence days",
                "Sales variability",
                "Period shift",
            ]

            st.dataframe(
                flagged_display,
                hide_index=True,
                use_container_width=True,
                height=430,
                column_config={
                    "Anomaly score": st.column_config.NumberColumn(format="%.3f"),
                    "Sales variability": st.column_config.NumberColumn(format="%.3f"),
                    "Period shift": st.column_config.NumberColumn(format="%.3f"),
                },
            )

        with p2:
            risk_chart = flagged_sorted.sort_values("Anomaly_Score", ascending=True)
            fig3 = px.bar(
                risk_chart,
                x="Anomaly_Score",
                y="Merchant",
                orientation="h",
                labels={
                    "Anomaly_Score": "Isolation Forest anomaly score",
                    "Merchant": "",
                },
                title="Flagged merchants by anomaly score",
            )
            fig3.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=45, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            """
            <div class="small-note">
            <b>Interpretation:</b> the portfolio table is a review-prioritisation tool.
            A higher anomaly score indicates that a merchant's engineered behavioural
            profile is more unusual relative to the rest of the portfolio; it is not,
            by itself, proof of fraud or misconduct.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # TAB 3 — Model evaluation
    # ============================================================
    with tab3:
        st.subheader("External model evaluator — new dataset")
        st.caption(
            "Performance against the assignment's concealed ground-truth labels."
        )

        performance = pd.DataFrame(
            [
                ["Logistic Regression", 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 15, 0, 0, 90],
                ["Random Forest", 0.981, 1.000, 0.867, 0.929, 1.000, 0.934, 13, 0, 2, 90],
                ["Isolation Forest", 0.981, 1.000, 0.867, 0.929, 1.000, 0.934, 13, 0, 2, 90],
                ["Gradient Boosting", 0.962, 0.923, 0.800, 0.857, 0.989, 0.895, 12, 1, 3, 89],
            ],
            columns=[
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Specificity",
                "Balanced accuracy",
                "TP",
                "FP",
                "FN",
                "TN",
            ],
        )

        e1, e2 = st.columns([1.25, 1])

        with e1:
            st.dataframe(
                performance,
                hide_index=True,
                use_container_width=True,
                height=270,
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "Precision": st.column_config.NumberColumn(format="%.3f"),
                    "Recall": st.column_config.NumberColumn(format="%.3f"),
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                    "Specificity": st.column_config.NumberColumn(format="%.3f"),
                    "Balanced accuracy": st.column_config.NumberColumn(format="%.3f"),
                },
            )

        with e2:
            perf_long = performance.melt(
                id_vars=["Model"],
                value_vars=["Recall", "F1", "Balanced accuracy"],
                var_name="Metric",
                value_name="Score",
            )
            fig4 = px.bar(
                perf_long,
                x="Model",
                y="Score",
                barmode="group",
                facet_col="Metric",
                category_orders={
                    "Model": [
                        "Logistic Regression",
                        "Random Forest",
                        "Isolation Forest",
                        "Gradient Boosting",
                    ]
                },
            )
            fig4.update_yaxes(range=[0, 1.05])
            fig4.update_layout(
                height=300,
                margin=dict(l=5, r=5, t=30, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.success(
            "Preferred supervised model: Logistic Regression. "
            "It achieved perfect classification on the supplied synthetic hidden "
            "evaluation dataset (105/105 merchants correctly classified)."
        )

        st.info(
            "Production recommendation: use Logistic Regression as the primary "
            "classifier for known patterns, retain Isolation Forest as an unsupervised "
            "challenger for emerging or structurally unusual behaviour, and use the "
            "day/business drill-down to explain alerts."
        )

    # ---------- Methodology footer ----------
    st.divider()
    with st.expander("Methodology and interpretation notes"):
        st.markdown(
            """
            **1. Sales definition**  
            Only transactions with `Status = Captured` are treated as sales.

            **2. Merchant-level screening**  
            Isolation Forest is applied to engineered behavioural features such as
            sales variability, extreme deviations, robust residual scores and period shift.

            **3. Day-level localisation**  
            Expected sales use prior-only historical information. High-confidence days
            combine a robust Z-score threshold with a material deviation threshold.

            **4. Business-category explanation**  
            On a selected anomaly date, each category's actual sales are compared with
            its own historical expectation to identify the largest likely driver.

            **5. Decision use**  
            An anomaly is a review signal, not proof of fraud. Final investigation
            should combine model output with merchant and business context.
            """
        )

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merchant anomaly detection assignment tool")
    parser.add_argument("--mode", choices=["batch", "app"], default="app", help="Run batch outputs or Streamlit app")
    parser.add_argument("--data-dir", default=".", help="Folder containing source CSV files")
    parser.add_argument("--output-dir", default=".", help="Folder where prepared outputs are stored")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if args.mode == "app":
        render_streamlit_app(data_dir, output_dir)
    else:
        run_batch(data_dir, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

