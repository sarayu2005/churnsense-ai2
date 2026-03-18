import io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _detect_columns(df):
    duration_col = None
    event_col = None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["tenure", "duration", "months", "days", "time", "age"]):
            if duration_col is None:
                duration_col = c
        if any(k in cl for k in ["churn", "event", "churned"]):
            if event_col is None:
                event_col = c
    if duration_col is None:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        duration_col = numeric[0] if numeric else df.columns[0]
    if event_col is None:
        event_col = df.columns[-1]
    return duration_col, event_col


def run_survival(df: pd.DataFrame):
    df = df.copy()

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.factorize(df[c])[0]

    df.fillna(df.median(numeric_only=True), inplace=True)
    duration_col, event_col = _detect_columns(df)

    T = df[duration_col].clip(lower=0)
    E = df[event_col].astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label="Overall")
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel(duration_col)
    ax.set_ylabel("Survival Probability")
    km_plot = _fig_to_b64(fig)

    cox_summary = None
    cox_plot = None
    try:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_to_use = [c for c in num_cols if c != event_col][:8] + [event_col]
        cox_df = df[cols_to_use].copy()
        if duration_col in cox_df.columns:
            cox_df = cox_df.rename(columns={duration_col: "__T__"})
            dur_key = "__T__"
        else:
            dur_key = duration_col
        if event_col in cox_df.columns:
            cox_df = cox_df.rename(columns={event_col: "__E__"})
            evt_key = "__E__"
        else:
            evt_key = event_col
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=dur_key, event_col=evt_key)
        cox_summary = cph.summary.round(4).reset_index().to_dict(orient="records")
        fig, ax = plt.subplots(figsize=(8, 5))
        cph.plot(ax=ax)
        ax.set_title("Cox PH - Hazard Ratios")
        cox_plot = _fig_to_b64(fig)
    except Exception as e:
        cox_summary = [{"error": str(e)}]

    survival_at = kmf.survival_function_at_times(T).values
    risk_df = df[[duration_col]].copy()
    risk_df["survival_score"] = survival_at
    risk_df["churn_risk"] = 1 - risk_df["survival_score"]
    high_risk = (
        risk_df.nlargest(10, "churn_risk")
        .reset_index()
        .rename(columns={"index": "customer_id"})
        .round(4)
        .to_dict(orient="records")
    )

    return {
        "km_plot": km_plot,
        "cox_summary": cox_summary,
        "cox_plot": cox_plot,
        "high_risk_customers": high_risk,
        "duration_col": duration_col,
        "event_col": event_col,
    }
