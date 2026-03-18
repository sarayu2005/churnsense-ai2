import os, io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def run_eda(df: pd.DataFrame):
    result = {}

    desc = df.describe(include="all").fillna("").astype(str)
    result["summary"] = desc.to_dict()

    missing = df.isnull().sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    missing_nonzero = missing[missing > 0].sort_values(ascending=False)
    if len(missing_nonzero) > 0:
        missing_nonzero.plot(kind="bar", ax=ax, color="coral")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="green")
    ax.set_title("Missing Values per Column")
    result["missing_plot"] = _fig_to_b64(fig)
    result["missing"] = missing.to_dict()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]
    dist_plots = {}
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        df[col].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Distribution: {col}")
        dist_plots[col] = _fig_to_b64(fig)
    result["distribution_plots"] = dist_plots

    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        result["correlation_plot"] = _fig_to_b64(fig)
    else:
        result["correlation_plot"] = None

    return result


def generate_pdf_report(df: pd.DataFrame, eda_result: dict, out_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ChurnSense AI - EDA Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Descriptive Statistics", ln=True)
    pdf.set_font("Helvetica", size=8)
    desc = df.describe().round(2)
    col_w = min(30, (pdf.w - 20) / (len(desc.columns) + 1))
    header = ["stat"] + list(desc.columns)
    for h in header:
        pdf.cell(col_w, 6, str(h)[:12], border=1)
    pdf.ln()
    for idx, row in desc.iterrows():
        pdf.cell(col_w, 6, str(idx), border=1)
        for v in row:
            pdf.cell(col_w, 6, f"{v:.2f}" if isinstance(v, float) else str(v)[:8], border=1)
        pdf.ln()

    for key in ("missing_plot", "correlation_plot"):
        b64 = eda_result.get(key)
        if not b64:
            continue
        img_data = base64.b64decode(b64)
        tmp = os.path.join(PLOTS_DIR, f"_tmp_{key}.png")
        with open(tmp, "wb") as f:
            f.write(img_data)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, key.replace("_", " ").title(), ln=True)
        pdf.image(tmp, x=10, w=180)
        os.remove(tmp)

    pdf.output(out_path)
