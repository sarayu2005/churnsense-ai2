import io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _detect_treatment_outcome(df):
    outcome = None
    for c in df.columns:
        if "churn" in c.lower():
            outcome = c
            break
    if outcome is None:
        outcome = df.columns[-1]

    treatment = None
    for c in df.columns:
        if any(k in c.lower() for k in ["fee", "charge", "price", "cost", "monthly", "spend"]):
            treatment = c
            break
    if treatment is None:
        numeric = [c for c in df.select_dtypes(include=np.number).columns if c != outcome]
        treatment = numeric[0] if numeric else df.columns[0]

    return treatment, outcome


def run_causal(df: pd.DataFrame):
    df = df.copy()

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.factorize(df[c])[0]
    df.fillna(df.median(numeric_only=True), inplace=True)

    treatment, outcome = _detect_treatment_outcome(df)

    median_val = df[treatment].median()
    df["_treatment_bin"] = (df[treatment] > median_val).astype(int)

    causal_estimate = None
    graph_plot = None
    method_used = None

    if DOWHY_AVAILABLE:
        try:
            common_causes = [c for c in df.columns
                             if c not in (treatment, outcome, "_treatment_bin")][:5]

            gml = (
                "graph [directed 1 node [id \"_treatment_bin\" label \"_treatment_bin\"] "
                + " ".join(
                    f'node [id "{c}" label "{c}"]' for c in common_causes
                )
                + f' node [id "{outcome}" label "{outcome}"]'
                + " ".join(
                    f'edge [source "{c}" target "_treatment_bin"] edge [source "{c}" target "{outcome}"]'
                    for c in common_causes
                )
                + f' edge [source "_treatment_bin" target "{outcome}"]'
                + " ]"
            )

            model = CausalModel(
                data=df,
                treatment="_treatment_bin",
                outcome=outcome,
                graph=gml,
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )
            causal_estimate = float(estimate.value)
            method_used = "DoWhy - Backdoor Linear Regression"

            fig, ax = plt.subplots(figsize=(8, 5))
            cols_to_plot = common_causes + ["_treatment_bin", outcome]
            positions = {}
            for i, c in enumerate(cols_to_plot):
                positions[c] = (i * 2, 0 if c not in ("_treatment_bin", outcome) else -2)
            for c in common_causes:
                ax.annotate("", xy=positions["_treatment_bin"], xytext=positions[c],
                            arrowprops=dict(arrowstyle="->", color="gray"))
                ax.annotate("", xy=positions[outcome], xytext=positions[c],
                            arrowprops=dict(arrowstyle="->", color="gray"))
            ax.annotate("", xy=positions[outcome], xytext=positions["_treatment_bin"],
                        arrowprops=dict(arrowstyle="->", color="red", lw=2))
            for c, pos in positions.items():
                ax.text(pos[0], pos[1], c, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
            ax.set_xlim(-2, max(p[0] for p in positions.values()) + 2)
            ax.set_ylim(-4, 2)
            ax.axis("off")
            ax.set_title("Causal Graph")
            graph_plot = _fig_to_b64(fig)

        except Exception as e:
            causal_estimate = None
            method_used = f"DoWhy error: {e}"
    else:
        from sklearn.linear_model import LinearRegression
        X = df[["_treatment_bin"]].values
        y = df[outcome].values
        lr = LinearRegression().fit(X, y)
        causal_estimate = float(lr.coef_[0])
        method_used = "Fallback - Linear Regression (DoWhy not available)"

    return {
        "treatment": treatment,
        "outcome": outcome,
        "causal_estimate": causal_estimate,
        "method": method_used,
        "graph_plot": graph_plot,
    }
