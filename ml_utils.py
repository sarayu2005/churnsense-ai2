import os, io, base64, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _detect_churn_col(df):
    for c in df.columns:
        if "churn" in c.lower():
            return c
    for c in reversed(df.columns):
        uniq = df[c].dropna().unique()
        if len(uniq) == 2:
            return c
    return df.columns[-1]


def run_ml(df: pd.DataFrame):
    df = df.copy()
    target = _detect_churn_col(df)

    for c in df.columns:
        if df[c].dtype == object:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))

    df.fillna(df.median(numeric_only=True), inplace=True)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                  random_state=42, verbosity=0),
    }

    metrics_rows = []
    roc_data = {}
    best_name, best_model, best_auc = None, None, -1

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        metrics_rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(auc, 4),
        })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc, 4)}
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = clf

    model_path = os.path.join(MODELS_DIR, "churn_predictor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "features": list(X.columns)}, f)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
        kind="bar", ax=ax, rot=0
    )
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    comparison_plot = _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rd in roc_data.items():
        ax.plot(rd["fpr"], rd["tpr"], label=f"{name} (AUC={rd['auc']})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    roc_plot = _fig_to_b64(fig)

    return {
        "metrics": metrics_rows,
        "best_model": best_name,
        "comparison_plot": comparison_plot,
        "roc_plot": roc_plot,
    }
