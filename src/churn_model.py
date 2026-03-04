import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

RANDOM_STATE = 42
DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn_Data (5).csv")
REPORTS_DIR = Path("reports")

try:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score as sk_accuracy,
        auc as sk_auc,
        confusion_matrix as sk_confusion_matrix,
        precision_score as sk_precision,
        recall_score as sk_recall,
        roc_curve as sk_roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def save_roc_svg(fpr_lr, tpr_lr, fpr_rf, tpr_rf, out_path):
    w, h, pad = 640, 460, 60
    tx = lambda x: pad + x * (w - 2 * pad)
    ty = lambda y: h - pad - y * (h - 2 * pad)
    def poly(points, color):
        pts = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" />'
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{pad}" y2="{pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{pad}" stroke="#aaaaaa" stroke-dasharray="5,5"/>',
        '<text x="320" y="30" text-anchor="middle" font-size="20">ROC Curve</text>',
        poly(list(zip(fpr_lr, tpr_lr)), "#1f77b4"),
        poly(list(zip(fpr_rf, tpr_rf)), "#d62728"),
        '</svg>'
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix_svg(cm, out_path):
    w, h = 420, 360
    x0, y0, s = 90, 70, 100
    vals = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
    maxv = max(vals) if max(vals) else 1
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="white"/>']
    idx = 0
    for r in range(2):
        for c in range(2):
            v = vals[idx]
            shade = int(255 - 170 * (v / maxv))
            x, y = x0 + c * s, y0 + r * s
            lines.append(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="rgb({shade},{shade},255)" stroke="black"/>')
            lines.append(f'<text x="{x+s/2}" y="{y+s/2+5}" text-anchor="middle" font-size="20">{v}</text>')
            idx += 1
    lines.append('</svg>')
    out_path.write_text("\n".join(lines), encoding="utf-8")


def high_value_customer_rule(customer):
    return customer["TotalCharges"] >= 3000 and customer["tenure"] >= 24 and customer["Contract"] in {"One year", "Two year"}


def run_sklearn_pipeline():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_train_prob = lr.predict_proba(X_train)[:, 1]
    lr_test_prob = lr.predict_proba(X_test)[:, 1]
    rf_train_prob = rf.predict_proba(X_train)[:, 1]
    rf_test_prob = rf.predict_proba(X_test)[:, 1]

    fpr_lr, tpr_lr, _ = sk_roc_curve(y_test, lr_test_prob)
    fpr_rf, tpr_rf, _ = sk_roc_curve(y_test, rf_test_prob)
    lr_train_auc, lr_test_auc = sk_auc(*sk_roc_curve(y_train, lr_train_prob)[:2]), sk_auc(fpr_lr, tpr_lr)
    rf_train_auc, rf_test_auc = sk_auc(*sk_roc_curve(y_train, rf_train_prob)[:2]), sk_auc(fpr_rf, tpr_rf)

    best_name = "random_forest" if rf_test_auc >= lr_test_auc else "logistic_regression"
    best_prob = rf_test_prob if best_name == "random_forest" else lr_test_prob
    thresholds = [i / 100 for i in range(20, 81)]
    best_thr, best_f1 = 0.5, -1
    for thr in thresholds:
        pred = (best_prob >= thr).astype(int)
        p, r = sk_precision(y_test, pred), sk_recall(y_test, pred)
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    y_pred = (best_prob >= best_thr).astype(int)
    cm = sk_confusion_matrix(y_test, y_pred).tolist()

    # feature diagnostics
    lr_pre = lr.named_steps["pre"]
    lr_clf = lr.named_steps["clf"]
    feature_names = list(lr_pre.get_feature_names_out())
    coefs = lr_clf.coef_[0]
    top_lr = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:15]

    rf_pre = rf.named_steps["pre"]
    rf_clf = rf.named_steps["clf"]
    top_rf = sorted(zip(rf_pre.get_feature_names_out(), rf_clf.feature_importances_), key=lambda x: x[1], reverse=True)[:15]

    all_scores = (rf if best_name == "random_forest" else lr).predict_proba(X)[:, 1]
    hv_flags = df.apply(lambda r: high_value_customer_rule(r), axis=1)
    hv_count = int(hv_flags.sum())
    hv_risk = int(((hv_flags) & (all_scores >= best_thr)).sum())

    metrics = {
        "backend": "sklearn",
        "data": {"n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "models": {
            "logistic_regression": {"train_auc": float(lr_train_auc), "test_auc": float(lr_test_auc)},
            "random_forest": {"train_auc": float(rf_train_auc), "test_auc": float(rf_test_auc)},
        },
        "selected_model": best_name,
        "decision_threshold": best_thr,
        "best_f1": best_f1,
        "test_metrics_selected_model": {
            "accuracy": float(sk_accuracy(y_test, y_pred)),
            "precision": float(sk_precision(y_test, y_pred)),
            "recall": float(sk_recall(y_test, y_pred)),
            "auc": float(sk_auc(*sk_roc_curve(y_test, best_prob)[:2])),
            "confusion_matrix": cm,
        },
        "top_logistic_coefficients": [{"feature": f, "coefficient": float(c)} for f, c in top_lr],
        "top_tree_importances": [{"feature": f, "importance": float(i)} for f, i in top_rf],
        "high_value_summary": {
            "definition_selected": "Rule A: TotalCharges >= 3000, tenure >= 24 months, contract in {One year, Two year}",
            "high_value_pct": hv_count / len(df), "high_value_count": hv_count,
            "high_value_high_risk_pct": hv_risk / hv_count if hv_count else 0.0,
            "high_value_high_risk_count": hv_risk,
        },
    }
    save_roc_svg(fpr_lr.tolist(), tpr_lr.tolist(), fpr_rf.tolist(), tpr_rf.tolist(), REPORTS_DIR / "roc_curve.svg")
    save_confusion_matrix_svg(cm, REPORTS_DIR / "confusion_matrix.svg")
    return metrics


# ---- fallback (custom implementation from previous revision) ----
def fallback_metrics():
    from churn_model_fallback import main as fallback_main
    fallback_main()
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    metrics["backend"] = "custom_python_fallback"
    return metrics


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics = run_sklearn_pipeline() if SKLEARN_AVAILABLE else fallback_metrics()
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved outputs with backend={metrics['backend']}")


if __name__ == "__main__":
    main()
