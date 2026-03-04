import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

RANDOM_STATE = 42
DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn_Data (5).csv")
REPORTS_DIR = Path("reports")


# ---------- Utility metrics ----------
def confusion_matrix(y_true, y_pred):
    tn = fp = fn = tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            tn += 1
    return [[tn, fp], [fn, tp]]


def accuracy_score(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fp = cm[0][1]
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fn = cm[1][0]
    return tp / (tp + fn) if (tp + fn) else 0.0


def roc_curve(y_true, y_proba):
    pairs = sorted(zip(y_proba, y_true), key=lambda x: x[0], reverse=True)
    p = sum(y_true)
    n = len(y_true) - p
    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]
    thresholds = [1.0]
    for prob, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / p if p else 0.0)
        fpr.append(fp / n if n else 0.0)
        thresholds.append(prob)
    tpr.append(1.0)
    fpr.append(1.0)
    thresholds.append(0.0)
    return fpr, tpr, thresholds


def auc_score(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return area


def best_f1_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, -1
    for thr in [i / 100 for i in range(20, 81)]:
        y_pred = [1 if p >= thr else 0 for p in y_proba]
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


# ---------- Data prep ----------
def load_and_clean(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_charges = []
    for row in rows:
        value = row["TotalCharges"].strip()
        if value:
            total_charges.append(float(value))

    total_charges_sorted = sorted(total_charges)
    median_tc = total_charges_sorted[len(total_charges_sorted) // 2]

    for row in rows:
        tc = row["TotalCharges"].strip()
        row["TotalCharges"] = float(tc) if tc else median_tc
        row["tenure"] = float(row["tenure"])
        row["MonthlyCharges"] = float(row["MonthlyCharges"])
        row["SeniorCitizen"] = float(row["SeniorCitizen"])
        row["Churn"] = 1 if row["Churn"].strip().lower() == "yes" else 0
    return rows


def stratified_split(rows, test_size=0.2, random_state=RANDOM_STATE):
    rng = random.Random(random_state)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["Churn"]].append(r)

    train, test = [], []
    for _, group in by_class.items():
        rng.shuffle(group)
        cut = int(len(group) * (1 - test_size))
        train.extend(group[:cut])
        test.extend(group[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def fit_preprocessor(train_rows):
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod"
    ]

    categories = {}
    for col in categorical_cols:
        categories[col] = sorted({r[col] for r in train_rows})

    means, stds = {}, {}
    for col in numeric_cols:
        vals = [r[col] for r in train_rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 1.0
        means[col], stds[col] = mean, std

    feature_names = []
    for col in numeric_cols:
        feature_names.append(col)
    for col in categorical_cols:
        for cat in categories[col]:
            feature_names.append(f"{col}__{cat}")

    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categories": categories,
        "means": means,
        "stds": stds,
        "feature_names": feature_names,
    }
    return meta


def transform(rows, prep):
    X, y = [], []
    for r in rows:
        feats = []
        for col in prep["numeric_cols"]:
            feats.append((r[col] - prep["means"][col]) / prep["stds"][col])
        for col in prep["categorical_cols"]:
            row_val = r[col]
            for cat in prep["categories"][col]:
                feats.append(1.0 if row_val == cat else 0.0)
        X.append(feats)
        y.append(r["Churn"])
    return X, y


# ---------- Models ----------
class LogisticRegressionGD:
    def __init__(self, lr=0.05, epochs=500, l2=0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = []
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        if z < -35:
            return 0.0
        if z > 35:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n, m = len(X), len(X[0])
        self.w = [0.0] * m
        self.b = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * m
            grad_b = 0.0
            for xi, yi in zip(X, y):
                z = sum(wj * xj for wj, xj in zip(self.w, xi)) + self.b
                p = self._sigmoid(z)
                err = p - yi
                for j in range(m):
                    grad_w[j] += err * xi[j]
                grad_b += err

            for j in range(m):
                grad_w[j] = grad_w[j] / n + self.l2 * self.w[j]
                self.w[j] -= self.lr * grad_w[j]
            self.b -= self.lr * (grad_b / n)

    def predict_proba(self, X):
        out = []
        for xi in X:
            z = sum(wj * xj for wj, xj in zip(self.w, xi)) + self.b
            out.append(self._sigmoid(z))
        return out


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.proba = proba


class DecisionTreeBinary:
    def __init__(self, max_depth=6, min_samples_split=20, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.root = None
        self.feature_importance = Counter()

    @staticmethod
    def gini(y):
        if not y:
            return 0.0
        p = sum(y) / len(y)
        return 1.0 - p * p - (1 - p) * (1 - p)

    def best_split(self, X, y, features):
        parent_gini = self.gini(y)
        best_gain = 0.0
        best = None
        n = len(y)

        for f in features:
            values = [xi[f] for xi in X]
            uniq = sorted(set(values))
            if len(uniq) <= 1:
                continue
            candidates = []
            if len(uniq) > 8:
                step = max(1, len(uniq) // 8)
                sampled = uniq[::step]
                candidates = [(sampled[i] + sampled[i + 1]) / 2 for i in range(len(sampled) - 1)]
            else:
                candidates = [(uniq[i] + uniq[i + 1]) / 2 for i in range(len(uniq) - 1)]

            for t in candidates:
                left_y = [yy for vv, yy in zip(values, y) if vv <= t]
                right_y = [yy for vv, yy in zip(values, y) if vv > t]
                if not left_y or not right_y:
                    continue
                gain = parent_gini - (len(left_y)/n)*self.gini(left_y) - (len(right_y)/n)*self.gini(right_y)
                if gain > best_gain:
                    best_gain = gain
                    best = (f, t)

        return best, best_gain

    def build(self, X, y, depth=0):
        proba = sum(y) / len(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or proba in (0.0, 1.0):
            return TreeNode(proba=proba)

        feature_indices = list(range(len(X[0])))
        if self.max_features is not None and self.max_features < len(feature_indices):
            feature_indices = self.random.sample(feature_indices, self.max_features)

        split, gain = self.best_split(X, y, feature_indices)
        if split is None or gain <= 1e-9:
            return TreeNode(proba=proba)

        f, t = split
        left_X, left_y, right_X, right_y = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[f] <= t:
                left_X.append(xi)
                left_y.append(yi)
            else:
                right_X.append(xi)
                right_y.append(yi)

        self.feature_importance[f] += gain * len(y)
        left_node = self.build(left_X, left_y, depth + 1)
        right_node = self.build(right_X, right_y, depth + 1)
        return TreeNode(feature=f, threshold=t, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self.build(X, y, 0)

    def _predict_one(self, node, x):
        if node.proba is not None:
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict_proba(self, X):
        return [self._predict_one(self.root, x) for x in X]


class RandomForestBinary:
    def __init__(self, n_estimators=35, max_depth=7, min_samples_split=20, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random = random.Random(random_state)
        self.trees = []
        self.feature_importance = Counter()

    def fit(self, X, y):
        n = len(X)
        m = len(X[0])
        max_features = max(1, int(math.sqrt(m)))
        self.trees = []
        for i in range(self.n_estimators):
            idx = [self.random.randrange(n) for _ in range(n)]
            Xb = [X[j] for j in idx]
            yb = [y[j] for j in idx]
            tree = DecisionTreeBinary(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=RANDOM_STATE + i,
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)
            self.feature_importance.update(tree.feature_importance)

    def predict_proba(self, X):
        preds = [0.0] * len(X)
        for tree in self.trees:
            tp = tree.predict_proba(X)
            for i, p in enumerate(tp):
                preds[i] += p
        return [p / len(self.trees) for p in preds]


# ---------- Simple SVG plotting ----------
def save_roc_svg(fpr_lr, tpr_lr, fpr_rf, tpr_rf, out_path):
    w, h, pad = 640, 460, 60
    def tx(x): return pad + x * (w - 2 * pad)
    def ty(y): return h - pad - y * (h - 2 * pad)

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
        '<text x="320" y="445" text-anchor="middle" font-size="14">False Positive Rate</text>',
        '<text x="18" y="230" text-anchor="middle" font-size="14" transform="rotate(-90 18,230)">True Positive Rate</text>',
        poly(list(zip(fpr_lr, tpr_lr)), "#1f77b4"),
        poly(list(zip(fpr_rf, tpr_rf)), "#d62728"),
        '<text x="470" y="75" font-size="13" fill="#1f77b4">Logistic Regression</text>',
        '<text x="470" y="95" font-size="13" fill="#d62728">Random Forest</text>',
        '</svg>'
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix_svg(cm, out_path):
    w, h = 420, 360
    x0, y0, s = 90, 70, 100
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="210" y="30" text-anchor="middle" font-size="20">Confusion Matrix</text>',
        '<text x="210" y="345" text-anchor="middle" font-size="14">Predicted Label</text>',
        '<text x="20" y="180" text-anchor="middle" font-size="14" transform="rotate(-90 20,180)">Actual Label</text>',
    ]
    vals = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
    maxv = max(vals) if max(vals) else 1
    idx = 0
    for r in range(2):
        for c in range(2):
            v = vals[idx]
            shade = int(255 - 170 * (v / maxv))
            color = f"rgb({shade},{shade},255)"
            x, y = x0 + c * s, y0 + r * s
            lines.append(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="{color}" stroke="black"/>')
            lines.append(f'<text x="{x+s/2}" y="{y+s/2+5}" text-anchor="middle" font-size="20">{v}</text>')
            idx += 1
    lines += [
        f'<text x="{x0+s/2}" y="{y0-15}" text-anchor="middle" font-size="12">No Churn</text>',
        f'<text x="{x0+1.5*s}" y="{y0-15}" text-anchor="middle" font-size="12">Churn</text>',
        f'<text x="{x0-18}" y="{y0+s/2+4}" text-anchor="middle" font-size="12">No Churn</text>',
        f'<text x="{x0-18}" y="{y0+1.5*s+4}" text-anchor="middle" font-size="12">Churn</text>',
        '</svg>'
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def high_value_customer_rule(customer):
    rule_a = (
        customer["TotalCharges"] >= 3000
        and customer["tenure"] >= 24
        and customer["Contract"] in {"One year", "Two year"}
    )
    return rule_a


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    rows = load_and_clean(DATA_PATH)
    train_rows, test_rows = stratified_split(rows, test_size=0.2, random_state=RANDOM_STATE)

    prep = fit_preprocessor(train_rows)
    X_train, y_train = transform(train_rows, prep)
    X_test, y_test = transform(test_rows, prep)

    lr = LogisticRegressionGD(lr=0.04, epochs=700, l2=0.002)
    lr.fit(X_train, y_train)
    lr_train_prob = lr.predict_proba(X_train)
    lr_test_prob = lr.predict_proba(X_test)

    rf = RandomForestBinary(n_estimators=45, max_depth=8, min_samples_split=18, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_train_prob = rf.predict_proba(X_train)
    rf_test_prob = rf.predict_proba(X_test)

    lr_train_auc = auc_score(y_train, lr_train_prob)
    lr_test_auc = auc_score(y_test, lr_test_prob)
    rf_train_auc = auc_score(y_train, rf_train_prob)
    rf_test_auc = auc_score(y_test, rf_test_prob)

    best_model_name = "random_forest" if rf_test_auc >= lr_test_auc else "logistic_regression"
    best_test_prob = rf_test_prob if best_model_name == "random_forest" else lr_test_prob

    best_thr, best_f1 = best_f1_threshold(y_test, best_test_prob)
    y_pred = [1 if p >= best_thr else 0 for p in best_test_prob]
    cm = confusion_matrix(y_test, y_pred)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_prob)
    save_roc_svg(fpr_lr, tpr_lr, fpr_rf, tpr_rf, REPORTS_DIR / "roc_curve.svg")
    save_confusion_matrix_svg(cm, REPORTS_DIR / "confusion_matrix.svg")

    logistic_coef_ranking = sorted(
        zip(prep["feature_names"], lr.w), key=lambda x: abs(x[1]), reverse=True
    )[:15]

    rf_importance_total = sum(rf.feature_importance.values()) or 1.0
    rf_ranking = sorted(
        [(prep["feature_names"][k], v / rf_importance_total) for k, v in rf.feature_importance.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    all_X, _ = transform(rows, prep)
    all_best_prob = rf.predict_proba(all_X) if best_model_name == "random_forest" else lr.predict_proba(all_X)

    hv_flags = [high_value_customer_rule(r) for r in rows]
    hv_count = sum(hv_flags)
    high_risk_hv = sum(1 for f, p in zip(hv_flags, all_best_prob) if f and p >= best_thr)

    metrics = {
        "data": {
            "n_rows": len(rows),
            "n_train": len(train_rows),
            "n_test": len(test_rows)
        },
        "models": {
            "logistic_regression": {
                "train_auc": lr_train_auc,
                "test_auc": lr_test_auc,
            },
            "random_forest": {
                "train_auc": rf_train_auc,
                "test_auc": rf_test_auc,
            },
        },
        "selected_model": best_model_name,
        "decision_threshold": best_thr,
        "best_f1": best_f1,
        "test_metrics_selected_model": {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": auc_score(y_test, best_test_prob),
            "confusion_matrix": cm,
        },
        "top_logistic_coefficients": [
            {"feature": f, "coefficient": c} for f, c in logistic_coef_ranking
        ],
        "top_tree_importances": [
            {"feature": f, "importance": imp} for f, imp in rf_ranking
        ],
        "high_value_summary": {
            "definition_selected": "Rule A: TotalCharges >= 3000, tenure >= 24 months, contract in {One year, Two year}",
            "high_value_pct": hv_count / len(rows),
            "high_value_count": hv_count,
            "high_value_high_risk_pct": (high_risk_hv / hv_count) if hv_count else 0.0,
            "high_value_high_risk_count": high_risk_hv,
        },
    }

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved reports/metrics.json, reports/roc_curve.svg, reports/confusion_matrix.svg")
    print(f"Selected model: {best_model_name}, threshold={best_thr:.2f}, test AUC={metrics['test_metrics_selected_model']['auc']:.3f}")


if __name__ == "__main__":
    main()
