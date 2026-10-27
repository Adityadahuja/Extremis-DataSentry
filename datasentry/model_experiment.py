"""ML Reliability Experiment: Baseline vs Cleaned model benchmark.

Evaluates the tangible impact of data quality issues by comparing model performance
trained on raw dirty data versus remediated data on an untouched test split.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_ml_reliability_experiment(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    model_type: str = "Random Forest",
    test_size: float = 0.25,
    random_state: int = 42,
    filter_anomalies: bool = True,
    filter_label_errors: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end Before vs After ML training experiment on an untouched test set."""
    if target_col not in df.columns:
        return {"status": "error", "message": f"Target column '{target_col}' not found."}

    valid_mask = df[target_col].notna()
    data = df[valid_mask].copy().reset_index(drop=True)

    if feature_cols is None:
        numeric = data.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in numeric if c != target_col]

    if not feature_cols:
        return {"status": "error", "message": "No numeric feature columns available for modeling."}

    unique_targets = np.unique(data[target_col])
    if len(unique_targets) < 2:
        return {"status": "error", "message": "Target must contain at least 2 distinct classes."}

    # Map classes to 0..K-1
    target_map = {val: idx for idx, val in enumerate(unique_targets)}
    y_all = np.array([target_map[v] for v in data[target_col]])
    X_all = data[feature_cols].copy()

    # Create train and untouched test split
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )

    # 1. Baseline Model Pipeline (Dirty Training Data)
    # Naive median imputation for NaNs, keep duplicates and outliers
    train_medians = X_train_raw.median().fillna(0)
    X_tr_dirty = X_train_raw.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    X_te_clean = X_test_raw.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    scaler_baseline = StandardScaler()
    X_tr_dirty_s = scaler_baseline.fit_transform(X_tr_dirty)
    X_te_s = scaler_baseline.transform(X_te_clean)

    def get_model():
        if model_type == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)

    clf_baseline = get_model()
    clf_baseline.fit(X_tr_dirty_s, y_train_raw)

    y_pred_baseline = clf_baseline.predict(X_te_s)
    y_prob_baseline = clf_baseline.predict_proba(X_te_s)

    # 2. Cleaned Model Pipeline (Remediated Training Data)
    # Reconstruct training dataframe to clean
    train_df = X_train_raw.copy()
    train_df["_target_"] = y_train_raw

    # Step A: Drop duplicate rows in training set
    initial_tr_len = len(train_df)
    train_df = train_df.drop_duplicates()
    dups_dropped = initial_tr_len - len(train_df)

    # Step B: Median imputation for training features
    X_tr_clean = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    clean_medians = X_tr_clean.median().fillna(0)
    X_tr_clean = X_tr_clean.fillna(clean_medians)
    y_tr_clean = train_df["_target_"].values

    # Step C: Filter top outlier anomalies from training data
    outliers_filtered = 0
    if filter_anomalies and len(X_tr_clean) >= 20:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.04, random_state=random_state)
        anom_preds = iso.fit_predict(X_tr_clean)
        normal_mask = anom_preds == 1
        outliers_filtered = int((anom_preds == -1).sum())
        X_tr_clean = X_tr_clean[normal_mask]
        y_tr_clean = y_tr_clean[normal_mask]

    # Step D: Filter confident label errors from training data
    label_errors_filtered = 0
    if filter_label_errors and len(X_tr_clean) >= 30 and len(np.unique(y_tr_clean)) >= 2:
        from datasentry.label_quality import detect_label_issues
        temp_tr_df = X_tr_clean.copy()
        temp_tr_df["_target_"] = y_tr_clean
        label_res = detect_label_issues(
            temp_tr_df,
            target_col="_target_",
            feature_cols=feature_cols,
            cv_folds=3,
            confidence_threshold=0.75,
            random_state=random_state,
        )
        if label_res["status"] == "success" and label_res["suspicious_count"] > 0:
            susp_idx = set(label_res["suspicious_table"]["source_row_index"].tolist())
            keep_mask = [i not in susp_idx for i in range(len(temp_tr_df))]
            label_errors_filtered = len(susp_idx)
            X_tr_clean = X_tr_clean.iloc[keep_mask]
            y_tr_clean = y_tr_clean[keep_mask]

    # Train Cleaned Model
    scaler_cleaned = StandardScaler()
    X_tr_clean_s = scaler_cleaned.fit_transform(X_tr_clean)
    X_te_clean_s = scaler_cleaned.transform(X_te_clean)

    clf_cleaned = get_model()
    clf_cleaned.fit(X_tr_clean_s, y_tr_clean)

    y_pred_cleaned = clf_cleaned.predict(X_te_clean_s)
    y_prob_cleaned = clf_cleaned.predict_proba(X_te_clean_s)

    # Calculate metrics on untouched test set
    def compute_metrics(y_true, y_pred, y_prob):
        acc = float(accuracy_score(y_true, y_pred))
        b_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        roc = None
        if len(np.unique(y_true)) == 2:
            try:
                roc = float(roc_auc_score(y_true, y_prob[:, 1]))
            except Exception:
                roc = None
        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc, 4) if roc is not None else None,
        }

    metrics_baseline = compute_metrics(y_test, y_pred_baseline, y_prob_baseline)
    metrics_cleaned = compute_metrics(y_test, y_pred_cleaned, y_prob_cleaned)

    # Comparison DataFrame
    comp_rows = []
    for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        b_val = metrics_baseline.get(k)
        c_val = metrics_cleaned.get(k)
        if b_val is not None and c_val is not None:
            delta = round(c_val - b_val, 4)
            pct_change = round((delta / max(abs(b_val), 1e-4)) * 100, 2)
            comp_rows.append({
                "metric": k.replace("_", " ").title(),
                "baseline_model": b_val,
                "cleaned_data_model": c_val,
                "absolute_gain": f"+{delta}" if delta >= 0 else str(delta),
                "relative_improvement": f"{pct_change:+.1f}%",
            })

    comparison_df = pd.DataFrame(comp_rows)

    return {
        "status": "success",
        "model_type": model_type,
        "test_samples": len(y_test),
        "raw_train_samples": len(X_train_raw),
        "cleaned_train_samples": len(X_tr_clean),
        "remediation_summary": {
            "duplicates_removed": dups_dropped,
            "outliers_filtered": outliers_filtered,
            "label_errors_filtered": label_errors_filtered,
        },
        "metrics_baseline": metrics_baseline,
        "metrics_cleaned": metrics_cleaned,
        "comparison_table": comparison_df,
        "confusion_matrix_baseline": confusion_matrix(y_test, y_pred_baseline).tolist(),
        "confusion_matrix_cleaned": confusion_matrix(y_test, y_pred_cleaned).tolist(),
        "classes": [str(c) for c in unique_targets],
    }
