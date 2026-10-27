"""Label quality auditing and suspicious label detection module for Extremis DataSentry.

Implements Out-Of-Fold (OOF) cross-validation confident learning to identify
mislabeled records and candidate annotation errors.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def detect_label_issues(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    cv_folds: int = 5,
    random_state: int = 42,
    confidence_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Identify potential label noise and incorrect class annotations using OOF predictions.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column to audit.
        feature_cols: List of features to use as predictors (if None, all usable features used).
        cv_folds: Number of cross-validation splits.
        random_state: Random seed for reproducibility.
        confidence_threshold: Minimum predicted probability for alternative class to flag.

    Returns:
        Dictionary with audit summary, suspicious samples DataFrame, and label quality scores.
    """
    if target_col not in df.columns:
        return {
            "status": "error",
            "message": f"Target column '{target_col}' not found in dataset.",
            "suspicious_count": 0,
            "suspicious_table": pd.DataFrame(),
        }

    # Clean target: drop nulls in target
    valid_mask = df[target_col].notna()
    sub_df = df[valid_mask].copy().reset_index(drop=True)
    y_raw = sub_df[target_col]

    unique_classes, counts = np.unique(y_raw, return_counts=True)
    if len(unique_classes) < 2:
        return {
            "status": "error",
            "message": f"Target '{target_col}' must have at least 2 distinct classes. Found: {len(unique_classes)}.",
            "suspicious_count": 0,
            "suspicious_table": pd.DataFrame(),
        }

    # Require minimum samples per class for CV
    if min(counts) < cv_folds:
        cv_folds = max(2, min(counts))
        if cv_folds < 2:
            return {
                "status": "error",
                "message": f"Class '{unique_classes[np.argmin(counts)]}' has only {min(counts)} instances; insufficient for cross-validation.",
                "suspicious_count": 0,
                "suspicious_table": pd.DataFrame(),
            }

    # Select predictor features
    if feature_cols is None:
        numeric_features = sub_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in numeric_features if c != target_col]

    if not feature_cols:
        return {
            "status": "error",
            "message": "No numeric predictor features available to audit label consistency.",
            "suspicious_count": 0,
            "suspicious_table": pd.DataFrame(),
        }

    # Prepare feature matrix
    X_mat = sub_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = X_mat.median()
    X_mat = X_mat.fillna(medians).fillna(0)

    # Encode target to numeric labels 0..K-1
    class_map = {val: idx for idx, val in enumerate(unique_classes)}
    inv_class_map = {idx: val for val, idx in class_map.items()}
    y_encoded = np.array([class_map[val] for val in y_raw])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_probs = np.zeros((len(sub_df), len(unique_classes)), dtype=float)

    for train_idx, val_idx in skf.split(X_mat, y_encoded):
        X_tr, y_tr = X_mat.iloc[train_idx], y_encoded[train_idx]
        X_val = X_mat.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        oof_probs[val_idx] = clf.predict_proba(X_val_s)

    # Calculate label quality metrics per sample
    # Label quality score = OOF probability assigned to the sample's given label
    given_label_prob = oof_probs[np.arange(len(sub_df)), y_encoded]
    predicted_classes = np.argmax(oof_probs, axis=1)
    max_prob = np.max(oof_probs, axis=1)

    # Candidate label errors: model is confident in a different class
    is_suspicious = (predicted_classes != y_encoded) & (max_prob >= confidence_threshold)
    suspicious_indices = np.where(is_suspicious)[0]

    records = []
    for idx in suspicious_indices:
        records.append({
            "source_row_index": idx,
            "given_label": inv_class_map[y_encoded[idx]],
            "suggested_label": inv_class_map[predicted_classes[idx]],
            "model_confidence": round(float(max_prob[idx]), 3),
            "given_label_confidence": round(float(given_label_prob[idx]), 3),
            "label_discrepancy": round(float(max_prob[idx] - given_label_prob[idx]), 3),
        })

    suspicious_df = pd.DataFrame(records)
    if not suspicious_df.empty:
        suspicious_df = suspicious_df.sort_values("label_discrepancy", ascending=False)

    avg_label_quality = float(np.mean(given_label_prob))

    return {
        "status": "success",
        "message": f"Audited {len(sub_df)} labeled samples across {cv_folds}-fold cross validation.",
        "target_column": target_col,
        "classes": [str(c) for c in unique_classes],
        "total_audited": len(sub_df),
        "suspicious_count": len(suspicious_indices),
        "suspicious_fraction": round(len(suspicious_indices) / max(len(sub_df), 1), 4),
        "avg_label_quality_score": round(avg_label_quality, 3),
        "suspicious_table": suspicious_df,
        "oof_probabilities": oof_probs,
        "given_probabilities": given_label_prob,
    }
