"""Numeric anomaly candidate detection module for Extremis DataSentry.

Implements Isolation Forest multi-feature anomaly detection, feature contribution
explainability (z-score deviations), PCA 2D projection for visual clustering,
and candidate ranking.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def run_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 150,
) -> Dict[str, Any]:
    """Execute Isolation Forest anomaly detection on numeric columns.

    Returns:
        Dictionary containing:
        - status: "success" or "skipped" with reason
        - anomaly_count: number of flagged rows
        - anomaly_indices: list of row indices flagged
        - anomaly_table: DataFrame of flagged records sorted by anomaly score
        - pca_df: DataFrame with PC1, PC2, is_anomaly, anomaly_score for 2D plotting
        - feature_contributions: explanations of top deviating features per flagged record
    """
    rows, _ = df.shape
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        return {
            "status": "skipped",
            "message": "No numeric columns found in dataset. Isolation Forest requires at least one numeric feature.",
            "anomaly_count": 0,
            "anomaly_indices": [],
            "anomaly_table": pd.DataFrame(),
            "pca_df": pd.DataFrame(),
            "feature_contributions": {},
        }

    if rows < 10:
        return {
            "status": "skipped",
            "message": f"Dataset has only {rows} rows. Minimum 10 rows required for reliable anomaly ranking.",
            "anomaly_count": 0,
            "anomaly_indices": [],
            "anomaly_table": pd.DataFrame(),
            "pca_df": pd.DataFrame(),
            "feature_contributions": {},
        }

    # Prepare feature matrix: replace infs, impute NaNs with median, filter zero-variance features
    mat = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    mat = mat.dropna(axis=1, how="all")
    medians = mat.median()
    mat = mat.fillna(medians).fillna(0)

    # Filter features that have strictly zero variance
    variable_cols = [c for c in mat.columns if mat[c].nunique() > 1]
    if not variable_cols:
        return {
            "status": "skipped",
            "message": "All numeric columns are constant or zero variance. Anomaly detection cannot operate on zero-variance features.",
            "anomaly_count": 0,
            "anomaly_indices": [],
            "anomaly_table": pd.DataFrame(),
            "pca_df": pd.DataFrame(),
            "feature_contributions": {},
        }

    mat_eval = mat[variable_cols]

    # Fit Isolation Forest
    clf = IsolationForest(
        contamination=float(np.clip(contamination, 0.005, 0.30)),
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1,
    )
    predictions = clf.fit_predict(mat_eval)
    raw_scores = clf.score_samples(mat_eval)
    # Higher anomaly score = more anomalous
    anomaly_scores = -raw_scores

    anomaly_indices = np.where(predictions == -1)[0].tolist()
    anomaly_count = len(anomaly_indices)

    # Calculate Z-Scores to explain top contributing feature deviations for each anomaly
    scaler = StandardScaler()
    scaled_mat = scaler.fit_transform(mat_eval)
    scaled_df = pd.DataFrame(scaled_mat, columns=variable_cols, index=mat_eval.index)

    feature_explanations: Dict[int, str] = {}
    for idx in anomaly_indices:
        row_z = scaled_df.iloc[idx].abs()
        top_feature = row_z.idxmax()
        top_z = row_z.max()
        top_val = df.iloc[idx][top_feature]
        feature_explanations[idx] = f"{top_feature} = {top_val} (|z| = {top_z:.1f})"

    # Build anomaly candidates table
    if anomaly_count > 0:
        anom_rows = df.iloc[anomaly_indices].copy()
        anom_rows.insert(0, "source_row_index", anomaly_indices)
        anom_rows.insert(1, "anomaly_score", [round(float(anomaly_scores[i]), 4) for i in anomaly_indices])
        anom_rows.insert(2, "key_driver", [feature_explanations.get(i, "Multi-feature combination") for i in anomaly_indices])
        anom_table = anom_rows.sort_values("anomaly_score", ascending=False)
    else:
        anom_table = pd.DataFrame()

    # Compute PCA 2D embedding for scatter visualization
    pca_df = pd.DataFrame()
    if len(variable_cols) >= 2:
        try:
            pca = PCA(n_components=2, random_state=random_state)
            coords = pca.fit_transform(scaled_mat)
            pca_df = pd.DataFrame({
                "PC1": coords[:, 0].round(3),
                "PC2": coords[:, 1].round(3),
                "source_index": list(range(rows)),
                "is_anomaly": ["Flagged Anomaly" if i in anomaly_indices else "Normal Pattern" for i in range(rows)],
                "anomaly_score": anomaly_scores.round(3),
            })
        except Exception:
            pca_df = pd.DataFrame()

    return {
        "status": "success",
        "message": f"Successfully evaluated {len(variable_cols)} numeric features across {rows} rows.",
        "evaluated_features": variable_cols,
        "anomaly_count": anomaly_count,
        "anomaly_fraction": round(anomaly_count / max(rows, 1), 4),
        "anomaly_indices": anomaly_indices,
        "anomaly_scores": anomaly_scores,
        "anomaly_table": anom_table,
        "pca_df": pca_df,
        "feature_contributions": feature_explanations,
    }
