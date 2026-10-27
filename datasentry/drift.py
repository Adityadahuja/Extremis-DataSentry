"""Dataset distribution drift monitoring module for Extremis DataSentry.

Compares current dataset distributions with a reference baseline using:
- Standardized Mean Difference (SMD / Cohen's d)
- Two-sample Kolmogorov-Smirnov (KS) test statistic and p-value
- Population Stability Index (PSI)
- Variance dispersion ratio
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples."""
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Calculate quantile bin edges from reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    try:
        bin_edges = np.percentile(reference, quantiles)
    except Exception:
        return 0.0

    # Ensure unique and strictly increasing edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Extend endpoints to catch all data
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (np.sum(cur_counts) + epsilon * len(cur_counts))

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_val))


def compute_drift_metrics(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare shared numeric features between current and reference datasets.

    Returns:
        Dictionary containing:
        - status: 'success' or 'warning'
        - shared_features: list of shared numeric features evaluated
        - drift_table: DataFrame with SMD, KS statistic, KS p-value, PSI, and drift status
        - summary: counts of high, moderate, and stable features
    """
    if current_df is None or reference_df is None:
        return {
            "status": "warning",
            "message": "Both current and reference datasets must be provided.",
            "shared_features": [],
            "drift_table": pd.DataFrame(),
            "summary": {"high": 0, "moderate": 0, "stable": 0},
        }

    cur_num = current_df.select_dtypes(include=np.number).columns.tolist()
    ref_num = reference_df.select_dtypes(include=np.number).columns.tolist()
    shared = [c for c in cur_num if c in ref_num]

    if not shared:
        return {
            "status": "warning",
            "message": "No shared numeric features found between current and reference datasets.",
            "shared_features": [],
            "drift_table": pd.DataFrame(),
            "summary": {"high": 0, "moderate": 0, "stable": 0},
        }

    results: List[Dict[str, Any]] = []

    for col in shared:
        ref_s = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        cur_s = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        if len(ref_s) < 3 or len(cur_s) < 3:
            continue

        ref_mean = float(np.mean(ref_s))
        cur_mean = float(np.mean(cur_s))
        ref_std = float(np.std(ref_s, ddof=1)) if len(ref_s) > 1 else 0.0
        cur_std = float(np.std(cur_s, ddof=1)) if len(cur_s) > 1 else 0.0

        # Standardized Mean Difference (Cohen's d with pooled std)
        pooled_std = float(np.sqrt((ref_std**2 + cur_std**2) / 2.0))
        if pooled_std > 1e-9:
            smd = float(abs(cur_mean - ref_mean) / pooled_std)
        else:
            smd = 0.0

        # Kolmogorov-Smirnov 2-sample test
        try:
            ks_res = stats.ks_2samp(ref_s, cur_s)
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)
        except Exception:
            ks_stat, ks_pvalue = 0.0, 1.0

        # Population Stability Index (PSI)
        psi_score = calculate_psi(ref_s, cur_s)

        # Variance ratio
        var_ratio = float((cur_std**2) / (ref_std**2)) if ref_std > 1e-9 else 1.0

        # Drift severity evaluation
        if smd >= 0.5 or psi_score >= 0.25 or (ks_stat >= 0.25 and ks_pvalue < 0.01):
            severity = "High Shift"
            action = "Substantial distribution divergence. Review feature relevance and re-tune baseline model."
        elif smd >= 0.2 or psi_score >= 0.10 or (ks_stat >= 0.15 and ks_pvalue < 0.05):
            severity = "Moderate Shift"
            action = "Noticeable distribution shift. Monitor incoming batches for trend progression."
        else:
            severity = "Stable"
            action = "Distribution is consistent with reference baseline."

        results.append({
            "feature": col,
            "drift_severity": severity,
            "smd": round(smd, 3),
            "psi": round(psi_score, 3),
            "ks_statistic": round(ks_stat, 3),
            "ks_pvalue": round(ks_pvalue, 4),
            "ref_mean": round(ref_mean, 2),
            "current_mean": round(cur_mean, 2),
            "ref_std": round(ref_std, 2),
            "current_std": round(cur_std, 2),
            "variance_ratio": round(var_ratio, 2),
            "action": action,
        })

    if not results:
        return {
            "status": "warning",
            "message": "Insufficient valid observations in shared numeric features.",
            "shared_features": shared,
            "drift_table": pd.DataFrame(),
            "summary": {"high": 0, "moderate": 0, "stable": 0},
        }

    drift_df = pd.DataFrame(results)
    # Sort by SMD descending
    drift_df = drift_df.sort_values("smd", ascending=False)

    high_count = int((drift_df["drift_severity"] == "High Shift").sum())
    mod_count = int((drift_df["drift_severity"] == "Moderate Shift").sum())
    stable_count = int((drift_df["drift_severity"] == "Stable").sum())

    return {
        "status": "success",
        "message": f"Successfully evaluated distribution shift across {len(drift_df)} shared features.",
        "shared_features": shared,
        "drift_table": drift_df,
        "summary": {
            "high": high_count,
            "moderate": mod_count,
            "stable": stable_count,
            "total": len(drift_df),
        },
    }
