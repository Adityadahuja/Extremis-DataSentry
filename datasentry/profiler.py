"""Dataset profiling and quality auditing engine for Extremis DataSentry.

Computes comprehensive schema metadata, missingness metrics, duplicate row counts,
constant/quasi-constant feature flags, cardinality issues, distribution statistics,
and an overall dataset health score.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import io
import numpy as np
import pandas as pd


def load_dataset(file_or_path: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Safely load a CSV dataset from a file object, buffer, or file path.

    Returns:
        (DataFrame, None) on success, or (None, error_message) on failure.
    """
    if file_or_path is None:
        return None, "No file provided."

    try:
        if isinstance(file_or_path, pd.DataFrame):
            df = file_or_path.copy()
        elif isinstance(file_or_path, str):
            df = pd.read_csv(file_or_path)
        elif hasattr(file_or_path, "read"):
            # Streamlit UploadedFile or BytesIO / StringIO
            try:
                df = pd.read_csv(file_or_path)
            except UnicodeDecodeError:
                if hasattr(file_or_path, "seek"):
                    file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding="latin-1")
        else:
            return None, "Unsupported file format."

        if df is None or df.empty:
            return None, "Dataset is empty (0 rows or 0 columns)."

        if df.shape[1] == 0:
            return None, "Dataset has no columns."

        return df, None
    except Exception as exc:
        return None, f"Failed to parse CSV: {str(exc)}"


def compute_column_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed column-level metadata and quality statistics."""
    rows, _ = df.shape
    records = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = round(float((missing_count / max(rows, 1)) * 100), 2)
        unique_vals = int(s.nunique(dropna=True))
        is_num = bool(pd.api.types.is_numeric_dtype(s))

        # Constant check (<= 1 distinct value excluding nulls)
        is_constant = bool(unique_vals <= 1)

        # Quasi-constant check (> 98% of valid values are the single mode, but nunique > 1)
        is_quasi = False
        if unique_vals > 1 and rows >= 20:
            val_counts = s.value_counts(dropna=True, normalize=True)
            if not val_counts.empty and val_counts.iloc[0] >= 0.98:
                is_quasi = True

        # High cardinality for non-numeric features (identifiers, hashes, timestamps)
        is_high_card = False
        if not is_num:
            if unique_vals > max(50, int(0.5 * rows)):
                is_high_card = True

        # Numeric statistics
        col_min, col_max, col_mean, col_std, col_median = None, None, None, None, None
        col_skew, zero_count, inf_count = None, 0, 0

        if is_num:
            clean_num = pd.to_numeric(s, errors="coerce")
            inf_count = int(np.isinf(clean_num).sum())
            clean_num = clean_num.replace([np.inf, -np.inf], np.nan).dropna()
            zero_count = int((clean_num == 0).sum())

            if not clean_num.empty:
                col_min = round(float(clean_num.min()), 2)
                col_max = round(float(clean_num.max()), 2)
                col_mean = round(float(clean_num.mean()), 2)
                col_std = round(float(clean_num.std()), 2) if len(clean_num) > 1 else 0.0
                col_median = round(float(clean_num.median()), 2)
                if len(clean_num) > 2:
                    try:
                        col_skew = round(float(clean_num.skew()), 2)
                    except Exception:
                        col_skew = None

        records.append({
            "column": str(col),
            "dtype": str(s.dtype),
            "is_numeric": is_num,
            "missing_count": missing_count,
            "missing_percent": missing_pct,
            "unique_values": unique_vals,
            "is_constant": is_constant,
            "is_quasi_constant": is_quasi,
            "is_high_cardinality": is_high_card,
            "zero_count": zero_count,
            "infinite_count": inf_count,
            "mean": col_mean,
            "std": col_std,
            "median": col_median,
            "min": col_min,
            "max": col_max,
            "skewness": col_skew,
        })

    return pd.DataFrame(records)


def extract_data_findings(
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
    duplicate_rows: int,
) -> pd.DataFrame:
    """Extract structured data findings with severity levels and actionable recommendations."""
    rows, _ = df.shape
    findings: List[Dict[str, Any]] = []

    # 1. Duplicate rows
    if duplicate_rows > 0:
        dup_pct = round(duplicate_rows / max(rows, 1) * 100, 2)
        severity = "High" if dup_pct >= 10 else "Medium"
        findings.append({
            "severity": severity,
            "issue": "Duplicate rows",
            "column": "(dataset-wide)",
            "count": duplicate_rows,
            "percentage": dup_pct,
            "detail": f"{duplicate_rows:,} exact identical rows ({dup_pct}%) present in dataset.",
            "recommended_action": "Investigate source pipeline for ingestion retries or key duplication; deduplicate if unintentional.",
        })

    # 2. Column-level issues
    for _, r in profile_df.iterrows():
        col = r["column"]

        # Missingness
        if r["missing_count"] > 0:
            pct = r["missing_percent"]
            if pct >= 50.0:
                sev = "Critical"
                act = "High missingness (>50%). Consider dropping column or validating collection mechanism."
            elif pct >= 20.0:
                sev = "High"
                act = "Significant missingness (20-50%). Use domain-specific imputation (median/KNN) or missing indicator."
            elif pct >= 5.0:
                sev = "Medium"
                act = "Moderate missingness (5-20%). Check missingness mechanism (MCAR vs MAR) before imputing."
            else:
                sev = "Low"
                act = "Low missingness (<5%). Safe for simple median/mode imputation or row omission if rows are abundant."

            findings.append({
                "severity": sev,
                "issue": "Missing values",
                "column": col,
                "count": int(r["missing_count"]),
                "percentage": float(pct),
                "detail": f"{int(r['missing_count']):,} missing cells ({pct}% of rows).",
                "recommended_action": act,
            })

        # Infinite values
        if r["infinite_count"] > 0:
            findings.append({
                "severity": "Critical",
                "issue": "Infinite numeric values",
                "column": col,
                "count": int(r["infinite_count"]),
                "percentage": round(r["infinite_count"] / max(rows, 1) * 100, 2),
                "detail": f"{int(r['infinite_count'])} infinite (+/- inf) values caused by division by zero or log(0).",
                "recommended_action": "Clip or replace with NaN prior to statistical/ML modeling to prevent estimator crash.",
            })

        # Constant feature
        if r["is_constant"]:
            findings.append({
                "severity": "Medium",
                "issue": "Constant column",
                "column": col,
                "count": int(r["unique_values"]),
                "percentage": 0.0,
                "detail": f"Feature carries only {int(r['unique_values'])} distinct non-null value (zero variance).",
                "recommended_action": "Zero predictive information for ML models; drop to reduce dimension and avoid collinearity.",
            })
        elif r["is_quasi_constant"]:
            findings.append({
                "severity": "Low",
                "issue": "Quasi-constant column",
                "column": col,
                "count": int(r["unique_values"]),
                "percentage": 98.0,
                "detail": "Over 98% of valid records share a single identical value.",
                "recommended_action": "Examine feature relevance; may cause near-zero variance penalty in tree and linear models.",
            })

        # High cardinality categorical
        if r["is_high_cardinality"]:
            findings.append({
                "severity": "Low",
                "issue": "High cardinality categorical",
                "column": col,
                "count": int(r["unique_values"]),
                "percentage": round(r["unique_values"] / max(rows, 1) * 100, 2),
                "detail": f"{int(r['unique_values']):,} distinct categories relative to {rows:,} rows.",
                "recommended_action": "Likely a primary key, tracking ID, or free-text column. Drop before one-hot encoding or hash encode.",
            })

        # Extreme skewness
        if r["is_numeric"] and r["skewness"] is not None and abs(r["skewness"]) > 3.0:
            findings.append({
                "severity": "Low",
                "issue": "Extreme numeric skewness",
                "column": col,
                "count": 1,
                "percentage": 0.0,
                "detail": f"Sample skewness is {r['skewness']:.2f} (heavy right/left tail).",
                "recommended_action": "Consider logarithmic, Box-Cox, or quantile power transformation if fitting linear/distance models.",
            })

    if not findings:
        return pd.DataFrame(columns=["severity", "issue", "column", "count", "percentage", "detail", "recommended_action"])

    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Review": 4}
    findings_df = pd.DataFrame(findings)
    findings_df["order"] = findings_df["severity"].map(order).fillna(5)
    findings_df = findings_df.sort_values(["order", "percentage"], ascending=[True, False]).drop(columns=["order"])
    return findings_df


def compute_health_score(
    rows: int,
    cols: int,
    missing_cells: int,
    duplicate_rows: int,
    constant_cols_count: int,
    critical_issues_count: int,
    high_issues_count: int,
) -> Tuple[int, str, str]:
    """Calculate an objective 0-100 dataset health score with letter grade and rating description."""
    total_cells = max(rows * cols, 1)
    missing_ratio = missing_cells / total_cells
    dup_ratio = duplicate_rows / max(rows, 1)

    score = 100.0

    # Penalties
    score -= min(35.0, missing_ratio * 120.0)
    score -= min(25.0, dup_ratio * 100.0)
    score -= min(15.0, constant_cols_count * 5.0)
    score -= min(20.0, critical_issues_count * 10.0)
    score -= min(15.0, high_issues_count * 4.0)

    final_score = int(np.clip(round(score), 0, 100))

    if final_score >= 90:
        grade = "A"
        desc = "Excellent quality. Ready for ML modeling with standard pipeline preprocessing."
    elif final_score >= 80:
        grade = "B"
        desc = "Good quality with minor issues. Safe to proceed after standard imputation/deduplication."
    elif final_score >= 65:
        grade = "C"
        desc = "Fair quality with noticeable defects. Targeted cleaning required before training."
    elif final_score >= 50:
        grade = "D"
        desc = "Poor quality. High missingness or duplicates detected; data sanitization is essential."
    else:
        grade = "F"
        desc = "Critical data quality risk. Substantial corruption or missingness that will severely bias models."

    return final_score, grade, desc


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Run complete profile audit on a dataframe.

    Returns structured dictionary containing dimensions, column profile DataFrame,
    findings DataFrame, and overall dataset health metrics.
    """
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    missing_cells = int(df.isna().sum().sum())
    total_cells = max(rows * cols, 1)
    missing_pct = round((missing_cells / total_cells) * 100, 2)
    duplicate_rows = int(df.duplicated().sum())
    dup_pct = round((duplicate_rows / max(rows, 1)) * 100, 2)
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024.0, 1)

    col_profiles = compute_column_profiles(df)
    constant_cols_count = int(col_profiles["is_constant"].sum())

    findings_df = extract_data_findings(df, col_profiles, duplicate_rows)

    critical_count = int((findings_df["severity"] == "Critical").sum()) if not findings_df.empty else 0
    high_count = int((findings_df["severity"] == "High").sum()) if not findings_df.empty else 0

    health_score, grade, rating_desc = compute_health_score(
        rows=rows,
        cols=cols,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        constant_cols_count=constant_cols_count,
        critical_issues_count=critical_count,
        high_issues_count=high_count,
    )

    # Compute correlation matrix if there are at least 2 numeric features
    corr_df = None
    if len(numeric_cols) >= 2:
        clean_num = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        corr_df = clean_num.corr().round(3)

    return {
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "memory_kb": memory_kb,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": dup_pct,
        "column_profiles": col_profiles,
        "findings": findings_df,
        "health_score": health_score,
        "health_grade": grade,
        "health_description": rating_desc,
        "correlation_matrix": corr_df,
    }
