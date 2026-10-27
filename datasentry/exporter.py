"""Audit report serialization and export utilities for Extremis DataSentry.

Generates machine-readable JSON audit summaries and CSV findings compliant
with FR-07, FR-08, and FR-10.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd


def generate_audit_json(
    profile_results: Dict[str, Any],
    anomaly_results: Optional[Dict[str, Any]] = None,
    drift_results: Optional[Dict[str, Any]] = None,
    ml_experiment_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile an audit report in JSON format."""
    now_utc = datetime.now(timezone.utc).isoformat()
    findings_df = profile_results.get("findings", pd.DataFrame())

    # Count findings by severity
    severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Review": 0}
    if not findings_df.empty and "severity" in findings_df.columns:
        counts = findings_df["severity"].value_counts().to_dict()
        for k, v in counts.items():
            if k in severity_counts:
                severity_counts[k] = int(v)

    col_profiles_df = profile_results.get("column_profiles", pd.DataFrame())
    col_profiles_list = col_profiles_df.to_dict(orient="records") if not col_profiles_df.empty else []

    # Clean col_profiles for JSON serialization (convert NaNs to None)
    clean_col_profiles = []
    for rec in col_profiles_list:
        clean_rec = {}
        for k, v in rec.items():
            if pd.isna(v):
                clean_rec[k] = None
            elif isinstance(v, (np.integer, int)):
                clean_rec[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_rec[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                clean_rec[k] = bool(v)
            else:
                clean_rec[k] = str(v)
        clean_col_profiles.append(clean_rec)

    # Clean findings
    clean_findings = []
    if not findings_df.empty:
        for rec in findings_df.to_dict(orient="records"):
            clean_rec = {}
            for k, v in rec.items():
                if pd.isna(v):
                    clean_rec[k] = None
                elif isinstance(v, (np.integer, int)):
                    clean_rec[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_rec[k] = float(v)
                else:
                    clean_rec[k] = str(v)
            clean_findings.append(clean_rec)

    # Anomaly section
    anomaly_summary = {"evaluated": False}
    if anomaly_results and anomaly_results.get("status") == "success":
        anom_table = anomaly_results.get("anomaly_table", pd.DataFrame())
        preview_records = []
        if not anom_table.empty:
            for rec in anom_table.head(50).to_dict(orient="records"):
                preview_records.append({k: (None if pd.isna(v) else v) for k, v in rec.items()})

        anomaly_summary = {
            "evaluated": True,
            "algorithm": "Isolation Forest (scikit-learn)",
            "evaluated_features": anomaly_results.get("evaluated_features", []),
            "flagged_candidate_count": int(anomaly_results.get("anomaly_count", 0)),
            "flagged_fraction": float(anomaly_results.get("anomaly_fraction", 0.0)),
            "candidates_preview": preview_records,
        }

    # Drift section
    drift_summary = {"evaluated": False}
    if drift_results and drift_results.get("status") == "success":
        drift_table = drift_results.get("drift_table", pd.DataFrame())
        records = []
        if not drift_table.empty:
            for rec in drift_table.to_dict(orient="records"):
                records.append({k: (None if pd.isna(v) else v) for k, v in rec.items()})

        drift_summary = {
            "evaluated": True,
            "screening_metrics": [
                "Standardized Mean Difference (SMD / Cohen's d)",
                "Kolmogorov-Smirnov 2-sample test",
                "Population Stability Index (PSI)",
                "Variance Dispersion Ratio",
            ],
            "shift_counts": drift_results.get("summary", {}),
            "shared_features_drift": records,
        }

    # ML Experiment section
    ml_summary = {"evaluated": False}
    if ml_experiment_results and ml_experiment_results.get("status") == "success":
        ml_summary = {
            "evaluated": True,
            "model_type": ml_experiment_results.get("model_type"),
            "test_sample_count": ml_experiment_results.get("test_samples"),
            "baseline_metrics": ml_experiment_results.get("metrics_baseline"),
            "cleaned_data_metrics": ml_experiment_results.get("metrics_cleaned"),
            "remediation_actions": ml_experiment_results.get("remediation_summary"),
        }

    report = {
        "metadata": {
            "product": "Extremis DataSentry",
            "version": "1.0.0",
            "generated_at_utc": now_utc,
            "framework": "Python 3.12, scikit-learn, Pandas, Streamlit",
            "audit_type": "Tabular Dataset Health & ML Reliability Audit",
        },
        "dataset_summary": {
            "rows": int(profile_results["rows"]),
            "columns": int(profile_results["cols"]),
            "total_cells": int(profile_results["total_cells"]),
            "memory_kb": float(profile_results["memory_kb"]),
            "missing_cells": int(profile_results["missing_cells"]),
            "missing_percent": float(profile_results["missing_pct"]),
            "duplicate_rows": int(profile_results["duplicate_rows"]),
            "duplicate_percent": float(profile_results["duplicate_pct"]),
            "numeric_columns": profile_results["numeric_cols"],
            "categorical_columns": profile_results["categorical_cols"],
            "health_score": int(profile_results["health_score"]),
            "health_grade": str(profile_results["health_grade"]),
            "health_description": str(profile_results["health_description"]),
        },
        "column_profiles": clean_col_profiles,
        "findings_summary": {
            "total_findings": len(clean_findings),
            "severity_breakdown": severity_counts,
            "findings_list": clean_findings,
        },
        "anomaly_audit": anomaly_summary,
        "distribution_drift_audit": drift_summary,
        "ml_reliability_experiment": ml_summary,
        "methodology_and_limitations": [
            "Anomaly candidates flagged by Isolation Forest are statistical review items, not confirmed errors. Domain experts must verify candidates before alteration.",
            "Distribution drift screening employs Standardized Mean Difference (Cohen's d), Kolmogorov-Smirnov test, and Population Stability Index (PSI). Mean and variance shifts denote population shifts, not necessarily data corruption.",
            "Label noise detection uses Out-of-Fold cross-validation predicted probabilities. Mislabeled candidates should be verified by annotators.",
            "Extremis DataSentry follows a non-destructive audit principle: original dataset files are never modified or deleted without explicit user consent.",
        ],
    }

    return report


def export_findings_to_csv(findings_df: pd.DataFrame) -> bytes:
    """Serialize findings DataFrame to UTF-8 CSV bytes."""
    if findings_df.empty:
        empty_df = pd.DataFrame(columns=["severity", "issue", "column", "count", "percentage", "detail", "recommended_action"])
        return empty_df.to_csv(index=False).encode("utf-8")
    return findings_df.to_csv(index=False).encode("utf-8")
