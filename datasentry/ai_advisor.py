"""AI Suggestion Provider & Diagnostic Advisor for Extremis DataSentry.

Synthesizes data quality profiles, anomaly scores, distribution drift, and label noise
to produce prioritized root-cause hypotheses, downstream ML impact analyses,
and actionable remediation recommendations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def generate_ai_suggestions(
    profile_results: Dict[str, Any],
    anomaly_results: Optional[Dict[str, Any]] = None,
    drift_results: Optional[Dict[str, Any]] = None,
    label_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Analyze all audit signals and generate human-readable AI diagnostic recommendations."""
    suggestions: List[Dict[str, Any]] = []

    rows = profile_results.get("rows", 0)
    col_profiles = profile_results.get("column_profiles", pd.DataFrame())
    findings = profile_results.get("findings", pd.DataFrame())

    # 1. Missingness Diagnostics & AI Hypotheses
    if not col_profiles.empty:
        for _, r in col_profiles.iterrows():
            col = r["column"]
            miss_pct = r["missing_percent"]
            miss_cnt = r["missing_count"]

            if miss_pct >= 40.0:
                suggestions.append({
                    "id": f"sug_miss_{col}",
                    "category": "Missingness & Collection",
                    "severity": "Critical",
                    "target_feature": col,
                    "finding": f"{col} is missing in {miss_cnt:,} rows ({miss_pct:.1f}%).",
                    "root_cause_hypothesis": (
                        f"High missingness (>40%) in '{col}' indicates an optional form field, "
                        "a late-stage pipeline schema migration, or a broken API integration upstream."
                    ),
                    "model_impact": (
                        "Imputing over 40% of records artificially reduces natural variance and introduces severe synthetic bias. "
                        "Tree models and neural networks will struggle with feature attribution."
                    ),
                    "recommended_action": (
                        f"Drop '{col}' if feature importance is non-critical, or create a binary indicator column "
                        f"'{col}_is_missing' to preserve information about the missingness mechanism (MNAR)."
                    ),
                    "confidence_score": 96,
                })
            elif 5.0 <= miss_pct < 40.0:
                suggestions.append({
                    "id": f"sug_miss_mod_{col}",
                    "category": "Imputation Strategy",
                    "severity": "High" if miss_pct >= 20 else "Medium",
                    "target_feature": col,
                    "finding": f"{col} has {miss_cnt:,} missing entries ({miss_pct:.1f}%).",
                    "root_cause_hypothesis": (
                        f"Moderate missingness in '{col}' typically reflects Missing At Random (MAR) conditions, "
                        "such as user drop-off or system timeouts during data transmission."
                    ),
                    "model_impact": (
                        "Default zero-imputation will create an artificial spike at 0, skewing gradient descent and linear coefficients."
                    ),
                    "recommended_action": (
                        f"Use median imputation for skewed numerics, or K-Nearest Neighbors (KNN) iterative imputation "
                        f"conditioned on related cohort features."
                    ),
                    "confidence_score": 91,
                })

    # 2. Duplicate Records Analysis
    dup_rows = profile_results.get("duplicate_rows", 0)
    dup_pct = profile_results.get("duplicate_pct", 0.0)
    if dup_rows > 0:
        suggestions.append({
            "id": "sug_duplicates",
            "category": "Data Pipeline Redundancy",
            "severity": "High" if dup_pct >= 5.0 else "Medium",
            "target_feature": "Dataset-Wide",
            "finding": f"Detected {dup_rows:,} exact identical duplicate rows ({dup_pct:.1f}% of dataset).",
            "root_cause_hypothesis": (
                "Identical rows usually occur from duplicate batch ingestion jobs, webhook retry storms, "
                "or missing primary key deduplication in SQL ETL extract queries."
            ),
            "model_impact": (
                "Causes data leakage when duplicate rows leak into both train and test splits, producing "
                "dangerously optimistic cross-validation scores that collapse in production."
            ),
            "recommended_action": (
                "Apply exact deduplication immediately before train/test splitting. Audit ingestion pipelines for idempotent keys."
            ),
            "confidence_score": 98,
        })

    # 3. Constant & High-Cardinality Feature Pruning
    if not col_profiles.empty:
        for _, r in col_profiles.iterrows():
            col = r["column"]
            if r["is_constant"]:
                suggestions.append({
                    "id": f"sug_const_{col}",
                    "category": "Dimensionality & Schema",
                    "severity": "Medium",
                    "target_feature": col,
                    "finding": f"'{col}' has strictly 1 unique value across all rows (zero variance).",
                    "root_cause_hypothesis": (
                        "Likely an obsolete filter flag, hardcoded configuration parameter, or single-tenant database extract."
                    ),
                    "model_impact": (
                        "Zero variance contributes zero mutual information or predictive entropy, wasting RAM and compute cycles."
                    ),
                    "recommended_action": f"Safely drop '{col}' from feature store and training pipelines.",
                    "confidence_score": 99,
                })
            elif r["is_high_cardinality"]:
                suggestions.append({
                    "id": f"sug_card_{col}",
                    "category": "Feature Representation",
                    "severity": "Medium",
                    "target_feature": col,
                    "finding": f"'{col}' has {r['unique_values']:,} unique string categories across {rows:,} rows.",
                    "root_cause_hypothesis": (
                        "Column appears to be a unique transaction ID, customer GUID, device UUID, or free-form text note."
                    ),
                    "model_impact": (
                        "One-hot encoding this feature will cause extreme dimensional explosion and memory exhaustion. "
                        "Tree models will overfit on individual IDs."
                    ),
                    "recommended_action": (
                        f"Exclude '{col}' from model predictors or extract frequency / target encodings."
                    ),
                    "confidence_score": 94,
                })

    # 4. Outlier & Anomaly Hypotheses (Isolation Forest Signals)
    if anomaly_results and anomaly_results.get("status") == "success":
        anom_cnt = anomaly_results.get("anomaly_count", 0)
        anom_frac = anomaly_results.get("anomaly_fraction", 0.0)
        driver_exps = anomaly_results.get("feature_contributions", {})

        if anom_cnt > 0:
            # Find the most frequent driver feature
            driver_counts: Dict[str, int] = {}
            for exp in driver_exps.values():
                feat_name = exp.split(" = ")[0].strip()
                driver_counts[feat_name] = driver_counts.get(feat_name, 0) + 1

            top_driver = max(driver_counts, key=driver_counts.get) if driver_counts else "Numeric Features"

            suggestions.append({
                "id": "sug_anomalies",
                "category": "Multi-Feature Outliers",
                "severity": "High" if anom_frac >= 0.08 else "Medium",
                "target_feature": top_driver,
                "finding": f"Isolation Forest identified {anom_cnt} unusual candidate records ({anom_frac*100:.1f}%). Primary driver: '{top_driver}'.",
                "root_cause_hypothesis": (
                    f"Extreme deviations in '{top_driver}' frequently correspond to sentinel test values (e.g., 9999, -1), "
                    "sensor communication spikes, or rare high-value transactions."
                ),
                "model_impact": (
                    "High-leverage outliers severely distort gradient descent, pulling decision boundaries away from typical cohorts."
                ),
                "recommended_action": (
                    f"Inspect the top anomaly candidates in Anomaly Review. Apply quantile clipping (Winsorization at 1st/99th percentiles) "
                    "or filter verified data errors in Data Correction Studio."
                ),
                "confidence_score": 89,
            })

    # 5. Covariate Shift & Distribution Drift Hypotheses
    if drift_results and drift_results.get("status") == "success":
        drift_table = drift_results.get("drift_table", pd.DataFrame())
        high_shift_feats = drift_table[drift_table["drift_severity"] == "High Shift"] if not drift_table.empty else pd.DataFrame()

        if not high_shift_feats.empty:
            for _, r in high_shift_feats.iterrows():
                f_name = r["feature"]
                smd_val = r["smd"]
                psi_val = r["psi"]
                suggestions.append({
                    "id": f"sug_drift_{f_name}",
                    "category": "Distribution Shift (Drift)",
                    "severity": "Critical" if smd_val >= 0.8 else "High",
                    "target_feature": f_name,
                    "finding": f"High distribution shift detected in '{f_name}' (SMD = {smd_val:.2f}, PSI = {psi_val:.2f}).",
                    "root_cause_hypothesis": (
                        f"Significant divergence between baseline and current data indicates a macroeconomic shift, "
                        "a new product pricing tier, or demographic population changes since baseline capture."
                    ),
                    "model_impact": (
                        f"Features with PSI >= 0.25 violate the i.i.d. assumption. Models trained on baseline weights "
                        f"will experience severe prediction drift on current '{f_name}' values."
                    ),
                    "recommended_action": (
                        "Re-baseline model features with recent production data. Consider continuous retraining or weighting recent samples."
                    ),
                    "confidence_score": 93,
                })

    # 6. Label Noise Diagnostics (Confident Learning)
    if label_results and label_results.get("status") == "success":
        susp_cnt = label_results.get("suspicious_count", 0)
        target_name = label_results.get("target_column", "Target")
        if susp_cnt > 0:
            suggestions.append({
                "id": "sug_label_noise",
                "category": "Label Quality (Annotation)",
                "severity": "High",
                "target_feature": target_name,
                "finding": f"Identified {susp_cnt} suspicious label candidate errors in target '{target_name}'.",
                "root_cause_hypothesis": (
                    "Human labeler ambiguity, inconsistent labeling guidelines, or flawed automated business heuristics."
                ),
                "model_impact": (
                    "Label noise forces estimators to memorize contradictory samples, penalizing generalization and precision."
                ),
                "recommended_action": (
                    "Review flagged rows in Label Quality tab. Prune or re-annotate suspicious labels prior to model training."
                ),
                "confidence_score": 90,
            })

    # Sort suggestions by severity: Critical > High > Medium > Low
    sev_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    suggestions.sort(key=lambda s: sev_rank.get(s["severity"], 4))
    return suggestions
