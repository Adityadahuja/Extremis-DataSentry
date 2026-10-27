"""Non-destructive dataset remediation and sanitization engine for Extremis DataSentry.

Allows users to preview and apply safe remediation strategies (deduplication, imputation,
outlier clipping/filtering, constant removal, label error correction) without modifying
the source dataset.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def apply_remediation(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    numeric_imputation: str = "median",  # "none", "median", "mean", "zero"
    categorical_imputation: str = "mode",  # "none", "mode", "constant"
    drop_constant_cols: bool = True,
    drop_high_cardinality: bool = False,
    clip_outliers: bool = False,
    anomaly_indices_to_drop: Optional[List[int]] = None,
    label_corrections: Optional[Dict[int, Any]] = None,
    target_col: Optional[str] = None,
    drop_cols_custom: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a sanitized copy of the dataset according to specified recipe.

    The original dataframe is NEVER modified.

    Returns:
        (Cleaned DataFrame, Remediation Audit Log)
    """
    cleaned = df.copy()
    initial_rows, initial_cols = cleaned.shape
    log: Dict[str, Any] = {
        "initial_shape": (initial_rows, initial_cols),
        "steps_applied": [],
        "dropped_rows": 0,
        "dropped_cols": [],
        "imputed_cells": 0,
        "clipped_cells": 0,
        "corrected_labels": 0,
    }

    # Step 1: Drop Specific Custom Columns
    if drop_cols_custom:
        valid_drops = [c for c in drop_cols_custom if c in cleaned.columns]
        if valid_drops:
            cleaned = cleaned.drop(columns=valid_drops)
            log["dropped_cols"].extend(valid_drops)
            log["steps_applied"].append(f"Pruned custom selected columns: {valid_drops}")

    # Step 2: Drop Duplicates
    if drop_duplicates:
        dups_count = int(cleaned.duplicated().sum())
        if dups_count > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            log["dropped_rows"] += dups_count
            log["steps_applied"].append(f"Deduplicated dataset: removed {dups_count:,} duplicate rows.")

    # Step 3: Filter Anomaly Indices
    if anomaly_indices_to_drop:
        valid_indices = [i for i in anomaly_indices_to_drop if i in cleaned.index]
        if valid_indices:
            cleaned = cleaned.drop(index=valid_indices).reset_index(drop=True)
            log["dropped_rows"] += len(valid_indices)
            log["steps_applied"].append(f"Filtered {len(valid_indices):,} user-confirmed outlier records.")

    # Step 4: Prune Constant Columns
    if drop_constant_cols:
        constant_cols = [c for c in cleaned.columns if cleaned[c].nunique(dropna=True) <= 1]
        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)
            log["dropped_cols"].extend(constant_cols)
            log["steps_applied"].append(f"Dropped {len(constant_cols)} zero-variance constant columns: {constant_cols}")

    # Step 5: Prune High Cardinality Categoricals
    if drop_high_cardinality:
        non_num = cleaned.select_dtypes(exclude=np.number).columns
        high_card_cols = [c for c in non_num if cleaned[c].nunique(dropna=True) > max(50, int(0.5 * len(cleaned)))]
        if high_card_cols:
            cleaned = cleaned.drop(columns=high_card_cols)
            log["dropped_cols"].extend(high_card_cols)
            log["steps_applied"].append(f"Dropped {len(high_card_cols)} high-cardinality categorical columns: {high_card_cols}")

    # Step 6: Clip Extreme Outliers (Winsorization at 1st and 99th percentiles)
    if clip_outliers:
        num_cols = cleaned.select_dtypes(include=np.number).columns
        clipped_count = 0
        for col in num_cols:
            s = pd.to_numeric(cleaned[col], errors="coerce").dropna()
            if len(s) > 10:
                p01 = np.percentile(s, 1)
                p99 = np.percentile(s, 99)
                out_mask = (cleaned[col] < p01) | (cleaned[col] > p99)
                clipped_count += int(out_mask.sum())
                cleaned[col] = cleaned[col].clip(lower=p01, upper=p99)
        if clipped_count > 0:
            log["clipped_cells"] += clipped_count
            log["steps_applied"].append(f"Winsorized {clipped_count:,} numeric values into 1st-99th percentile bounds.")

    # Step 7: Numeric Imputation
    num_cols = cleaned.select_dtypes(include=np.number).columns
    if numeric_imputation != "none":
        for col in num_cols:
            missing_count = int(cleaned[col].isna().sum())
            if missing_count > 0:
                log["imputed_cells"] += missing_count
                if numeric_imputation == "median":
                    fill_val = cleaned[col].median()
                elif numeric_imputation == "mean":
                    fill_val = cleaned[col].mean()
                elif numeric_imputation == "zero":
                    fill_val = 0.0
                else:
                    fill_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(fill_val)
        if numeric_imputation != "none" and log["imputed_cells"] > 0:
            log["steps_applied"].append(f"Imputed missing numeric features using {numeric_imputation} strategy.")

    # Step 8: Categorical Imputation
    cat_cols = cleaned.select_dtypes(exclude=np.number).columns
    if categorical_imputation != "none":
        cat_imputed = 0
        for col in cat_cols:
            missing_count = int(cleaned[col].isna().sum())
            if missing_count > 0:
                cat_imputed += missing_count
                if categorical_imputation == "mode":
                    mode_series = cleaned[col].mode(dropna=True)
                    fill_val = mode_series.iloc[0] if not mode_series.empty else "Missing"
                else:
                    fill_val = "Missing"
                cleaned[col] = cleaned[col].fillna(fill_val)
        if cat_imputed > 0:
            log["imputed_cells"] += cat_imputed
            log["steps_applied"].append(f"Imputed {cat_imputed} missing categorical cells using {categorical_imputation} strategy.")

    # Step 9: Correct Label Errors
    if label_corrections and target_col and target_col in cleaned.columns:
        corr_count = 0
        for row_idx, new_lbl in label_corrections.items():
            if row_idx in cleaned.index:
                cleaned.loc[row_idx, target_col] = new_lbl
                corr_count += 1
        if corr_count > 0:
            log["corrected_labels"] = corr_count
            log["steps_applied"].append(f"Applied {corr_count} AI-verified label corrections to target '{target_col}'.")

    final_rows, final_cols = cleaned.shape
    log["final_shape"] = (final_rows, final_cols)

    return cleaned, log
