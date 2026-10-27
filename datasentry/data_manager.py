"""Data Manager module for Extremis DataSentry.

Provides interactive filtering, column searching, type inspection,
and schema management utilities for tabular datasets.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def filter_dataset(
    df: pd.DataFrame,
    search_query: str = "",
    column_filter: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> pd.DataFrame:
    """Filter dataset based on search text or specific column values."""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # Search across all string representation of cells
    if search_query.strip():
        q = search_query.strip().lower()
        mask = filtered.astype(str).apply(lambda row: row.str.lower().str.contains(q, regex=False)).any(axis=1)
        filtered = filtered[mask]

    # Specific column value filter
    if column_filter and column_filter in filtered.columns and filter_value is not None:
        if filter_value == "(Missing / Null)":
            filtered = filtered[filtered[column_filter].isna()]
        else:
            filtered = filtered[filtered[column_filter].astype(str) == str(filter_value)]

    return filtered


def get_column_summary(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Retrieve detailed statistics and sample values for a single column."""
    if column not in df.columns:
        return {}

    s = df[column]
    rows = len(df)
    missing = int(s.isna().sum())
    unique = int(s.nunique(dropna=True))

    res: Dict[str, Any] = {
        "column": column,
        "dtype": str(s.dtype),
        "total_rows": rows,
        "missing_count": missing,
        "missing_pct": round(missing / max(rows, 1) * 100, 2),
        "unique_values": unique,
        "uniqueness_ratio": round(unique / max(rows, 1) * 100, 2),
        "sample_values": s.dropna().head(5).tolist(),
    }

    if pd.api.types.is_numeric_dtype(s):
        clean_s = pd.to_numeric(s, errors="coerce").dropna()
        if not clean_s.empty:
            res["min"] = round(float(clean_s.min()), 2)
            res["max"] = round(float(clean_s.max()), 2)
            res["mean"] = round(float(clean_s.mean()), 2)
            res["median"] = round(float(clean_s.median()), 2)
            res["std"] = round(float(clean_s.std()), 2) if len(clean_s) > 1 else 0.0

    return res
