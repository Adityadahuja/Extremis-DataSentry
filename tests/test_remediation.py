"""Tests for datasentry.remediation and datasentry.exporter modules."""

import json
import numpy as np
import pandas as pd
from datasentry.exporter import export_findings_to_csv, generate_audit_json
from datasentry.profiler import profile_dataset
from datasentry.remediation import apply_remediation


def test_apply_remediation():
    df = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 40.0],
        "cat": ["A", "B", np.nan, "A"],
        "const": [1, 1, 1, 1],
    })
    # Add duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    orig_shape = df.shape

    cleaned, log = apply_remediation(
        df,
        drop_duplicates=True,
        numeric_imputation="median",
        categorical_imputation="mode",
        drop_constant_cols=True,
    )

    # Verify original dataframe is untouched
    assert df.shape == orig_shape
    assert df["num"].isna().sum() == 1

    # Verify cleaned dataframe
    assert cleaned.shape[0] == 4  # Duplicate removed
    assert "const" not in cleaned.columns  # Constant column removed
    assert cleaned["num"].isna().sum() == 0  # Missing imputed
    assert cleaned["cat"].isna().sum() == 0


def test_generate_audit_json():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "b": [10, 20, 30, 40, np.nan, 60, 70, 80, 90, 100],
    })
    prof = profile_dataset(df)
    report = generate_audit_json(prof)

    assert report["metadata"]["product"] == "Extremis DataSentry"
    assert report["dataset_summary"]["rows"] == 10
    assert "findings_summary" in report
    assert "methodology_and_limitations" in report

    # Test serialization to valid JSON
    json_str = json.dumps(report)
    assert len(json_str) > 100


def test_export_findings_to_csv():
    df = pd.DataFrame({
        "severity": ["High"],
        "issue": ["Missing values"],
        "column": ["test_col"],
        "count": [5],
        "percentage": [10.0],
        "detail": ["5 rows missing"],
        "recommended_action": ["Impute"],
    })
    csv_bytes = export_findings_to_csv(df)
    assert b"Missing values" in csv_bytes
