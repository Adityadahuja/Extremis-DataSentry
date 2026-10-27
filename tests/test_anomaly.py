"""Tests for datasentry.anomaly module."""

import numpy as np
import pandas as pd
from datasentry.anomaly import run_anomaly_detection


def test_anomaly_detection_with_outlier():
    rng = np.random.default_rng(42)
    x1 = rng.normal(50, 5, size=100)
    x2 = rng.normal(100, 10, size=100)

    # Inject extreme outlier
    x1[10] = 9999.0
    x2[10] = 9999.0

    df = pd.DataFrame({"x1": x1, "x2": x2})
    res = run_anomaly_detection(df, contamination=0.05, random_state=42)

    assert res["status"] == "success"
    assert res["anomaly_count"] > 0
    assert 10 in res["anomaly_indices"]
    assert not res["pca_df"].empty
    assert 10 in res["feature_contributions"]


def test_anomaly_detection_skips_small_dataset():
    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    res = run_anomaly_detection(df)
    assert res["status"] == "skipped"
    assert res["anomaly_count"] == 0


def test_anomaly_detection_no_numeric():
    df = pd.DataFrame({"cat1": ["a", "b", "c"] * 10, "cat2": ["x", "y", "z"] * 10})
    res = run_anomaly_detection(df)
    assert res["status"] == "skipped"
