"""Tests for datasentry.drift module."""

import numpy as np
import pandas as pd
from datasentry.drift import compute_drift_metrics


def test_drift_identical_distributions():
    rng = np.random.default_rng(42)
    vals = rng.normal(100, 15, size=200)

    ref = pd.DataFrame({"metric": vals})
    cur = pd.DataFrame({"metric": vals})

    res = compute_drift_metrics(cur, ref)
    assert res["status"] == "success"
    drift_df = res["drift_table"]
    assert len(drift_df) == 1
    row = drift_df.iloc[0]
    assert row["smd"] < 0.05
    assert row["drift_severity"] == "Stable"


def test_drift_detected_shift():
    rng = np.random.default_rng(42)
    ref_vals = rng.normal(50, 10, size=250)
    cur_vals = rng.normal(90, 12, size=250)  # Strong mean shift

    ref = pd.DataFrame({"metric": ref_vals})
    cur = pd.DataFrame({"metric": cur_vals})

    res = compute_drift_metrics(cur, ref)
    assert res["status"] == "success"
    drift_df = res["drift_table"]
    row = drift_df.iloc[0]
    assert row["smd"] >= 0.5
    assert row["drift_severity"] == "High Shift"


def test_drift_no_shared_features():
    ref = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    cur = pd.DataFrame({"b": [1, 2, 3, 4, 5]})
    res = compute_drift_metrics(cur, ref)
    assert res["status"] == "warning"
    assert "No shared numeric features" in res["message"]
