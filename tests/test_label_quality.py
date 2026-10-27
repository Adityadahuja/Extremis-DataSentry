"""Tests for datasentry.label_quality and model_experiment modules."""

import numpy as np
import pandas as pd
from datasentry.label_quality import detect_label_issues
from datasentry.model_experiment import run_ml_reliability_experiment


def test_detect_label_issues():
    rng = np.random.default_rng(42)
    # Generate linearly separable data
    n = 120
    x1 = np.concatenate([rng.normal(20, 2, n // 2), rng.normal(80, 2, n // 2)])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    df = pd.DataFrame({"feat": x1, "target": y})

    # Deliberately flip label of index 0
    df.loc[0, "target"] = 1

    res = detect_label_issues(df, target_col="target", cv_folds=3, confidence_threshold=0.7)
    assert res["status"] == "success"
    assert res["suspicious_count"] >= 1
    flagged_indices = res["suspicious_table"]["source_row_index"].tolist()
    assert 0 in flagged_indices


def test_run_ml_reliability_experiment():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(20, 5, n)
    prob = 1 / (1 + np.exp(-(0.05 * (x1 - 50) + 0.1 * (x2 - 20))))
    y = rng.binomial(1, prob)

    df = pd.DataFrame({"feat1": x1, "feat2": x2, "target": y})

    # Add duplicates and extreme outliers
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    df.loc[5, "feat1"] = 9999.0

    res = run_ml_reliability_experiment(df, target_col="target", test_size=0.25, random_state=42)
    assert res["status"] == "success"
    assert "metrics_baseline" in res
    assert "metrics_cleaned" in res
    assert not res["comparison_table"].empty
