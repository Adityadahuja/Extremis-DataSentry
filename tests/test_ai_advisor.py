"""Tests for AI advisor and data manager modules."""

import numpy as np
import pandas as pd
from datasentry.ai_advisor import generate_ai_suggestions
from datasentry.data_manager import filter_dataset, get_column_summary
from datasentry.profiler import profile_dataset


def test_ai_advisor_generates_suggestions():
    df = pd.DataFrame({
        "age": [20, 30, np.nan, np.nan, 50, 60, 70, 80, 90, 100],  # 20% missing
        "const": ["X"] * 10,  # Constant
        "spend": [100, 200, 300, 400, 500, 600, 700, 800, 900, 99999],  # Outlier
    })
    prof = profile_dataset(df)
    suggestions = generate_ai_suggestions(prof)

    assert len(suggestions) > 0
    categories = [s["category"] for s in suggestions]
    assert any("Missing" in c or "Imputation" in c for c in categories)
    assert any("Dimensionality" in c or "Constant" in c for c in categories)


def test_data_manager_filtering():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David"],
        "dept": ["HR", "IT", "IT", "Finance"],
    })
    # Search query
    f1 = filter_dataset(df, search_query="char")
    assert len(f1) == 1
    assert f1.iloc[0]["name"] == "Charlie"

    # Column filter
    f2 = filter_dataset(df, column_filter="dept", filter_value="IT")
    assert len(f2) == 2

    # Column summary
    summary = get_column_summary(df, "dept")
    assert summary["unique_values"] == 3
    assert summary["total_rows"] == 4
