"""Tests for datasentry.profiler module."""

import numpy as np
import pandas as pd
import pytest
from datasentry.profiler import (
    compute_column_profiles,
    compute_health_score,
    extract_data_findings,
    load_dataset,
    profile_dataset,
)


def test_load_dataset_valid():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    loaded, err = load_dataset(df)
    assert err is None
    assert loaded.shape == (3, 2)


def test_load_dataset_empty():
    df = pd.DataFrame()
    loaded, err = load_dataset(df)
    assert loaded is None
    assert "empty" in err.lower()


def test_load_dataset_none():
    loaded, err = load_dataset(None)
    assert loaded is None
    assert "no file" in err.lower()


def test_compute_column_profiles():
    df = pd.DataFrame({
        "num_col": [10.0, 20.0, 30.0, np.nan, 50.0],
        "const_col": ["A", "A", "A", "A", "A"],
        "id_col": [f"ID_{i}" for i in range(5)],
    })
    profiles = compute_column_profiles(df)
    assert len(profiles) == 3

    # Check num_col
    num_prof = profiles[profiles["column"] == "num_col"].iloc[0]
    assert num_prof["missing_count"] == 1
    assert num_prof["missing_percent"] == 20.0
    assert bool(num_prof["is_numeric"]) is True
    assert num_prof["mean"] == 27.5

    # Check const_col
    const_prof = profiles[profiles["column"] == "const_col"].iloc[0]
    assert bool(const_prof["is_constant"]) is True
    assert const_prof["unique_values"] == 1


def test_profile_dataset_full():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, np.nan, 50, 55, 60, 65, 70],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        "dept": ["HR", "IT", "IT", "HR", "IT", "Finance", "IT", "Finance", "IT", "HR"],
        "const": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    res = profile_dataset(df)
    assert res["rows"] == 10
    assert res["cols"] == 4
    assert res["missing_cells"] == 1
    assert 0 <= res["health_score"] <= 100
    assert res["health_grade"] in ["A", "B", "C", "D", "F"]
    assert not res["findings"].empty
