"""Sample dataset generation and loading utilities for Extremis DataSentry.

Provides realistic tabular datasets with controlled data quality anomalies,
distribution drift, label noise, and edge cases for auditing and testing.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def generate_churn_datasets(
    n_current: int = 600, n_ref: int = 500, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic Customer Churn datasets (Current with anomalies, Reference baseline)."""
    rng = np.random.default_rng(seed)

    # Reference dataset (Clean baseline distribution)
    ref_age = rng.integers(18, 75, size=n_ref)
    ref_tenure = rng.integers(1, 72, size=n_ref)
    ref_monthly = np.round(rng.normal(65.0, 15.0, size=n_ref), 2)
    ref_monthly = np.clip(ref_monthly, 18.0, 125.0)
    ref_total = np.round(ref_monthly * ref_tenure + rng.normal(0, 50, size=n_ref), 2)
    ref_total = np.clip(ref_total, 20.0, None)
    ref_tickets = rng.poisson(1.2, size=n_ref)
    ref_contracts = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_ref)
    ref_payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.35, 0.20, 0.25, 0.20],
        size=n_ref,
    )
    # Target churn probability
    ref_logit = -2.0 + 0.02 * (ref_monthly - 65) - 0.04 * ref_tenure + 0.4 * ref_tickets
    ref_prob = 1 / (1 + np.exp(-ref_logit))
    ref_churn = rng.binomial(1, ref_prob)

    ref_df = pd.DataFrame({
        "customer_id": [f"REF-{1000 + i}" for i in range(n_ref)],
        "age": ref_age,
        "tenure_months": ref_tenure,
        "monthly_charges": ref_monthly,
        "total_charges": ref_total,
        "support_tickets": ref_tickets,
        "contract_type": ref_contracts,
        "payment_method": ref_payment,
        "constant_status": ["ACTIVE"] * n_ref,
        "churn": ref_churn,
    })

    # Current dataset (Contains drift, missing values, outliers, duplicates, label noise)
    cur_age = rng.integers(18, 75, size=n_current)
    cur_tenure = rng.integers(1, 72, size=n_current)
    # Notice drift in monthly charges: shifted mean from 65 to 88
    cur_monthly = np.round(rng.normal(88.0, 22.0, size=n_current), 2)
    cur_monthly = np.clip(cur_monthly, 20.0, 160.0)
    cur_total = np.round(cur_monthly * cur_tenure + rng.normal(0, 60, size=n_current), 2)
    cur_total = np.clip(cur_total, 20.0, None)
    # Tickets shifted higher
    cur_tickets = rng.poisson(2.1, size=n_current)
    cur_contracts = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.70, 0.18, 0.12], size=n_current)
    cur_payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.45, 0.15, 0.20, 0.20],
        size=n_current,
    )

    cur_logit = -1.5 + 0.025 * (cur_monthly - 70) - 0.035 * cur_tenure + 0.45 * cur_tickets
    cur_prob = 1 / (1 + np.exp(-cur_logit))
    cur_churn = rng.binomial(1, cur_prob)

    cur_df = pd.DataFrame({
        "customer_id": [f"CUST-{2000 + i}" for i in range(n_current)],
        "age": cur_age,
        "tenure_months": cur_tenure,
        "monthly_charges": cur_monthly,
        "total_charges": cur_total,
        "support_tickets": cur_tickets,
        "contract_type": cur_contracts,
        "payment_method": cur_payment,
        "constant_status": ["ACTIVE"] * n_current,
        "tracking_guid": [f"TRK-{rng.integers(100000, 999999)}-{i}" for i in range(n_current)],
        "churn": cur_churn,
    })

    # Inject data quality defects into Current Dataset:
    # 1. Missing values
    missing_age_idx = rng.choice(n_current, size=int(0.08 * n_current), replace=False)
    cur_df.loc[missing_age_idx, "age"] = np.nan

    missing_total_idx = rng.choice(n_current, size=int(0.12 * n_current), replace=False)
    cur_df.loc[missing_total_idx, "total_charges"] = np.nan

    missing_ticket_idx = rng.choice(n_current, size=int(0.05 * n_current), replace=False)
    cur_df.loc[missing_ticket_idx, "support_tickets"] = np.nan

    # 2. Extreme Outliers / Anomalies
    outlier_idx = rng.choice(n_current, size=6, replace=False)
    cur_df.loc[outlier_idx[0], "monthly_charges"] = 9999.0
    cur_df.loc[outlier_idx[1], "monthly_charges"] = 8500.0
    cur_df.loc[outlier_idx[2], "support_tickets"] = 45
    cur_df.loc[outlier_idx[3], "total_charges"] = 185000.0
    cur_df.loc[outlier_idx[4], "age"] = 142
    cur_df.loc[outlier_idx[5], "monthly_charges"] = -150.0

    # 3. Duplicate rows (inject 8 duplicate rows)
    dup_rows = cur_df.iloc[10:18].copy()
    cur_df = pd.concat([cur_df, dup_rows], ignore_index=True)

    # 4. Label noise (flip 8% of target labels to test label quality detection)
    flip_idx = rng.choice(len(cur_df), size=int(0.08 * len(cur_df)), replace=False)
    cur_df.loc[flip_idx, "churn"] = 1 - cur_df.loc[flip_idx, "churn"]

    return cur_df, ref_df


def generate_transaction_datasets(
    n_current: int = 600, n_ref: int = 500, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Financial Fraud & Transaction dataset with risk indicators and drift."""
    rng = np.random.default_rng(seed)

    # Reference baseline
    ref_amt = np.round(rng.exponential(scale=65.0, size=n_ref) + 5.0, 2)
    ref_balance = np.round(rng.lognormal(mean=7.5, sigma=0.8, size=n_ref), 2)
    ref_velocity = rng.poisson(lam=3.0, size=n_ref)
    ref_device_trust = np.round(rng.beta(8, 2, size=n_ref) * 100, 1)
    ref_channel = rng.choice(["Web", "Mobile App", "POS Terminal", "ATM"], p=[0.35, 0.45, 0.15, 0.05], size=n_ref)
    ref_fraud_prob = 1 / (1 + np.exp(-(-4.0 + 0.015 * ref_amt + 0.3 * ref_velocity - 0.04 * ref_device_trust)))
    ref_fraud = rng.binomial(1, ref_fraud_prob)

    ref_df = pd.DataFrame({
        "tx_id": [f"TX-REF-{10000 + i}" for i in range(n_ref)],
        "amount": ref_amt,
        "account_balance": ref_balance,
        "velocity_24h": ref_velocity,
        "device_trust_score": ref_device_trust,
        "channel": ref_channel,
        "currency": ["USD"] * n_ref,
        "is_fraud": ref_fraud,
    })

    # Current dataset (With surge in transaction amounts, velocity drift, missing balance, outliers)
    cur_amt = np.round(rng.exponential(scale=110.0, size=n_current) + 10.0, 2)
    cur_balance = np.round(rng.lognormal(mean=7.6, sigma=0.9, size=n_current), 2)
    cur_velocity = rng.poisson(lam=5.2, size=n_current)
    cur_device_trust = np.round(rng.beta(6, 3, size=n_current) * 100, 1)
    cur_channel = rng.choice(["Web", "Mobile App", "POS Terminal", "ATM"], p=[0.40, 0.45, 0.10, 0.05], size=n_current)
    cur_fraud_prob = 1 / (1 + np.exp(-(-3.5 + 0.012 * cur_amt + 0.28 * cur_velocity - 0.035 * cur_device_trust)))
    cur_fraud = rng.binomial(1, cur_fraud_prob)

    cur_df = pd.DataFrame({
        "tx_id": [f"TX-CUR-{20000 + i}" for i in range(n_current)],
        "amount": cur_amt,
        "account_balance": cur_balance,
        "velocity_24h": cur_velocity,
        "device_trust_score": cur_device_trust,
        "channel": cur_channel,
        "currency": ["USD"] * n_current,
        "session_hash": [f"HASH-{rng.integers(10000, 99999)}-{i}" for i in range(n_current)],
        "is_fraud": cur_fraud,
    })

    # Inject issues
    cur_df.loc[rng.choice(n_current, size=int(0.09 * n_current), replace=False), "account_balance"] = np.nan
    cur_df.loc[rng.choice(n_current, size=int(0.06 * n_current), replace=False), "device_trust_score"] = np.nan
    # Outliers
    outlier_idx = rng.choice(n_current, size=5, replace=False)
    cur_df.loc[outlier_idx[0], "amount"] = 89999.00
    cur_df.loc[outlier_idx[1], "velocity_24h"] = 120
    cur_df.loc[outlier_idx[2], "amount"] = 55000.00
    # Duplicates
    cur_df = pd.concat([cur_df, cur_df.iloc[5:12].copy()], ignore_index=True)
    # Label flips
    flip_idx = rng.choice(len(cur_df), size=int(0.07 * len(cur_df)), replace=False)
    cur_df.loc[flip_idx, "is_fraud"] = 1 - cur_df.loc[flip_idx, "is_fraud"]

    return cur_df, ref_df


def generate_healthcare_datasets(
    n_current: int = 500, n_ref: int = 400, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Healthcare Diagnostic Patient dataset with clinical features."""
    rng = np.random.default_rng(seed)

    # Reference
    ref_glucose = np.round(rng.normal(115, 25, size=n_ref), 1)
    ref_bp = np.round(rng.normal(72, 11, size=n_ref), 1)
    ref_bmi = np.round(rng.normal(28.5, 5.2, size=n_ref), 1)
    ref_age = rng.integers(21, 80, size=n_ref)
    ref_insulin = np.round(rng.exponential(80, size=n_ref) + 15, 1)
    ref_prob = 1 / (1 + np.exp(-(-5.0 + 0.03 * ref_glucose + 0.02 * ref_bp + 0.06 * ref_bmi + 0.02 * ref_age)))
    ref_outcome = rng.binomial(1, np.clip(ref_prob, 0.01, 0.99))

    ref_df = pd.DataFrame({
        "patient_id": [f"PAT-REF-{100 + i}" for i in range(n_ref)],
        "glucose": np.clip(ref_glucose, 45, 240),
        "blood_pressure": np.clip(ref_bp, 40, 130),
        "bmi": np.clip(ref_bmi, 15.0, 55.0),
        "age": ref_age,
        "insulin": ref_insulin,
        "clinic_region": rng.choice(["North", "South", "East", "West"], size=n_ref),
        "standard_protocol": ["V2.4"] * n_ref,
        "high_risk_flag": ref_outcome,
    })

    # Current
    cur_glucose = np.round(rng.normal(138, 35, size=n_current), 1)  # Drift in glucose
    cur_bp = np.round(rng.normal(76, 14, size=n_current), 1)
    cur_bmi = np.round(rng.normal(31.2, 6.1, size=n_current), 1)
    cur_age = rng.integers(21, 80, size=n_current)
    cur_insulin = np.round(rng.exponential(110, size=n_current) + 20, 1)
    cur_prob = 1 / (1 + np.exp(-(-4.2 + 0.03 * cur_glucose + 0.02 * cur_bp + 0.05 * cur_bmi + 0.02 * cur_age)))
    cur_outcome = rng.binomial(1, np.clip(cur_prob, 0.01, 0.99))

    cur_df = pd.DataFrame({
        "patient_id": [f"PAT-CUR-{500 + i}" for i in range(n_current)],
        "glucose": np.clip(cur_glucose, 40, 350),
        "blood_pressure": np.clip(cur_bp, 35, 150),
        "bmi": np.clip(cur_bmi, 14.0, 65.0),
        "age": cur_age,
        "insulin": cur_insulin,
        "clinic_region": rng.choice(["North", "South", "East", "West"], size=n_current),
        "standard_protocol": ["V2.4"] * n_current,
        "sample_barcode": [f"BC-{rng.integers(100000, 999999)}" for _ in range(n_current)],
        "high_risk_flag": cur_outcome,
    })

    # Missing values & extreme physiology values
    cur_df.loc[rng.choice(n_current, size=int(0.10 * n_current), replace=False), "insulin"] = np.nan
    cur_df.loc[rng.choice(n_current, size=int(0.07 * n_current), replace=False), "bmi"] = np.nan
    # Extreme physiology
    outliers = rng.choice(n_current, size=4, replace=False)
    cur_df.loc[outliers[0], "glucose"] = 999.0
    cur_df.loc[outliers[1], "blood_pressure"] = 320.0
    # Duplicates
    cur_df = pd.concat([cur_df, cur_df.iloc[3:8].copy()], ignore_index=True)
    # Label noise
    flip_idx = rng.choice(len(cur_df), size=int(0.06 * len(cur_df)), replace=False)
    cur_df.loc[flip_idx, "high_risk_flag"] = 1 - cur_df.loc[flip_idx, "high_risk_flag"]

    return cur_df, ref_df


SAMPLE_DATASETS = {
    "Customer Churn & Billing": generate_churn_datasets,
    "Financial Transactions & Fraud": generate_transaction_datasets,
    "Healthcare Patient Diagnostics": generate_healthcare_datasets,
}


def get_available_presets() -> list[str]:
    """Return list of sample preset names."""
    return list(SAMPLE_DATASETS.keys())


def load_preset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate preset current and reference datasets."""
    generator = SAMPLE_DATASETS.get(name, generate_churn_datasets)
    return generator()
