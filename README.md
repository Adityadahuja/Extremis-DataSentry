# ◈ Extremis DataSentry Enterprise
> **Local-First Tabular Dataset Quality Auditing & ML Reliability Platform**  
> *Make your data worthy of your model.*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E.svg)](https://scikit-learn.org/)
[![Tests: Pytest](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)](https://pytest.org/)

---

## 1. Executive Summary
**Extremis DataSentry Enterprise** is an industrial-grade, local-first tabular data platform inspired by **[ToltIQ](https://toltiq.com/)**. It acts as a comprehensive **Data Manager**, **Quality & Schema Analyzer**, **Data Correction Studio**, and **AI-Based Suggestion Provider**, proving the value of clean data on an **untouched holdout benchmark**.

---

## 2. Key Architecture & 10 Core Workspaces

DataSentry automatically pre-loads an industrial benchmark on launch so **all 10 workspaces are immediately active and interactive** without requiring any manual setup:

1. **📊 Executive Overview:** Executive KPI cards (Rows, Features, Missingness, Duplicates, Health Score A–F), missingness chart, column mix pie, and prioritized findings queue.
2. **📁 Data Manager:** Real-time global text search across all cells, column-specific filtering, table sorting, and single-column metadata inspector.
3. **🔍 Quality & Schema Analyzer:** Column profile table, numerical descriptive distributions, and Pearson correlation heatmap.
4. **🎯 Anomaly Detection Studio:** Unsupervised Isolation Forest, z-score explainability key drivers, 2D PCA cluster projection, and candidate CSV export.
5. **🌊 Distribution Drift Monitor:** Cohen's d (SMD), Kolmogorov-Smirnov test, Population Stability Index (PSI), and distribution overlay histograms.
6. **🧠 AI Label Quality & ML Benchmark:** Confident learning label error detection and before/after ML benchmark on an **untouched test split** proving model accuracy/F1 gains.
7. **🛠️ Data Correction Studio:** Non-destructive interactive data cleansing: deduplication, median/mean imputation, outlier Winsorization, constant column removal, and sanitized CSV download.
8. **🤖 AI Suggestion Provider:** Automated diagnostic cards with root-cause hypotheses, downstream ML impact, and 1-click recommendations.
9. **📥 Data Ingestion & Sources (Dedicated Workspace):** Full-screen studio to switch between 1-click benchmarks (Customer Churn, Financial Transactions, Healthcare) or upload custom CSV datasets.
10. **📖 User Instruction Manual:** Complete in-app operational guide and documentation.

---

## 3. Top-Right Settings & Customization (⚙️)
To maintain an uncluttered workspace, all configuration settings are located in the **⚙️ Settings** button at the **top right corner**:
- **Interface Palette:** 1-click toggle between **🌙 Dark Mode** and **☀️ Light Mode** (WCAG AAA compliant).
- **Isolation Forest Contamination:** Slider to tune outlier sensitivity.
- **Random State Seed:** Fixed seed for reproducible audits.
- **Label Noise Discrepancy Threshold:** Minimum classifier certainty required to flag candidate annotation errors.
- **Cross-Validation Folds:** Number of CV splits (3, 5, or 10).
- **Reset Session State:** 1-click memory reset.

---

## 4. Quickstart Guide (Local Execution)

In your Command Prompt or PowerShell:
```cmd
# 1. Run tests (18 tests passing)
run_tests.bat

# 2. Launch the Web Application
run_app.bat
```
*(or simply double-click `run_app.bat` in File Explorer)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 5. Vercel & Cloud Deployment
- **Vercel Deployment:** See [`docs/VERCEL_DEPLOYMENT_GUIDE.md`](file:///c:/Users/adity/Downloads/Extremis%20DataSentry/docs/VERCEL_DEPLOYMENT_GUIDE.md) for deploying via Stlite (WebAssembly) or Vercel CLI.
- **User Instruction Manual:** See [`docs/USER_INSTRUCTION_MANUAL.md`](file:///c:/Users/adity/Downloads/Extremis%20DataSentry/docs/USER_INSTRUCTION_MANUAL.md).

---

## 6. License
Distributed under the MIT License. Developed for AI/ML Industrial Training.
