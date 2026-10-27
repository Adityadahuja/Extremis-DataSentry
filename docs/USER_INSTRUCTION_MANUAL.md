# Extremis DataSentry: User Instruction Manual
**Enterprise AI Tabular Data Manager, Analyzer, Correction & Reliability Platform**  
*ToltIQ-Inspired SaaS Architecture | Version 1.0.0*

---

## 1. Product Overview
**Extremis DataSentry** is an all-in-one tabular data platform engineered to inspect, diagnose, correct, and validate tabular datasets before they enter Machine Learning pipelines.

Unlike static data tools, DataSentry acts as a **Data Manager**, **Quality Analyzer**, **Data Correction Studio**, and **AI-Based Suggestion Provider**, proving the value of clean data on an **untouched holdout test benchmark**.

---

## 2. Getting Started & Launching

### Quick Local Launch
In your Windows Command Prompt or PowerShell:
```cmd
run_app.bat
```
*(or double-click `run_app.bat` in File Explorer)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 3. UI Navigation & Top-Right Settings

### Top-Right Settings & Customization (⚙️)
To keep the interface clean, all system settings are consolidated in the **⚙️ Settings & System** button at the **top right corner**:
- **Interface Palette:** Instant toggle between **🌙 Dark Mode** and **☀️ Light Mode** (WCAG AAA contrast compliant).
- **Isolation Forest Contamination:** Control the proportion of records flagged as anomaly candidates.
- **Random State Seed:** Fixed seed (default: `42`) ensuring 100% reproducible results.
- **Label Noise Confidence Threshold:** Minimum classifier certainty required to flag candidate annotation errors.
- **Cross-Validation Folds:** Number of CV splits (3, 5, or 10) for out-of-fold probabilistic evaluation.
- **Reset Session State:** 1-click memory reset to start an audit from scratch.

---

## 4. Workspaces Walkthrough

### 1. 📊 Executive Overview
- **KPI Cards:** Scanned rows, feature count, missingness percentage, duplicate row count, and objective 0–100 Health Score.
- **Visual Analytics:** Interactive horizontal missingness bar chart and feature type distribution donut chart.
- **Prioritized Findings Queue:** Filterable queue categorizing all detected defects by severity rank (Critical, High, Medium, Low, Review).

### 2. 📁 Data Manager
- **Global Search:** Type any text or number in the search bar to filter rows across all columns.
- **Column-Specific Filter:** Filter by exact categories or isolate rows where a specific column is `(Missing / Null)`.
- **Column Metadata Inspector:** Select any feature to view its data type, uniqueness ratio, descriptive statistics (mean, std, min, max), and sample values.

### 3. 🔍 Quality & Schema Analyzer
- **Full Column Schema Table:** Detailed breakdown of missing counts, null percentages, distinct counts, constant column flags, and quasi-constant flags.
- **Feature Distribution Explorer:** Select any numeric feature to render an interactive histogram with a marginal boxplot.
- **Pearson Correlation Heatmap:** Color-coded matrix showing pairwise linear relationships between numeric predictors.

### 4. 🎯 Anomaly Detection Studio
- **Unsupervised Isolation Forest:** Recursively partitions multidimensional observations in random decision trees.
- **Key Driver Explainability:** For each flagged anomaly candidate, DataSentry calculates z-score deviations to explain *which* specific feature caused the row to be anomalous.
- **2D PCA Projection:** Projects all dimensions onto orthogonal principal axes to visually inspect how outliers cluster relative to the normal population.
- **Export:** 1-click download of all flagged candidates as CSV.

### 5. 🌊 Distribution Drift Monitor
- **Multi-Metric Drift Screening:**
  - **Cohen's d (Standardized Mean Difference):** Measures mean displacement divided by pooled standard deviation.
  - **Kolmogorov-Smirnov Test:** Non-parametric two-sample test measuring cumulative distribution divergence.
  - **Population Stability Index (PSI):** Quantile-binned divergence metric standard in financial credit risk.
- **Visual Distribution Overlay:** Overlays histograms of reference vs current datasets for any shared feature.

### 6. 🧠 AI Label Quality & ML Benchmark
- **Confident Learning:** Calculates Out-Of-Fold (OOF) cross-validation class posterior probabilities to identify mislabeled training records.
- **ML Reliability Benchmark:** Trains identical model architectures (Random Forest or Logistic Regression) on raw dirty data versus remediated data and evaluates both on an **exact, untouched test partition**.
- **Proof of Value:** Generates a side-by-side comparison table showing accuracy, F1, precision, recall, and ROC-AUC gains, alongside confusion matrix heatmaps.

### 7. 🛠️ Data Correction Studio
- **Non-Destructive Guarantee (FR-09):** The uploaded original CSV is never modified or overwritten.
- **Interactive Remediation Recipe:**
  - Deduplicate identical records.
  - Impute missing numeric values using **median**, **mean**, or **zero**.
  - Impute missing categorical values using **mode** or **"Missing"**.
  - Prune zero-variance constant columns.
  - Drop high-cardinality ID features.
  - Winsorize / clip extreme numeric outliers into 1st–99th percentile bounds.
  - Filter confirmed Isolation Forest anomaly candidates.
- **Live Preview & Downloads:**
  - View before vs after shape transformations and step-by-step audit logs.
  - Download sanitized dataset as CSV.
  - Download findings log as CSV.
  - Download machine-readable audit report as JSON.

### 8. 🤖 AI Suggestion Provider
- Synthesizes all audit signals into human-readable diagnostic cards.
- **For each finding, the AI Advisor provides:**
  - **💡 Root-Cause Hypothesis:** Why this issue occurred in the data engineering pipeline.
  - **📉 Downstream ML Impact:** How this defect will damage model weights, loss functions, and inference.
  - **🛠️ Recommended Action:** The statistically sound strategy to remediate the defect.
  - **AI Confidence Score:** Probabilistic confidence rating (85%–99%).

---

## 5. Frequently Asked Questions (FAQ)

**Q: Are flagged anomalies guaranteed to be errors?**  
*A:* No. An anomaly indicates a statistically unusual record. In fraud detection or healthcare, an anomaly may represent a critical rare event. Always inspect key drivers in Anomaly Review before filtering.

**Q: Does DataSentry send my data to external cloud APIs?**  
*A:* No. DataSentry executes 100% locally on your machine. No telemetry or dataset contents are sent to external servers.

**Q: Can I use this for non-English datasets?**  
*A:* Yes. DataSentry's CSV parser includes automatic fallback encoding support (`UTF-8` and `Latin-1`).
