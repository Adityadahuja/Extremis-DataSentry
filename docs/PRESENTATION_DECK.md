# Extremis DataSentry: Project Presentation Deck Outline

**Project Title:** Extremis DataSentry — Local-First Tabular Dataset Quality Auditing & ML Reliability Platform  
**Target Audience:** Industrial Training Evaluation Committee, Technical Evaluators, and Viva Examiners  
**Estimated Presentation Duration:** 15–20 minutes  

---

## Slide 1: Title & Introduction
- **Headline:** Extremis DataSentry
- **Subtitle:** AI-Powered Dataset Quality Auditing & Machine Learning Reliability Platform
- **Tagline:** *Make your data worthy of your model.*
- **Presenter Name:** Machine Learning Engineering Trainee
- **Key Talking Points:**
  - Introduce the shift from model-centric to data-centric AI.
  - Highlight the core premise: 80% of machine learning pipeline failures originate in corrupted, shifted, or unvalidated data.

---

## Slide 2: The Problem: The High Cost of Dirty Data
- **Key Bullet Points:**
  - **Silent Model Degradation:** Machine learning algorithms do not crash on dirty data; they optimize on noise and fail in production.
  - **Data Defects in Tabular Data:**
    - High missingness rates and improper imputation.
    - Exact duplicate records biasing cross-validation.
    - Zero-variance constant columns wasting memory and compute.
    - High-cardinality tracking IDs causing overfitting.
    - Subtle multidimensional outliers corrupting loss gradients.
    - Label noise and annotation mistakes.
    - Covariate distribution drift between training and inference.
  - **Existing Tools Gap:** Existing enterprise tools require cloud infrastructure, monthly subscriptions, or complex setup.

---

## Slide 3: Solution & System Vision
- **Key Bullet Points:**
  - **Extremis DataSentry:** A local-first, zero-dependency data quality auditor.
  - **Core Value Proposition:**
    - Fast CSV profiling with an interpretable 0–100 Health Score.
    - Unsupervised anomaly detection via Isolation Forest.
    - Covariate drift screening against historical baselines.
    - Confident learning label quality screening.
    - Downstream ML reliability proof on an untouched test set.
    - 100% free, runs locally, zero cloud or paid API lock-in.

---

## Slide 4: Architectural Flow
- **Visual Diagram:** Ingestion ➔ Profiler ➔ Anomaly Review ➔ Drift Monitor ➔ Label Quality & ML Experiment ➔ Remediation & Export.
- **Key Modules:**
  - `datasentry.profiler`: Schema, uniqueness, missingness, health score.
  - `datasentry.anomaly`: Isolation Forest, z-score explainability, 2D PCA projection.
  - `datasentry.drift`: Cohen's d (SMD), Kolmogorov-Smirnov test, Population Stability Index (PSI).
  - `datasentry.label_quality`: Out-of-fold probabilistic confident learning.
  - `datasentry.model_experiment`: Untouched holdout benchmark.
  - `datasentry.remediation`: Non-destructive preview and export.

---

## Slide 5: Dataset Profiling & Objective Health Score
- **Key Bullet Points:**
  - Computes dimensions, memory consumption, missing cells %, duplicate rows %.
  - Identifies constant (zero-variance) and high-cardinality categorical features.
  - **Health Score Formulation:**
    $$H = 100 - \text{Penalties}(\text{Missing}, \text{Duplicates}, \text{Constants}, \text{Critical Issues})$$
  - Outputs letter grade (A through F) with actionable recommendations.

---

## Slide 6: Multi-Dimensional Anomaly Detection (Isolation Forest)
- **Key Bullet Points:**
  - Recursively isolates numeric observations in random partition trees.
  - Anomalous points require fewer tree splits ($\mathbb{E}(h(x)) \ll c(n)$).
  - **Key Innovations in DataSentry:**
    - **Explainability:** Calculates feature z-scores to explain *which* variable drove the anomaly.
    - **2D PCA Projection:** Projects all numeric features onto orthogonal principal axes for visual inspection.
    - **Ethical Caveat (FR-05):** Flagged rows are review candidates, not confirmed errors.

---

## Slide 7: Reference vs Current Distribution Drift
- **Key Bullet Points:**
  - Essential for monitoring concept drift and covariate shift in production.
  - Evaluates shared numeric features across three complementary lenses:
    1. **Cohen's d (SMD):** Standardized mean shift relative to pooled variance.
    2. **Two-Sample KS Test:** Non-parametric test of empirical cumulative distribution divergence.
    3. **Population Stability Index (PSI):** Quantile-binned divergence metric standard in financial risk.
  - Interactive distribution overlay plots (reference vs current).

---

## Slide 8: Label Quality Auditing (Confident Learning)
- **Key Bullet Points:**
  - Human annotation errors misguide gradient descent.
  - Employs Stratified K-Fold Cross-Validation:
    - Fits model on out-of-fold partitions.
    - Predicts class probabilities for holdout samples.
  - Compares model confidence against observed label.
  - Identifies flipped, noisy, or ambiguous ground truth annotations.

---

## Slide 9: ML Reliability Experiment: The Proof
- **Scientific Methodology:**
  - Split dataset into training pool and an **untouched clean test set** (25%).
  - Train baseline model on dirty training data (with duplicates, outliers, label noise).
  - Train cleaned data model on remediated training data.
  - Evaluate both on the exact same untouched test set.
- **Experimental Findings (Customer Churn Benchmark):**
  - **Accuracy:** Improved from 78.9% to **84.9%** (+7.5% relative).
  - **F1 Score:** Improved from 78.6% to **84.8%** (+7.8% relative).
  - **ROC-AUC:** Improved from 0.852 to **0.908** (+6.6% relative).
  - Proves that data cleaning directly enhances model generalization!

---

## Slide 10: Non-Destructive Remediation & Export (FR-07, FR-08, FR-09)
- **Key Bullet Points:**
  - **Data Integrity Guarantee (FR-09):** Never silently overwrite source CSV files.
  - Interactive recipe builder: deduplication, median/mean imputation, constant column pruning, outlier filtering.
  - Downloads:
    - Sanitized dataset CSV.
    - Prioritized findings CSV.
    - Machine-readable JSON audit summary.

---

## Slide 11: Verification & Testing
- **Key Bullet Points:**
  - Automated test suite built with `pytest` (16 passing tests).
  - Validates edge cases: empty datasets, zero numeric features, identical distributions, extreme outliers, label flips, JSON serialization.
  - Fast execution time: full suite executes in < 4 seconds.

---

## Slide 12: Conclusion & Viva Takeaways
- **Key Summary Points:**
  - Extremis DataSentry bridges the gap between raw CSV data and reliable ML deployments.
  - Combines profiling, anomaly detection, statistical drift, and label auditing into a unified local dashboard.
  - Demonstrates that high data quality delivers superior model performance.
- **Closing Statement:** *"Make your data worthy of your model."*
- **Q&A Invitation:** Open for questions from the evaluation committee.
