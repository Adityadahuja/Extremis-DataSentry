# Extremis DataSentry: Industrial Training Technical Report
**Project Title:** AI-Powered Tabular Dataset Quality Auditing & Machine Learning Reliability Platform  
**Candidate Domain:** Machine Learning Engineering & Data Quality Operations (Data-Centric AI)  
**Training Duration:** 45 Days | **Type:** AI/ML Industrial Training | **Version:** 1.0.0  

---

## Abstract
In contemporary Applied Machine Learning, the dominant failure mode of production systems stems not from sub-optimal model architectures, but from corrupted, drifted, or noisy training data. Data problems such as unhandled missingness mechanisms, duplicated observations, zero-variance constant features, high-cardinality keys, multi-feature outliers, label annotation errors, and covariate distribution shifts silently degrade model generalization. 

**Extremis DataSentry** is an industrial-grade, local-first dataset quality auditing platform engineered to diagnose, quantify, and remediate these defects prior to model deployment. DataSentry profiles tabular data, computes an objective penalized Health Score (0–100), isolates multi-feature numeric anomalies via Isolation Forests with fixed-seed reproducibility, screens distribution drift using Standardized Mean Difference (Cohen's d), Kolmogorov-Smirnov statistics, and Population Stability Index (PSI), identifies label noise through Out-Of-Fold (OOF) cross-validation probabilities, and empirically demonstrates the value of data cleaning on an **untouched holdout benchmark**. Operating entirely on local hardware without paid third-party APIs, DataSentry provides a robust framework for reliable Data-Centric AI.

---

## Table of Contents
1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [Project Objectives and Scope](#2-project-objectives-and-scope)
3. [System Architecture and Component Design](#3-system-architecture-and-component-design)
4. [Mathematical Formulations and Algorithmic Deep Dive](#4-mathematical-formulations-and-algorithmic-deep-dive)
5. [Implementation Details](#5-implementation-details)
6. [Empirical Evaluation and ML Reliability Experiments](#6-empirical-evaluation-and-ml-reliability-experiments)
7. [45-Day Training Schedule and Milestones Log](#7-45-day-training-schedule-and-milestones-log)
8. [Known Limitations, Ethical Considerations, and Future Scope](#8-known-limitations-ethical-considerations-and-future-scope)
9. [Viva Voce Examination Questions and Comprehensive Answers](#9-viva-voce-examination-questions-and-comprehensive-answers)

---

## 1. Introduction and Problem Statement

### 1.1 The Data-Centric AI Paradigm
For decades, machine learning research focused almost exclusively on model-centric development: fixing the dataset and iteratively searching for hyperparameter configurations or neural architectures. In industrial settings, however, real-world data is dynamic, noisy, and subject to upstream schema mutations and pipeline failures.

Andrew Ng and the Data-Centric AI movement emphasized that holding the model constant while systematically improving data quality yields substantially higher performance gains than model tuning.

### 1.2 The Problem
Organizations face severe risks when dirty data enters ML pipelines:
- **Silent Degradation:** Models train without throwing exceptions, but produce skewed predictions in production.
- **Biased Sampling:** Undetected duplicates artificially inflate cross-validation scores while degrading generalization.
- **Collinearity and Wasted Compute:** Zero-variance constant features and high-cardinality tracking IDs cause matrix instability and over-parameterization.
- **Covariate Shift:** Discrepancies between training and inference distributions invalidate statistical assumptions.
- **Label Corruption:** Human annotators and automated heuristics introduce noisy ground-truth labels that misguide gradient descent.

DataSentry addresses these challenges through automated local auditing and validation.

---

## 2. Project Objectives and Scope

### 2.1 Core Objectives
1. **Automated Schema & Quality Profiling:** Inspect dimensions, dtypes, missingness rates, duplicate rows, constant columns, and cardinality.
2. **Objective Health Scoring:** Formulate an interpretable 0–100 quality metric with letter grades (A through F).
3. **Unsupervised Anomaly Review:** Rank multidimensional numeric outliers using Isolation Forests with top-contributing z-score explainability and 2D PCA projections.
4. **Statistical Drift Screening:** Quantify feature divergence between current datasets and historical baselines using Cohen's d, KS tests, and PSI.
5. **Label Quality Auditing:** Identify candidate annotation errors using cross-validated prediction probabilities.
6. **Empirical Model Benchmarking:** Benchmark baseline vs cleaned data models on an untouched holdout test split.
7. **Adherence to Non-Destructive Principles (FR-09):** Never silently alter source records.

---

## 3. System Architecture and Component Design

```
                     ┌──────────────────────────────────────────┐
                     │          Input Datasets (CSV)            │
                     │   Current Dataset + Reference Baseline   │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │            datasentry.profiler           │
                     │  - Ingestion & Schema Type Validation    │
                     │  - Missingness & Duplication Penalties   │
                     │  - Constant & Cardinality Detection      │
                     │  - 0–100 Dataset Health Score Engine     │
                     └────────────────────┬─────────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
┌──────────────────┐            ┌───────────────────┐            ┌───────────────────┐
│datasentry.anomaly│            │ datasentry.drift  │            │datasentry.label_qu│
│- Isolation Forest│            │ - Cohen's d (SMD) │            │- Out-of-fold CV   │
│- Z-score drivers │            │ - KS 2-Sample Test│            │- Confident Learn  │
│- PCA 2D scatter  │            │ - PSI Screening   │            │- Discrepancy Rank │
└────────┬─────────┘            └─────────┬─────────┘            └─────────┬─────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │       datasentry.model_experiment        │
                     │  - Untouched Holdout Test Partition      │
                     │  - Baseline vs Cleaned Data Benchmark    │
                     │  - Accuracy, F1, ROC-AUC, Confusion Mat  │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │   datasentry.remediation & exporter      │
                     │  - In-Memory Non-Destructive Preview     │
                     │  - JSON Audit Report Serializer          │
                     │  - Actionable Findings CSV Export        │
                     └──────────────────────────────────────────┘
```

---

## 4. Mathematical Formulations and Algorithmic Deep Dive

### 4.1 Penalized Dataset Health Scoring
The Health Score $H \in [0, 100]$ provides an executive summary of dataset hygiene:
$$H = \max\left(0, \min\left(100, 100 - \lambda_{\text{miss}} R_{\text{miss}} - \lambda_{\text{dup}} R_{\text{dup}} - \lambda_{\text{const}} N_{\text{const}} - \lambda_{\text{crit}} N_{\text{crit}} - \lambda_{\text{high}} N_{\text{high}}\right)\right)$$

Where:
- $R_{\text{miss}} = \frac{\text{missing cells}}{\text{total cells}}$ (weight $\lambda_{\text{miss}} = 120$, capped at 35)
- $R_{\text{dup}} = \frac{\text{duplicate rows}}{\text{total rows}}$ (weight $\lambda_{\text{dup}} = 100$, capped at 25)
- $N_{\text{const}}$ = count of zero-variance columns (weight $\lambda_{\text{const}} = 5$, capped at 15)
- $N_{\text{crit}}, N_{\text{high}}$ = count of Critical and High severity findings.

### 4.2 Isolation Forest Anomaly Detection
Isolation Forest recursively partitions features by selecting a random split value between the minimum and maximum of a randomly chosen feature.
The anomaly score $s(x, n)$ for an instance $x$ across a sample size $n$ is:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
where $h(x)$ is the path length to isolate $x$, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree:
$$c(n) = 2\left(\ln(n - 1) + \gamma\right) - \frac{2(n - 1)}{n}, \quad \gamma \approx 0.5772156649$$

### 4.3 Distribution Drift Screening Metrics
1. **Standardized Mean Difference (Cohen's d):**
$$\text{SMD} = \frac{|\bar{y}_{\text{current}} - \bar{x}_{\text{reference}}|}{\sqrt{\frac{s_{\text{ref}}^2 + s_{\text{cur}}^2}{2}}}$$
$\text{SMD} \ge 0.50$ flags High Shift; $0.20 \le \text{SMD} < 0.50$ flags Moderate Shift.

2. **Two-Sample Kolmogorov-Smirnov Test:**
Evaluates empirical cumulative distributions $F_{\text{cur}}(t)$ and $F_{\text{ref}}(t)$:
$$D = \sup_{t} |F_{\text{cur}}(t) - F_{\text{ref}}(t)|$$

3. **Population Stability Index (PSI):**
Measures shifts across $K$ quantile buckets:
$$\text{PSI} = \sum_{k=1}^{K} (P_k - Q_k) \ln\left(\frac{P_k}{Q_k}\right)$$
$\text{PSI} < 0.10$ signifies stability; $\text{PSI} \ge 0.25$ indicates critical distribution divergence.

### 4.4 Confident Learning Label Quality Screening
For an observed dataset $(X_i, \tilde{y}_i)_{i=1}^N$, Out-Of-Fold (OOF) cross-validation estimates class posterior probabilities $\hat{P}(Y = j \mid X_i)$. The Label Quality Score is defined as:
$$Q_i = \hat{P}(Y = \tilde{y}_i \mid X_i)$$
Samples where $\operatorname{argmax}_j \hat{P}(Y = j \mid X_i) \ne \tilde{y}_i$ with confidence above threshold $\tau$ are prioritized for annotator review.

---

## 5. Implementation Details

- **Language & Runtime:** Python 3.12, local-first execution.
- **Frontend / UI:** Streamlit with custom CSS design tokens (`Space Grotesk`, `DM Sans`, mint accents).
- **Visualization:** Plotly Express and Graph Objects for interactive histograms, marginal boxplots, 2D PCA scatter plots, and confusion matrices.
- **Machine Learning Core:** scikit-learn (`IsolationForest`, `RandomForestClassifier`, `LogisticRegression`, `StandardScaler`, `StratifiedKFold`).
- **Statistical Analytics:** SciPy (`scipy.stats.ks_2samp`).
- **Testing:** Pytest with 16 automated tests covering edge cases, empty datasets, and mathematical bounds.

---

## 6. Empirical Evaluation and ML Reliability Experiments

To validate DataSentry, experiments were conducted on the synthetic **Customer Churn Benchmark**:
- Current dataset: 608 instances (corrupted with $9,999 outliers, duplicate rows, 8% missing age/charges, and 8% flipped labels).
- Holdout test set: 25% untouched clean partition ($N=152$).

### Experimental Results

| Metric | Baseline Model (Dirty Training Data) | Cleaned Data Model (Remediated) | Gain ($\Delta$) | Relative Improvement |
|---|---|---|---|---|
| **Accuracy** | 78.95% | **84.87%** | +5.92% | **+7.5%** |
| **Balanced Accuracy** | 76.12% | **82.35%** | +6.23% | **+8.2%** |
| **Weighted Precision**| 78.50% | **84.70%** | +6.20% | **+7.9%** |
| **Weighted Recall**   | 78.95% | **84.87%** | +5.92% | **+7.5%** |
| **Weighted F1 Score** | 78.61% | **84.75%** | +6.14% | **+7.8%** |
| **ROC-AUC**           | 0.8521 | **0.9084** | +0.0563| **+6.6%** |

**Conclusion:** Removing unhandled duplicates, extreme outliers, and mislabeled training instances improved test set generalization by over 7.5% without modifying the model architecture.

---

## 7. 45-Day Training Schedule and Milestones Log

| Sprints | Phase / Focus | Key Milestones & Deliverables | Status |
|---|---|---|---|
| **Days 1–5** | Scope & Requirements | PRD draft, architecture design, UI wireframes | Completed |
| **Days 6–12** | Profiling Engine | Missingness, duplicate detection, health scoring, schema profiling | Completed |
| **Days 13–19** | Anomaly Detection | Isolation Forest engine, z-score explainability, 2D PCA projection | Completed |
| **Days 20–25** | Distribution Drift | Cohen's d SMD, KS 2-sample test, PSI implementation | Completed |
| **Days 26–32** | Interactive UI & Reports | Streamlit tabs, JSON audit serialization, CSV export | Completed |
| **Days 33–38** | Label Quality & ML Experiment | Out-of-fold label auditing, untouched holdout ML benchmark | Completed |
| **Days 39–42** | Quality Assurance & Testing | Pytest test suite (16 tests), edge-case validation | Completed |
| **Days 43–45** | Documentation & Viva Prep | Industrial report, presentation deck, viva defense prep | Completed |

---

## 8. Known Limitations, Ethical Considerations, and Future Scope

### 8.1 Method Limitations
1. **False Positives in Anomalies:** Isolation Forest flags statistically rare rows. In high-wealth banking or healthcare, rare cases are often valid critical events. Human review is mandatory.
2. **Drift as Screening:** Statistical drift does not inherently imply model degradation; covariate shift may reflect business growth or seasonal patterns.
3. **Local Compute Trade-offs:** Computing OOF cross-validation on millions of rows requires distributed engines (e.g., PySpark or Ray), exceeding lightweight single-node Streamlit execution.

### 8.2 Future Scope
- Integration with Cleanlab open-source library for advanced multi-annotator consensus.
- Automated SQL database connector (PostgreSQL, Snowflake, BigQuery).
- SQLite review history to track dataset version quality over time.

---

## 9. Viva Voce Examination Questions and Comprehensive Answers

**Q1: What is the fundamental difference between model-centric AI and data-centric AI?**  
*Answer:* Model-centric AI treats the training dataset as static and focuses on tuning model architectures, loss functions, and hyperparameters. Data-centric AI holds the model architecture fixed and focuses systematically on improving data quality, fixing label noise, removing uninformative features, and resolving distribution shifts.

**Q2: How does Isolation Forest differ from distance-based outlier detectors like DBSCAN or k-NN?**  
*Answer:* Distance-based methods require computing pairwise Euclidean or Manhattan distances ($O(N^2)$ complexity), which suffers severely from the curse of dimensionality. Isolation Forest builds ensembles of random trees ($O(t \cdot \psi \log \psi)$ complexity), isolating anomalies in fewer partitions because outliers reside in sparse, low-density regions.

**Q3: Why must the ML Reliability experiment be evaluated on an untouched test set?**  
*Answer:* If the test set is cleaned using the same pipeline as the training set, the evaluation becomes biased. Evaluating both the baseline model (trained on dirty data) and the sanitized model (trained on cleaned data) on the exact same untouched holdout split guarantees an honest, scientific measurement of generalization improvement.

**Q4: Explain the difference between Standardized Mean Difference (Cohen's d) and Population Stability Index (PSI).**  
*Answer:* Cohen's d measures the standardized shift in central tendency (difference of means divided by pooled standard deviation). PSI evaluates changes across the entire distribution shape by binning the continuous variable into quantiles and computing Kullback-Leibler-style divergence between reference and current proportions.

**Q5: How does DataSentry adhere to Functional Requirement FR-09?**  
*Answer:* FR-09 mandates that source data must never be silently modified or deleted. DataSentry generates clean datasets strictly in-memory for user preview and export, leaving the original CSV completely immutable.
