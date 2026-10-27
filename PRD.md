# Extremis DataSentry
## Product Requirements Document (PRD)

**Version:** 1.0 | **Duration:** 45 days | **Budget target:** ₹0 | **Type:** AI/ML industrial training

## 1. Executive summary
Extremis DataSentry is a local-first dataset quality auditing platform. It profiles tabular data, highlights potential data-quality issues, detects unusual numeric records, compares a current dataset with a reference dataset, and exports an audit report. It is an educational prototype, not a production data-governance service.

## 2. Problem statement
ML models depend on the quality and stability of their data. Missing values, duplicates, constant features, suspicious outliers, and distribution shifts can quietly damage training and inference. DataSentry makes these potential issues visible before a dataset is used to train a model.

## 3. Goals
- Profile uploaded CSV datasets.
- Identify missing values, duplicate rows, constant columns, and high-cardinality columns.
- Rank unusual numeric rows using Isolation Forest.
- Compare shared numeric features between reference and current datasets.
- Present findings with severity and recommended actions.
- Export findings as CSV and an audit summary as JSON.
- Provide a base for future label-quality checks and before/after model experiments.

## 4. Target users
Students and ML trainees, data analysts, and ML engineers performing first-pass checks on tabular datasets.

## 5. MVP features
### Included
1. CSV upload and validation.
2. Dataset size, column types, missingness, duplicate counts.
3. Constant and high-cardinality column checks.
4. Numeric anomaly candidates via Isolation Forest.
5. Optional reference-vs-current numeric drift screening.
6. Dashboard with KPI cards, charts, schema table, and findings.
7. Downloadable findings CSV and JSON report.
8. Local-first execution without paid APIs.

### Future scope
- Suspicious-label ranking using out-of-fold model predictions and Cleanlab.
- Baseline-vs-cleaned model experiments on the same untouched test set.
- SQLite review history, FastAPI, richer reports, robust drift metrics, dataset versioning.

### Out of scope
- Automatically deleting or silently changing data.
- Treating every anomaly as an error.
- Production monitoring guarantees or regulated data processing.
- Paid APIs and cloud-only services.

## 6. User journey
Upload CSV → profile schema and quality → inspect missingness/duplicates → review anomaly candidates → optionally compare with reference data → download report → verify findings before changing data.

## 7. Technology stack
Python 3.11/3.12, Streamlit, Pandas, NumPy, scikit-learn, Plotly. Future options: Cleanlab, FastAPI, SQLite, SHAP.

## 8. Architecture
CSV input → validation → profiling → quality checks → anomaly detection → optional drift comparison → dashboard → CSV/JSON exports.

## 9. Functional requirements
- **FR-01:** Read CSV and show clear errors for invalid or empty files.
- **FR-02:** Report rows, columns, dtypes, missing counts, duplicates, and unique values.
- **FR-03:** Flag missingness, constant columns, and high-cardinality categoricals.
- **FR-04:** Run Isolation Forest when enough usable numeric data exists.
- **FR-05:** Mark anomalies as review candidates, not confirmed errors.
- **FR-06:** Compare shared numeric columns with a reference dataset.
- **FR-07:** Provide findings table and downloadable CSV.
- **FR-08:** Export aggregate audit summary as JSON.
- **FR-09:** Never silently modify or delete source data.
- **FR-10:** Explain method limitations in the interface.

## 10. Non-functional requirements
- Free local execution, no API key required.
- Reproducible anomaly detection with fixed seed.
- Readable interface and clear error messages.
- Avoid persisting uploaded data outside the running session.
- Reasonable performance for typical training-sized CSVs.

## 11. Methods and caveats
- Missingness: count and percentage by column.
- Duplicates: exact repeated rows.
- Constant columns: one unique non-null value.
- High cardinality: many distinct categorical values relative to row count.
- Anomalies: Isolation Forest candidates across numeric features.
- Drift screening: absolute standardized mean difference for shared numeric features.

These checks can produce false positives. A human should verify findings. A mean shift alone does not establish harmful drift.

## 12. Success criteria
- User can upload a CSV and view the profile without writing code.
- At least four types of potential issues are surfaced.
- Anomaly ranking works for sufficiently large numeric datasets.
- Optional reference/current comparison and exports work.
- README setup steps run in a clean local environment.
- Results and limitations are reported honestly.

## 13. 45-day plan
| Days | Work | Deliverable |
|---|---|---|
| 1–5 | Scope, setup, dataset research, UX plan | Requirements and first prototype |
| 6–12 | Profiling and validation | Quality profile |
| 13–19 | Anomaly detection and issue ranking | Detection module |
| 20–25 | Reference/current drift comparison | Drift view |
| 26–32 | Dashboard refinement and exports | Integrated MVP |
| 33–38 | Label-quality experiment and model comparison | ML reliability experiment |
| 39–42 | Tests, edge cases, report generation | Release candidate |
| 43–45 | Documentation, screenshots, viva | Training submission |

## 14. Risks and mitigation
- False positives: clearly label candidates and require human review.
- Different schemas: compare only shared numeric columns.
- Small datasets: skip or warn when there are too few rows/features.
- Misleading drift signal: describe it as a screening metric.
- Scope creep: complete core profiling, anomalies, drift, and exports first.

## 15. Deliverables
Source code, requirements, PRD, README, sample audit report, screenshots, test cases, known limitations, industrial training report, and presentation.

## 16. References
- Streamlit: https://docs.streamlit.io/
- scikit-learn: https://scikit-learn.org/stable/
- Cleanlab: https://cleanlab.ai/
- Plotly: https://plotly.com/python/
