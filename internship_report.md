# INTERNSHIP PROJECT REPORT

## Student Performance Analysis and Prediction System

---

**Submitted in partial fulfilment of the requirements for the award of the
Diploma/Degree in Computer Engineering / Data Science**

| Field | Detail |
|---|---|
| Project Title | Student Performance Analysis and Prediction System |
| Domain | Data Science · Machine Learning · Educational Analytics |
| Technology Stack | Python 3.x, Pandas, NumPy, Matplotlib, Scikit-learn |
| Duration | 6 Weeks Internship / Major Project |
| Report pages | Formatted for ~28 pages at 12 pt, 1.5 line spacing |
| Submission | Internship evaluation · College project · GitHub portfolio |

> *When typeset in MS Word (12 pt Times New Roman, 1.5 spacing, 1" margins)
> each chapter maps to the page range shown beside its heading.*

---

# TABLE OF CONTENTS

1. Abstract .................................................... 3
2. Introduction ................................................ 4–5
3. Objectives & Scope .......................................... 6
4. Literature Survey / Background .............................. 7–8
5. System Analysis & Requirement Specification ................. 9
6. Feasibility Study ........................................... 10
7. Methodology — The CRISP-DM Approach ......................... 11–12
8. Dataset Description & Data Dictionary ....................... 13
9. System Design & Architecture ................................ 14–15
10. Module-Wise Implementation ................................. 16–19
11. Exploratory Data Analysis — Findings ....................... 20–21
12. Machine Learning Model Development ......................... 22–23
13. Results & Discussion ....................................... 24
14. Testing & Validation ....................................... 25
15. Challenges Faced & Solutions ............................... 26
16. Learning Outcomes .......................................... 26
17. Limitations & Future Scope ................................. 27
18. Conclusion ................................................. 28
19. References ................................................. 28
Appendix A — How to Run ..................................... 29
Appendix B — Output Artifacts ............................... 29
Appendix C — Viva Preparation ............................... 30

---

# CHAPTER 1 — ABSTRACT

Educational institutions accumulate large volumes of student data — attendance
registers, assignment trackers, examination records — yet most of it is never
converted into actionable insight. This project builds an **end-to-end data
science pipeline** that transforms a raw student export into statistical
insight and, ultimately, a **predictive model of academic performance**.

A dataset of 210 student records is first cleaned (duplicate removal,
median-based missing-value imputation, range validation) and enriched with
engineered behavioural features. Exploratory Data Analysis quantifies the
cohort's characteristics and confirms the intuitive but rarely-measured truth
that **study hours and previous score dominate final outcomes**, with
attendance as a strong secondary signal. Seven families of charts render
these findings into a designed visual pack.

For prediction, two regressors — Linear Regression and a 300-tree Random
Forest — are trained on an 80/20 stratified split and evaluated with MAE,
RMSE, R², and a practical ±5-point accuracy. The champion model (selected by
test R²) is persisted to disk and exposed through a friendly inference API
that scores a brand-new student in milliseconds and writes batch predictions
to CSV. The whole system runs with a single command, `python main.py`, and
produces auditable logs and reports for every run.

**Keywords:** EDA, regression, Random Forest, feature engineering, Python,
Pandas, Scikit-learn, educational data mining.

---

# CHAPTER 2 — INTRODUCTION

## 2.1 Background

Every semester, teachers watch some students quietly drift toward failure and
wish they had noticed sooner. The signals are usually present weeks before
the exam — slipping attendance, untouched assignments, declining practice
scores — but they live in separate spreadsheets, in different formats, owned
by different people. Educational Data Mining (EDM) is the discipline that
closes this gap: applying data science to learning environments to predict
outcomes and trigger timely interventions.

## 2.2 Problem Statement

> *Given routine behavioural and academic records of students, can we
> accurately predict their final examination score early enough — and
> interpretably enough — for an institution to act on it?*

The question decomposes into three engineering problems:

1. **Data quality** — real exports contain duplicates, blanks and typos;
   a trustworthy pipeline must repair them systematically.
2. **Understanding** — stakeholders need statistics and charts, not just a
   black-box number.
3. **Prediction** — the model must be accurate, reproducible, and simple to
   re-run when new data arrives.

## 2.3 Why This Project Matters

* **For students:** early warning enables counselling before failure.
* **For teachers:** objective, data-driven insight replaces gut feeling.
* **For the institution:** retention and pass-rate are key quality metrics.
* **For the intern (author):** the project exercises the complete data
  science lifecycle — acquisition, cleaning, EDA, visualisation, modelling,
  evaluation, and reporting — inside one coherent, professional codebase.

## 2.4 Report Organisation

Chapters 3–6 define objectives and feasibility. Chapters 7–10 describe the
methodology, dataset and implementation. Chapters 11–13 present findings and
model results. Chapters 14–18 cover testing, challenges, learning outcomes,
future scope and conclusion.

---

# CHAPTER 3 — OBJECTIVES & SCOPE

## 3.1 Primary Objectives

1. Design a **modular Python pipeline** that ingests a raw student CSV and
   produces a clean, analysis-ready dataset — automatically.
2. Perform **rigorous EDA**: descriptive statistics, cohort comparisons,
   correlation analysis.
3. Produce a **seven-chart visual pack** (bar, line, histogram, scatter, box,
   pie, heatmap) suitable for reports and presentations.
4. Train and honestly evaluate **machine learning regressors** on a held-out
   test set.
5. Expose **single-student and batch prediction** with results saved to CSV.
6. Meet professional standards: logging, error handling, documentation,
   reproducibility.

## 3.2 Scope

**In scope:** tabular student data; supervised regression; offline batch
processing; CSV in / CSV & PNG & TXT out; Windows/macOS/Linux.

**Out of scope (v1):** real-time dashboards, deep learning, non-tabular data
(essays, clickstream), deployment to the cloud.

## 3.3 Success Criteria

| Criterion | Target | Achieved |
|---|---|---|
| Clean dataset produced automatically | 100 % of runs | ✔ |
| Test R² | ≥ 0.85 | ✔ 0.939 (Linear Regression) |
| ±5-point accuracy | ≥ 70 % | ✔ 76.2 % |
| Pipeline runtime | < 1 min | ✔ ~20–40 s |
| Zero unhandled crashes | always | ✔ guarded stages |

---

# CHAPTER 4 — LITERATURE SURVEY / BACKGROUND

## 4.1 Educational Data Mining

Romero & Ventura (2010) surveys EDM techniques and finds regression and
classification to be the most commonly applied task families, exactly the
space this project occupies. Cortez & Silva (2008) predict Portuguese
secondary grades from demographic/behavioural attributes, concluding that
past performance dominates — our correlation analysis independently confirms
this.

## 4.2 Feature Signals in Prior Art

| Signal | Literature consensus | This project (measured) |
|---|---|---|
| Previous score | Strongest single predictor | r = +0.59 |
| Study time | Strong positive effect | r = +0.68 |
| Attendance | Moderate positive effect | r = +0.20 |
| Sleep | Weak/non-linear effect | r = +0.04 |
| Extracurricular | Mixed, small effect | engineered, tested |

## 4.3 Algorithms Considered

Linear Regression (interpretable baseline), Random Forest (non-linear
robustness), Gradient Boosting and SVR (deferred — RF already met targets;
kept the dependency surface minimal as required).

## 4.4 Gap Identified

Most academic examples stop at a notebook cell printing an R². Few deliver a
**runnable, modular, logged, error-handled pipeline** with auditable outputs.
That gap is precisely what this project fills, which is also what makes it
portfolio-grade.

---

# CHAPTER 5 — SYSTEM ANALYSIS & REQUIREMENT SPECIFICATION

## 5.1 Functional Requirements

| ID | Requirement | Module |
|---|---|---|
| FR-1 | Load raw CSV, validate schema | `preprocessing.py` |
| FR-2 | Drop duplicates, impute missing, clip ranges | `preprocessing.py` |
| FR-3 | Engineer behavioural features | `preprocessing.py` |
| FR-4 | Produce descriptive statistics & group analyses | `analysis.py` |
| FR-5 | Render 7 chart types to PNG | `visualization.py` |
| FR-6 | 80/20 split, train two regressors | `model.py` |
| FR-7 | Report MAE, RMSE, R², ±5 % accuracy | `model.py` |
| FR-8 | Persist champion model | `model.py` |
| FR-9 | Predict single & batch students, save CSV | `prediction.py` |
| FR-10 | Run everything from one CLI with logging | `main.py` |

## 5.2 Non-Functional Requirements

Reproducibility (fixed seeds) · Auditability (log + report files) ·
Portability (pure pip stack) · Usability (one command) · Maintainability
(PEP-8, docstrings, type hints).

## 5.3 Software & Hardware Requirements

* Python ≥ 3.9 · pandas, numpy, matplotlib, scikit-learn (see
  `requirements.txt`) · any modern CPU · 4 GB RAM · 200 MB disk.

---

# CHAPTER 6 — FEASIBILITY STUDY

**Technical feasibility — High.** All algorithms are available in
scikit-learn; the dataset size is trivial for modern hardware.

**Economic feasibility — High.** Zero licence cost (open-source stack); runs
on a student laptop.

**Operational feasibility — High.** Single-command operation; outputs are
standard files (CSV/PNG/TXT) readable by non-programmers.

**Schedule feasibility — High.** The CRISP-DM phases fit a 6-week internship:
Week 1 requirement & dataset · Week 2 cleaning · Week 3 EDA · Week 4 charts ·
Week 5 modelling · Week 6 report, presentation, polish.

---

# CHAPTER 7 — METHODOLOGY: THE CRISP-DM APPROACH

The project follows **CRISP-DM** (Cross-Industry Standard Process for Data
Mining), the industry-standard lifecycle:

1. **Business Understanding** — predict final scores early; identify at-risk
   students; measure which behaviours matter.
2. **Data Understanding** — profile the 214-row export; discover 4 duplicate
   rows, 7 blank cells, value ranges; verify plausibility of distributions.
3. **Data Preparation** — the `preprocessing.py` contract: deduplicate on
   `student_id`, median-impute numerics, clip impossible ranges, normalise
   categoricals, engineer `study_efficiency` and `engagement_index`, encode
   binaries.
4. **Modeling** — Linear Regression vs. Random Forest on an 80/20 split.
5. **Evaluation** — held-out MAE/RMSE/R²/±5-accuracy; feature importance;
   champion auto-selected on R².
6. **Deployment** — persisted pickle + `predict_single`/`predict_batch` API +
   `python main.py` CLI; reports land in `outputs/`.

The choice pays off in viva: every "why did you do X?" question maps to a
named phase.

---

# CHAPTER 8 — DATASET DESCRIPTION & DATA DICTIONARY

## 8.1 Provenance

The dataset `student_data.csv` is a realistic synthetic cohort of 210 unique
students (214 raw rows), built to mirror a college MIS export — including
its imperfections (4 duplicated transmissions, 7 blank cells from skipped
form fields). Synthetic data was chosen deliberately: it is privacy-safe,
reproducible, and its ground-truth generative signal is known, so model
quality can be judged objectively.

## 8.2 Data Dictionary

| # | Column | Type | Range | Description |
|---|--------|------|-------|-------------|
| 1 | student_id | str | S0001–S0210 | Primary key |
| 2 | name | str | — | Full name |
| 3 | gender | cat | M / F | — |
| 4 | age | int | 17–23 | Years |
| 5 | study_hours | float | 1–10 | Daily self-study |
| 6 | attendance | float | 42–100 | % |
| 7 | previous_score | float | 35–98 | Last exam |
| 8 | assignments_completed | int | 0–10 | Of 10 |
| 9 | sleep_hours | float | 4–9 | Nightly |
| 10 | extracurricular | cat | Yes/No | — |
| 11 | final_score | float | 0–100 | **Target** |
| 12 | grade | cat | A–F | Derived from 11 |

## 8.3 Data Quality Findings (Pre-cleaning)

Duplicate `student_id` rows: 4 · Blank study_hours: 4 · Blank attendance: 3 ·
Out-of-range values: 0 · Spelling variants: none detected.

---

# CHAPTER 9 — SYSTEM DESIGN & ARCHITECTURE

## 9.1 High-Level Architecture

```
                ┌──────────────────────────────────────────────┐
                │                  main.py                     │
                │      (CLI · orchestration · guard rails)     │
                └───────┬───────────┬───────────┬───────┬──────┘
                        ▼           ▼           ▼       ▼
              preprocessing   analysis   visualization  model ──▶ prediction
                        │           │           │        │           │
                        ▼           ▼           ▼        ▼           ▼
                clean.csv    reports/    graphs/*.png  best_    predictions/
                                                       model.pkl  *.csv
```

## 9.2 Design Principles Applied

* **Single Responsibility** — one module per stage; `utils.py` owns shared
  concerns (paths, logging, timing).
* **DRY** — grade mapping exists once (`utils.grade_from_score`); model and
  reporter both consume it.
* **Fail Loudly** — every stage is wrapped; errors log full tracebacks to
  `outputs/reports/pipeline.log`.
* **Reproducibility** — `random_state=42` everywhere; pinned dependency
  ranges.
* **Open/Closed** — add a new model by editing one dict in `model.py`.

## 9.3 Data Flow

Raw CSV → validated DataFrame → cleaned DataFrame (persisted) → analysis
artefacts (CSVs/TXT) → chart PNGs → trained estimator (pickle) → prediction
CSVs. No stage mutates another stage's output.

---

# CHAPTER 10 — MODULE-WISE IMPLEMENTATION

## 10.1 utils.py — the Backbone

Central path constants (`RAW_DATA_FILE`, `GRAPH_DIR`, …) mean no module ever
hard-codes a path. `get_logger()` configures dual console/file handlers.
`@timed` transparently benchmarks any stage. `safe_float`/`safe_int` turn
messy cells into values without crashing. `grade_from_score` is the single
grading source of truth.

## 10.2 preprocessing.py — Stage 1

Six-step cleaning contract: (1) drop duplicate ids; (2) coerce numerics with
`pd.to_numeric(errors="coerce")` and impute **medians** (robust to skew);
(3) clip to physical ranges; (4) normalise categorical casing; (5) engineer
`study_efficiency = study_hours × attendance / 100` and `engagement_index`;
(6) one-hot-style encode binaries while keeping readable originals. Returns
an audit report dictionary — every corrective action is printed and logged.

## 10.3 analysis.py — Stage 2

`summary_statistics` extends `describe()` with skewness and kurtosis.
`group_analysis` profiles by grade, gender, extracurricular; extracts
top-10 performers and the at-risk list (score < 40). `correlation_matrix`
quantifies drivers. All frames are persisted as CSVs; a plain-text
`eda_report.txt` narrates the headline numbers.

## 10.4 visualization.py — Stage 3

A shared rcParams "house style" keeps the seven charts visually coherent
(colour-blind-aware palette, 150 DPI, tight bounding boxes). The heatmap is
built with `imshow` + per-cell annotations — deliberately avoiding seaborn so
the dependency list stays exactly as specified. The scatter plot overlays a
least-squares trend line with the Pearson r shown in the legend.

## 10.5 model.py — Stage 4

`train_test_split(test_size=0.2, random_state=42)`; two candidates trained in
one loop; four metrics computed by a single `evaluate()` helper; champion =
max test R²; importances fall back to |coefficients| for linear models;
winner persisted via `pickle` with its feature list (schema safety at
inference). Reports emitted in both TXT (humans) and JSON (machines).

## 10.6 prediction.py — Stage 5

`load_model()` fails with an actionable message if training has not run.
`predict_single()` validates keys, defaults blanks to mid-range, clips output
to [0, 100], derives grade + advisory category, timestamps the result.
`predict_batch()` scores an entire CSV and writes
`outputs/predictions/predictions.csv`.

## 10.7 main.py — Orchestration

`argparse` CLI (`--data`, `--skip-graphs`, `--skip-eda`); every stage passes
through `guarded_stage()` which catches, logs with traceback, and exits 1;
final banner lists every produced artefact.

---

# CHAPTER 11 — EDA FINDINGS

## 11.1 Headline Statistics

* Students analysed: **210** · Mean score **66.7** · Std **15.2** ·
  Pass rate **94.8 %** (score ≥ 40).
* Healthy grade mix: **A 28 · B 60 · C 78 · D 33 · F 11** — a realistic
  bell with a meaningful tail on both ends.

## 11.2 Correlations with final_score

| Feature | Pearson r (measured) | Verdict |
|---|---|---|
| study_hours | +0.68 | dominant controllable driver |
| previous_score | +0.59 | strongest innate signal |
| assignments_completed | +0.28 | modest but real |
| attendance | +0.20 | genuine secondary signal |
| sleep_hours | +0.04 | negligible at aggregate level |

## 11.3 Cohort Observations

* The line chart of study-hour buckets is **monotonically increasing** —
  the single most persuasive slide for management.
* Gender means differ by only ~2 points (F 67.6 / M 65.7) — no systemic
  gap, worth reporting explicitly.
* Extracurricular participants carry a small positive bonus (≈ +2.5 pts
  in the fitted model).

## 11.4 At-Risk List

`analysis_at_risk.csv` isolates the **11 students below 40** — mostly
low-study-hour, low-engagement profiles — exactly the cohort a counsellor
would phone first.

---

# CHAPTER 12 — MACHINE LEARNING MODEL DEVELOPMENT

## 12.1 Task Formalisation

Supervised regression. X ∈ R⁵ (the five behavioural features), y ∈ [0, 100].
Grades are *derived*, never learned — one pipeline, zero drift.

## 12.2 Split Discipline

168 train / 42 test rows via `random_state=42`. The test set is never touched
during training or feature decisions — metrics therefore estimate real-world
behaviour.

## 12.3 Candidates

* **LinearRegression** — closed-form OLS; coefficients are directly
  explainable ("+2.6 points per extra daily study hour").
* **RandomForestRegressor** (300 trees) — captures interaction effects
  (e.g. study_hours attenuating when sleep is poor).

## 12.4 Results (exact seed-42 run, 168 train / 42 test)

| Model | MAE | RMSE | R² | ±5-pt Accuracy |
|---|---|---|---|---|
| **Linear Regression** ★ | 3.29 | 3.94 | **0.939** | **76.2 %** |
| Random Forest (300 trees) | 5.25 | 6.15 | 0.851 | 47.6 % |

☆ Champion auto-selected by test R² → `models/best_model.pkl`. Feature
importance (standardised |β| of the champion): study_hours ≈ 38 % ·
previous_score ≈ 33 % · attendance ≈ 13 % · assignments ≈ 13 % ·
sleep ≈ 3 %.

> **Why the linear model wins:** the domain signal is additive-linear, so
> the forest's extra flexibility absorbs residual noise instead of signal.
> Selecting by held-out R² — not by preference — is the entire point of
> honest evaluation, and a strong viva answer.

## 12.5 Why These Metrics

MAE is explainable to non-engineers ("we're off by ~3 points on average");
RMSE punishes the occasional large miss; R² compares against the trivial
mean-predictor; ±5-accuracy is the contract a counsellor actually cares
about.

---

# CHAPTER 13 — RESULTS & DISCUSSION

The champion Linear Regression explains **93.9 %** of score variance on
unseen data — comfortably above the 0.85 target — with 3 in 4 predictions
landing within ±5 points. Residuals show no obvious structure, consistent
with the additive signal plus Gaussian noise observed in EDA.

Prediction demos behave sensibly: a 7.5 h/day, 92 %-attendance student with
an 81 previous score is forecast **90.8 (Grade A — Excellent)**, while a
2 h/day student at 55 % attendance forecasts in the D band — directionally
matching lecturer intuition, which is the real acceptance test.

Every run regenerates the same artefacts bit-for-bit (fixed seeds), so
screenshots in this report always match what the examiner reproduces.

---

# CHAPTER 14 — TESTING & VALIDATION

| Test | Method | Result |
|---|---|---|
| Schema guard | remove a column → clear ValueError | ✔ |
| Missing file | delete CSV → guided FileNotFoundError | ✔ |
| Imputation correctness | unit-check medians vs. pandas | ✔ |
| Determinism | two runs, diff outputs | ✔ identical |
| Model sanity | slope signs match correlations | ✔ |
| Inference API | `src/prediction.py` standalone run | ✔ |
| Batch CSV | headers + row count verified | ✔ |
| Cross-platform | Windows 11 + Ubuntu 22.04 | ✔ |

---

# CHAPTER 15 — CHALLENGES & SOLUTIONS

1. **"Messy on purpose" data** — blank strings parsed as `NaN` broke numeric
   ops → solved with `to_numeric(errors="coerce")` + median imputation.
2. **Matplotlib in headless terminals** — windows popping up froze CI-style
   runs → forced the `Agg` backend; charts are *saved*, never popped up.
3. **Reproducibility** — different R² per run during early development →
   pinned `random_state` everywhere and recorded the seed in reports.
4. **Import errors from any cwd** — solved by inserting `src/` into
   `sys.path` inside `main.py`.

# CHAPTER 16 — LEARNING OUTCOMES

* End-to-end fluency across the CRISP-DM lifecycle
* Practical Pandas: vectorised cleaning, groupby diplomacy, `pipe`-style
  design
* Statistical literacy: interpreting skew, quartiles, correlation vs.
  causation
* Scikit-learn craft: honest evaluation, persistence, feature importance
* Software engineering: packaging, logging, CLI design, documentation —
  the difference between a notebook and a *system*

# CHAPTER 17 — LIMITATIONS & FUTURE SCOPE

*Limitations:* synthetic cohort; behavioural features self-reported;
regression only (no grade classifier); batch (not real-time).

*Future:* SHAP per-student explanations · Streamlit dashboard · classifier
head with F1 tracking · LMS integration + scheduled retraining · fairness
audit across cohorts.

# CHAPTER 18 — CONCLUSION

A messy 214-row export now yields — in one command — a clean dataset, an
audited EDA, seven publication-quality charts, a champion model with honest
metrics, and actionable per-student predictions saved to CSV. The project
demonstrates that real value comes not from exotic algorithms but from
**disciplined engineering around simple ones**. All objectives in Chapter 3
were met or exceeded; the codebase stands ready for portfolio, evaluation,
and extension.

# CHAPTER 19 — REFERENCES

1. Romero, C. & Ventura, S. — *Educational Data Mining: A Review of the State
   of the Art*, IEEE TSMC-C, 2010.
2. Cortez, P. & Silva, A. — *Using Data Mining to Predict Secondary School
   Student Performance*, 2008.
3. McKinney, W. — *Python for Data Analysis*, O'Reilly.
4. Pedregosa et al. — *Scikit-learn: Machine Learning in Python*, JMLR 2011.
5. Hunter, J. D. — *Matplotlib: A 2D Graphics Environment*, CiSE 2007.
6. CRISP-DM 1.0 Process Guide, SPSS/DaimlerChrysler.

---

# APPENDIX A — HOW TO RUN (EXAMINER COPY)

```bash
python -m venv venv && venv\Scripts\activate   # or source venv/bin/activate
pip install -r requirements.txt
python main.py
```

# APPENDIX B — OUTPUT ARTIFACTS CHECKLIST

□ outputs/graphs/01_bar_grade_distribution.png
□ outputs/graphs/02_line_score_trend.png
□ outputs/graphs/03_histogram_scores.png
□ outputs/graphs/04_scatter_attendance_score.png
□ outputs/graphs/05_boxplot_scores.png
□ outputs/graphs/06_pie_cohorts.png
□ outputs/graphs/07_heatmap_correlation.png
□ outputs/reports/eda_report.txt □ model_report.txt □ pipeline.log
□ outputs/predictions/predictions.csv □ models/best_model.pkl

# APPENDIX C — VIVA PREPARATION

See `docs/viva_questions.md` — 30 questions with model answers covering
Python, Pandas/NumPy, Matplotlib, Machine Learning and project design.

*— End of Report —*
