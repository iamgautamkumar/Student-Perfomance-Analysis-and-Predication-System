# Student Performance Analysis & Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, production-style **Data Science mini-project** that analyses
the academic behaviour of 210 students and predicts their final exam score
with machine learning. Built to industry standards: modular code, logging,
error handling, reproducible seeds, auditable reports, and a one-command
pipeline runner.

> Predict a student's final score from five everyday signals —
> study hours, attendance, previous score, assignments completed, sleep hours —
> and never let an at-risk student go unnoticed again.

---

## Table of Contents

1. [Features](#features)
2. [Project Architecture](#project-architecture)
3. [Dataset](#dataset)
4. [Pipeline Overview](#pipeline-overview)
5. [Installation Guide](#installation-guide)
6. [How to Run](#how-to-run)
7. [Outputs & Screenshots](#outputs--screenshots)
8. [Model Performance](#model-performance)
9. [Troubleshooting](#troubleshooting)
10. [Future Scope](#future-scope)
11. [License](#license)

---

## Features

| # | Feature | Where |
|---|---------|-------|
| 1 | Sample dataset — 210 student records | `data/student_data.csv` |
| 2 | Data cleaning with an auditable action report | `src/preprocessing.py` |
| 3 | Median-based missing value imputation | `src/preprocessing.py` |
| 4 | Duplicate `student_id` removal | `src/preprocessing.py` |
| 5 | Encoding, clipping & feature engineering | `src/preprocessing.py` |
| 6 | Full exploratory data analysis (EDA) | `src/analysis.py` |
| 7 | Statistical summary (skew, kurtosis, quartiles) | `src/analysis.py` |
| 8 | Bar chart — grade distribution | `src/visualization.py` |
| 9 | Line chart — score vs. study-hour trend | `src/visualization.py` |
| 10 | Histogram — score distribution | `src/visualization.py` |
| 11 | Scatter plot — attendance vs. score + trend line | `src/visualization.py` |
| 12 | Box plot — score spread per grade | `src/visualization.py` |
| 13 | Pie chart — performance cohort share | `src/visualization.py` |
| 14 | Correlation heatmap | `src/visualization.py` |
| 15 | Two ML models (Linear Regression + Random Forest) | `src/model.py` |
| 16 | 80/20 train-test split, `random_state=42` | `src/model.py` |
| 17 | Model training with timing logs | `src/model.py` |
| 18 | Evaluation: MAE, RMSE, R², ±5-point accuracy | `src/model.py` |
| 19 | Predict a brand-new student's performance | `src/prediction.py` |
| 20 | Batch predictions saved to CSV | `src/prediction.py` |
| 21 | Guarded stages, logging, graceful failures | `main.py`, `src/utils.py` |
| 22 | Modular, PEP-8, docstring-first code | whole `src/` package |
| 23 | Professional docs: report, viva, slides outline | `docs/` |

## Project Architecture

```
Student_Performance_Project/
│
├── data/
│   ├── student_data.csv            # raw MIS export (214 rows, messy on purpose)
│   └── student_data_clean.csv      # generated — cleaned & enriched dataset
│
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb   # optional notebook walkthrough
│
├── src/                            # the importable pipeline package
│   ├── preprocessing.py            # Stage 1 — cleaning & feature engineering
│   ├── analysis.py                 # Stage 2 — statistics & cohort analysis
│   ├── visualization.py            # Stage 3 — all seven charts
│   ├── model.py                    # Stage 4 — train, evaluate, persist
│   ├── prediction.py               # Stage 5 — single & batch inference
│   └── utils.py                    # paths, logging, timing, helpers
│
├── outputs/
│   ├── graphs/                     # generated PNG charts (150 DPI)
│   ├── reports/                    # eda/model reports, CSV extracts, log
│   └── predictions/                # predictions.csv
│
├── models/                         # best_model.pkl (created on first run)
├── docs/                           # report, viva questions, slide outline
├── main.py                         # one-command pipeline runner
├── requirements.txt
├── README.md
└── LICENSE                         # MIT
```

## Dataset

`data/student_data.csv` — **214 raw rows**, of which 210 are unique students.
The export deliberately contains 4 duplicate rows and 7 blank cells so the
cleaning stage demonstrates real, verifiable work.

| Column | Type | Range / Values | Meaning |
|--------|------|----------------|---------|
| `student_id` | string | `S0001`–`S0210` | Unique identifier |
| `name` | string | — | Student's full name |
| `gender` | string | Male / Female | — |
| `age` | int | 17 – 23 | Years |
| `study_hours` | float | 1.0 – 10.0 | Self-study hours per day |
| `attendance` | float | 42 – 100 | Attendance percentage |
| `previous_score` | float | 35 – 98 | Previous exam score (0–100) |
| `assignments_completed` | int | 0 – 10 | Assignments submitted (of 10) |
| `sleep_hours` | float | 4.0 – 9.0 | Nightly sleep |
| `extracurricular` | string | Yes / No | Participation flag |
| `final_score` | float | 0 – 100 | **Target** — final exam score |
| `grade` | string | A / B / C / D / F | Letter grade derived from target |

**Grade scale:** A ≥ 85 · B ≥ 70 · C ≥ 55 · D ≥ 40 · F < 40.

## Pipeline Overview

```
 raw CSV ──▶ PREPROCESSING ──▶ EDA ──▶ VISUALIZATION ──▶ MODELING ──▶ PREDICTION
             dup removal         stats      7 charts         split 80/20     single + batch
             median impute       groupby    PNG export       train 2 models  CSV output
             feature eng.        corr       150 DPI          eval + pickle   recommendations
```

Each stage can also run standalone, e.g. `python src/visualization.py`.

## Installation Guide

### Prerequisites
- **Python 3.9 or newer** → check with `python --version`
- `pip` (bundled with Python) and 5 minutes

### Step 1 — Get the code
```bash
# Option A: clone
git clone https://github.com/<you>/Student_Performance_Project.git
cd Student_Performance_Project

# Option B: download ZIP from GitHub -> extract -> open terminal inside it
```

### Step 2 — Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the pipeline
```bash
python main.py
```

That's it. First run takes ~20–40 seconds (Random Forest training), and all
artefacts appear inside `outputs/`.

## How to Run

```bash
python main.py                    # full five-stage pipeline
python main.py --skip-graphs      # fast run without rendering charts
python main.py --data myfile.csv  # use your own export (same schema)

python src/preprocessing.py       # run any stage alone
python src/prediction.py          # predicts one demo student
```

**Predict your own student in 5 lines of Python:**

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("src").resolve()))
from prediction import predict_single

result = predict_single({
    "study_hours": 7.5, "attendance": 92.0, "previous_score": 81.0,
    "assignments_completed": 9, "sleep_hours": 7.0,
})
print(result["predicted_score"], result["predicted_grade"])   # e.g. 90.8 A
```

## Outputs & Screenshots

After `python main.py`, collect your submission screenshots from:

| Screenshot | Location |
|------------|----------|
| Pipeline console log | terminal running `main.py` |
| Bar / line / hist / scatter / box / pie / heatmap | `outputs/graphs/*.png` (7 files) |
| Statistical summary | `outputs/reports/eda_report.txt` + `summary_statistics.csv` |
| Model metrics | `outputs/reports/model_report.txt` |
| Batch predictions | `outputs/predictions/predictions.csv` |

> Tip: take screenshots at 100 % zoom; the PNGs are 150 DPI and look crisp
> in the printed internship report (see `docs/internship_report.md`, §8).

## Model Performance

Actual first-run results on the held-out 20 % test set (seed 42):

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Accuracy (±5 pts) ↑ |
|-------|------:|-------:|-----:|--------------------:|
| **Linear Regression** ★ | 3.29 | 3.94 | **0.939** | **76.2 %** |
| Random Forest (300 trees) | 5.25 | 6.15 | 0.851 | 47.6 % |

★ crowned champion automatically by test R² and saved to
`models/best_model.pkl`. A lovely teaching moment: the underlying signal is
additive-linear, so the forest's extra flexibility mostly absorbs residual
noise — the simple model wins honestly. Top drivers: **study hours** and
**previous score** — exactly what the correlation heatmap shows.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: pandas` | Activate the venv, then `pip install -r requirements.txt` |
| Charts don't open | They are **saved**, not popped up — check `outputs/graphs/` |
| `FileNotFoundError: student_data.csv` | Run from the project root folder, not inside `src/` |
| `SyntaxError` on start | Use Python 3.9+ (`python --version`) |
| Permission error on Windows | Close any open CSV/report files, then re-run |

## Future Scope

- Predict grades directly with a classifier (multiclass F1 evaluation)
- Streamlit dashboard for live teacher-facing predictions
- SHAP explainability per student ("why is this student at risk?")
- Connect to a real LMS export and schedule weekly retraining

## License

MIT — free for academic and commercial use. See `LICENSE`.

---

*Prepared as a diploma/engineering Data Science project. Documentation pack:
`docs/internship_report.md` · `docs/presentation_outline.md` · `docs/viva_questions.md`.*
