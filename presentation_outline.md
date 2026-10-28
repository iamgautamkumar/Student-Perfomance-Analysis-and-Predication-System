# Presentation Outline — 15 Slides
### Student Performance Analysis & Prediction System

Copy each slide into PowerPoint / Google Slides. One slide ≈ one screen of
content; speaker notes included. Screenshots: generate charts first with
`python main.py`, then insert from `outputs/graphs/`.

---

**Slide 1 — Title**
Student Performance Analysis and Prediction System
*A Data Science & Machine Learning Mini-Project* · Name · Roll No · Guide
Notes: greet, state the one-line goal: "predict final scores early enough
to help students before the exam, not after."

**Slide 2 — Agenda**
Problem → Objectives → Tech Stack → Pipeline → Dataset → Cleaning → EDA →
Charts → Models → Results → Live Demo → Future Work → Q&A

**Slide 3 — Problem Statement**
Institutions hold rich student data but act on it late. Can we predict the
final exam score from everyday behavioural signals — early and
interpretably? Bullets: duplicates/blanks in exports · insights need charts
· predictions must be reproducible.

**Slide 4 — Objectives**
1. Automated cleaning pipeline  2. Statistical EDA  3. 7-chart visual pack
4. ML regression with honest metrics  5. Single + batch prediction CSV
6. Production-grade code & docs

**Slide 5 — Technology Stack**
Python 3.x · Pandas (data frames) · NumPy (numerics) · Matplotlib (charts) ·
Scikit-learn (ML) · Why: open-source, industry-standard, examiner-friendly.
Diagram: logos in a row.

**Slide 6 — System Architecture / Pipeline**
Five stages drawing: raw CSV → Preprocessing → EDA → Visualization →
Modeling → Prediction; artefacts under each stage (clean.csv, reports,
graphs, model.pkl, predictions.csv). Notes: every stage logged + guarded.

**Slide 7 — Dataset Overview**
210 students × 12 columns; data dictionary table (id, demographics, 5
features, target, grade); grade scale A≥85…F<40; raw file intentionally
contains 4 duplicates + 7 blanks.

**Slide 8 — Data Cleaning**
Before/after table: 214 → 210 rows; duplicates removed: 4; medians imputed:
study_hours ×4, attendance ×3; range clipping; categorical normalisation;
engineered features study_efficiency, engagement_index.

**Slide 9 — EDA Highlights**
Mean 66.7, pass rate 94.8 %; correlation ranking: study_hours +0.68,
previous_score +0.59, attendance +0.20; gender gap ≈ 2 pts; grade mix
A 28 · B 60 · C 78 · D 33 · F 11. Insert `eda_report.txt` excerpt as
screenshot.

**Slide 10 — Chart Gallery**
2×4 grid of the seven PNGs: bar (grades), line (study-hour trend), histogram
(scores), scatter (attendance vs score + r), box (per grade), pie (cohorts),
heatmap (correlations).

**Slide 11 — Machine Learning Approach**
Task: regression (target = final_score). Split: 80/20 seed 42. Candidates:
Linear Regression (explainable baseline, +3.9 pts per study hour) vs
Random Forest 300 trees (non-linear). Grade derived from score — single
source of truth.

**Slide 12 — Results & Metrics**
Metric table per model: MAE / RMSE / R² / ±5-accuracy; champion = Linear
Regression, test R² = 0.939 (forest overfits residual noise at 0.85);
feature importance bars (study 38 %, previous 33 %, attendance 13 %);
screenshot `model_report.txt`.

**Slide 13 — Live Prediction Demo**
Input card: study 7.5 h · attendance 92 % · previous 81 · assignments 9 ·
sleep 7 h → Predicted 90.8 (Grade A, "Excellent"). Show
`outputs/predictions/predictions.csv` columns.

**Slide 14 — Future Scope & Limitations**
SHAP explanations · Streamlit dashboard · grade classifier · LMS
integration / fairness audit. Limitations: synthetic cohort, batch mode,
self-reported features.

**Slide 15 — Thank You / Q&A**
Recap one-liner: "Messy CSV → clean insight → honest model → actionable
predictions." + GitHub link + "Questions welcome."

---
*Build tip: duplicate Slide 13 and fill it with a screenshot of the terminal
running `python src/prediction.py` for a wow moment.*
