# Viva Questions & Model Answers

**30 questions examiners actually ask — grouped by topic, with the answers
that score marks.** Starred (★) questions are near-guaranteed.

---

## A. Python Fundamentals

**1. ★ Why Python for data science?**
Python offers a readable syntax plus a mature scientific stack — NumPy for
vectorised numerics, Pandas for tabular data, scikit-learn for ML,
Matplotlib for charts — all glued together without leaving one language, and
with zero licence cost.

**2. What is a virtual environment and why did you use one?**
An isolated Python install (`python -m venv venv`) with its own packages.
It locks dependency versions per project so `pip install -r requirements.txt`
always recreates the exact environment — on the examiner's machine too.

**3. Difference between a list and a NumPy array?**
Lists hold heterogeneous objects with Python-level loops; NumPy arrays are
homogeneous, C-contiguous memory with vectorised ops — typically 10–100×
faster, which matters once data grows past toy size.

**4. ★ How does your project handle errors?**
Three layers: (1) defensive coercion (`pd.to_numeric(errors="coerce")`,
`safe_float`) so messy cells never crash; (2) explicit validation with
actionable messages (schema check, missing model file); (3) a top-level
`guarded_stage()` wrapper in `main.py` that logs full tracebacks to
`outputs/reports/pipeline.log` and exits non-zero.

**5. What is pickling, and where do you use it?**
Serialising a Python object to bytes. We pickle the trained model bundle
(`models/best_model.pkl`) so inference later costs milliseconds instead of
a retrain.

## B. Pandas & NumPy

**6. ★ How do you read a CSV into Pandas?**
`pd.read_csv("data/student_data.csv")` — returns a DataFrame: labelled,
typed, columnar. We follow with a schema validation step.

**7. How do you detect and remove duplicates?**
`df.duplicated(subset=["student_id"])` for detection;
`df.drop_duplicates(subset=["student_id"], keep="first")` to drop,
keeping the first transmission.

**8. ★ Why median and not mean for imputation?**
Behavioural features (study hours) are right-skewed; the mean is dragged by
outliers. The median is robust — imputing it preserves the typical student,
not the extreme one.

**9. What is `groupby`? Show an example from your code.**
Split-apply-combine. Example: `df.groupby("grade")["final_score"].mean()`
splits by grade, applies mean to each group, combines into a series — the
"who performs better" question in one line.

**10. loc vs. iloc?**
`loc` selects by label (`df.loc[df.grade=="A"]`), `iloc` by integer position
(`df.iloc[:10]`). We use `loc` for cohort filtering, e.g. at-risk students.

**11. How did you handle an empty string in a numeric column?**
CSV blanks arrive as NaN/"" — `pd.to_numeric(errors="coerce")` converts
unparsable cells to NaN, then `fillna(df[col].median())` imputes.

**12. What did feature engineering add?**
`study_efficiency = study_hours × attendance / 100` (effort weighted by
presence) and `engagement_index = assignments + study_hours` (total academic
engagement) — interactions a linear model can't discover alone.

## C. Visualization

**13. ★ Which chart for which question?**
Distribution → histogram · category counts → bar · trend over ordered
buckets → line · relationship of two numerics → scatter · spread/outliers →
box · part-of-whole → pie · pairwise correlations → heatmap.

**14. Why set the Agg backend?**
`matplotlib.use("Agg")` renders to files without a display server — the
pipeline then runs identically in terminals, servers, and CI.

**15. How is the heatmap built without seaborn?**
`ax.imshow(corr, cmap=…, vmin=-1, vmax=1)` plus a nested loop annotating
each cell with the Pearson value — full control, one less dependency.

**16. What does the box plot show that the histogram hides?**
Per-group medians, quartiles and outliers side-by-side (score spread per
grade) — histograms collapse everything into one shape.

## D. Machine Learning

**17. ★ Why regression and not classification?**
The target `final_score` is continuous (0–100). Regression preserves ordinal
distance (predicting 55 vs 90 matters differently), and grades are derived
deterministically afterwards — one system, no drift between two models.

**18. ★ Explain your train-test split.**
`train_test_split(X, y, test_size=0.2, random_state=42)` → 168 train / 42
test rows. The seed freezes the shuffle so every examiner sees identical
metrics.

**19. Why is the test set sacred?**
It estimates performance on *unseen* students. Touching it during tuning
leaks information and inflates metrics — the classic academic sin.

**20. Overfitting — what is it, how do you control it?**
Memorising training noise → great train / poor test scores. Controls used:
held-out evaluation, Random Forest averaging (300 trees), honest metric
reporting. Gap between train/test performance is the tell-tale.

**21. ★ Compare your two models. Why did the linear model win?**
Linear Regression = one global hyperplane (fast, explainable: +3.9 pts per
study-hour). Random Forest = 300 decorrelated trees capturing interactions.
Measured on the held-out set: **R² 0.94 (LR) vs 0.85 (RF)** — the domain
signal is additive-linear, so the forest's flexibility absorbs residual noise
instead of signal. We crown by held-out R², not by preference — that's the
lesson.

**22. Explain MAE, RMSE, R² in one line each.**
MAE — average absolute miss (points). RMSE — same but quadratically punishes
big misses. R² — share of variance explained vs. predicting the mean; 0.93 ≈
"93 % of score variation explained".

**23. What is your ±5-point accuracy?**
The share of test students with |prediction − actual| ≤ 5 — a counsellor's
contract: "we call roughly 3 out of 4 shots within half a grade band."
Measured: **76.2 %** on the seed-42 test set.

**24. How is feature importance computed?**
For the forest: mean impurity decrease per feature, normalised to 100 %.
For the linear champion: |standardised coefficients|. Measured verdict:
study_hours (~38 %) > previous_score (~33 %) > attendance (~13 %) >
assignments (~13 %) > sleep (~3 %).

**25. Which supervised family do Linear Regression and Random Forest belong
to?** Both supervised — the first parametric/linear, the second
ensemble/bagging (bootstrap aggregating) of decision trees.

## E. Project Design & Workflow

**26. ★ Walk me through your pipeline.**
`main.py` → Stage 1 preprocessing (clean/dedupe/impute/engineer) → Stage 2
EDA (stats, group analysis, correlations) → Stage 3 seven charts → Stage 4
train/evaluate two models, persist champion → Stage 5 predict single + batch,
save CSV. Every stage is timed, logged, and guarded.

**27. Why modular code instead of one long script?**
Testable units, single-responsibility files, parallel teamwork, cheap
extension (add a model by editing one dict). Viva answer in one word:
maintainability.

**28. What is CRISP-DM and where did you use it?**
Cross-Industry Standard Process for Data Mining — Business understanding →
Data understanding → Preparation → Modeling → Evaluation → Deployment. Each
of our modules maps to a phase; the report's Chapter 7 traces it.

**29. If the dataset grew to 10 million rows, what changes?**
Chunked reads (`read_csv(chunksize=…)`), Dask/Polars for out-of-core frames,
Random Forest n_jobs already parallel, possibly sample the EDA; charts move
to aggregated bins. Architecture stays identical — that was the point of the
modular design.

**30. ★ What would you build next?**
Per-student SHAP explanations ("your attendance costs you ~6 points"), a
Streamlit dashboard for teachers, and a grade classifier head with F1
tracking — in that order of value.

---

*Tip: when unsure, anchor every answer to a file: "that's in
`src/preprocessing.py`, step 2 of the cleaning contract." Specificity reads
as competence.*
