"""
model.py
========
Stage 4 of the pipeline — Machine Learning.

Workflow
--------
1.  Feature/target separation (``X``, ``y``)
2.  Train-test split — 80 % train / 20 % test, ``random_state=42``
    (a fixed seed makes every run exactly reproducible)
3.  Train two candidate regressors:

    * ``LinearRegression``      — interpretable baseline
    * ``RandomForestRegressor`` — captures any non-linear residue

4.  Evaluate on the *held-out* test set with:

    * **MAE**   — mean absolute error (points)
    * **RMSE**  — root-mean-squared error (punishes big misses)
    * **R2**    — variance explained (1.0 is perfect)
    * **Accuracy ±5** — share of predictions within 5 points of truth

5.  Pick the winner by test R2, persist it with ``pickle``, and write a
    full text + JSON report to ``outputs/reports/``.

Design note: the model predicts the *numeric* final score; the letter grade
is then derived deterministically via :func:`utils.grade_from_score`. One
source of truth — no separate classifier to drift out of sync.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import (
    FEATURE_COLUMNS,
    MODEL_FILE,
    REPORT_DIR,
    TARGET_COLUMN,
    banner,
    get_logger,
    print_banner,
    timed,
)

logger = get_logger(__name__)

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute the four headline metrics for a regression model.

    ``accuracy_within_5`` is the college-friendly "accuracy": the percentage
    of students whose predicted score lands inside a ±5-point window of the
    actual exam outcome.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "accuracy_within_5": round(float(np.mean(np.abs(y_true - y_pred) <= 5.0)) * 100, 1),
    }


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #

@timed
def train_models(df: pd.DataFrame) -> dict:
    """
    Split the data, train both candidates, evaluate, and select the best.

    Returns a results dict containing the fitted best model, its name, the
    metrics table, feature importances, and the raw test-set arrays (used by
    the reporting functions and the demo notebooks).
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    logger.info("Feature matrix: %s | target: %s", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    logger.info("Split -> %d train / %d test (seed=%d)",
                len(X_train), len(X_test), RANDOM_STATE)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    metrics: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        fitted[name] = estimator
        metrics[name] = evaluate(y_test, estimator.predict(X_test))
        logger.info("%-24s MAE=%.3f RMSE=%.3f R2=%.4f ±5-acc=%.1f%%",
                    name, metrics[name]["MAE"], metrics[name]["RMSE"],
                    metrics[name]["R2"], metrics[name]["accuracy_within_5"])

    best_name = max(metrics, key=lambda n: metrics[n]["R2"])
    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test)

    # Feature importance — natives for the forest, |coefficients| for linear
    if hasattr(best_model, "feature_importances_"):
        raw_imp = np.asarray(best_model.feature_importances_, dtype=float)
    else:
        raw_imp = np.abs(np.asarray(best_model.coef_, dtype=float))
    importances = (raw_imp / raw_imp.sum()).round(4)
    importance_table = dict(zip(FEATURE_COLUMNS, importances.tolist()))

    # Persist the winning model
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump({"model": best_model, "features": FEATURE_COLUMNS,
                     "name": best_name}, fh)
    logger.info("Best model '%s' persisted -> %s", best_name, MODEL_FILE)

    return {
        "best_name": best_name,
        "best_model": best_model,
        "metrics": metrics,
        "importances": importance_table,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def write_model_report(results: dict) -> None:
    """Write the human-readable and machine-readable model reports."""
    lines = [
        "=" * 74,
        "  MODEL TRAINING & EVALUATION REPORT",
        "=" * 74,
        f"\nSamples: {results['n_train']} train / {results['n_test']} test",
        f"Selected model : {results['best_name']}",
        "\n--- TEST-SET METRICS " + "-" * 52,
        f"  {'Model':<26}{'MAE':>8}{'RMSE':>9}{'R2':>9}{'±5 acc':>9}",
    ]
    for name, m in results["metrics"].items():
        flag = " *" if name == results["best_name"] else ""
        lines.append(f"  {name:<26}{m['MAE']:>8}{m['RMSE']:>9}"
                     f"{m['R2']:>9}{m['accuracy_within_5']:>8.1f}%{flag}")
    lines.append("\n--- FEATURE IMPORTANCE " + "-" * 50)
    for feat, imp in sorted(results["importances"].items(),
                            key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {feat:<24}{'#' * int(imp * 40)} {imp * 100:.1f}%")
    lines.append("\nBest model file : models/best_model.pkl\n")
    (REPORT_DIR / "model_report.txt").write_text("\n".join(lines), encoding="utf-8")

    serialisable = {k: v for k, v in results.items()
                    if k not in {"best_model", "y_test", "y_pred"}}
    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8")
    logger.info("Model reports written -> outputs/reports/")


# --------------------------------------------------------------------------- #
# Stage entry-point                                                           #
# --------------------------------------------------------------------------- #

@timed
def run_modeling(df: pd.DataFrame) -> dict:
    """Train, evaluate, persist and report — the complete ML stage."""
    print_banner("Stage 4 — Model Training & Evaluation")
    results = train_models(df)
    write_model_report(results)

    best = results["metrics"][results["best_name"]]
    print(banner("Champion model"))
    print(f"  Model            : {results['best_name']}")
    print(f"  R2 (test)        : {best['R2']}")
    print(f"  RMSE (test)      : {best['RMSE']} points")
    print(f"  Accuracy (±5 pts): {best['accuracy_within_5']}%\n")
    return results


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    run_modeling(run_preprocessing())
