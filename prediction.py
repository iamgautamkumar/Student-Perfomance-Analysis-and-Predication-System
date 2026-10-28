"""
prediction.py
=============
Stage 5 of the pipeline — Inference.

Two prediction modes are provided:

* **Single student** — :func:`predict_single` scores one student described by
  a plain dict (used interactively and by the CLI demo).
* **Batch** — :func:`predict_batch` scores an entire CSV of new students and
  writes the enriched file to ``outputs/predictions/predictions.csv``.

Every prediction returns both the numeric score **and** the derived letter
grade, so downstream consumers never need to re-implement the grade mapping.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    FEATURE_COLUMNS,
    MODEL_FILE,
    PREDICTION_DIR,
    get_logger,
    grade_from_score,
    performance_category,
    print_banner,
    safe_float,
    timed,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #

def load_model(path: Path = MODEL_FILE) -> dict:
    """
    Load the persisted ``{model, features, name}`` bundle from disk.

    Raises
    ------
    FileNotFoundError
        With an actionable message when the pipeline has not been trained yet.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            "Run `python main.py` (or src/model.py) first to train it."
        )
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    logger.info("Loaded model '%s' from %s", bundle["name"], path)
    return bundle


def _validate_student(student: dict) -> list[float]:
    """
    Validate one raw student record and return the ordered feature vector.

    Unknown / empty values fall back to neutral mid-range defaults instead of
    crashing — a deliberate robustness choice for manual data entry.
    """
    vector = []
    for col in FEATURE_COLUMNS:
        value = safe_float(student.get(col), default=None)
        if value is None:
            logger.warning("Missing/invalid '%s' — defaulting to 5.0", col)
            value = 5.0
        vector.append(float(value))
    return vector


# --------------------------------------------------------------------------- #
# Single prediction                                                           #
# --------------------------------------------------------------------------- #

@timed
def predict_single(student: dict, bundle: dict | None = None) -> dict:
    """
    Predict the final score for one student.

    Parameters
    ----------
    student : dict
        Keys matching :data:`utils.FEATURE_COLUMNS`, e.g.::

            {"study_hours": 7.5, "attendance": 92.0, "previous_score": 81.0,
             "assignments_completed": 9, "sleep_hours": 7.0}

    bundle : dict, optional
        Pre-loaded model bundle (avoids re-reading the pickle in loops).
    """
    bundle = bundle or load_model()
    X = np.asarray([_validate_student(student)], dtype=float)
    score = float(np.clip(bundle["model"].predict(X)[0], 0.0, 100.0))
    grade = grade_from_score(score)
    result = {
        **student,
        "predicted_score": round(score, 1),
        "predicted_grade": grade,
        "category": performance_category(grade),
        "model_used": bundle["name"],
        "prediction_time": datetime.now().isoformat(timespec="seconds"),
    }
    logger.info("Prediction -> %.1f (%s)", score, grade)
    return result


# --------------------------------------------------------------------------- #
# Batch prediction                                                            #
# --------------------------------------------------------------------------- #

@timed
def predict_batch(input_csv: Path, output_csv: Path | None = None) -> pd.DataFrame:
    """
    Score every student in ``input_csv`` and save the enriched CSV.

    The output file lands in ``outputs/predictions/predictions.csv`` unless
    another location is requested — satisfying the project requirement
    "save predictions to CSV".
    """
    bundle = load_model()
    df = pd.read_csv(input_csv)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing feature columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(5.0)
    scores = np.clip(bundle["model"].predict(X.to_numpy(dtype=float)), 0, 100)

    out = df.copy()
    out["predicted_score"] = scores.round(1)
    out["predicted_grade"] = [grade_from_score(s) for s in scores]
    out["category"] = [performance_category(g) for g in out["predicted_grade"]]
    out["model_used"] = bundle["name"]

    output_csv = output_csv or (PREDICTION_DIR / "predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.info("Batch predictions (%d rows) saved -> %s", len(out), output_csv)
    return out


# --------------------------------------------------------------------------- #
# Stage entry-point                                                           #
# --------------------------------------------------------------------------- #

@timed
def run_prediction_demo(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Demonstrate inference on five freshly-sampled "new" students, print a
    recommendation for each, and persist the demo predictions to CSV.
    """
    print_banner("Stage 5 — New Student Prediction Demo")
    bundle = load_model()
    sample = df.sample(n=5, random_state=7)

    records = []
    for _, row in sample.iterrows():
        student = {c: row[c] for c in FEATURE_COLUMNS}
        records.append(predict_single(student, bundle))
        records[-1]["student_id"] = row["student_id"]
        records[-1]["name"] = row["name"]

    results = pd.DataFrame(records)
    cols = ["student_id", "name", "predicted_score", "predicted_grade",
            "category", "model_used"]
    print(results[cols].to_string(index=False))

    if save:
        out_path = PREDICTION_DIR / "predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\n  Predictions saved -> {out_path}\n")
    return results


if __name__ == "__main__":
    demo = {"study_hours": 7.5, "attendance": 92.0, "previous_score": 81.0,
            "assignments_completed": 9, "sleep_hours": 7.0}
    print(predict_single(demo))
