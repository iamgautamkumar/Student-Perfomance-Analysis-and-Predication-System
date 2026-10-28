"""
preprocessing.py
================
Stage 1 of the pipeline — turn the raw MIS export into an analysis-ready
DataFrame.

Cleaning contract
-----------------
1.  Exact duplicate rows (re-sent exports) are dropped, keeping the first
    occurrence of each ``student_id``.
2.  Missing numeric values are imputed with the **column median** — robust to
    the skew typically seen in behavioural variables such as study hours.
3.  Physically impossible values are clipped into valid ranges
    (e.g. attendance cannot exceed 100).
4.  Categorical spelling is normalised (``"yes"`` -> ``"Yes"``).
5.  Two engineered features are added:

    * ``study_efficiency``   = study_hours x attendance / 100
    * ``engagement_index``   = assignments_completed + study_hours

6.  Binary categoricals are encoded to 0/1 for the modelling stage.

The module can be used two ways::

    from src.preprocessing import run_preprocessing
    df = run_preprocessing()                    # full stage, saves clean CSV

    # or unit-style, for fine-grained control
    from src.preprocessing import load_dataset, clean_dataset
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils import (
    CLEAN_DATA_FILE,
    RAW_DATA_FILE,
    banner,
    get_logger,
    print_banner,
    timed,
)

logger = get_logger(__name__)

#: Columns that must exist in the raw export
EXPECTED_COLUMNS = {
    "student_id", "name", "gender", "age", "study_hours", "attendance",
    "previous_score", "assignments_completed", "sleep_hours",
    "extracurricular", "final_score", "grade",
}

#: Sensible physical bounds for the numeric fields  {column: (min, max)}
VALID_RANGES = {
    "age": (15, 40),
    "study_hours": (0.0, 16.0),
    "attendance": (0.0, 100.0),
    "previous_score": (0.0, 100.0),
    "assignments_completed": (0, 15),
    "sleep_hours": (0.0, 14.0),
    "final_score": (0.0, 100.0),
}


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_dataset(path=RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the raw CSV export with defensive error handling.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist — with a helpful message.
    ValueError
        If expected columns are missing from the file.
    """
    logger.info("Loading raw dataset from %s", path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. "
            "Place student_data.csv inside the data/ folder."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV file '{path}': {exc}") from exc

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    logger.info("Loaded %d rows x %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# Inspection                                                                  #
# --------------------------------------------------------------------------- #

def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Produce a quick data-quality snapshot (used by the cleaning report).

    Returns a dict with row counts, duplicates found, and per-column missing
    counts — everything the analyst needs before touching the data.
    """
    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated(subset=["student_id"]).sum()),
        "missing_values": df.isna().sum().to_dict(),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }
    return report


# --------------------------------------------------------------------------- #
# Cleaning                                                                    #
# --------------------------------------------------------------------------- #

@timed
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply the full cleaning contract and return ``(clean_df, report)``.

    The report dictionary records every corrective action taken, so the run
    is fully auditable — an important habit in industry data pipelines.
    """
    df = df.copy()
    report: dict = {"actions": []}

    # -- 1. Remove exact duplicate submissions --------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["student_id"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    report["duplicates_removed"] = removed
    report["actions"].append(f"Removed {removed} duplicate student_id rows")
    logger.info("Duplicate rows removed: %d", removed)

    # -- 2. Coerce numerics & impute missing values with the column median ----
    numeric_cols = [c for c in VALID_RANGES if c in df.columns]
    imputed: dict = {}
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")   # "", "N/A" -> NaN
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = float(df[col].median())
            df[col] = df[col].fillna(median)
            imputed[col] = {"count": n_missing, "median": round(median, 2)}
            logger.info("Imputed %d missing values in '%s' (median=%.2f)", n_missing, col, median)
    report["missing_imputed"] = imputed

    # -- 3. Clip out-of-range values to physical bounds -----------------------
    for col, (lo, hi) in VALID_RANGES.items():
        out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
        if out_of_range:
            df[col] = df[col].clip(lo, hi)
            report["actions"].append(f"Clipped {out_of_range} out-of-range values in '{col}'")

    # -- 4. Normalise categorical spelling ------------------------------------
    df["gender"] = df["gender"].astype(str).str.strip().str.title()
    df["extracurricular"] = (
        df["extracurricular"].astype(str).str.strip().str.capitalize()
        .replace({"Y": "Yes", "N": "No", "True": "Yes", "False": "No"})
    )

    # -- 5. Feature engineering ------------------------------------------------
    df["study_efficiency"] = (df["study_hours"] * df["attendance"] / 100.0).round(2)
    df["engagement_index"] = (df["assignments_completed"] + df["study_hours"]).round(2)

    # -- 6. Encode binary categoricals (kept alongside originals) -------------
    df["gender_enc"] = (df["gender"] == "Male").astype(int)
    df["extra_enc"] = (df["extracurricular"] == "Yes").astype(int)

    report["final_rows"] = int(len(df))
    report["actions"].append(f"Feature engineering added: study_efficiency, engagement_index")
    logger.info("Cleaning complete — %d clean rows", len(df))
    return df, report


# --------------------------------------------------------------------------- #
# Stage entry-point                                                           #
# --------------------------------------------------------------------------- #

@timed
def run_preprocessing(raw_path=RAW_DATA_FILE, save: bool = True) -> pd.DataFrame:
    """
    Execute the full preprocessing stage and (optionally) persist the clean
    dataset to ``data/student_data_clean.csv``.
    """
    print_banner("Stage 1 — Data Preprocessing & Cleaning")
    raw = load_dataset(raw_path)

    snapshot = inspect_dataset(raw)
    logger.info("Raw snapshot: %d rows, %d duplicates, missing=%s",
                snapshot["rows"], snapshot["duplicate_rows"],
                {k: v for k, v in snapshot["missing_values"].items() if v})

    clean, report = clean_dataset(raw)

    if save:
        clean.to_csv(CLEAN_DATA_FILE, index=False)
        logger.info("Clean dataset saved -> %s", CLEAN_DATA_FILE)

    print(banner("Cleaning report"))
    for action in report["actions"]:
        print(f"  - {action}")
    print(f"  - Final shape: {clean.shape[0]} rows x {clean.shape[1]} columns\n")
    return clean


if __name__ == "__main__":
    # Allow `python src/preprocessing.py` for quick manual runs
    run_preprocessing()
