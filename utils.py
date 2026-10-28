"""
utils.py
========
Shared utilities for the Student Performance Analysis & Prediction System.

Responsibilities
----------------
* Centralised project path constants (so every module agrees on locations)
* Directory bootstrap helper
* Standardised logging configuration
* Console banner helpers for readable pipeline output
* A lightweight ``@timed`` decorator used to benchmark pipeline stages
* Small defensive-casting helpers used across modules

Author : Student Performance Project Team
Python : 3.8+
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Path constants                                                              #
# --------------------------------------------------------------------------- #

#: Absolute path to the project root (the folder that contains main.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Raw input data
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_FILE: Path = DATA_DIR / "student_data.csv"
CLEAN_DATA_FILE: Path = DATA_DIR / "student_data_clean.csv"

#: All artefacts produced by the pipeline
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
GRAPH_DIR: Path = OUTPUT_DIR / "graphs"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
PREDICTION_DIR: Path = OUTPUT_DIR / "predictions"
MODEL_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILE: Path = MODEL_DIR / "best_model.pkl"

#: Every directory the pipeline expects to exist
REQUIRED_DIRS = (DATA_DIR, OUTPUT_DIR, GRAPH_DIR, REPORT_DIR, PREDICTION_DIR, MODEL_DIR)

#: Numeric feature columns used for modelling
FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
]

#: Target column for the regression task
TARGET_COLUMN = "final_score"


# --------------------------------------------------------------------------- #
# Filesystem helpers                                                          #
# --------------------------------------------------------------------------- #

def ensure_directories() -> None:
    """Create every directory the pipeline needs (idempotent)."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger that writes to stdout (and optionally a file).

    Parameters
    ----------
    name : str
        Name of the logger, normally ``__name__`` of the calling module.
    log_file : Path, optional
        When provided, log records are also appended to this file.
    level : int
        Standard ``logging`` level constant.
    """
    logger = logging.getLogger(name)
    if logger.handlers:                      # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Console cosmetics                                                           #
# --------------------------------------------------------------------------- #

def banner(title: str, width: int = 72) -> str:
    """Return a centred banner string used to delimit pipeline stages."""
    line = "=" * width
    return f"\n{line}\n  {title.upper()}\n{line}"


def print_banner(title: str, width: int = 72) -> None:
    """Print :func:`banner` straight to stdout."""
    print(banner(title, width))


def timed(func: Callable) -> Callable:
    """
    Decorator that logs the wall-clock time of the wrapped function.

    Example
    -------
    >>> @timed
    ... def train_model(): ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", func.__name__, elapsed)
        return result

    return wrapper


# --------------------------------------------------------------------------- #
# Defensive casting                                                           #
# --------------------------------------------------------------------------- #

def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Best-effort conversion to ``float``.

    Returns ``default`` (instead of raising) when conversion fails — handy for
    messy CSV cells such as ``""``, ``"N/A"`` or ``None``.
    """
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    """Best-effort conversion to ``int``; see :func:`safe_float`."""
    as_float = safe_float(value, None)
    return None if as_float is None else int(round(as_float))


def grade_from_score(score: float) -> str:
    """
    Map a 0-100 numeric score onto the letter-grade scale used campus-wide.

    ===  =================================
    >= 85  A (Excellent)
    >= 70  B (Good)
    >= 55  C (Average)
    >= 40  D (Below Average)
    <  40  F (Fail)
    ===  =================================
    """
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def performance_category(grade: str) -> str:
    """Group letter grades into advisory categories used in reporting."""
    return {"A": "Excellent", "B": "Good", "C": "Average", "D": "At Risk", "F": "At Risk"}.get(grade, "Unknown")
