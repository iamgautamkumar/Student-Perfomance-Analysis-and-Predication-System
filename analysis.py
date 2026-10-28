"""
analysis.py
===========
Stage 2 of the pipeline — Exploratory Data Analysis (EDA).

Produces the descriptive statistics and group comparisons that answer the
business questions of the project:

* How do students score overall?                 -> :func:`summary_statistics`
* Who performs better, and who is at risk?       -> :func:`group_analysis`
* Which behaviours drive the final score?        -> :func:`correlation_matrix`

All artefacts are written to ``outputs/reports/`` so every run leaves behind
an auditable paper trail (CSV + TXT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import REPORT_DIR, banner, get_logger, print_banner, timed

logger = get_logger(__name__)

NUMERICAL = [
    "age", "study_hours", "attendance", "previous_score",
    "assignments_completed", "sleep_hours", "final_score",
]


# --------------------------------------------------------------------------- #
# Descriptive statistics                                                      #
# --------------------------------------------------------------------------- #

@timed
def summary_statistics(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Extended descriptive statistics for every numeric column:
    count, mean, std, min, the three quartiles, max, skewness and kurtosis.
    """
    stats = df[NUMERICAL].describe().T
    stats["skewness"] = df[NUMERICAL].skew()
    stats["kurtosis"] = df[NUMERICAL].kurtosis()
    stats = stats.round(2)
    logger.info("Computed summary statistics for %d numeric columns", len(NUMERICAL))

    if save:
        path = REPORT_DIR / "summary_statistics.csv"
        stats.to_csv(path)
        logger.info("Summary statistics saved -> %s", path)
    return stats


# --------------------------------------------------------------------------- #
# Group analysis                                                              #
# --------------------------------------------------------------------------- #

@timed
def group_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compare performance across natural cohorts and detect at-risk students.

    Returns a dictionary of named DataFrames (each also saved as CSV).
    """
    results: dict[str, pd.DataFrame] = {}

    # Mean score per letter grade — validates that grades track reality
    results["by_grade"] = (
        df.groupby("grade")["final_score"]
        .agg(students="count", mean_score="mean", std="std")
        .round(2).sort_index()
    )

    # Gender gap check (should be ~0 in a healthy dataset)
    results["by_gender"] = (
        df.groupby("gender")["final_score"]
        .agg(students="count", mean_score="mean", median_score="median")
        .round(2)
    )

    # Extracurricular participation effect
    results["by_extracurricular"] = (
        df.groupby("extracurricular")["final_score"]
        .agg(students="count", mean_score="mean")
        .round(2)
    )

    # Top performers & students needing support
    results["top_performers"] = df.nlargest(10, "final_score")[
        ["student_id", "name", "study_hours", "attendance", "final_score", "grade"]
    ]
    results["at_risk"] = df[df["final_score"] < 40][
        ["student_id", "name", "attendance", "previous_score", "final_score", "grade"]
    ].sort_values("final_score")

    for name, frame in results.items():
        frame.to_csv(REPORT_DIR / f"analysis_{name}.csv")
        logger.info("Group analysis '%s' saved (%d rows)", name, len(frame))
    return results


# --------------------------------------------------------------------------- #
# Correlation                                                                 #
# --------------------------------------------------------------------------- #

@timed
def correlation_matrix(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Pearson correlation matrix across the behavioural features + target.

    The top drivers (typically ``study_hours`` and ``previous_score``) are the
    features the ML model will lean on most heavily.
    """
    cols = ["study_hours", "attendance", "previous_score",
            "assignments_completed", "sleep_hours", "final_score"]
    corr = df[cols].corr().round(3)

    if save:
        corr.to_csv(REPORT_DIR / "correlation_matrix.csv")
    logger.info("Correlation with final_score: %s",
                corr["final_score"].drop("final_score").to_dict())
    return corr


# --------------------------------------------------------------------------- #
# Text report                                                                 #
# --------------------------------------------------------------------------- #

def write_eda_report(df: pd.DataFrame, stats: pd.DataFrame,
                     groups: dict, corr: pd.DataFrame) -> None:
    """Render a human-readable EDA report to ``outputs/reports/eda_report.txt``."""
    path = REPORT_DIR / "eda_report.txt"
    lines = [
        "=" * 74,
        "  STUDENT PERFORMANCE — EXPLORATORY DATA ANALYSIS REPORT",
        "=" * 74,
        f"\nStudents analysed : {len(df)}",
        f"Mean final score  : {df['final_score'].mean():.2f}",
        f"Median final score: {df['final_score'].median():.2f}",
        f"Std deviation     : {df['final_score'].std():.2f}",
        f"Pass rate (>=40)  : {(df['final_score'] >= 40).mean() * 100:.1f}%",
        "\n--- GRADE DISTRIBUTION " + "-" * 50,
        groups["by_grade"].to_string(),
        "\n--- TOP 3 SCORE DRIVERS (|correlation|) " + "-" * 32,
    ]
    drivers = corr["final_score"].drop("final_score").abs().sort_values(ascending=False)
    for name in drivers.index[:3]:
        lines.append(f"  {name:<24} r = {corr.loc[name, 'final_score']:+.3f}")
    lines += [
        "\n--- FULL SUMMARY STATISTICS " + "-" * 45,
        stats.to_string(),
        "\n",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("EDA report written -> %s", path)


# --------------------------------------------------------------------------- #
# Stage entry-point                                                           #
# --------------------------------------------------------------------------- #

@timed
def run_analysis(df: pd.DataFrame) -> dict:
    """Run the complete EDA stage and return all produced artefacts."""
    print_banner("Stage 2 — Exploratory Data Analysis")
    stats = summary_statistics(df)
    groups = group_analysis(df)
    corr = correlation_matrix(df)
    write_eda_report(df, stats, groups, corr)

    print(banner("Key insights"))
    drivers = corr["final_score"].drop("final_score").abs().sort_values(ascending=False)
    print(f"  - Strongest score driver : {drivers.index[0]} "
          f"(r={corr.loc[drivers.index[0], 'final_score']:+.3f})")
    print(f"  - Students at risk (<40) : {(df['final_score'] < 40).sum()}")
    print(f"  - Pass rate              : {(df['final_score'] >= 40).mean() * 100:.1f}%\n")
    return {"stats": stats, "groups": groups, "corr": corr}


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    run_analysis(run_preprocessing())
