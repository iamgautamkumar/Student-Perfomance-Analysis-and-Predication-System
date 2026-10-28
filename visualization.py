"""
visualization.py
================
Stage 3 of the pipeline — every chart the project requires, rendered with
Matplotlib and saved at 150 DPI to ``outputs/graphs/``.

Required chart inventory
------------------------
============  ==================================  ============================
Chart         Function                            Insight
============  ==================================  ============================
Bar chart     :func:`plot_grade_distribution`     How grades are spread
Line chart    :func:`plot_score_trend`            Score vs. study hours trend
Histogram     :func:`plot_score_histogram`        Shape of the score curve
Scatter plot  :func:`plot_attendance_vs_score`    Attendance -> performance
Box plot      :func:`plot_score_boxplot`          Spread / outliers per grade
Pie chart     :func:`plot_category_pie`           Cohort share
Heatmap       :func:`plot_correlation_heatmap`    Feature correlations
============  ==================================  ============================

A consistent house style (colours, fonts, grid) is applied through
:func:`_apply_style` so the generated pack looks like a designed report,
not a pile of default plots.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — safe for servers / CI / terminals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from utils import GRAPH_DIR, get_logger, print_banner, timed

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# House style                                                                 #
# --------------------------------------------------------------------------- #

PALETTE = ["#4C8DFF", "#7BD88F", "#FFC857", "#FF8A5C", "#C77DFF", "#5CE1E6"]
GRADE_COLORS = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}
HEAT_CMAP = LinearSegmentedColormap.from_list("brand", ["#1d4ed8", "#f8fafc", "#dc2626"])
FIG_BG, AX_BG, TXT = "#ffffff", "#ffffff", "#0f172a"


def _apply_style() -> None:
    """Apply the report's shared Matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG,
        "axes.edgecolor": "#94a3b8", "axes.labelcolor": TXT,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.grid": True,
        "grid.color": "#e2e8f0", "grid.linewidth": 0.8,
        "xtick.color": TXT, "ytick.color": TXT, "text.color": TXT,
        "font.family": "DejaVu Sans", "figure.dpi": 150,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, filename: str) -> str:
    """Save ``fig`` into the graphs directory and close it (memory hygiene)."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    logger.info("Graph saved -> %s", path)
    return str(path)


# --------------------------------------------------------------------------- #
# 1. Bar chart                                                                #
# --------------------------------------------------------------------------- #

@timed
def plot_grade_distribution(df: pd.DataFrame) -> str:
    """Bar chart — how many students earned each grade."""
    _apply_style()
    counts = df["grade"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[GRADE_COLORS[g] for g in counts.index],
                  edgecolor="#0f172a", linewidth=0.6)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Grade Distribution")
    ax.set_xlabel("Letter Grade")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "01_bar_grade_distribution.png")


# --------------------------------------------------------------------------- #
# 2. Line chart                                                               #
# --------------------------------------------------------------------------- #

@timed
def plot_score_trend(df: pd.DataFrame) -> str:
    """Line chart — mean final score within each study-hours bucket."""
    _apply_style()
    buckets = pd.cut(df["study_hours"], bins=[0, 2, 4, 6, 8, 10],
                     labels=["0-2 h", "2-4 h", "4-6 h", "6-8 h", "8-10 h"])
    trend = df.groupby(buckets, observed=False)["final_score"].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trend.index.astype(str), trend["mean"], marker="o",
            color=PALETTE[0], linewidth=2.5, markersize=9,
            markerfacecolor="#ffffff", markeredgewidth=2)
    for x, (m, n) in enumerate(zip(trend["mean"], trend["count"])):
        ax.annotate(f"{m:.1f}", (x, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontweight="bold")
        ax.annotate(f"n={n}", (x, m), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=8, color="#64748b")
    ax.set_title("Average Final Score by Weekly Study Hours")
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Mean Final Score")
    return _save(fig, "02_line_score_trend.png")


# --------------------------------------------------------------------------- #
# 3. Histogram                                                                #
# --------------------------------------------------------------------------- #

@timed
def plot_score_histogram(df: pd.DataFrame) -> str:
    """Histogram — distribution of final scores with mean/median markers."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["final_score"], bins=16, color=PALETTE[0],
            edgecolor="#ffffff", alpha=0.9)
    mean, median = df["final_score"].mean(), df["final_score"].median()
    ax.axvline(mean, color="#dc2626", linestyle="--", linewidth=2,
               label=f"Mean = {mean:.1f}")
    ax.axvline(median, color="#16a34a", linestyle=":", linewidth=2,
               label=f"Median = {median:.1f}")
    ax.set_title("Distribution of Final Scores")
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Number of Students")
    ax.legend(frameon=False)
    return _save(fig, "03_histogram_scores.png")


# --------------------------------------------------------------------------- #
# 4. Scatter plot                                                             #
# --------------------------------------------------------------------------- #

@timed
def plot_attendance_vs_score(df: pd.DataFrame) -> str:
    """Scatter — attendance vs. final score, with least-squares fit line."""
    _apply_style()
    x, y = df["attendance"], df["final_score"]
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    r = x.corr(y)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=42, alpha=0.65, color=PALETTE[0], edgecolor="#ffffff",
               linewidth=0.4, label="Students")
    ax.plot(xs, slope * xs + intercept, color="#dc2626", linewidth=2,
            label=f"Trend (r = {r:+.2f})")
    ax.set_title("Attendance vs. Final Score")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Final Score")
    ax.legend(frameon=False)
    return _save(fig, "04_scatter_attendance_score.png")


# --------------------------------------------------------------------------- #
# 5. Box plot                                                                 #
# --------------------------------------------------------------------------- #

@timed
def plot_score_boxplot(df: pd.DataFrame) -> str:
    """Box plot — score spread per grade (median, quartiles, outliers)."""
    _apply_style()
    order = [g for g in ["A", "B", "C", "D", "F"] if g in set(df["grade"])]
    data = [df.loc[df["grade"] == g, "final_score"] for g in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="#0f172a", linewidth=2))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    for patch, grade in zip(bp["boxes"], order):
        patch.set_facecolor(GRADE_COLORS[grade])
        patch.set_alpha(0.85)
    ax.set_title("Score Spread by Grade (Box Plot)")
    ax.set_xlabel("Letter Grade")
    ax.set_ylabel("Final Score")
    return _save(fig, "05_boxplot_scores.png")


# --------------------------------------------------------------------------- #
# 6. Pie chart                                                                #
# --------------------------------------------------------------------------- #

@timed
def plot_category_pie(df: pd.DataFrame) -> str:
    """Pie chart — share of Excellent / Good / Average / At-Risk cohorts."""
    _apply_style()
    mapping = {"A": "Excellent", "B": "Good", "C": "Average",
               "D": "At Risk", "F": "At Risk"}
    share = df["grade"].map(mapping).value_counts()
    colors = {"Excellent": "#16a34a", "Good": "#84cc16",
              "Average": "#f59e0b", "At Risk": "#ef4444"}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(share.values, labels=share.index, autopct="%1.1f%%",
           colors=[colors[i] for i in share.index], startangle=90,
           wedgeprops=dict(width=0.55, edgecolor="#ffffff", linewidth=2),
           textprops=dict(fontweight="bold"))
    ax.set_title("Performance Cohort Share")
    return _save(fig, "06_pie_cohorts.png")


# --------------------------------------------------------------------------- #
# 7. Correlation heatmap                                                      #
# --------------------------------------------------------------------------- #

@timed
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Annotated correlation heatmap of every behavioural feature + target."""
    _apply_style()
    cols = ["study_hours", "attendance", "previous_score",
            "assignments_completed", "sleep_hours", "final_score"]
    corr = df[cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap=HEAT_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="#0f172a" if abs(corr.iloc[i, j]) < 0.75 else "#ffffff")
    ax.set_title("Feature Correlation Heatmap")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    return _save(fig, "07_heatmap_correlation.png")


# --------------------------------------------------------------------------- #
# Stage entry-point                                                           #
# --------------------------------------------------------------------------- #

@timed
def run_visualization(df: pd.DataFrame) -> list[str]:
    """Render all seven required charts; returns the list of saved paths."""
    print_banner("Stage 3 — Data Visualization")
    charts = [
        plot_grade_distribution(df),
        plot_score_trend(df),
        plot_score_histogram(df),
        plot_attendance_vs_score(df),
        plot_score_boxplot(df),
        plot_category_pie(df),
        plot_correlation_heatmap(df),
    ]
    print(f"  Generated {len(charts)} charts -> outputs/graphs/\n")
    return charts


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    run_visualization(run_preprocessing())
