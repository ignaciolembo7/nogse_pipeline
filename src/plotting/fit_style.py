from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_COLOR = "#1f77b4"
FIT_COLOR = "#c43c35"
FIT_POINT_COLOR = "#0f172a"
DATA_MARKER_SIZE = 5.5
FIT_POINT_SIZE = 7.0
FIT_LINEWIDTH = 2.2
FIGSIZE = (7.0, 5.0)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 10
GRID_ALPHA = 0.25


def start_fit_figure() -> None:
    plt.figure(figsize=FIGSIZE)


def plot_fit_data(x: np.ndarray, y: np.ndarray, *, label: str = "data") -> None:
    plt.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        markersize=DATA_MARKER_SIZE,
        color=DATA_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=label,
    )


def highlight_fit_points(x: np.ndarray, y: np.ndarray, *, label: str) -> None:
    if len(x) == 0:
        return
    plt.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        markersize=FIT_POINT_SIZE,
        markerfacecolor="none",
        markeredgecolor=FIT_POINT_COLOR,
        markeredgewidth=1.2,
        label=label,
    )


def plot_fit_curve(x: np.ndarray, y: np.ndarray, *, label: str) -> None:
    plt.plot(x, y, "-", color=FIT_COLOR, linewidth=FIT_LINEWIDTH, label=label)


def finish_fit_figure(*, title: str, xlabel: str, ylabel: str, out_png: Path) -> None:
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.grid(True, linestyle="--", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
