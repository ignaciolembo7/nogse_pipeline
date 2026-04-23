from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class XYSeries:
    x: np.ndarray
    y: np.ndarray
    label: str
    sigma: np.ndarray | None = None
    color: str | None = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compact_float(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}g}"


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    out: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "NA"


def distinct_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    golden_ratio = 0.618033988749895
    hue = 0.11
    colors: list[str] = []
    for _ in range(n):
        hue = (hue + golden_ratio) % 1.0
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(mcolors.to_hex(rgb))
    return colors


def render_xy_plot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    sigma: np.ndarray | None = None,
    data_label: str = "data",
    connect_data: bool = True,
    fit_x: np.ndarray | None = None,
    fit_y: np.ndarray | None = None,
    fit_label: str | None = None,
    highlight_x: np.ndarray | None = None,
    highlight_y: np.ndarray | None = None,
    highlight_label: str | None = None,
    text_lines: Sequence[str] | None = None,
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    if sigma is not None and sigma.shape[0] == y.shape[0]:
        fmt = "o-" if connect_data else "o"
        ax.errorbar(x, y, yerr=sigma, fmt=fmt, linewidth=2, markersize=6, capsize=3, label=data_label)
    else:
        fmt = "o-" if connect_data else "o"
        ax.plot(x, y, fmt, linewidth=2, markersize=6, label=data_label)

    if highlight_x is not None and highlight_y is not None and highlight_x.size and highlight_y.size:
        ax.plot(
            highlight_x,
            highlight_y,
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="#0f172a",
            markeredgewidth=1.2,
            label=(highlight_label or "fit points"),
        )

    if fit_x is not None and fit_y is not None and fit_x.size and fit_y.size:
        ax.plot(fit_x, fit_y, "-", linewidth=2.2, color="#c43c35", label=(fit_label or "fit"))

    if text_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    if yscale:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def render_multi_series_plot(
    *,
    series: Sequence[XYSeries],
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str | None = None,
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for curve in series:
        if curve.x.size == 0 or curve.y.size == 0:
            continue
        if curve.sigma is not None and curve.sigma.shape[0] == curve.y.shape[0]:
            ax.errorbar(
                curve.x,
                curve.y,
                yerr=curve.sigma,
                fmt="o-",
                linewidth=2,
                markersize=5,
                capsize=3,
                label=curve.label,
                color=curve.color,
            )
        else:
            ax.plot(curve.x, curve.y, "o-", linewidth=2, markersize=5, label=curve.label, color=curve.color)

    if yscale:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title=legend_title, frameon=False, fontsize=10, loc="best")

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
