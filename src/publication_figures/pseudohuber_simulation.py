from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tc_fittings.tc_td_pseudohuber import tc_linear_largeTd, tc_pseudohuber

from .tc_param_vars import DEFAULT_COLORS, _set_publication_rc


def _validate_positive(values: Sequence[float], name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out:
        raise ValueError(f"{name} must contain at least one value.")
    bad = [value for value in out if not np.isfinite(value) or value <= 0]
    if bad:
        raise ValueError(f"{name} values must be finite and > 0. Got: {bad}")
    return out


def _curve_table(
    td_ms: np.ndarray,
    *,
    c: float,
    alpha_macro: float,
    deltas_ms: Sequence[float],
    show_common_slope_guide: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for delta in deltas_ms:
        model_c = c
        tc_vals = tc_pseudohuber(td_ms, model_c, delta, alpha_macro)
        rows.append(
            pd.DataFrame(
                {
                    "td_ms": td_ms,
                    "tc_ms": tc_vals,
                    "td_normalized": td_ms / float(delta),
                    "tc_normalized": tc_vals / float(delta),
                    "linear_large_td_ms": tc_linear_largeTd(td_ms, model_c, delta, alpha_macro),
                    "c_asymptote_ms": float(c),
                    "c_model_ms": float(model_c),
                    "alpha_macro": float(alpha_macro),
                    "delta_ms": float(delta),
                    "show_common_slope_guide": bool(show_common_slope_guide),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_pseudohuber_transition_publication(
    *,
    out_stem: str | Path,
    c: float = 20.0,
    alpha_macro: float = 0.55,
    deltas_ms: Sequence[float] = (35.0, 60.0, 85.0),
    td_min_ms: float = 0.0,
    td_max_ms: float = 180.0,
    n_points: int = 600,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 400,
    figsize: tuple[float, float] = (6, 6),
    xlabel: str = r"Diffusion time $T_d / \delta$",
    ylabel: str = r"Correlation time $\tau_c / \delta$",
    title: str | None = None,
    legend_title: str = r"Transition parameter $\delta$ [ms]",
    show_common_slope_guide: bool = False,
    linear_start_factor: float = 1.45,
) -> list[Path]:
    _set_publication_rc()
    deltas_ms = _validate_positive(deltas_ms, "deltas_ms")
    c = float(c)
    alpha_macro = float(alpha_macro)
    td_min_ms = float(td_min_ms)
    td_max_ms = float(td_max_ms)
    n_points = int(n_points)

    if not np.isfinite(c):
        raise ValueError("c must be finite.")
    if not np.isfinite(alpha_macro):
        raise ValueError("alpha_macro must be finite.")
    if not np.isfinite(td_min_ms) or not np.isfinite(td_max_ms) or td_max_ms <= td_min_ms:
        raise ValueError("td_max_ms must be greater than td_min_ms.")
    if n_points < 10:
        raise ValueError("n_points must be at least 10.")
    linear_start_factor = float(linear_start_factor)
    if not np.isfinite(linear_start_factor) or linear_start_factor < 0:
        raise ValueError("linear_start_factor must be finite and >= 0.")

    td_ms = np.linspace(td_min_ms, td_max_ms, n_points)
    plotted = _curve_table(
        td_ms,
        c=c,
        alpha_macro=alpha_macro,
        deltas_ms=deltas_ms,
        show_common_slope_guide=bool(show_common_slope_guide),
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = tuple(DEFAULT_COLORS)
    visible_y: list[np.ndarray] = []
    for idx, delta in enumerate(deltas_ms):
        sub = plotted[plotted["delta_ms"] == float(delta)]
        color = colors[idx % len(colors)]
        label = f"{delta:g} ms"
        curve_y = sub["tc_normalized"].to_numpy(dtype=float)
        visible_y.append(curve_y)
        ax.plot(
            sub["td_normalized"],
            sub["tc_normalized"],
            color=color,
            linewidth=2.3,
            label=label,
        )
        # Commented out dotted lines for large-Td linear asymptote
        # line_mask = sub["td_ms"].to_numpy(dtype=float) >= linear_start_factor * float(delta)
        # line_y = sub.loc[line_mask, "linear_large_td_ms"].to_numpy(dtype=float)
        # visible_y.append(line_y)
        # ax.plot(
        #     sub.loc[line_mask, "td_ms"],
        #     sub.loc[line_mask, "linear_large_td_ms"],
        #     color=color,
        #     linewidth=1.75,
        #     linestyle=(0, (1.2, 2.2)),
        #     alpha=0.86,
        # )
    if show_common_slope_guide:
        common_line = c + alpha_macro * td_ms
        visible_y.append(common_line)
        ax.plot(
            td_ms,
            common_line,
            color="0.18",
            linewidth=1.45,
            linestyle=(0, (4.0, 2.4)),
            alpha=0.72,
            label=r"common slope guide",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    
    # Set xlim based on normalized values
    td_normalized_min = plotted["td_normalized"].min()
    td_normalized_max = plotted["td_normalized"].max()
    ax.set_xlim(td_normalized_min, td_normalized_max)

    y_values = np.concatenate([y for y in visible_y if y.size]) if visible_y else np.asarray([], dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size:
        ymin = float(np.nanmin(y_values))
        ymax = float(np.nanmax(y_values))
        pad = 0.07 * (ymax - ymin) if ymax > ymin else max(0.1 * abs(ymax), 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title=legend_title, frameon=False, loc="upper left")

    ax.text(
        0.98,
        0.04,
        rf"$c={c:g}$ ms, $\alpha={alpha_macro:g}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.20",
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.96 if title else 0.98, bottom=0.15)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{str(fmt).lower().lstrip('.')}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        outputs.append(out_path)
    plt.close(fig)

    plotted.to_csv(out_stem.parent / f"{out_stem.name}_plotted_curves.csv", index=False)
    return outputs


def default_output_stem(out_dir: str | Path) -> Path:
    return Path(out_dir) / "tc_vs_td_pseudohuber_delta_transition_publication"
