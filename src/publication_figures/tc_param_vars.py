from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BRAIN_MARKERS = ("o", "s", "^", "D", "v", "P")
DEFAULT_PHANTOM_MARKER = "D"
DEFAULT_COLORS = ("#2a6fbb", "#d55e00", "#009e73", "#cc79a7", "#000000", "#56b4e9")
DEFAULT_ROI_COLORS = ("#2a6fbb", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#f0e442", "#0072b2")
CC_ROI_COLORS = {
    "PostCC": "#7ed321",
    "MidPostCC": "#56b4e9",
    "CentralCC": "#d62728",
    "MidAntCC": "#e69f00",
    "AntCC": "#8e44ad",
}
CC_ROI_LABELS = {
    "PostCC": "Post",
    "MidPostCC": "MidPost",
    "CentralCC": "Central",
    "MidAntCC": "MidAnt",
    "AntCC": "Ant",
}


@dataclass(frozen=True)
class TcParamVarsSpec:
    name: str
    label: str
    table: Path
    roi_order: tuple[str, ...]
    direction_order: tuple[str, ...]
    direction_aliases: Mapping[str, str]
    subject_aliases: Mapping[str, str] | None = None


VAR_SPECS: dict[str, tuple[str, str | None, str]] = {
    "q_quad": ("q_quad", "q_quad_se", r"$q = \alpha/(2\delta)$"),
    "alpha_macro": ("alpha_macro", "alpha_macro_se", "Macroscopic tortuosity\ncoefficient $\\alpha$"),
    "delta": ("delta", "delta_se", "Transition parameter\n$\\delta$ [ms]"),
    "A": ("A", "A_se", r"$A = \alpha/\delta$"),
    "c": ("c", "c_se", r"$c$"),
    "sqrt_q": ("sqrt_q", "sqrt_q_se", r"$\sqrt{q}$"),
}


def _set_publication_rc() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _normalize_direction(value: object, aliases: Mapping[str, str]) -> str:
    text = str(value).strip()
    return str(aliases.get(text, text)).strip()


def _sanitize_token(value: object) -> str:
    text = str(value).strip()
    keep = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    return "".join(keep).strip("_") or "NA"


def _subject_display_map(spec: TcParamVarsSpec, subjects: Sequence[str]) -> dict[str, str]:
    if spec.subject_aliases:
        return {str(subj): str(spec.subject_aliases.get(str(subj), subj)) for subj in subjects}
    ordered = sorted(str(x) for x in subjects)
    if spec.name.startswith("brains"):
        return {subj: f"subj{idx + 1}" for idx, subj in enumerate(ordered)}
    return {subj: "phantom" for subj in ordered}


def _subject_marker_map(spec: TcParamVarsSpec, subjects: Sequence[str]) -> dict[str, str]:
    ordered = sorted(str(x) for x in subjects)
    if spec.name.startswith("brains"):
        return {subj: DEFAULT_BRAIN_MARKERS[idx % len(DEFAULT_BRAIN_MARKERS)] for idx, subj in enumerate(ordered)}
    return {subj: DEFAULT_PHANTOM_MARKER for subj in ordered}


def _subject_color_map(subjects: Sequence[str]) -> dict[str, str]:
    ordered = sorted(str(x) for x in subjects)
    return {subj: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx, subj in enumerate(ordered)}


def _roi_color_map(rois: Sequence[str]) -> dict[str, str]:
    return {
        str(roi): CC_ROI_COLORS.get(str(roi), DEFAULT_ROI_COLORS[idx % len(DEFAULT_ROI_COLORS)])
        for idx, roi in enumerate(rois)
    }


def _roi_display_name(roi: object) -> str:
    text = str(roi)
    return CC_ROI_LABELS.get(text, text)


def _aggregate_duplicate_directions(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    value_cols = [VAR_SPECS[var][0] for var in variables if var in VAR_SPECS]
    err_cols = [VAR_SPECS[var][1] for var in variables if var in VAR_SPECS and VAR_SPECS[var][1] is not None]
    err_cols = [col for col in err_cols if col in df.columns]
    rows: list[dict[str, object]] = []
    for key, sub in df.groupby(["subj", "roi", "direction"], sort=False, dropna=False):
        row: dict[str, object] = {"subj": key[0], "roi": key[1], "direction": key[2]}
        for col in value_cols:
            if col in sub.columns:
                row[col] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        for col in err_cols:
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[col] = float(np.sqrt(np.mean(finite**2)) / np.sqrt(len(finite))) if finite.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "A" not in out.columns and {"alpha_macro", "delta"}.issubset(out.columns):
        alpha = pd.to_numeric(out["alpha_macro"], errors="coerce").to_numpy(dtype=float)
        delta = pd.to_numeric(out["delta"], errors="coerce").to_numpy(dtype=float)
        out["A"] = np.where(np.isfinite(alpha) & np.isfinite(delta) & (delta != 0), alpha / delta, np.nan)

    if "A_se" not in out.columns:
        out["A_se"] = np.nan
        required = {"alpha_macro", "delta", "alpha_macro_se", "delta_se"}
        if required.issubset(out.columns):
            alpha = pd.to_numeric(out["alpha_macro"], errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(out["delta"], errors="coerce").to_numpy(dtype=float)
            alpha_se = pd.to_numeric(out["alpha_macro_se"], errors="coerce").to_numpy(dtype=float)
            delta_se = pd.to_numeric(out["delta_se"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(alpha) & np.isfinite(delta) & (delta > 0) & np.isfinite(alpha_se) & np.isfinite(delta_se)
            variance = np.full(len(out), np.nan, dtype=float)
            variance[ok] = (alpha_se[ok] / delta[ok]) ** 2 + ((alpha[ok] * delta_se[ok]) / (delta[ok] ** 2)) ** 2
            out["A_se"] = np.sqrt(variance)

    if "sqrt_q" not in out.columns and "q_quad" in out.columns:
        q = pd.to_numeric(out["q_quad"], errors="coerce").to_numpy(dtype=float)
        out["sqrt_q"] = np.where(q > 0, np.sqrt(q), np.nan)
    if "sqrt_q_se" not in out.columns:
        out["sqrt_q_se"] = np.nan
        if {"q_quad", "q_quad_se", "sqrt_q"}.issubset(out.columns):
            q = pd.to_numeric(out["q_quad"], errors="coerce").to_numpy(dtype=float)
            q_se = pd.to_numeric(out["q_quad_se"], errors="coerce").to_numpy(dtype=float)
            sqrt_q = pd.to_numeric(out["sqrt_q"], errors="coerce").to_numpy(dtype=float)
            ok = (q > 0) & np.isfinite(q_se) & np.isfinite(sqrt_q) & (sqrt_q > 0)
            values = np.full(len(out), np.nan, dtype=float)
            values[ok] = q_se[ok] / (2.0 * sqrt_q[ok])
            out["sqrt_q_se"] = values

    return out


def load_param_vars(spec: TcParamVarsSpec, variables: Sequence[str]) -> pd.DataFrame:
    df = _ensure_derived_columns(_read_table(spec.table).copy())
    required = {"subj", "roi", "direction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{spec.table} is missing columns: {missing}")

    df["subj"] = df["subj"].astype(str).str.strip()
    df["roi"] = df["roi"].astype(str).str.replace("_norm", "", regex=False).str.strip()
    df["direction"] = df["direction"].map(lambda x: _normalize_direction(x, spec.direction_aliases))
    if spec.roi_order:
        df = df[df["roi"].isin([str(x) for x in spec.roi_order])].copy()
    if spec.direction_order:
        df = df[df["direction"].isin([str(x) for x in spec.direction_order])].copy()

    for var in variables:
        if var not in VAR_SPECS:
            raise ValueError(f"Unknown variable {var!r}. Expected one of {sorted(VAR_SPECS)}.")
        val_col, err_col, _ = VAR_SPECS[var]
        if val_col in df.columns:
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        if err_col and err_col in df.columns:
            df[err_col] = pd.to_numeric(df[err_col], errors="coerce")

    return _aggregate_duplicate_directions(df, variables)


def _row_ylim(df: pd.DataFrame, variables: Sequence[str], var: str, show_errorbars: bool) -> tuple[float, float] | None:
    val_col, err_col, _ = VAR_SPECS[var]
    if val_col not in df.columns:
        return None
    values = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
    if err_col and err_col in df.columns and show_errorbars:
        errors = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
    else:
        errors = np.zeros_like(values)
    m = np.isfinite(values) & np.isfinite(errors)
    if not np.any(m):
        return None
    lo = float(np.nanmin(values[m] - errors[m]))
    hi = float(np.nanmax(values[m] + errors[m]))
    pad = 0.08 * (hi - lo) if hi > lo else max(0.1 * abs(hi), 1.0)
    return lo - pad, hi + pad


def plot_tc_param_vars_publication(
    spec: TcParamVarsSpec,
    *,
    out_stem: str | Path,
    variables: Sequence[str] = ("q_quad", "alpha_macro", "delta", "A", "c", "sqrt_q"),
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 400,
    show_errorbars: bool = True,
    figsize: tuple[float, float] | None = None,
) -> list[Path]:
    _set_publication_rc()
    variables = tuple(str(v) for v in variables)
    df = load_param_vars(spec, variables)
    if df.empty:
        raise ValueError(f"No rows available for {spec.label}.")

    regions = [roi for roi in spec.roi_order if roi in set(df["roi"].astype(str))]
    if not regions:
        regions = sorted(df["roi"].astype(str).unique().tolist())
    directions = [direction for direction in spec.direction_order if direction in set(df["direction"].astype(str))]
    if not directions:
        directions = sorted(df["direction"].astype(str).unique().tolist())
    subjects = sorted(df["subj"].astype(str).unique().tolist())
    display = _subject_display_map(spec, subjects)
    markers = _subject_marker_map(spec, subjects)
    colors = _subject_color_map(subjects)

    nrows = len(variables)
    ncols = len(directions)
    if figsize is None:
        figsize = (5.4 * ncols, 2.55 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey="row", squeeze=False)
    x = np.arange(len(regions))

    for col, direction in enumerate(directions):
        df_dir = df[df["direction"].astype(str) == str(direction)].copy()
        for row_idx, var in enumerate(variables):
            ax = axes[row_idx, col]
            val_col, err_col, y_label = VAR_SPECS[var]
            for subj in subjects:
                sub = df_dir[df_dir["subj"].astype(str) == subj].set_index("roi").reindex(regions)
                y = pd.to_numeric(sub[val_col], errors="coerce").to_numpy(dtype=float) if val_col in sub.columns else np.full(len(regions), np.nan)
                if err_col and err_col in sub.columns:
                    yerr = pd.to_numeric(sub[err_col], errors="coerce").to_numpy(dtype=float)
                else:
                    yerr = np.full(len(regions), np.nan)
                if show_errorbars:
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        color=colors[subj],
                        marker=markers[subj],
                        linestyle="-",
                        linewidth=1.9,
                        markersize=5.5,
                        capsize=2.8,
                        label=display[subj],
                    )
                else:
                    ax.plot(x, y, color=colors[subj], marker=markers[subj], linestyle="-", linewidth=1.9, markersize=5.5, label=display[subj])
            ax.grid(True, alpha=0.22, linewidth=0.8)
            ax.set_title(str(direction), fontsize=16)
            if col == 0:
                ax.set_ylabel(y_label)

            limits = _row_ylim(df, variables, var, show_errorbars)
            if limits is not None:
                ax.set_ylim(*limits)

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=25, ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.12 if handles else 0.08, hspace=0.34, wspace=0.13)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{str(fmt).lower().lstrip('.')}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        outputs.append(out_path)
    plt.close(fig)

    plotted = out_stem.parent / f"{out_stem.name}_plotted_rows.csv"
    df.to_csv(plotted, index=False)
    return outputs


def default_output_stem(out_dir: str | Path, spec: TcParamVarsSpec, variables: Sequence[str]) -> Path:
    dirs = "_".join(_sanitize_token(x) for x in spec.direction_order)
    return (
        Path(out_dir)
        / f"vars_sameY_dirs={dirs}_pseudohuber_fixed_macro_k=None_y=tc_peak_resampled_data_ms_mode=fixed_macro_publication_{spec.name}"
    )


def plot_delta_vs_alpha_publication(
    spec: TcParamVarsSpec,
    *,
    out_stem: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 400,
    show_errorbars: bool = True,
    label_points: bool = True,
    figsize: tuple[float, float] | None = None,
) -> list[Path]:
    _set_publication_rc()
    df = load_param_vars(spec, ("alpha_macro", "delta"))
    if df.empty:
        raise ValueError(f"No rows available for {spec.label}.")

    regions = [roi for roi in spec.roi_order if roi in set(df["roi"].astype(str))]
    if not regions:
        regions = sorted(df["roi"].astype(str).unique().tolist())
    directions = [direction for direction in spec.direction_order if direction in set(df["direction"].astype(str))]
    if not directions:
        directions = sorted(df["direction"].astype(str).unique().tolist())
    subjects = sorted(df["subj"].astype(str).unique().tolist())
    display = _subject_display_map(spec, subjects)
    markers = _subject_marker_map(spec, subjects)
    roi_colors = _roi_color_map(regions)
    show_panel_errorbars = False
    show_point_labels = bool(label_points)

    if figsize is None:
        figsize = (5.9 * len(directions), 5.5)
    fig, axes = plt.subplots(1, len(directions), figsize=figsize, sharey=True, squeeze=False)
    axes_row = axes[0]

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for ax, direction in zip(axes_row, directions):
        panel = df[df["direction"].astype(str) == str(direction)].copy()
        for subj in subjects:
            sub = panel[panel["subj"].astype(str) == subj].set_index("roi").reindex(regions)
            for roi in regions:
                row = sub.loc[roi] if roi in sub.index else None
                if row is None:
                    continue
                x = float(pd.to_numeric(pd.Series([row.get("alpha_macro")]), errors="coerce").iloc[0])
                y = float(pd.to_numeric(pd.Series([row.get("delta")]), errors="coerce").iloc[0])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                xerr = float(pd.to_numeric(pd.Series([row.get("alpha_macro_se")]), errors="coerce").iloc[0]) if "alpha_macro_se" in sub.columns else np.nan
                yerr = float(pd.to_numeric(pd.Series([row.get("delta_se")]), errors="coerce").iloc[0]) if "delta_se" in sub.columns else np.nan
                color = roi_colors.get(roi, "#000000")
                if show_panel_errorbars and (np.isfinite(xerr) or np.isfinite(yerr)):
                    ax.errorbar(
                        [x],
                        [y],
                        xerr=None if not np.isfinite(xerr) else [xerr],
                        yerr=None if not np.isfinite(yerr) else [yerr],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.1,
                        capsize=2.5,
                        alpha=0.58,
                        zorder=2,
                    )
                ax.scatter(
                    [x],
                    [y],
                    s=100.0,
                    marker=markers[subj],
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.92,
                    zorder=3,
                )
                if show_point_labels:
                    ax.annotate(
                        _roi_display_name(roi),
                        (x, y),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=14,
                        color="0.10",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="none", alpha=0.24),
                        zorder=4,
                    )
                x_all.append(np.asarray([x], dtype=float))
                y_all.append(np.asarray([y], dtype=float))

        ax.set_title(str(direction), fontsize=16)
        ax.set_xlabel("Macroscopic tortuosity coefficient $\\alpha$")
        ax.grid(True, alpha=0.24, linewidth=0.8)

    axes_row[0].set_ylabel("Transition parameter $\\delta$ [ms]")
    if x_all:
        xs = np.concatenate(x_all)
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        xpad = 0.08 * (xmax - xmin) if xmax > xmin else max(0.02 * abs(xmax), 0.02)
        for ax in axes_row:
            ax.set_xlim(xmin - xpad, xmax + xpad)
    if y_all:
        ys = np.concatenate(y_all)
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        ypad = 0.10 * (ymax - ymin) if ymax > ymin else max(0.05 * abs(ymax), 10.0)
        for ax in axes_row:
            ax.set_ylim(ymin - ypad, ymax + ypad)

    import matplotlib.lines as mlines

    subject_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker=markers[subj],
            linestyle="None",
            color="0.1",
            markerfacecolor="0.1",
            markeredgecolor="white",
            markersize=8,
            label=display[subj],
        )
        for subj in subjects
    ]
    handles = subject_handles
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.20 if handles else 0.12, wspace=0.12)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{str(fmt).lower().lstrip('.')}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        outputs.append(out_path)
    plt.close(fig)

    plotted = out_stem.parent / f"{out_stem.name}_plotted_rows.csv"
    df.to_csv(plotted, index=False)
    return outputs


def default_delta_vs_alpha_output_stem(out_dir: str | Path, spec: TcParamVarsSpec) -> Path:
    dirs = "_".join(_sanitize_token(x) for x in spec.direction_order)
    return Path(out_dir) / f"delta_vs_alpha_macro_pseudohuber_fixed_macro_k=None_publication_{spec.name}_dirs={dirs}"
