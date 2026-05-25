from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_DIRECTION_COLORS = {
    "longitudinal": "#2a6fbb",
    "transversal": "#c44e52",
    "long": "#2a6fbb",
    "tra": "#c44e52",
    "1": "#c44e52",
    "3": "#2a6fbb",
}
DEFAULT_BRAIN_MARKERS = ("o", "s", "^", "P", "X")
DEFAULT_PHANTOM_MARKER = "D"


@dataclass(frozen=True)
class DatasetFigureSpec:
    name: str
    label: str
    alpha_table: Path
    delta_table: Path
    roi_order: tuple[str, ...]
    direction_order: tuple[str, ...]
    direction_aliases: Mapping[str, str]


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
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


def _pick_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lower_to_name = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        col = lower_to_name.get(candidate.lower())
        if col is not None:
            return col
    if required:
        raise KeyError(f"Could not resolve any of these columns: {list(candidates)}")
    return None


def _normalize_direction(value: object, aliases: Mapping[str, str]) -> str:
    text = str(value).strip()
    return str(aliases.get(text, text)).strip()


def _is_phantom_dataset(dataset: str) -> bool:
    return str(dataset).startswith("phantoms")


def _collapse_phantom_rows(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out

    collapsed = out.copy()
    collapsed["_priority"] = collapsed["subj"].astype(str).str.contains("DDE", case=False, na=False).astype(int)
    collapsed = collapsed.sort_values(["dataset", "metric", "roi", "direction", "_priority", "subj"], kind="stable")
    collapsed = collapsed.drop_duplicates(subset=["dataset", "metric", "roi", "direction"], keep="first")
    collapsed = collapsed.drop(columns=["_priority"])
    collapsed["subj"] = "phantom"
    collapsed["error"] = np.nan
    return collapsed.reset_index(drop=True)


def _single_centered_jitter(width: float, n_items: int, fraction: float) -> np.ndarray:
    if n_items <= 1:
        return np.array([0.0])
    return np.linspace(-width * fraction, width * fraction, n_items)


def _filter_preferred_alpha_rows(df: pd.DataFrame, direction_order: Sequence[str]) -> pd.DataFrame:
    if "direction_kind" not in df.columns:
        return df

    derived = df[df["direction_kind"].astype(str).str.lower() == "derived"].copy()
    if not derived.empty:
        return derived
    return df[df["direction_kind"].astype(str).str.lower() != "raw_ignored"].copy()


def load_metric_table(
    path: str | Path,
    *,
    dataset: str,
    metric: str,
    value_candidates: Sequence[str],
    error_candidates: Sequence[str],
    roi_order: Sequence[str],
    direction_order: Sequence[str],
    direction_aliases: Mapping[str, str],
    prefer_alpha_derived_rows: bool = False,
) -> pd.DataFrame:
    df = _read_table(path)
    if prefer_alpha_derived_rows:
        df = _filter_preferred_alpha_rows(df, direction_order)

    subj_col = _pick_column(df, ["subj", "subject", "brain"], required=False)
    roi_col = _pick_column(df, ["roi", "region"])
    direction_col = _pick_column(df, ["direction", "direccion", "dir"])
    value_col = _pick_column(df, value_candidates)
    error_col = _pick_column(df, error_candidates, required=False)

    out = pd.DataFrame(
        {
            "dataset": dataset,
            "metric": metric,
            "subj": df[subj_col].astype(str).str.strip() if subj_col else dataset,
            "roi": df[roi_col].astype(str).str.strip().str.replace("_norm", "", regex=False),
            "direction": df[direction_col].map(lambda x: _normalize_direction(x, direction_aliases)),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
            "source_table": str(Path(path)),
        }
    )
    if error_col is not None:
        out["error"] = pd.to_numeric(df[error_col], errors="coerce")
    else:
        out["error"] = np.nan

    roi_set = {str(x) for x in roi_order}
    direction_set = {str(x) for x in direction_order}
    out = out[out["roi"].isin(roi_set) & out["direction"].isin(direction_set)].copy()
    out = out.dropna(subset=["value"])
    if _is_phantom_dataset(dataset):
        out = _collapse_phantom_rows(out)
    return out.reset_index(drop=True)


def load_dataset_metrics(spec: DatasetFigureSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = load_metric_table(
        spec.alpha_table,
        dataset=spec.name,
        metric="alpha_macro",
        value_candidates=["alpha_macro", "alpha"],
        error_candidates=["alpha_macro_error", "alpha_macro_se", "alpha_error", "alpha_se"],
        roi_order=spec.roi_order,
        direction_order=spec.direction_order,
        direction_aliases=spec.direction_aliases,
        prefer_alpha_derived_rows=True,
    )
    delta = load_metric_table(
        spec.delta_table,
        dataset=spec.name,
        metric="delta",
        value_candidates=["delta"],
        error_candidates=["delta_se", "delta_error", "delta_err"],
        roi_order=spec.roi_order,
        direction_order=spec.direction_order,
        direction_aliases=spec.direction_aliases,
    )
    return delta, alpha


def aggregate_metric(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, sub in df.groupby(["dataset", "metric", "roi", "direction"], sort=False):
        values = sub["value"].to_numpy(dtype=float)
        errors = sub["error"].to_numpy(dtype=float) if "error" in sub.columns else np.full(len(sub), np.nan)
        n = int(np.isfinite(values).sum())
        if n == 0:
            continue

        mean = float(np.nanmean(values))
        if n == 1:
            err = float(errors[np.isfinite(errors)][0]) if np.isfinite(errors).any() else np.nan
        else:
            between = float(np.nanstd(values, ddof=1) / np.sqrt(n))
            source = float(np.sqrt(np.nanmean(errors[np.isfinite(errors)] ** 2)) / np.sqrt(n)) if np.isfinite(errors).any() else 0.0
            err = float(np.sqrt(between**2 + source**2))

        rows.append(
            {
                "dataset": key[0],
                "metric": key[1],
                "roi": key[2],
                "direction": key[3],
                "value_mean": mean,
                "value_error": err,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def build_plot_tables(specs: Sequence[DatasetFigureSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        delta, alpha = load_dataset_metrics(spec)
        frames.extend([delta, alpha])
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    agg = aggregate_metric(raw) if not raw.empty else pd.DataFrame()
    return raw, agg


def _set_publication_rc() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def _metric_label(metric: str) -> str:
    if metric == "delta":
        return r"$\delta$ [ms]"
    if metric == "alpha_macro":
        return r"$\alpha$"
    return metric


def _subject_display_map(spec: DatasetFigureSpec, subjects: Sequence[str]) -> dict[str, str]:
    ordered = sorted(str(x) for x in subjects)
    if spec.name.startswith("brains"):
        return {subj: f"subj{idx + 1}" for idx, subj in enumerate(ordered)}
    return {subj: "phantom" for subj in ordered}


def _subject_marker_map(spec: DatasetFigureSpec, subjects: Sequence[str]) -> dict[str, str]:
    ordered = sorted(str(x) for x in subjects)
    if spec.name.startswith("brains"):
        return {subj: DEFAULT_BRAIN_MARKERS[idx % len(DEFAULT_BRAIN_MARKERS)] for idx, subj in enumerate(ordered)}
    return {subj: DEFAULT_PHANTOM_MARKER for subj in ordered}


def _metric_limits(raw: pd.DataFrame, metric: str, *, delta_ymin: float | None = None, alpha_ymin: float = 0.0) -> tuple[float, float] | None:
    sub = raw[raw["metric"] == metric].copy()
    if sub.empty:
        return None

    values = pd.to_numeric(sub["value"], errors="coerce").to_numpy(float)
    errors = pd.to_numeric(sub.get("error", np.nan), errors="coerce").to_numpy(float)
    finite = np.isfinite(values)
    if not finite.any():
        return None

    clean_errors = np.where(np.isfinite(errors), errors, 0.0)
    lows = values[finite] - clean_errors[finite]
    highs = values[finite] + clean_errors[finite]

    if metric == "delta" and delta_ymin is not None:
        ymin = float(delta_ymin)
    elif metric == "alpha_macro":
        ymin = float(alpha_ymin)
    else:
        ymin = float(np.nanmin(lows))

    ymax = float(np.nanmax(highs))
    if not np.isfinite(ymax):
        return None
    span = max(ymax - ymin, abs(ymax) * 0.1, 1e-9)
    if metric == "delta" and delta_ymin is None:
        ymin = float(np.nanmin(lows) - 0.08 * span)
    return ymin, ymax + 0.08 * span


def _plot_points_metric_panel(
    ax: plt.Axes,
    *,
    raw: pd.DataFrame,
    spec: DatasetFigureSpec,
    metric: str,
    colors: Mapping[str, str],
    point_width: float,
) -> list[object]:
    roi_order = list(spec.roi_order)
    direction_order = list(spec.direction_order)
    x = np.arange(len(roi_order), dtype=float)
    n_dirs = max(1, len(direction_order))
    offsets = (np.arange(n_dirs) - (n_dirs - 1) / 2.0) * point_width

    handles: list[object] = []
    cell_raw = raw[(raw["dataset"] == spec.name) & (raw["metric"] == metric)].copy()
    if cell_raw.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return handles

    subj_order = sorted(cell_raw["subj"].dropna().astype(str).unique().tolist())
    display_labels = _subject_display_map(spec, subj_order)
    markers = _subject_marker_map(spec, subj_order)
    subj_jitter = _single_centered_jitter(point_width, len(subj_order), 0.22)
    jitter = dict(zip(subj_order, subj_jitter))

    for idx, direction in enumerate(direction_order):
        color = colors.get(direction, DEFAULT_DIRECTION_COLORS.get(direction, None))
        sub_raw = cell_raw[cell_raw["direction"] == direction]
        for roi_idx, roi in enumerate(roi_order):
            points = sub_raw[sub_raw["roi"] == roi]
            for _, point in points.iterrows():
                subj = str(point["subj"])
                value = float(point["value"])
                error = float(point["error"]) if np.isfinite(float(point.get("error", np.nan))) else np.nan
                ax.errorbar(
                    x[roi_idx] + offsets[idx] + jitter.get(subj, 0.0),
                    value,
                    yerr=None if not np.isfinite(error) else error,
                    marker=markers.get(subj, "o"),
                    markersize=7.5,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    linestyle="none",
                    zorder=5,
                    label=display_labels.get(subj, subj),
                )

    ax.set_xticks(x)
    ax.set_xticklabels(roi_order, rotation=35, ha="right")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.04)
    return handles


def _plot_bars_metric_panel(
    ax: plt.Axes,
    *,
    raw: pd.DataFrame,
    agg: pd.DataFrame,
    spec: DatasetFigureSpec,
    metric: str,
    colors: Mapping[str, str],
    bar_width: float,
) -> list[object]:
    roi_order = list(spec.roi_order)
    direction_order = list(spec.direction_order)
    x = np.arange(len(roi_order), dtype=float)
    n_dirs = max(1, len(direction_order))
    offsets = (np.arange(n_dirs) - (n_dirs - 1) / 2.0) * bar_width

    cell_agg = agg[(agg["dataset"] == spec.name) & (agg["metric"] == metric)].copy()
    if cell_agg.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return []

    cell_raw = raw[(raw["dataset"] == spec.name) & (raw["metric"] == metric)].copy()
    subj_order = sorted(cell_raw["subj"].dropna().astype(str).unique().tolist())
    markers = _subject_marker_map(spec, subj_order)
    subj_jitter = _single_centered_jitter(bar_width, len(subj_order), 0.18)
    jitter = dict(zip(subj_order, subj_jitter))

    for idx, direction in enumerate(direction_order):
        color = colors.get(direction, DEFAULT_DIRECTION_COLORS.get(direction, None))
        for roi_idx, roi in enumerate(roi_order):
            sub = cell_agg[(cell_agg["direction"] == direction) & (cell_agg["roi"] == roi)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            value = float(row["value_mean"])
            error = float(row["value_error"]) if np.isfinite(float(row.get("value_error", np.nan))) else np.nan
            ax.bar(
                x[roi_idx] + offsets[idx],
                value,
                width=bar_width * 0.82,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.72,
                zorder=3,
            )
            if np.isfinite(error):
                ax.errorbar(
                    x[roi_idx] + offsets[idx],
                    value,
                    yerr=error,
                    color="black",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=4,
                    linestyle="none",
                    zorder=8,
                )
            points = cell_raw[(cell_raw["direction"] == direction) & (cell_raw["roi"] == roi)]
            for _, point in points.iterrows():
                subj = str(point["subj"])
                value = float(point["value"])
                ax.plot(
                    x[roi_idx] + offsets[idx] + jitter.get(subj, 0.0),
                    value,
                    marker=markers.get(subj, "o"),
                    markersize=7.2,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    alpha=0.72,
                    linestyle="none",
                    zorder=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(roi_order, rotation=35, ha="right")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.margins(x=0.04)
    return []


def plot_delta_alpha_figure(
    raw: pd.DataFrame,
    agg: pd.DataFrame,
    specs: Sequence[DatasetFigureSpec],
    *,
    out_stem: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 400,
    colors: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    delta_ymin: float | None = None,
    phantoms_delta_ymin: float | None = None,
    alpha_ymin: float = 0.0,
    brains_alpha_ymax: float | None = None,
    plot_mode: str = "points",
) -> list[Path]:
    _set_publication_rc()
    colors = dict(DEFAULT_DIRECTION_COLORS if colors is None else colors)
    plot_mode = str(plot_mode).strip().lower()
    if plot_mode not in {"points", "bars"}:
        raise ValueError(f"plot_mode must be 'points' or 'bars'. Received: {plot_mode!r}")

    specs = list(specs)
    ncols = len(specs)
    if ncols == 0:
        raise ValueError("At least one dataset spec is required.")
    width_ratios = [max(1, len(spec.roi_order)) for spec in specs]
    if figsize is None:
        figsize = (1.45 * sum(width_ratios) + 2.0, 6.8)

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    for col, spec in enumerate(specs):
        for row, metric in enumerate(["delta", "alpha_macro"]):
            ax = axes[row, col]
            if plot_mode == "bars":
                _plot_bars_metric_panel(
                    ax,
                    raw=raw,
                    agg=agg,
                    spec=spec,
                    metric=metric,
                    colors=colors,
                    bar_width=0.34 if len(spec.direction_order) <= 2 else 0.25,
                )
            else:
                _plot_points_metric_panel(
                    ax,
                    raw=raw,
                    spec=spec,
                    metric=metric,
                    colors=colors,
                    point_width=0.34 if len(spec.direction_order) <= 2 else 0.25,
                )
            if row == 0:
                ax.set_title(spec.label, fontsize=18, pad=9)

    for row, metric in enumerate(["delta", "alpha_macro"]):
        if metric == "delta":
            for col, spec in enumerate(specs):
                dataset_raw = raw[raw["dataset"] == spec.name]
                dataset_delta_ymin = phantoms_delta_ymin if _is_phantom_dataset(spec.name) else delta_ymin
                limits = _metric_limits(dataset_raw, metric, delta_ymin=dataset_delta_ymin, alpha_ymin=alpha_ymin)
                if limits is not None:
                    axes[row, col].set_ylim(*limits)
            continue

        for col, spec in enumerate(specs):
            dataset_raw = raw[raw["dataset"] == spec.name]
            limits = _metric_limits(dataset_raw, metric, delta_ymin=delta_ymin, alpha_ymin=alpha_ymin)
            if limits is None:
                continue
            if spec.name.startswith("brains") and brains_alpha_ymax is not None:
                limits = (limits[0], float(brains_alpha_ymax))
            axes[row, col].set_ylim(*limits)

    direction_handles = [
        Line2D([0], [0], color=colors.get(direction, DEFAULT_DIRECTION_COLORS.get(direction, "black")), lw=2, label=direction)
        for direction in specs[0].direction_order
    ]
    marker_handles: list[Line2D] = []
    if plot_mode in {"points", "bars"}:
        seen_marker_labels: set[str] = set()
        for spec in specs:
            subjects = sorted(raw.loc[raw["dataset"] == spec.name, "subj"].dropna().astype(str).unique().tolist())
            display_labels = _subject_display_map(spec, subjects)
            markers = _subject_marker_map(spec, subjects)
            for subj in subjects:
                label = display_labels.get(subj, subj)
                if label in seen_marker_labels:
                    continue
                seen_marker_labels.add(label)
                marker_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=markers.get(subj, "o"),
                        color="black",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        lw=0,
                        markersize=9,
                        label=label,
                    )
                )

    fig.legend(
        direction_handles + marker_handles,
        [h.get_label() for h in direction_handles + marker_handles],
        loc="upper center",
        ncol=max(1, min(6, len(direction_handles) + len(marker_handles))),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        fmt_clean = str(fmt).lower().lstrip(".")
        out_path = out_stem.with_suffix(f".{fmt_clean}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt_clean == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def write_plot_tables(raw: pd.DataFrame, agg: pd.DataFrame, out_dir: str | Path, *, prefix: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{prefix}.raw_points.csv"
    agg_path = out_dir / f"{prefix}.point_summary.csv"
    raw.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)
    return [raw_path, agg_path]
