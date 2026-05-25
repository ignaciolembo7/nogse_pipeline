from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ogse_fitting.contrast_fit_panels import (
    _build_fit_curve,
    _extract_plot_arrays,
    _load_contrast_table_cached,
    _resolve_contrast_parquet,
    _subset_group,
)
from ogse_fitting.contrast_tc_peak_panels import _transform_x
from tc_fittings.contrast_fit_table import load_contrast_fit_params
from tc_fittings.tc_td_pseudohuber import tc_pseudohuber


DEFAULT_DIRECTION_COLORS = {
    "longitudinal": "#2a6fbb",
    "transversal": "#c44e52",
    "long": "#2a6fbb",
    "tra": "#c44e52",
}

TD_COLORS = ("#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#000000", "#56B4E9")


@dataclass(frozen=True)
class ContrastDatasetSpec:
    name: str
    label: str
    fits: Path
    contrast_root: Path
    tc_table: Path
    roi: str
    roi_label: str | None
    subj: str
    directions: tuple[str, ...]
    direction_aliases: Mapping[str, str]
    peak_D0_fix: float
    exclude_td_ms: tuple[float, ...] = ()
    fit_pattern: str = "**/fit_params.*"
    model: str = "rest"


def _set_publication_rc() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )


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


def _normalize_direction(value: object, aliases: Mapping[str, str]) -> str:
    text = str(value).strip()
    return str(aliases.get(text, text)).strip()


def _exclude_td(df: pd.DataFrame, exclude_td_ms: Sequence[float]) -> pd.DataFrame:
    if not exclude_td_ms or df.empty or "td_ms" not in df.columns:
        return df
    td = pd.to_numeric(df["td_ms"], errors="coerce").to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    for td_excl in exclude_td_ms:
        keep &= ~np.isclose(td, float(td_excl), atol=1e-3, equal_nan=False)
    return df.loc[keep].copy()


def _prefer_complete_direction_sets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "analysis_id" not in df.columns:
        return df

    rows: list[pd.Series] = []
    group_cols = [c for c in ["subj", "roi", "direction", "td_ms", "model", "gbase", "ycol", "xplot"] if c in df.columns]
    for _, sub in df.groupby(group_cols, sort=False, dropna=False):
        if len(sub) == 1:
            rows.append(sub.iloc[0])
            continue

        analysis = sub["analysis_id"].astype(str)
        complete = sub[analysis.str.contains("dir1-2-3|dirlong-tra", regex=True, na=False)]
        rows.append((complete if not complete.empty else sub).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def load_contrast_rows(spec: ContrastDatasetSpec) -> pd.DataFrame:
    df = load_contrast_fit_params(
        [spec.fits],
        pattern=spec.fit_pattern,
        models=[spec.model],
        subjs=[spec.subj] if spec.subj else None,
        directions=list(spec.directions),
        rois=[spec.roi],
        ok_only=True,
    )
    df = _exclude_td(df, spec.exclude_td_ms)
    df = _prefer_complete_direction_sets(df)
    if df.empty:
        raise ValueError(f"No contrast fits found for {spec.label}: subj={spec.subj}, roi={spec.roi}.")
    return df.sort_values(["direction", "td_ms"], kind="stable").reset_index(drop=True)


def load_tc_params(spec: ContrastDatasetSpec) -> pd.DataFrame:
    df = _read_table(spec.tc_table)
    required = {"subj", "roi", "direction", "c", "delta", "alpha_macro"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{spec.tc_table} is missing columns: {missing}")

    out = df.copy()
    out["subj"] = out["subj"].astype(str).str.strip()
    out["roi"] = out["roi"].astype(str).str.replace("_norm", "", regex=False).str.strip()
    out["direction_raw"] = out["direction"].astype(str).str.strip()
    out["direction_label"] = out["direction_raw"].map(lambda x: _normalize_direction(x, spec.direction_aliases))
    out = out[(out["subj"] == spec.subj) & (out["roi"] == spec.roi)].copy()
    raw_dirs = {str(x) for x in spec.directions}
    label_dirs = {_normalize_direction(x, spec.direction_aliases) for x in spec.directions}
    out = out[out["direction_raw"].isin(raw_dirs) | out["direction_label"].isin(label_dirs)].copy()
    for col in ["c", "delta", "alpha_macro", "c_se", "td_ms", "tc_peak_ms", "tc_peak_resampled_ms"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def _tc_param_row(tc_params: pd.DataFrame, raw_direction: str, aliases: Mapping[str, str]) -> pd.Series | None:
    label = _normalize_direction(raw_direction, aliases)
    sub = tc_params[
        (tc_params["direction_raw"].astype(str) == str(raw_direction))
        | (tc_params["direction_label"].astype(str) == str(label))
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def _td_color_map(rows: pd.DataFrame) -> tuple[dict[float, str], list[float]]:
    td_values = sorted(pd.to_numeric(rows["td_ms"], errors="coerce").dropna().astype(float).unique().tolist())
    if not td_values:
        raise ValueError("No td_ms values available for plotting.")
    colors = {float(td): TD_COLORS[idx % len(TD_COLORS)] for idx, td in enumerate(td_values)}
    return colors, [float(td) for td in td_values]


def _td_label(td: float) -> str:
    return f"{td:g}"


def _display_roi(spec: ContrastDatasetSpec) -> str:
    return spec.roi_label or spec.roi


def _plot_contrast_panel(
    ax: plt.Axes,
    *,
    rows: pd.DataFrame,
    spec: ContrastDatasetSpec,
    direction: str,
    tc_params: pd.DataFrame,
    contrast_cache: dict[Path, pd.DataFrame],
    td_colors: Mapping[float, str],
    peak_gamma: float,
    point_size: float,
    line_width: float,
) -> list[str]:
    missing: list[str] = []
    panel = rows[rows["direction"].astype(str) == str(direction)].sort_values("td_ms", kind="stable")
    direction_label = _normalize_direction(direction, spec.direction_aliases)

    if panel.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return missing

    x_bounds: list[np.ndarray] = []
    y_bounds: list[np.ndarray] = []
    for _, row in panel.iterrows():
        td = float(row["td_ms"])
        color = td_colors.get(td, "black")
        try:
            contrast_path = _resolve_contrast_parquet(
                analysis_id=str(row["analysis_id"]),
                sheet=row.get("sheet", None),
                contrast_root=spec.contrast_root,
            )
            contrast_df = _load_contrast_table_cached(contrast_path, contrast_cache)
            df_group = _subset_group(contrast_df, row)
            if df_group.empty:
                missing.append(f"{spec.label} {row['analysis_id']} {spec.roi} {direction}: empty contrast group")
                continue

            x_corr, y, g1, g2 = _extract_plot_arrays(df_group, row)
            x_plot = _transform_x(
                xvar="lcf",
                active_corr=x_corr,
                fit_row=row,
                peak_D0_fix=spec.peak_D0_fix,
                peak_gamma=peak_gamma,
            )
            m = np.isfinite(x_plot) & np.isfinite(y)
            if not np.any(m):
                missing.append(f"{spec.label} {row['analysis_id']} {spec.roi} {direction}: no finite lcf data")
                continue
            ax.scatter(
                x_plot[m],
                y[m],
                s=point_size,
                marker="o",
                color=color,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.98,
                zorder=3,
            )
            x_bounds.append(x_plot[m])
            y_bounds.append(y[m])

            x_fit_corr, y_fit = _build_fit_curve(row, g1, g2)
            x_fit = _transform_x(
                xvar="lcf",
                active_corr=x_fit_corr,
                fit_row=row,
                peak_D0_fix=spec.peak_D0_fix,
                peak_gamma=peak_gamma,
            )
            mf = np.isfinite(x_fit) & np.isfinite(y_fit)
            if np.any(mf):
                order = np.argsort(x_fit[mf])
                ax.plot(
                    x_fit[mf][order],
                    y_fit[mf][order],
                    color=color,
                    linewidth=line_width,
                    alpha=0.98,
                    zorder=2,
                )
                y_bounds.append(y_fit[mf])
        except Exception as exc:
            missing.append(f"{spec.label} {row.get('analysis_id', 'NA')} {spec.roi} {direction}: {exc}")

    if x_bounds:
        x_all = np.concatenate([x[np.isfinite(x)] for x in x_bounds if x.size])
        if x_all.size:
            xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
            pad = 0.06 * (xmax - xmin) if xmax > xmin else 0.5
            ax.set_xlim(max(0.0, xmin - pad), xmax + pad)
    if y_bounds:
        y_all = np.concatenate([y[np.isfinite(y)] for y in y_bounds if y.size])
        if y_all.size:
            ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
            pad = 0.10 * (ymax - ymin) if ymax > ymin else 0.05
            ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(f"{spec.label} | {_display_roi(spec)} | {direction_label}", pad=8, fontsize=16)
    ax.set_xlabel(r"center filter length $l_{cf}$ [$\mu$m]")
    ax.set_ylabel("NOGSE contrast")
    ax.grid(True, alpha=0.24, linewidth=0.8)
    return missing


def _plot_tc_panel(
    ax: plt.Axes,
    *,
    rows: pd.DataFrame,
    direction: str,
    aliases: Mapping[str, str],
    tc_params: pd.DataFrame,
    td_colors: Mapping[float, str],
    show_errorbars: bool = False,
    marker_size: float = 74.0,
) -> None:
    panel = rows[rows["direction"].astype(str) == str(direction)].sort_values("td_ms", kind="stable")
    td = pd.to_numeric(panel["td_ms"], errors="coerce").to_numpy(dtype=float)
    tc_col = "tc_peak_resampled_ms" if "tc_peak_resampled_ms" in panel.columns else "tc_peak_ms"
    tc = pd.to_numeric(panel[tc_col], errors="coerce").to_numpy(dtype=float)
    tc_err = pd.to_numeric(panel.get("tc_err_ms", pd.Series(np.nan, index=panel.index)), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(td) & np.isfinite(tc)

    if np.any(m):
        colors = [td_colors.get(float(x), "black") for x in td[m]]
        yerr = tc_err[m]
        if show_errorbars and np.isfinite(yerr).any():
            ax.errorbar(
                td[m],
                tc[m],
                yerr=np.where(np.isfinite(yerr), yerr, 0.0),
                fmt="none",
                ecolor="0.25",
                elinewidth=1.4,
                capsize=2.5,
                zorder=2,
            )
        ax.scatter(td[m], tc[m], s=marker_size, c=colors, edgecolors="white", linewidths=0.7, zorder=3)

    fit_row = _tc_param_row(tc_params, direction, aliases)
    if fit_row is not None and np.any(m):
        xmin, xmax = float(np.nanmin(td[m])), float(np.nanmax(td[m]))
        xx = np.linspace(max(0.0, xmin - 0.04 * (xmax - xmin)), xmax + 0.04 * (xmax - xmin), 200)
        yy = tc_pseudohuber(xx, float(fit_row["c"]), float(fit_row["delta"]), float(fit_row["alpha_macro"]))
        ax.plot(xx, yy, color="black", linewidth=2.5, zorder=4)

    if np.any(m):
        xmin, xmax = float(np.nanmin(td[m])), float(np.nanmax(td[m]))
        ymin, ymax = float(np.nanmin(tc[m])), float(np.nanmax(tc[m]))
        xpad = 0.06 * (xmax - xmin) if xmax > xmin else 1.0
        ypad = 0.08 * (ymax - ymin) if ymax > ymin else 0.4
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_xlabel(r"$T_D$ [ms]", fontsize=16, labelpad=1)
    ax.set_ylabel(r"$tc_{peak}$ [ms]", fontsize=16, labelpad=1)
    ax.tick_params(axis="both", labelsize=14, length=3.5, width=1.0)
    ax.grid(True, alpha=0.20, linewidth=0.8)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.92))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def write_summary_table(rows_by_spec: Mapping[str, pd.DataFrame], out_dir: str | Path, prefix: str) -> Path:
    frames: list[pd.DataFrame] = []
    for name, df in rows_by_spec.items():
        sub = df.copy()
        sub.insert(0, "dataset", name)
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keep = [
        c
        for c in [
            "dataset",
            "subj",
            "roi",
            "direction",
            "td_ms",
            "analysis_id",
            "tc_peak_ms",
            "tc_peak_resampled_ms",
            "tc_err_ms",
            "x_peak_corr_mTm",
            "g_peak_resampled_corr_mTm",
            "signal_peak",
        ]
        if c in out.columns
    ]
    path = Path(out_dir) / f"{prefix}_plotted_rows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_csv(path, index=False)
    return path


def _plot_one_dataset(
    spec: ContrastDatasetSpec,
    *,
    rows: pd.DataFrame,
    tc_params: pd.DataFrame,
    out_stem: Path,
    formats: Sequence[str],
    dpi: int,
    peak_gamma: float,
    point_size: float,
    line_width: float,
    figsize: tuple[float, float],
    inset_errorbars: bool,
) -> list[Path]:
    td_colors, td_values = _td_color_map(rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        len(spec.directions),
        1,
        height_ratios=[1.0] * len(spec.directions),
        hspace=0.46,
    )
    contrast_cache: dict[Path, pd.DataFrame] = {}
    missing: list[str] = []

    for idx, direction in enumerate(spec.directions):
        main_ax = fig.add_subplot(gs[idx, 0])
        missing.extend(
            _plot_contrast_panel(
                main_ax,
                rows=rows,
                spec=spec,
                direction=direction,
                tc_params=tc_params,
                contrast_cache=contrast_cache,
                td_colors=td_colors,
                peak_gamma=peak_gamma,
                point_size=point_size,
                line_width=line_width,
            )
        )
        tc_ax = main_ax.inset_axes([0.64, 0.51, 0.32, 0.40])
        _plot_tc_panel(
            tc_ax,
            rows=rows,
            direction=direction,
            aliases=spec.direction_aliases,
            tc_params=tc_params,
            td_colors=td_colors,
            show_errorbars=bool(inset_errorbars),
            marker_size=max(64.0, point_size * 0.90),
        )

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=td_colors[td],
            marker="o",
            linestyle="-",
            linewidth=2.7,
            markersize=8,
            markeredgecolor="white",
            label=_td_label(td),
        )
        for td in td_values
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        title=r"$T_D$ [ms]",
        title_fontsize=16,
        fontsize=15,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.13)

    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{str(fmt).lower().lstrip('.')}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        outputs.append(out_path)
    plt.close(fig)

    if missing:
        warnings.warn("Some contrast curves could not be plotted:\n" + "\n".join(missing[:20]), stacklevel=2)
    return outputs


def plot_contrast_lcf_tcpeak_summary(
    specs: Sequence[ContrastDatasetSpec],
    *,
    out_stem: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 400,
    peak_gamma: float = 267.5221900,
    point_size: float = 54.0,
    line_width: float = 2.2,
    figsize: tuple[float, float] = (7.8, 10.5),
    inset_errorbars: bool = False,
) -> list[Path]:
    _set_publication_rc()
    rows_by_spec = {spec.name: load_contrast_rows(spec) for spec in specs}
    tc_by_spec = {spec.name: load_tc_params(spec) for spec in specs}

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for spec in specs:
        outputs.extend(
            _plot_one_dataset(
                spec,
                rows=rows_by_spec[spec.name],
                tc_params=tc_by_spec[spec.name],
                out_stem=out_stem.parent / f"{out_stem.name}_{spec.name}",
                formats=formats,
                dpi=dpi,
                peak_gamma=peak_gamma,
                point_size=point_size,
                line_width=line_width,
                figsize=figsize,
                inset_errorbars=inset_errorbars,
            )
        )

    write_summary_table(rows_by_spec, out_stem.parent, out_stem.name)
    return outputs
