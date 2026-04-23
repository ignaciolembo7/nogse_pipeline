from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plottings.core import XYSeries, compact_float, ensure_dir, render_multi_series_plot, render_xy_plot


def load_long_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_gradient_axis(values: pd.Series, *, xcol: str, use_sqrt2: bool = True) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if xcol.startswith("g_thorsten") and use_sqrt2:
        x = np.sqrt(2.0) * np.abs(x)
    return x


def _prepare_avg_std(df: pd.DataFrame, *, xcol: str, ycol: str, stat: str) -> pd.DataFrame:
    keys = ["direction", "b_step", "roi"]

    avg = df[df["stat"].astype(str) == str(stat)].copy()
    std = df[df["stat"].astype(str) == "std"].copy()

    if avg.empty:
        raise ValueError(f"No rows with stat={stat!r} were found in the input table.")
    if std.empty:
        raise ValueError("No rows with stat='std' were found in the input table.")

    required_avg = set(keys + [xcol, ycol])
    missing_avg = required_avg - set(avg.columns)
    if missing_avg:
        raise ValueError(f"Missing required columns in avg rows: {sorted(missing_avg)}")

    required_std = set(keys + ["value"])
    missing_std = required_std - set(std.columns)
    if missing_std:
        raise ValueError(f"Missing required columns in std rows: {sorted(missing_std)}")

    avg = avg[keys + [xcol, ycol]].rename(columns={ycol: "y_mean", xcol: "x_raw"})
    std = std[keys + ["value"]].rename(columns={"value": "y_std"})

    merged = avg.merge(std, on=keys, how="left")
    merged["y_mean"] = pd.to_numeric(merged["y_mean"], errors="coerce")
    merged["y_std"] = pd.to_numeric(merged["y_std"], errors="coerce")
    merged = merged.sort_values(["direction", "roi", "b_step"], kind="stable")
    return merged


def _build_group_title(*, roi: str, direction: str, stat: str, model: str | None = None, fit_row: dict | None = None) -> str:
    parts = [f"ROI={roi}", f"direction={direction}", f"stat={stat}"]
    if model:
        parts.append(f"model={model}")
    if fit_row and bool(fit_row.get("ok", True)):
        if "M0" in fit_row:
            parts.append(f"M0={compact_float(fit_row.get('M0'))}")
        if "D0_mm2_s" in fit_row:
            parts.append(f"D0={compact_float(fit_row.get('D0_mm2_s'))} mm2/s")
        elif "D0_m2_ms" in fit_row:
            parts.append(f"D0={compact_float(fit_row.get('D0_m2_ms'))} m2/ms")
    return " | ".join(parts)


def plot_ogse_signal_summary(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    xcol: str = "g_thorsten",
    ycol: str = "value_norm",
    stat: str = "avg",
    use_sqrt2: bool = True,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> list[Path]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    merged = _prepare_avg_std(df, xcol=xcol, ycol=ycol, stat=stat)
    merged["x"] = compute_gradient_axis(merged["x_raw"], xcol=xcol, use_sqrt2=use_sqrt2)

    directions = sorted(merged["direction"].dropna().astype(str).unique().tolist())
    rois = sorted(merged["roi"].dropna().astype(str).unique().tolist())
    out_paths: list[Path] = []

    y_limits = None if ycol == "value" else ylim
    xlabel = "Modulation gradient strength G [mT/m]"
    ylabel = f"OGSE signal ({ycol})"

    # A) One plot per (direction, roi)
    for direction in directions:
        for roi in rois:
            group = merged[(merged["direction"].astype(str) == direction) & (merged["roi"].astype(str) == roi)].copy()
            group = group.dropna(subset=["x", "y_mean"]).sort_values("x")
            if group.empty:
                continue

            out_png = out_dir / f"OGSE_vs_G_roi={roi}_dir={direction}.png"
            render_xy_plot(
                x=group["x"].to_numpy(dtype=float),
                y=group["y_mean"].to_numpy(dtype=float),
                sigma=group["y_std"].to_numpy(dtype=float),
                out_png=out_png,
                title=_build_group_title(roi=roi, direction=direction, stat=stat),
                xlabel=xlabel,
                ylabel=ylabel,
                ylim=y_limits,
                data_label="signal",
                connect_data=True,
            )
            out_paths.append(out_png)

    # B) One plot per ROI with all directions
    for roi in rois:
        series: list[XYSeries] = []
        for direction in directions:
            group = merged[(merged["roi"].astype(str) == roi) & (merged["direction"].astype(str) == direction)].copy()
            group = group.dropna(subset=["x", "y_mean"]).sort_values("x")
            if group.empty:
                continue
            series.append(
                XYSeries(
                    x=group["x"].to_numpy(dtype=float),
                    y=group["y_mean"].to_numpy(dtype=float),
                    sigma=group["y_std"].to_numpy(dtype=float),
                    label=direction,
                )
            )
        if not series:
            continue
        out_png = out_dir / f"OGSE_vs_G_roi={roi}_all_dirs.png"
        render_multi_series_plot(
            series=series,
            out_png=out_png,
            title=f"ROI={roi} | stat={stat}",
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title="direction",
            ylim=y_limits,
        )
        out_paths.append(out_png)

    # C) One plot per direction with all ROIs
    for direction in directions:
        series = []
        for roi in rois:
            group = merged[(merged["direction"].astype(str) == direction) & (merged["roi"].astype(str) == roi)].copy()
            group = group.dropna(subset=["x", "y_mean"]).sort_values("x")
            if group.empty:
                continue
            series.append(
                XYSeries(
                    x=group["x"].to_numpy(dtype=float),
                    y=group["y_mean"].to_numpy(dtype=float),
                    sigma=group["y_std"].to_numpy(dtype=float),
                    label=roi,
                )
            )
        if not series:
            continue
        out_png = out_dir / f"OGSE_vs_G_dir={direction}_all_rois.png"
        render_multi_series_plot(
            series=series,
            out_png=out_png,
            title=f"direction={direction} | stat={stat}",
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title="ROI",
            ylim=y_limits,
        )
        out_paths.append(out_png)

    return out_paths


def build_ogse_signal_fit_label(fit_row: dict[str, object]) -> str:
    model = str(fit_row.get("model", "monoexp"))
    parts = [model]
    if "M0" in fit_row:
        parts.append(f"M0={compact_float(fit_row.get('M0'))}")
    if "D0_mm2_s" in fit_row:
        parts.append(f"D0={compact_float(fit_row.get('D0_mm2_s'))} mm2/s")
    elif "D0_m2_ms" in fit_row:
        parts.append(f"D0={compact_float(fit_row.get('D0_m2_ms'))} m2/ms")
    return ", ".join(parts)


def plot_ogse_signal_fit(
    *,
    x: np.ndarray,
    y: np.ndarray,
    fit_row: dict[str, object],
    out_png: Path,
    ycol: str,
    x_label: str,
    fit_points: int,
    fit_x: np.ndarray | None,
    fit_y: np.ndarray | None,
) -> None:
    roi = str(fit_row.get("roi", "roi"))
    direction = str(fit_row.get("direction", "direction"))
    td_txt = compact_float(fit_row.get("td_ms"))
    n_txt = compact_float(fit_row.get("N"), digits=0)

    text_lines = [
        f"model={fit_row.get('model', 'monoexp')}",
        f"M0={compact_float(fit_row.get('M0'))}",
        (
            f"D0={compact_float(fit_row.get('D0_mm2_s'))} mm2/s"
            if "D0_mm2_s" in fit_row
            else f"D0={compact_float(fit_row.get('D0_m2_ms'))} m2/ms"
        ),
        f"rmse={compact_float(fit_row.get('rmse'), digits=4)}",
        f"R2={compact_float(fit_row.get('r2'), digits=4)}",
    ]

    k = max(0, int(fit_points))
    fit_label = build_ogse_signal_fit_label(fit_row)
    render_xy_plot(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        out_png=out_png,
        title=(
            f"OGSE signal fit | ROI={roi} | direction={direction} | "
            f"{fit_label} | td_ms={td_txt} | N={n_txt}"
        ),
        xlabel=str(x_label),
        ylabel=ycol,
        data_label="data",
        connect_data=False,
        fit_x=None if fit_x is None else np.asarray(fit_x, dtype=float),
        fit_y=None if fit_y is None else np.asarray(fit_y, dtype=float),
        fit_label=fit_label,
        highlight_x=np.asarray(x[:k], dtype=float) if k > 0 else None,
        highlight_y=np.asarray(y[:k], dtype=float) if k > 0 else None,
        highlight_label=f"fit first {k}",
        text_lines=text_lines,
        yscale="log",
    )
