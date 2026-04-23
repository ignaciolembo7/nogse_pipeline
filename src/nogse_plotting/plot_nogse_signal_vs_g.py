from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from plottings.core import compact_float, render_xy_plot


def analysis_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def split_all_or_values(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    out = [str(v) for v in values]
    if len(out) == 1 and out[0].upper() == "ALL":
        return None
    return out


def unique_scalar(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    values = pd.Series(df[col]).dropna().unique().tolist()
    if len(values) != 1:
        return None
    return values[0]


def prepare_signal_series(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    xcol: str,
    ycol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = avg_df.copy()
    data[xcol] = pd.to_numeric(data[xcol], errors="coerce")
    data[ycol] = pd.to_numeric(data[ycol], errors="coerce")
    data = data.dropna(subset=[xcol, ycol]).sort_values(xcol)
    if data.empty:
        return np.array([]), np.array([]), None

    sigma = None
    if std_df is not None and not std_df.empty:
        err = std_df.copy()
        err[xcol] = pd.to_numeric(err[xcol], errors="coerce")
        err["value"] = pd.to_numeric(err["value"], errors="coerce")
        err = err.dropna(subset=[xcol, "value"]).sort_values(xcol)
        if not err.empty:
            sigma = err["value"].to_numpy(dtype=float)
            if ycol == "value_norm" and "S0" in data.columns:
                s0 = pd.to_numeric(data["S0"], errors="coerce").to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma = sigma / s0
            if sigma.shape[0] != data.shape[0]:
                sigma = None

    return data[xcol].to_numpy(dtype=float), data[ycol].to_numpy(dtype=float), sigma


def fit_parameter_fragments(fit_row: dict[str, object]) -> list[str]:
    fragments: list[str] = []
    if "M0" in fit_row:
        fragments.append(f"M0={compact_float(fit_row.get('M0'))}")
    if "D0_m2_ms" in fit_row:
        fragments.append(f"D0={compact_float(fit_row.get('D0_m2_ms'))} m2/ms")
    elif "D0_mm2_s" in fit_row:
        fragments.append(f"D0={compact_float(fit_row.get('D0_mm2_s'))} mm2/s")
    return fragments


def build_fit_label(fit_row: dict[str, object]) -> str:
    model = str(fit_row.get("model", "fit"))
    fragments = fit_parameter_fragments(fit_row)
    return ", ".join([model] + fragments) if fragments else model


def build_title(
    *,
    analysis_id: str,
    roi: str,
    direction: str,
    signal_type: str | None,
    model: str | None,
    fit_row: dict[str, object] | None,
) -> str:
    parts = [analysis_id, f"ROI={roi}", f"direction={direction}"]
    if signal_type:
        parts.append(f"type={signal_type}")
    if model:
        parts.append(f"model={model}")
    if fit_row and bool(fit_row.get("ok", True)):
        parts.extend(fit_parameter_fragments(fit_row))
    return " | ".join(parts)


def plot_nogse_signal_group(
    *,
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    xcol: str,
    ycol: str,
    out_png: Path,
    analysis_id: str,
    roi: str,
    direction: str,
    signal_type: str | None,
    fit_row: dict[str, object] | None = None,
    fit_curve: np.ndarray | None = None,
    x_data: np.ndarray | None = None,
    y_data: np.ndarray | None = None,
    fit_points: int | None = None,
    data_label: str = "signal",
    connect_data: bool = True,
) -> None:
    if x_data is None or y_data is None:
        x_vals, y_vals, sigma = prepare_signal_series(avg_df, std_df, xcol=xcol, ycol=ycol)
    else:
        x_vals = np.asarray(x_data, dtype=float)
        y_vals = np.asarray(y_data, dtype=float)
        _x, _y, sigma = prepare_signal_series(avg_df, std_df, xcol=xcol, ycol=ycol)

    if x_vals.size == 0 or y_vals.size == 0:
        return

    model = str(fit_row.get("model")) if fit_row is not None and fit_row.get("model") is not None else None
    title = build_title(
        analysis_id=analysis_id,
        roi=str(roi),
        direction=str(direction),
        signal_type=None if signal_type is None else str(signal_type),
        model=model,
        fit_row=fit_row,
    )

    fit_x = None
    fit_y = None
    fit_label = None
    if fit_curve is not None and fit_curve.size:
        fit_x = np.asarray(fit_curve[:, 0], dtype=float)
        fit_y = np.asarray(fit_curve[:, 1], dtype=float)
        if fit_row is not None:
            fit_label = build_fit_label(fit_row)

    highlight_x = None
    highlight_y = None
    highlight_label = None
    if fit_points is not None and fit_points > 0:
        k = min(int(fit_points), len(x_vals))
        highlight_x = np.asarray(x_vals[:k], dtype=float)
        highlight_y = np.asarray(y_vals[:k], dtype=float)
        highlight_label = f"fit first {k}"

    text_lines = None
    if fit_row is not None:
        text_lines = [
            f"model={fit_row.get('model', 'fit')}",
            *fit_parameter_fragments(fit_row),
            f"rmse={compact_float(fit_row.get('rmse'), digits=4)}",
            f"R2={compact_float(fit_row.get('r2'), digits=4)}",
        ]

    render_xy_plot(
        x=x_vals,
        y=y_vals,
        sigma=sigma,
        out_png=out_png,
        title=title,
        xlabel=xcol,
        ylabel=ycol,
        data_label=data_label,
        connect_data=connect_data,
        fit_x=fit_x,
        fit_y=fit_y,
        fit_label=fit_label,
        highlight_x=highlight_x,
        highlight_y=highlight_y,
        highlight_label=highlight_label,
        text_lines=text_lines,
    )


def plot_nogse_signal_table(
    df: pd.DataFrame,
    *,
    out_root: Path,
    analysis_id: str,
    xcol: str,
    ycol: str,
    stat: str,
    rois: list[str] | None,
    directions: list[str] | None,
) -> list[Path]:
    avg_df = df[df["stat"].astype(str) == str(stat)].copy()
    if avg_df.empty:
        raise ValueError(f"No rows found for stat={stat!r}.")

    std_df = df[df["stat"].astype(str) == "std"].copy()

    if rois is not None:
        avg_df = avg_df[avg_df["roi"].astype(str).isin(rois)].copy()
        std_df = std_df[std_df["roi"].astype(str).isin(rois)].copy()
    if directions is not None:
        avg_df = avg_df[avg_df["direction"].astype(str).isin(directions)].copy()
        std_df = std_df[std_df["direction"].astype(str).isin(directions)].copy()

    if avg_df.empty:
        raise ValueError("No data remains after ROI/direction filtering.")

    out_dir = out_root / analysis_id
    out_paths: list[Path] = []

    for (roi, direction), group in avg_df.groupby(["roi", "direction"], sort=False):
        std_group = None
        if not std_df.empty:
            std_group = std_df[
                (std_df["roi"].astype(str) == str(roi))
                & (std_df["direction"].astype(str) == str(direction))
            ].copy()

        signal_type = unique_scalar(group, "type")
        out_png = out_dir / f"{analysis_id}.roi-{roi}.dir-{direction}.{ycol}_vs_{xcol}.png"
        plot_nogse_signal_group(
            avg_df=group,
            std_df=std_group,
            xcol=xcol,
            ycol=ycol,
            out_png=out_png,
            analysis_id=analysis_id,
            roi=str(roi),
            direction=str(direction),
            signal_type=None if signal_type is None else str(signal_type),
            connect_data=True,
        )
        out_paths.append(out_png)

    return out_paths
