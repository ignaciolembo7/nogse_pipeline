from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_processing.master_table import filter_master_rows, load_master_table, split_selector_values
from fitting.gradient_correction import (
    SignalCorrectionLookupSpec,
    build_signal_direction_factors,
)
from tools.scalar import unique_float, unique_float_any, unique_text
from tools.strict_columns import raise_on_unrecognized_column_names


@dataclass(frozen=True)
class CurveData:
    """One physical signal curve selected for a global signal fit."""

    curve_id: int
    source_file: str
    analysis_id: str
    subj: str
    roi: str
    direction: str
    stat: str
    td_ms: float
    n_value: float
    x_model_ms: float
    sequence: str
    sheet: str
    protocol: str
    contrast_analysis_id: str
    contrast_source_file: str
    contrast_side: int
    contrast_N_1: float
    contrast_N_2: float
    pair_key: str
    g: np.ndarray
    f_corr: float
    corr_status: str
    y: np.ndarray
    b_step: np.ndarray


def split_values(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    if not out or (len(out) == 1 and out[0].upper() == "ALL"):
        return None
    return out


def load_master_signal_input(args: Any) -> pd.DataFrame:
    """Select signal rows directly from the master table."""
    master = load_master_table(args.master_parquet)
    selectors: dict[str, object] = {"row_kind": str(args.row_kind)}
    for arg_name, col_name in [
        ("subjs", "subj"),
        ("sheets", "sheet"),
        ("rois", "roi"),
        ("directions", "direction"),
    ]:
        values = split_selector_values(getattr(args, arg_name, None))
        if values is not None:
            selectors[col_name] = values
    for arg_name, col_name in [
        ("td_ms", "td_ms"),
        ("N", "N"),
        ("Hz", "Hz"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            selectors[col_name] = float(value)
    df = filter_master_rows(master, **selectors)
    if df.empty:
        raise ValueError(f"No master rows matched selectors: {selectors}")
    raise_on_unrecognized_column_names(df.columns, context=f"fit_global_signal({args.master_parquet})")
    return df



def preferred_correction_side(group: pd.DataFrame) -> int | None:
    signal_type = unique_text(group, "type").strip().upper()
    if signal_type == "CPMG":
        return 1
    if signal_type == "HAHN":
        return 2
    return None


def prepare_signal_curves(
    df: pd.DataFrame,
    *,
    ycol: str,
    g_type: str,
    stat: str | None,
    rois: list[str] | None,
    directions: list[str] | None,
    n_fit: int | None,
    min_points: int,
    corr: pd.DataFrame | None,
    corr_roi: str,
    corr_tol_ms: float,
    corr_sheet: str | None,
    corr_missing: str,
) -> list[CurveData]:
    """Build the physical curves that enter the global least-squares problem."""
    work = df.copy()
    required = {"subj", "roi", "direction", "N", ycol, g_type}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available columns: {list(work.columns)}")
    if not any(col in work.columns for col in ("td_ms", "TN", "TE")):
        raise ValueError("Missing required time column. Expected one of: td_ms, TN, TE.")

    work["roi"] = work["roi"].astype(str)
    work["direction"] = work["direction"].astype(str)
    work["subj"] = work["subj"].astype(str)
    if "stat" in work.columns:
        work["stat"] = work["stat"].astype(str)

    if stat is not None and str(stat).upper() != "ALL" and "stat" in work.columns:
        work = work[work["stat"].astype(str) == str(stat)].copy()
    if rois is not None:
        work = work[work["roi"].isin([str(v) for v in rois])].copy()
    if directions is not None:
        work = work[work["direction"].isin([str(v) for v in directions])].copy()
    if work.empty:
        raise ValueError("No signal rows remain after filtering.")

    group_cols = [
        "analysis_id",
        "source_file",
        "subj",
        "roi",
        "direction",
        "td_ms",
        "N",
    ]
    if "stat" in work.columns:
        group_cols.append("stat")
    for optional_col in ("type", "TN", "TE", "x", "y"):
        if optional_col in work.columns:
            group_cols.append(optional_col)

    curves: list[CurveData] = []
    for curve_id, (_key, group) in enumerate(work.groupby(group_cols, sort=False, dropna=False)):
        g_raw = pd.to_numeric(group[g_type], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[ycol], errors="coerce").to_numpy(dtype=float)
        b_step = pd.to_numeric(group.get("b_step", pd.Series(np.arange(len(group)))), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(g_raw) & np.isfinite(y)
        g_raw = g_raw[mask]
        y = y[mask]
        b_step = b_step[mask]

        td_value = float(unique_float_any(group, ("td_ms", "TN", "TE")))
        n_value = float(unique_float(group, "N"))
        x_value = float(unique_float(group, "x"))
        if not np.isfinite(x_value):
            x_value = float(td_value) / float(n_value)
        source_file = unique_text(group, "source_file")
        sheet = unique_text(group, "sheet")
        direction = unique_text(group, "direction")
        roi = unique_text(group, "roi")
        subj = unique_text(group, "subj")
        stat_value = unique_text(group, "stat", default=str(stat or ""))

        contrast_analysis_id = unique_text(group, "contrast_analysis_id")
        contrast_source_file = unique_text(group, "contrast_source_file")
        contrast_side_value = unique_float(group, "contrast_side")
        contrast_side = int(contrast_side_value) if np.isfinite(contrast_side_value) else 0
        contrast_n1 = float(unique_float(group, "contrast_N_1"))
        contrast_n2 = float(unique_float(group, "contrast_N_2"))
        pair_key = f"{subj}|{sheet}|{roi}|{direction}|{stat_value}|td={td_value:.6g}"

        f_corr = 1.0
        corr_status = "not_requested"
        if corr is not None:
            try:
                factors = build_signal_direction_factors(
                    corr,
                    spec=SignalCorrectionLookupSpec(
                        roi_ref=str(corr_roi),
                        td_ms=float(td_value),
                        signal_n=int(round(float(n_value))) if np.isfinite(n_value) else None,
                        tol_ms=float(corr_tol_ms),
                        sheet=corr_sheet or sheet or None,
                        signal_source_file=source_file,
                        preferred_side=preferred_correction_side(group),
                    ),
                )
                if direction not in factors:
                    raise ValueError(
                        f"No correction factor for direction={direction!r}, "
                        f"td_ms={td_value}, N={n_value}, source_file={source_file}."
                    )
                f_corr = float(factors[direction])
                corr_status = "applied"
            except Exception:
                if corr_missing == "error":
                    raise
                if corr_missing == "skip":
                    continue
                f_corr = 1.0
                corr_status = "missing_identity"

        g = g_raw * float(f_corr)
        if g.size:
            order = np.argsort(g)
            g = g[order]
            y = y[order]
            b_step = b_step[order]
        if n_fit is not None:
            k = int(n_fit)
            g = g[:k]
            y = y[:k]
            b_step = b_step[:k]
        if len(y) < int(min_points):
            continue

        curves.append(
            CurveData(
                curve_id=int(curve_id),
                source_file=source_file,
                analysis_id=unique_text(group, "analysis_id"),
                subj=subj,
                roi=roi,
                direction=direction,
                stat=stat_value,
                td_ms=td_value,
                n_value=n_value,
                x_model_ms=x_value,
                sequence=unique_text(group, "sequence"),
                sheet=sheet,
                protocol=unique_text(group, "protocol"),
                contrast_analysis_id=contrast_analysis_id,
                contrast_source_file=contrast_source_file,
                contrast_side=contrast_side,
                contrast_N_1=contrast_n1,
                contrast_N_2=contrast_n2,
                pair_key=pair_key,
                g=g,
                f_corr=float(f_corr),
                corr_status=str(corr_status),
                y=y,
                b_step=b_step,
            )
        )
    if not curves:
        raise ValueError("No valid curves remained after filtering and min-points checks.")
    return curves


def group_curves_by_subject_roi_direction(curves: Sequence[CurveData]) -> dict[tuple[str, str, str, str], list[CurveData]]:
    """Fit independently inside each subject, ROI, direction, and statistic."""
    group_map: dict[tuple[str, str, str, str], list[CurveData]] = {}
    for curve in curves:
        group_map.setdefault((curve.subj, curve.roi, curve.direction, curve.stat), []).append(curve)
    return group_map
