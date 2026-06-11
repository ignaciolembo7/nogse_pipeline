from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_processing.master_table import filter_master_rows, load_master_table, split_selector_values
from fitting.gradient_correction import (
    SignalCorrectionLookupSpec,
    build_signal_direction_factors,
)
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


def source_key(text: object) -> str:
    name = Path(str(text)).name.strip()
    lower = name.lower()
    for suffix in (".rot_tensor.long.parquet", ".long.parquet", ".parquet", ".xlsx", ".xls"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def split_values(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    if not out or (len(out) == 1 and out[0].upper() == "ALL"):
        return None
    return out


def analysis_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".rot_tensor.long.parquet", ".long.parquet", ".parquet"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_signal_inputs(paths: Sequence[Path]) -> pd.DataFrame:
    """Load legacy signal tables. Prefer ``load_master_signal_input`` in new workflows."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        raise_on_unrecognized_column_names(df.columns, context=f"fit_global_signal({path})")
        if "source_file" not in df.columns:
            df["source_file"] = path.name
        if "analysis_id" not in df.columns:
            df["analysis_id"] = analysis_id_from_path(path)
        df["source_path"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("At least one input parquet is required.")
    return pd.concat(frames, ignore_index=True, sort=False)


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
    if "source_path" not in df.columns:
        df["source_path"] = str(args.master_parquet)
    if "source_file" not in df.columns:
        df["source_file"] = Path(args.master_parquet).name
    return df


def build_contrast_side_index(contrast_root: Path | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Index legacy contrast tables so signal curves can be labeled by contrast side."""
    if contrast_root is None:
        return {}
    table_root = Path(contrast_root) / "tables" if (Path(contrast_root) / "tables").is_dir() else Path(contrast_root)
    if not table_root.is_dir():
        raise FileNotFoundError(f"contrast_root does not exist or has no tables: {contrast_root}")

    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    columns = [
        "analysis_id",
        "source_file_1",
        "source_file_2",
        "N_1",
        "N_2",
        "td_ms_1",
        "td_ms_2",
        "sheet",
        "subj",
        "roi",
        "direction",
        "stat",
    ]
    for path in sorted(table_root.glob("**/*.parquet")):
        if path.name.endswith(".signal_fit_params.parquet") or path.name.endswith(".signal_fit_points.parquet"):
            continue
        try:
            df = pd.read_parquet(path, columns=[c for c in columns if c])
        except Exception:
            df = pd.read_parquet(path)
        needed = {"source_file_1", "source_file_2", "roi", "direction"}
        if not needed.issubset(df.columns):
            continue
        if "analysis_id" not in df.columns:
            df["analysis_id"] = path.name[: -len(".long.parquet")] if path.name.endswith(".long.parquet") else path.stem
        if "stat" not in df.columns:
            df["stat"] = ""
        meta_cols = [c for c in columns if c in df.columns]
        for _, row in df[meta_cols].drop_duplicates().iterrows():
            roi = str(row.get("roi", ""))
            direction = str(row.get("direction", ""))
            stat = str(row.get("stat", ""))
            for side in (1, 2):
                source = row.get(f"source_file_{side}", "")
                key = (source_key(source), roi, direction, stat)
                if key in index:
                    continue
                index[key] = {
                    "contrast_analysis_id": str(row.get("analysis_id", "")),
                    "contrast_source_file": path.name,
                    "contrast_side": int(side),
                    "contrast_N_1": float(row.get("N_1", np.nan)),
                    "contrast_N_2": float(row.get("N_2", np.nan)),
                }
                wildcard_key = (source_key(source), roi, direction, "")
                index.setdefault(wildcard_key, index[key])
    return index


def unique_text(df: pd.DataFrame, col: str, default: str = "") -> str:
    if col not in df.columns:
        return default
    values = pd.Series(df[col]).dropna().astype(str).unique().tolist()
    if len(values) == 1:
        return str(values[0])
    if len(values) == 0:
        return default
    return "|".join(str(v) for v in values)


def unique_float(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(values) == 1:
        return float(values[0])
    if len(values) == 0:
        return np.nan
    raise ValueError(f"Column {col!r} is not unique inside a curve: {values[:10].tolist()}")


def unique_float_any(df: pd.DataFrame, cols: Sequence[str]) -> float:
    for col in cols:
        value = unique_float(df, col)
        if np.isfinite(value):
            return float(value)
    return np.nan


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
    contrast_index: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
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
        "source_path",
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
    skipped_unmatched_contrast = 0
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

        contrast_meta: dict[str, Any] = {}
        if contrast_index:
            key = (source_key(source_file), roi, direction, stat_value)
            contrast_meta = contrast_index.get(key, {})
            if not contrast_meta and stat_value:
                key = (source_key(source_file), roi, direction, "")
                contrast_meta = contrast_index.get(key, {})
            if not contrast_meta:
                skipped_unmatched_contrast += 1
                continue

        contrast_analysis_id = str(contrast_meta.get("contrast_analysis_id", ""))
        contrast_side = int(contrast_meta.get("contrast_side", 0) or 0)
        contrast_n1 = float(contrast_meta.get("contrast_N_1", np.nan))
        contrast_n2 = float(contrast_meta.get("contrast_N_2", np.nan))
        if contrast_analysis_id:
            pair_key = f"contrast:{contrast_analysis_id}|{roi}|{direction}|{stat_value}"
        else:
            pair_key = f"fallback:{subj}|{sheet}|{roi}|{direction}|{stat_value}|td={td_value:.6g}"

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
                contrast_source_file=str(contrast_meta.get("contrast_source_file", "")),
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
    if skipped_unmatched_contrast:
        print("Skipped curves without a matching contrast table:", int(skipped_unmatched_contrast))
    return curves


def group_curves_by_subject_roi_direction(curves: Sequence[CurveData]) -> dict[tuple[str, str, str, str], list[CurveData]]:
    """Fit independently inside each subject, ROI, direction, and statistic."""
    group_map: dict[tuple[str, str, str, str], list[CurveData]] = {}
    for curve in curves:
        group_map.setdefault((curve.subj, curve.roi, curve.direction, curve.stat), []).append(curve)
    return group_map
