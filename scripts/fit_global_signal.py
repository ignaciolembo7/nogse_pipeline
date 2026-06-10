from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import repo_bootstrap  # noqa: F401

from data_processing.io import write_table_outputs
from fitting.b_from_g import normalize_axis_base
from fitting.core import chi2 as _chi2
from fitting.core import rmse as _rmse
from fitting.gradient_correction import (
    SignalCorrectionLookupSpec,
    build_signal_direction_factors,
    read_correction_table,
)
from fitting.model_registry import (
    SignalModelSpec,
    evaluate_signal_model,
    get_signal_model,
    signal_model_names,
)
from fitting.parameter_modes import (
    FitParameterConfig,
    VALID_PARAMETER_MODES,
    finite_or_none,
    fixed_parameter_value,
    from_optimization_values,
    make_parameter_config,
    normalize_parameter_mode,
    parameter_mode_summary,
    to_optimization_values,
)
from tools.strict_columns import raise_on_unrecognized_column_names


@dataclass(frozen=True)
class CurveData:
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


def _source_key(text: object) -> str:
    name = Path(str(text)).name.strip()
    lower = name.lower()
    for suffix in (".rot_tensor.long.parquet", ".long.parquet", ".parquet", ".xlsx", ".xls"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _contrast_table_root(contrast_root: Path) -> Path:
    return contrast_root / "tables" if (contrast_root / "tables").is_dir() else contrast_root


def _build_contrast_side_index(contrast_root: Path | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if contrast_root is None:
        return {}
    table_root = _contrast_table_root(Path(contrast_root))
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
                key = (_source_key(source), roi, direction, stat)
                if key in index:
                    continue
                index[key] = {
                    "contrast_analysis_id": str(row.get("analysis_id", "")),
                    "contrast_source_file": path.name,
                    "contrast_side": int(side),
                    "contrast_N_1": float(row.get("N_1", np.nan)),
                    "contrast_N_2": float(row.get("N_2", np.nan)),
                }
                wildcard_key = (_source_key(source), roi, direction, "")
                index.setdefault(wildcard_key, index[key])
    return index


def _split_values(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    if not out or (len(out) == 1 and out[0].upper() == "ALL"):
        return None
    return out


def _analysis_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".rot_tensor.long.parquet", ".long.parquet", ".parquet"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _unique_text(df: pd.DataFrame, col: str, default: str = "") -> str:
    if col not in df.columns:
        return default
    values = pd.Series(df[col]).dropna().astype(str).unique().tolist()
    if len(values) == 1:
        return str(values[0])
    if len(values) == 0:
        return default
    return "|".join(str(v) for v in values)


def _unique_float(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(values) == 1:
        return float(values[0])
    if len(values) == 0:
        return np.nan
    raise ValueError(f"Column {col!r} is not unique inside a curve: {values[:10].tolist()}")


def _unique_float_any(df: pd.DataFrame, cols: Sequence[str]) -> float:
    for col in cols:
        value = _unique_float(df, col)
        if np.isfinite(value):
            return float(value)
    return np.nan


def _preferred_correction_side(group: pd.DataFrame) -> int | None:
    signal_type = _unique_text(group, "type").strip().upper()
    if signal_type == "CPMG":
        return 1
    if signal_type == "HAHN":
        return 2
    return None


def _load_inputs(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        raise_on_unrecognized_column_names(df.columns, context=f"fit_global_signal({path})")
        if "source_file" not in df.columns:
            df["source_file"] = path.name
        if "analysis_id" not in df.columns:
            df["analysis_id"] = _analysis_id_from_path(path)
        df["source_path"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("At least one input parquet is required.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _prepare_curves(
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

        td_value = float(_unique_float_any(group, ("td_ms", "TN", "TE")))
        n_value = float(_unique_float(group, "N"))
        x_value = float(_unique_float(group, "x"))
        if not np.isfinite(x_value):
            x_value = float(td_value) / float(n_value)
        source_file = _unique_text(group, "source_file")
        sheet = _unique_text(group, "sheet")
        direction = _unique_text(group, "direction")
        roi = _unique_text(group, "roi")
        subj = _unique_text(group, "subj")
        stat_value = _unique_text(group, "stat", default=str(stat or ""))

        contrast_meta: dict[str, Any] = {}
        if contrast_index:
            key = (_source_key(source_file), roi, direction, stat_value)
            contrast_meta = contrast_index.get(key, {})
            if not contrast_meta and stat_value:
                key = (_source_key(source_file), roi, direction, "")
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
            pair_key = (
                f"fallback:{subj}|{sheet}|{roi}|{direction}|{stat_value}|"
                f"td={td_value:.6g}"
            )

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
                        preferred_side=_preferred_correction_side(group),
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
                source_file=_unique_text(group, "source_file"),
                analysis_id=_unique_text(group, "analysis_id"),
                subj=subj,
                roi=roi,
                direction=direction,
                stat=stat_value,
                td_ms=td_value,
                n_value=n_value,
                x_model_ms=x_value,
                sequence=_unique_text(group, "sequence"),
                sheet=sheet,
                protocol=_unique_text(group, "protocol"),
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
        print(
            "Skipped curves without a matching contrast table:",
            int(skipped_unmatched_contrast),
        )
    return curves


ALL_PARAM_ORDER = ("tc_ms", "alpha", "RN", "M0", "C", "D0_m2_ms")


def _model_curve(curve: CurveData, *, model_spec: SignalModelSpec, params: dict[str, float]) -> np.ndarray:
    return evaluate_signal_model(
        model_spec,
        td_ms=float(curve.td_ms),
        G=np.asarray(curve.g, dtype=float),
        N=float(curve.n_value),
        params=params,
        x_ms=float(curve.x_model_ms),
    )


def _param_order(model_spec: SignalModelSpec) -> tuple[str, ...]:
    return tuple(name for name in ALL_PARAM_ORDER if name in set(model_spec.param_names))


def _curves_y_seed(curves: Sequence[CurveData], bounds: tuple[float, float]) -> float:
    arrays = [np.asarray(curve.y, dtype=float) for curve in curves if len(curve.y)]
    if not arrays:
        value = 1.0
    else:
        y_values = np.concatenate(arrays)
        finite = y_values[np.isfinite(y_values)]
        if finite.size == 0:
            value = 1.0
        else:
            positive = finite[finite > 0]
            value = float(positive[0]) if positive.size else float(np.nanmax(finite))
    return float(np.clip(value, bounds[0], bounds[1]))


def _initial_value_for_scope(param_name: str, config: FitParameterConfig, curves: Sequence[CurveData]) -> float:
    init = finite_or_none(config.init)
    if init is not None:
        return float(np.clip(init, config.bounds[0], config.bounds[1]))
    if param_name == "M0":
        return _curves_y_seed(curves, config.bounds)
    raise ValueError(f"No finite initial value provided for free parameter {param_name}.")


def _scope_name(config: FitParameterConfig, curve: CurveData, pair_param_ids: dict[str, int]) -> str | None:
    if config.mode == "fixed":
        return None
    if config.mode == "global_td":
        return config.name
    if config.mode == "global_contrast":
        return f"{config.name}__pair_{pair_param_ids[curve.pair_key]}"
    if config.mode == "free":
        return f"{config.name}__curve_{curve.curve_id}"
    raise ValueError(f"Unsupported mode {config.mode!r} for {config.name}.")


def _scope_members(
    curves: Sequence[CurveData],
    *,
    config: FitParameterConfig,
    pair_param_ids: dict[str, int],
) -> list[tuple[str, list[CurveData]]]:
    if config.mode == "fixed":
        return []
    if config.mode == "global_td":
        return [(config.name, list(curves))]
    if config.mode == "global_contrast":
        pairs = _curves_by_pair(curves)
        return [(f"{config.name}__pair_{pair_param_ids[key]}", members) for key, members in pairs.items()]
    if config.mode == "free":
        return [(f"{config.name}__curve_{curve.curve_id}", [curve]) for curve in curves]
    raise ValueError(f"Unsupported mode {config.mode!r} for {config.name}.")


def _pack_fit_problem(
    curves: Sequence[CurveData],
    *,
    model_spec: SignalModelSpec,
    param_configs: dict[str, FitParameterConfig],
    pair_param_ids: dict[str, int],
) -> tuple[list[str], np.ndarray, tuple[np.ndarray, np.ndarray], set[str]]:
    names: list[str] = []
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    log_params: set[str] = set()

    for param_name in _param_order(model_spec):
        config = param_configs[param_name]
        if config.mode == "fixed":
            fixed_parameter_value(config)
            continue
        for scoped_name, members in _scope_members(curves, config=config, pair_param_ids=pair_param_ids):
            names.append(scoped_name)
            p0.append(_initial_value_for_scope(param_name, config, members))
            lower.append(float(config.bounds[0]))
            upper.append(float(config.bounds[1]))
            if config.log:
                log_params.add(scoped_name)

    return names, np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)), log_params


def _curves_by_pair(curves: Sequence[CurveData]) -> dict[str, list[CurveData]]:
    pairs: dict[str, list[CurveData]] = {}
    for curve in curves:
        pairs.setdefault(curve.pair_key, []).append(curve)
    return pairs


def _param_value(
    *,
    values: dict[str, float],
    param_configs: dict[str, FitParameterConfig],
    param_name: str,
    curve: CurveData,
    pair_param_ids: dict[str, int],
) -> float:
    if param_name not in param_configs:
        return np.nan
    config = param_configs[param_name]
    scoped_name = _scope_name(config, curve, pair_param_ids)
    if scoped_name is None:
        return fixed_parameter_value(config)
    return float(values.get(scoped_name, np.nan))


def _param_error(
    *,
    errors: dict[str, float],
    param_configs: dict[str, FitParameterConfig],
    param_name: str,
    curve: CurveData,
    pair_param_ids: dict[str, int],
) -> float:
    if param_name not in param_configs:
        return np.nan
    scoped_name = _scope_name(param_configs[param_name], curve, pair_param_ids)
    if scoped_name is None:
        return np.nan
    return float(errors.get(scoped_name, np.nan))


def _mode_summary(param_configs: dict[str, FitParameterConfig], mode: str, *, model_spec: SignalModelSpec) -> str:
    return parameter_mode_summary(param_configs, mode, ordered_names=_param_order(model_spec))


def _free_param_count(
    param_configs: dict[str, FitParameterConfig],
    curves: Sequence[CurveData],
    *,
    model_spec: SignalModelSpec,
) -> int:
    pair_count = len(_curves_by_pair(curves))
    count = 0
    for name in _param_order(model_spec):
        config = param_configs[name]
        if config.mode == "global_td":
            count += 1
        elif config.mode == "global_contrast":
            count += pair_count
        elif config.mode == "free":
            count += len(curves)
    return int(count)


def _fit_group(
    curves: Sequence[CurveData],
    *,
    model_spec: SignalModelSpec,
    param_configs: dict[str, FitParameterConfig],
    max_nfev: int,
) -> tuple[dict[str, float], dict[str, float], bool, str, str]:
    pair_param_ids = {key: members[0].curve_id for key, members in _curves_by_pair(curves).items()}
    names, p0, bounds, log_params = _pack_fit_problem(
        curves,
        model_spec=model_spec,
        param_configs=param_configs,
        pair_param_ids=pair_param_ids,
    )
    lower_opt = to_optimization_values(bounds[0], names, log_params)
    upper_opt = to_optimization_values(bounds[1], names, log_params)
    p0_opt = to_optimization_values(np.clip(p0, bounds[0], bounds[1]), names, log_params)

    def residual(opt_values: np.ndarray) -> np.ndarray:
        params = from_optimization_values(opt_values, names, log_params)
        residuals: list[np.ndarray] = []
        for curve in curves:
            curve_params = {
                name: _param_value(
                    values=params,
                    param_configs=param_configs,
                    param_name=name,
                    curve=curve,
                    pair_param_ids=pair_param_ids,
                )
                for name in model_spec.param_names
            }
            yhat = _model_curve(curve, model_spec=model_spec, params=curve_params)
            if yhat.shape != curve.y.shape or not np.all(np.isfinite(yhat)):
                return np.full(sum(len(c.y) for c in curves), 1e12, dtype=float)
            residuals.append(yhat - curve.y)
        return np.concatenate(residuals)

    mode_desc = "; ".join(
        f"{mode}={_mode_summary(param_configs, mode, model_spec=model_spec) or '-'}"
        for mode in ("fixed", "global_td", "global_contrast", "free")
    )
    method = f"least_squares({model_spec.name}: {mode_desc})"
    if not names:
        try:
            for curve in curves:
                curve_params = {
                    name: _param_value(
                        values={},
                        param_configs=param_configs,
                        param_name=name,
                        curve=curve,
                        pair_param_ids=pair_param_ids,
                    )
                    for name in model_spec.param_names
                }
                yhat = _model_curve(curve, model_spec=model_spec, params=curve_params)
                if yhat.shape != curve.y.shape or not np.all(np.isfinite(yhat)):
                    return {}, {}, False, "fixed-parameter model returned non-finite values", method
            return {}, {}, True, "", method
        except Exception as exc:
            return {}, {}, False, str(exc), method
    try:
        opt = least_squares(
            residual,
            p0_opt,
            bounds=(lower_opt, upper_opt),
            x_scale="jac",
            max_nfev=int(max_nfev),
        )
        if not opt.success:
            return from_optimization_values(opt.x, names, log_params), {}, False, str(opt.message), method
        values = from_optimization_values(opt.x, names, log_params)
        errors: dict[str, float] = {}
        jac = np.asarray(opt.jac, dtype=float)
        res = np.asarray(opt.fun, dtype=float)
        dof = int(res.size) - int(len(names))
        if dof > 0 and jac.size:
            try:
                cov = np.linalg.pinv(jac.T @ jac) * float(np.sum(res**2) / dof)
                opt_err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
                for name, err in zip(names, opt_err):
                    if name in log_params:
                        errors[name] = float(abs(values[name]) * err)
                    else:
                        errors[name] = float(err)
            except np.linalg.LinAlgError:
                pass
        return values, errors, True, "", method
    except Exception as exc:
        return {}, {}, False, str(exc), method


def _build_outputs(
    curves: Sequence[CurveData],
    *,
    model_spec: SignalModelSpec,
    values: dict[str, float],
    errors: dict[str, float],
    param_configs: dict[str, FitParameterConfig],
    ok: bool,
    msg: str,
    method: str,
    ycol: str,
    g_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    pair_param_ids = {key: members[0].curve_id for key, members in _curves_by_pair(curves).items()}

    def mode_for(param_name: str) -> str:
        config = param_configs.get(param_name)
        return config.mode if config is not None else ""

    for curve in curves:
        tc = _param_value(values=values, param_configs=param_configs, param_name="tc_ms", curve=curve, pair_param_ids=pair_param_ids)
        alpha = _param_value(values=values, param_configs=param_configs, param_name="alpha", curve=curve, pair_param_ids=pair_param_ids)
        RN = _param_value(values=values, param_configs=param_configs, param_name="RN", curve=curve, pair_param_ids=pair_param_ids)
        M0 = _param_value(values=values, param_configs=param_configs, param_name="M0", curve=curve, pair_param_ids=pair_param_ids)
        C = _param_value(values=values, param_configs=param_configs, param_name="C", curve=curve, pair_param_ids=pair_param_ids)
        D0 = _param_value(values=values, param_configs=param_configs, param_name="D0_m2_ms", curve=curve, pair_param_ids=pair_param_ids)
        curve_params = {
            name: _param_value(
                values=values,
                param_configs=param_configs,
                param_name=name,
                curve=curve,
                pair_param_ids=pair_param_ids,
            )
            for name in model_spec.param_names
        }
        yhat = (
            _model_curve(curve, model_spec=model_spec, params=curve_params)
            if ok
            else np.full_like(curve.y, np.nan, dtype=float)
        )
        fit_rows.append(
            {
                "source_file": curve.source_file,
                "analysis_id": curve.analysis_id,
                "subj": curve.subj,
                "roi": curve.roi,
                "direction": curve.direction,
                "stat": curve.stat,
                "td_ms": curve.td_ms,
                "N": curve.n_value,
                "x_model_ms": curve.x_model_ms,
                "sequence": curve.sequence,
                "sheet": curve.sheet,
                "protocol": curve.protocol,
                "fit_kind": f"{model_spec.family}_signal",
                "model": model_spec.name,
                "fit_scope": "global_subj_roi_direction",
                "fixed_params": _mode_summary(param_configs, "fixed", model_spec=model_spec),
                "global_params": _mode_summary(param_configs, "global_td", model_spec=model_spec),
                "global_contrast_params": _mode_summary(param_configs, "global_contrast", model_spec=model_spec),
                "local_params": _mode_summary(param_configs, "free", model_spec=model_spec),
                "pair_key": curve.pair_key,
                "contrast_analysis_id": curve.contrast_analysis_id,
                "contrast_source_file": curve.contrast_source_file,
                "contrast_side": curve.contrast_side,
                "N_1": curve.contrast_N_1,
                "N_2": curve.contrast_N_2,
                "tc_mode": mode_for("tc_ms"),
                "alpha_mode": mode_for("alpha"),
                "RN_mode": mode_for("RN"),
                "M0_mode": mode_for("M0"),
                "C_mode": mode_for("C"),
                "D0_mode": mode_for("D0_m2_ms"),
                "gbase": g_type,
                "xcol": g_type,
                "plot_xcol": f"{g_type}_{curve.contrast_side}" if curve.contrast_side in (1, 2) else g_type,
                "xplot": str(curve.contrast_side) if curve.contrast_side in (1, 2) else "1",
                "ycol": ycol,
                "f_corr": curve.f_corr,
                "corr_status": curve.corr_status,
                "n_points": int(len(curve.y)),
                "n_fit": int(len(curve.y)),
                "M0": M0,
                "M0_err": _param_error(errors=errors, param_configs=param_configs, param_name="M0", curve=curve, pair_param_ids=pair_param_ids),
                "tc_ms": tc,
                "tc_err_ms": _param_error(errors=errors, param_configs=param_configs, param_name="tc_ms", curve=curve, pair_param_ids=pair_param_ids),
                "alpha": alpha,
                "alpha_err": _param_error(errors=errors, param_configs=param_configs, param_name="alpha", curve=curve, pair_param_ids=pair_param_ids),
                "RN": RN,
                "RN_err": _param_error(errors=errors, param_configs=param_configs, param_name="RN", curve=curve, pair_param_ids=pair_param_ids),
                "RN_fixed": mode_for("RN") == "fixed",
                "D0_m2_ms": D0,
                "D0_err_m2_ms": _param_error(errors=errors, param_configs=param_configs, param_name="D0_m2_ms", curve=curve, pair_param_ids=pair_param_ids),
                "D0_mm2_s": D0 * 1e9 if np.isfinite(D0) else np.nan,
                "C": C,
                "C_err": _param_error(errors=errors, param_configs=param_configs, param_name="C", curve=curve, pair_param_ids=pair_param_ids),
                "rmse": _rmse(curve.y, yhat) if ok else np.nan,
                "chi2": _chi2(curve.y, yhat) if ok else np.nan,
                "method": method,
                "ok": bool(ok),
                "msg": msg,
            }
        )
        for b_step, g, y, yh in zip(curve.b_step, curve.g, curve.y, yhat):
            point_rows.append(
                {
                    "source_file": curve.source_file,
                    "analysis_id": curve.analysis_id,
                    "subj": curve.subj,
                    "roi": curve.roi,
                    "direction": curve.direction,
                    "stat": curve.stat,
                    "td_ms": curve.td_ms,
                    "N": curve.n_value,
                    "x_model_ms": curve.x_model_ms,
                    "b_step": b_step,
                    g_type: g,
                    ycol: y,
                    "f_corr": curve.f_corr,
                    "corr_status": curve.corr_status,
                    "yhat": yh,
                    "residual": y - yh if np.isfinite(yh) else np.nan,
                }
            )
    return fit_rows, point_rows


def _build_contrast_fit_rows(signal_rows: list[dict[str, Any]], *, model_spec: SignalModelSpec) -> list[dict[str, Any]]:
    df = pd.DataFrame(signal_rows)
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    group_cols = ["pair_key", "roi", "direction", "stat"]
    for _, sub in df.groupby(group_cols, sort=True, dropna=False):
        side_rows = {
            int(row["contrast_side"]): row
            for _, row in sub.iterrows()
            if int(row.get("contrast_side", 0) or 0) in (1, 2)
        }
        if 1 not in side_rows or 2 not in side_rows:
            continue
        row1 = side_rows[1]
        row2 = side_rows[2]
        base = row1.to_dict()
        base["source_file"] = str(row1.get("contrast_source_file") or row1.get("source_file", ""))
        base["analysis_id"] = str(row1.get("contrast_analysis_id") or row1.get("analysis_id", ""))
        base["fit_kind"] = f"{model_spec.family}_contrast_from_global_signal_fit"
        base["model"] = model_spec.name
        base["fit_scope"] = "global_subj_roi_direction"
        base["global_params"] = row1.get("global_params", "")
        base["global_contrast_params"] = row1.get("global_contrast_params", "")
        base["local_params"] = row1.get("local_params", "")
        base["fixed_params"] = row1.get("fixed_params", "")
        base["N_1"] = row1.get("N", np.nan)
        base["N_2"] = row2.get("N", np.nan)
        base["td_ms"] = row1.get("td_ms", np.nan)
        base["td_ms_1"] = row1.get("td_ms", np.nan)
        base["td_ms_2"] = row2.get("td_ms", np.nan)
        base["source_file_1"] = row1.get("source_file", "")
        base["source_file_2"] = row2.get("source_file", "")
        base["f_corr_1"] = row1.get("f_corr", np.nan)
        base["f_corr_2"] = row2.get("f_corr", np.nan)
        base["corr_status_1"] = row1.get("corr_status", "")
        base["corr_status_2"] = row2.get("corr_status", "")
        base["xplot"] = "1"
        base["plot_xcol"] = f"{row1.get('gbase', row1.get('xcol', 'g'))}_1"
        base["n_points"] = int(row1.get("n_points", 0) or 0) + int(row2.get("n_points", 0) or 0)
        base["n_fit"] = int(row1.get("n_fit", 0) or 0) + int(row2.get("n_fit", 0) or 0)
        base["rmse_side_1"] = row1.get("rmse", np.nan)
        base["rmse_side_2"] = row2.get("rmse", np.nan)
        base["chi2_side_1"] = row1.get("chi2", np.nan)
        base["chi2_side_2"] = row2.get("chi2", np.nan)
        base["rmse"] = np.nanmean([base["rmse_side_1"], base["rmse_side_2"]])
        base["chi2"] = np.nansum([base["chi2_side_1"], base["chi2_side_2"]])
        base["ok"] = bool(row1.get("ok", False)) and bool(row2.get("ok", False))
        rows.append(base)
    return rows


def _sanitize_token(value: object) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "NA"


def _plot_curve(row: pd.Series, points: pd.DataFrame, *, out_png: Path, ycol: str, g_type: str) -> None:
    pts = points.copy()
    pts = pts.sort_values(g_type, kind="stable")
    x = pd.to_numeric(pts[g_type], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pts[ycol], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(pts["yhat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    x = x[mask]
    y = y[mask]
    yhat = yhat[mask]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, y, "o", color="#2c6fbb", label="data", markersize=5)
    if np.isfinite(yhat).any():
        order = np.argsort(x)
        ax.plot(x[order], yhat[order], "-", color="#c43c35", linewidth=2.0, label="fit")
    ax.set_xlabel(f"{g_type} corrected [mT/m]" if float(row.get("f_corr", 1.0) or 1.0) != 1.0 else f"{g_type} [mT/m]")
    ax.set_ylabel(str(ycol))
    ax.set_title(
        " | ".join(
            [
                f"{row.get('analysis_id')}",
                f"ROI={row.get('roi')}",
                f"dir={row.get('direction')}",
                f"N={float(row.get('N')):g}",
                f"td={float(row.get('td_ms')):g} ms",
            ]
        ),
        fontsize=10,
    )
    subtitle = (
        f"tc={float(row.get('tc_ms')):.4g} ms, alpha={float(row.get('alpha')):.4g}, "
        f"M0={float(row.get('M0')):.4g}, C={float(row.get('C')):.4g}, "
        f"f_corr={float(row.get('f_corr', 1.0) or 1.0):.5g}"
    )
    ax.text(0.01, 0.99, subtitle, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_per_analysis_outputs(
    *,
    fit_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    points_df: pd.DataFrame,
    out_root: Path,
    model_name: str,
    ycol: str,
    g_type: str,
    write_csv: bool,
    write_signal_tables: bool,
) -> None:
    if fit_df.empty:
        return
    for analysis_id, sub in fit_df.groupby("analysis_id", sort=True):
        analysis_dir = out_root / str(analysis_id)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        fit_name = f"fit_params.{model_name}.{g_type}.{ycol}.parquet"
        fit_path = analysis_dir / fit_name
        write_table_outputs(
            sub.reset_index(drop=True),
            fit_path,
            xlsx_path=fit_path.with_suffix(".xlsx"),
            csv_path=fit_path.with_suffix(".csv") if write_csv else None,
        )

        sig_sub = signal_df[
            (signal_df["contrast_analysis_id"].astype(str) == str(analysis_id))
            | (signal_df["analysis_id"].astype(str) == str(analysis_id))
        ].copy()
        if write_signal_tables and not sig_sub.empty:
            sig_path = analysis_dir / f"signal_fit_params.{model_name}.{g_type}.{ycol}.parquet"
            write_table_outputs(
                sig_sub.reset_index(drop=True),
                sig_path,
                xlsx_path=sig_path.with_suffix(".xlsx"),
                csv_path=sig_path.with_suffix(".csv") if write_csv else None,
            )

        pts_sub = points_df[points_df["analysis_id"].astype(str) == str(analysis_id)].copy()
        if pts_sub.empty and not sig_sub.empty:
            sig_ids = set(sig_sub["analysis_id"].astype(str).tolist())
            pts_sub = points_df[points_df["analysis_id"].astype(str).isin(sig_ids)].copy()
        if write_signal_tables and not pts_sub.empty:
            pts_path = analysis_dir / f"signal_fit_points.{model_name}.{g_type}.{ycol}.parquet"
            write_table_outputs(
                pts_sub.reset_index(drop=True),
                pts_path,
                csv_path=pts_path.with_suffix(".csv") if write_csv else None,
            )

        for _, row in sig_sub.iterrows():
            mask = (
                (pts_sub["source_file"].astype(str) == str(row["source_file"]))
                & (pts_sub["roi"].astype(str) == str(row["roi"]))
                & (pts_sub["direction"].astype(str) == str(row["direction"]))
                & np.isclose(pd.to_numeric(pts_sub["td_ms"], errors="coerce"), float(row["td_ms"]), atol=1e-6)
                & np.isclose(pd.to_numeric(pts_sub["N"], errors="coerce"), float(row["N"]), atol=1e-6)
            )
            pts_curve = pts_sub[mask].copy()
            if pts_curve.empty:
                continue
            out_png = analysis_dir / (
                f"{_sanitize_token(row['roi'])}.{_sanitize_token(model_name)}."
                f"{_sanitize_token(g_type)}.{_sanitize_token(ycol)}."
                f"direction_{_sanitize_token(row['direction'])}.png"
            )
            _plot_curve(row, pts_curve, out_png=out_png, ycol=ycol, g_type=g_type)


def _parse_bounds(values: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values.")
    lo, hi = float(values[0]), float(values[1])
    if not lo < hi:
        raise ValueError(f"{name} must be increasing. Received {values}.")
    return lo, hi


def _build_param_configs(args: argparse.Namespace, *, model_spec: SignalModelSpec) -> dict[str, FitParameterConfig]:
    mode_args = {
        "tc_ms": args.tc_mode,
        "alpha": args.alpha_mode,
        "RN": args.RN_mode,
        "M0": args.M0_mode,
        "C": args.C_mode,
        "D0_m2_ms": args.D0_mode,
    }
    init_args = {
        "tc_ms": args.tc_init,
        "alpha": args.alpha_init,
        "RN": args.RN_init,
        "M0": args.M0_init,
        "C": args.C_init,
        "D0_m2_ms": args.D0_fixed,
    }
    fixed_args = {
        "tc_ms": args.tc_fixed,
        "alpha": args.alpha_fixed,
        "RN": args.RN_fixed,
        "M0": args.M0_fixed,
        "C": args.C_fixed,
        "D0_m2_ms": args.D0_fixed if normalize_parameter_mode(str(args.D0_mode), param_name="D0_m2_ms") == "fixed" else None,
    }
    bounds_args = {
        "tc_ms": (args.tc_bounds, "tc_bounds"),
        "alpha": (args.alpha_bounds, "alpha_bounds"),
        "RN": (args.RN_bounds, "RN_bounds"),
        "M0": (args.M0_bounds, "M0_bounds"),
        "C": (args.C_bounds, "C_bounds"),
        "D0_m2_ms": (args.D0_bounds, "D0_bounds"),
    }

    configs: dict[str, FitParameterConfig] = {}
    for name in _param_order(model_spec):
        bounds_values, bounds_name = bounds_args[name]
        configs[name] = make_parameter_config(
            name=name,
            mode=str(mode_args[name]),
            init=float(init_args[name]),
            fixed=fixed_args[name],
            bounds=_parse_bounds(bounds_values, name=bounds_name),
            log=name in set(model_spec.log_params),
        )
    return configs


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("parquet", nargs="+", type=Path, help="Input signal *.long.parquet tables.")
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--family", choices=["ogse", "nogse"], default="ogse")
    ap.add_argument("--model", default=None, choices=signal_model_names())
    ap.add_argument("--contrast_root", type=Path, default=None, help="Existing contrast root containing tables/, used to reuse contrast analysis_id/source metadata.")
    ap.add_argument("--ycol", default="value", help="Signal column to fit. Use value or value_norm.")
    ap.add_argument("--g_type", default="g_thorsten", help="Gradient column used as G in mT/m.")
    ap.add_argument("--directions", nargs="*", default=None)
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--n_fit", type=int, default=None, help="Use first n points after sorting by G. Default: all valid points.")
    ap.add_argument("--min_points", type=int, default=4)
    mode_choices = list(VALID_PARAMETER_MODES)
    ap.add_argument("--tc_mode", default="global_td", choices=mode_choices)
    ap.add_argument("--alpha_mode", default="global_td", choices=mode_choices)
    ap.add_argument("--RN_mode", default="global_td", choices=mode_choices)
    ap.add_argument("--M0_mode", default="global_contrast", choices=mode_choices)
    ap.add_argument("--C_mode", default="global_contrast", choices=mode_choices)
    ap.add_argument("--D0_mode", choices=mode_choices, default="fixed")
    ap.add_argument("--D0_fixed", type=float, default=2.3e-12, help="D0 in m^2/ms; used as fixed value or initial seed.")
    ap.add_argument("--tc_init", type=float, default=5.0)
    ap.add_argument("--tc_fixed", type=float, default=None)
    ap.add_argument("--alpha_init", type=float, default=0.5)
    ap.add_argument("--alpha_fixed", type=float, default=None)
    ap.add_argument("--M0_init", type=float, default=np.nan)
    ap.add_argument("--M0_fixed", type=float, default=None)
    ap.add_argument("--C_init", type=float, default=0.0)
    ap.add_argument("--C_fixed", type=float, default=None)
    ap.add_argument("--RN_init", type=float, default=0.0)
    ap.add_argument("--RN_fixed", type=float, default=None)
    ap.add_argument("--tc_bounds", nargs=2, type=float, default=(0.1, 1000.0))
    ap.add_argument("--alpha_bounds", nargs=2, type=float, default=(0.0, 1.0))
    ap.add_argument("--M0_bounds", nargs=2, type=float, default=(0.0, 1e9))
    ap.add_argument("--C_bounds", nargs=2, type=float, default=(-1e9, 1e9))
    ap.add_argument("--RN_bounds", nargs=2, type=float, default=(0.0, 1e9))
    ap.add_argument("--D0_bounds", nargs=2, type=float, default=(1e-16, 1e-10))
    ap.add_argument("--max_nfev", type=int, default=400000)
    corr_group = ap.add_mutually_exclusive_group()
    corr_group.add_argument("--apply_grad_corr", action="store_true")
    corr_group.add_argument("--no_grad_corr", action="store_true")
    ap.add_argument("--corr_xlsx", type=Path, default=None)
    ap.add_argument("--corr_roi", default="Syringe")
    ap.add_argument("--corr_tol_ms", type=float, default=1e-3)
    ap.add_argument("--corr_sheet", default=None)
    ap.add_argument(
        "--corr_missing",
        choices=["error", "identity", "skip"],
        default="error",
        help="Policy when a requested gradient correction factor is missing.",
    )
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--write_csv", action="store_true", help="Write optional CSV siblings. Default: parquet/xlsx only.")
    ap.add_argument(
        "--write_signal_tables",
        action="store_true",
        help="Write curve-level signal_fit_params and signal_fit_points audit tables. Default: only contrast-level fit_params.",
    )
    args = ap.parse_args()

    g_type = normalize_axis_base(str(args.g_type))
    model_name = str(args.model or f"{args.family}_mixed_offset")
    model_spec = get_signal_model(model_name)
    if model_spec.family != str(args.family):
        raise ValueError(f"Model {model_spec.name!r} belongs to family {model_spec.family!r}, not {args.family!r}.")
    param_configs = _build_param_configs(args, model_spec=model_spec)
    df = _load_inputs(args.parquet)
    contrast_index = _build_contrast_side_index(args.contrast_root)
    apply_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    corr = None
    if apply_corr:
        if args.corr_xlsx is None:
            raise ValueError("--apply_grad_corr requires --corr_xlsx.")
        corr = read_correction_table(args.corr_xlsx)
    curves = _prepare_curves(
        df,
        ycol=str(args.ycol),
        g_type=g_type,
        stat=None if str(args.stat).upper() == "ALL" else str(args.stat),
        rois=_split_values(args.rois),
        directions=_split_values(args.directions),
        n_fit=args.n_fit,
        min_points=int(args.min_points),
        corr=corr,
        corr_roi=str(args.corr_roi),
        corr_tol_ms=float(args.corr_tol_ms),
        corr_sheet=args.corr_sheet,
        corr_missing=str(args.corr_missing),
        contrast_index=contrast_index,
    )

    fit_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    group_map: dict[tuple[str, str, str, str], list[CurveData]] = {}
    for curve in curves:
        group_map.setdefault((curve.subj, curve.roi, curve.direction, curve.stat), []).append(curve)

    for group_key, group_curves in sorted(group_map.items()):
        values, errors, ok, msg, method = _fit_group(
            group_curves,
            model_spec=model_spec,
            param_configs=param_configs,
            max_nfev=int(args.max_nfev),
        )
        rows, points = _build_outputs(
            group_curves,
            model_spec=model_spec,
            values=values,
            errors=errors,
            param_configs=param_configs,
            ok=ok,
            msg=msg,
            method=method,
            ycol=str(args.ycol),
            g_type=g_type,
        )
        fit_rows.extend(rows)
        point_rows.extend(points)
        print(
            "Fitted group:",
            f"subj={group_key[0]}",
            f"roi={group_key[1]}",
            f"direction={group_key[2]}",
            f"stat={group_key[3]}",
            f"curves={len(group_curves)}",
            f"ok={ok}",
        )

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    signal_df = pd.DataFrame(fit_rows)
    pair_fit_df = pd.DataFrame(_build_contrast_fit_rows(fit_rows, model_spec=model_spec))
    fit_df = pair_fit_df if not pair_fit_df.empty else signal_df
    points_df = pd.DataFrame(point_rows)
    fit_path = out_root / f"fit_params.{model_spec.name}.{g_type}.{args.ycol}.parquet"
    signal_path = out_root / f"signal_fit_params.{model_spec.name}.{g_type}.{args.ycol}.parquet"
    points_path = out_root / f"signal_fit_points.{model_spec.name}.{g_type}.{args.ycol}.parquet"
    write_table_outputs(
        fit_df,
        fit_path,
        xlsx_path=fit_path.with_suffix(".xlsx"),
        csv_path=fit_path.with_suffix(".csv") if args.write_csv else None,
    )
    if args.write_signal_tables:
        write_table_outputs(
            signal_df,
            signal_path,
            xlsx_path=signal_path.with_suffix(".xlsx"),
            csv_path=signal_path.with_suffix(".csv") if args.write_csv else None,
        )
        write_table_outputs(
            points_df,
            points_path,
            csv_path=points_path.with_suffix(".csv") if args.write_csv else None,
        )
    if not args.no_plots:
        _write_per_analysis_outputs(
            fit_df=fit_df,
            signal_df=signal_df,
            points_df=points_df,
            out_root=out_root,
            model_name=model_spec.name,
            ycol=str(args.ycol),
            g_type=g_type,
            write_csv=bool(args.write_csv),
            write_signal_tables=bool(args.write_signal_tables),
        )
    print("Saved fit table:", fit_path)
    if args.write_signal_tables:
        print("Saved signal-fit table:", signal_path)
        print("Saved point table:", points_path)
    if not args.no_plots:
        print("Saved per-analysis tables and plots in:", out_root)


if __name__ == "__main__":
    main()
