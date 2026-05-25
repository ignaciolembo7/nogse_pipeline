from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from fitting.b_from_g import (
    axis_uses_bvalue,
    build_axis_bundle,
    bvalue_from_gradient,
    gradient_base_for_axis,
    gradient_column_name,
    normalize_axis_base,
)
from fitting.contrast import ContrastResult, make_contrast
from ogse_fitting.fit_ogse_contrast_vs_g import _tc_peak_from_notebook_formula
from monoexp_fitting.fit_monoexp_signal_vs_bval import (
    _b_from_mode as _monoexp_b_from_mode,
    _select_fit_result as _select_monoexp_fit_result,
    monoexp,
)
from ogse_fitting.fit_ogse_signal_vs_g import (
    AUTO_FIT_ERR_FLOOR,
    AUTO_FIT_MAX_POINTS,
    AUTO_FIT_MIN_POINTS,
    AUTO_FIT_REL_TOL,
    OGSE_SIGNAL_MODEL_FREE,
    OGSE_SIGNAL_MODEL_MONOEXP,
    OGSE_SIGNAL_MODEL_REST,
    OGSE_SIGNAL_MODEL_REST_OFFSET,
    VALID_OGSE_SIGNAL_MODELS,
    _ogse_signal_free,
    _ogse_signal_rest,
    _select_fit_result as _select_ogse_signal_fit_result,
)
from tools.strict_columns import raise_on_unrecognized_column_names
from tools.value_formatting import compact_unique_values


@dataclass(frozen=True)
class FittedResampledContrastResult:
    df: pd.DataFrame
    signal_fit_params: pd.DataFrame
    signal_fit_points: pd.DataFrame
    signal_fit_curves: pd.DataFrame
    contrast_peak_params: pd.DataFrame


def _unique_float(df: pd.DataFrame, col: str, *, required: bool) -> float | None:
    if col not in df.columns:
        if required:
            raise ValueError(f"Missing required column {col!r}.")
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(values) == 0:
        if required:
            raise ValueError(f"Column {col!r} is empty.")
        return None
    if len(values) != 1:
        raise ValueError(f"Expected one unique value in {col!r}, found {values[:10]}.")
    return float(values[0])


def _compact_scalar(series: pd.Series):
    vals = pd.Series(series).dropna()
    if vals.empty:
        return np.nan
    unique = vals.unique().tolist()
    if len(unique) == 1:
        return unique[0]
    return compact_unique_values(unique)


def _side_scalar_metadata(df: pd.DataFrame, *, side: int, skip_cols: set[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for col in df.columns:
        if col in skip_cols:
            continue
        out[f"{col}_{side}"] = _compact_scalar(df[col])
    return out


def _associated_bvalue_column(axis: str, df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    base = normalize_axis_base(axis)
    if base == "bvalue":
        return "bvalue"
    if base == "bvalue_g":
        return "bvalue_g"
    if base == "bvalue_g_lin_max":
        return "bvalue_g_lin_max"
    if base == "bvalue_thorsten":
        return "bvalue_thorsten"
    if base == "g_lin_max":
        return "bvalue_g_lin_max"
    if base == "g_thorsten":
        return "bvalue_thorsten"
    if "bvalue_g" in df_a.columns or "bvalue_g" in df_b.columns:
        return "bvalue_g"
    return "bvalue"


@dataclass(frozen=True)
class _SignalCurveFit:
    model: str
    row: dict[str, object]
    gradient_values: np.ndarray
    evaluate: Callable[[np.ndarray], np.ndarray]
    bvalue_from_gradient: Callable[[np.ndarray], np.ndarray]
    n_value: float
    td_ms: float
    delta_ms: float | None
    Delta_app_ms: float | None


def _fit_one_signal_curve(
    df: pd.DataFrame,
    *,
    model: str,
    ycol: str,
    g_type: str,
    fit_points: int | None,
    auto_fit_points: bool,
    auto_fit_min_points: int,
    auto_fit_max_points: int | None,
    auto_fit_rel_tol: float,
    auto_fit_err_floor: float,
    free_M0: bool,
    fix_M0: float,
    D0_init: float,
    gamma: float,
    delta_ms: float | None,
    Delta_app_ms: float | None,
    td_ms: float | None,
) -> _SignalCurveFit:
    d = df.sort_values("b_step", kind="stable").copy()
    if ycol not in d.columns:
        raise KeyError(f"Missing ycol {ycol!r}. Columns={list(d.columns)}")

    n_value = _unique_float(d, "N", required=True)
    if n_value is None:
        raise ValueError("Could not infer N.")
    delta_value = delta_ms if delta_ms is not None else _unique_float(d, "delta_ms", required=False)
    Delta_app_value = Delta_app_ms if Delta_app_ms is not None else _unique_float(d, "Delta_app_ms", required=False)
    td_value = td_ms if td_ms is not None else _unique_float(d, "td_ms", required=True)
    if td_value is None:
        raise ValueError("Could not infer td_ms.")

    y = pd.to_numeric(d[ycol], errors="coerce").to_numpy(dtype=float)
    bundle = build_axis_bundle(
        d,
        axis=g_type,
        gamma=float(gamma),
        N=float(n_value),
        delta_ms=delta_value,
        Delta_app_ms=Delta_app_value,
    )
    gradients = np.asarray(bundle.gradient_corr, dtype=float)

    def b_from_common_gradient(g_common: np.ndarray) -> np.ndarray:
        return bvalue_from_gradient(
            np.asarray(g_common, dtype=float),
            axis=g_type,
            N=float(n_value),
            gamma=float(gamma),
            delta_ms=delta_value,
            Delta_app_ms=Delta_app_value,
        )

    model_name = str(model)
    if model_name == OGSE_SIGNAL_MODEL_MONOEXP:
        b = _monoexp_b_from_mode(
            d,
            g_type=g_type,
            gamma=float(gamma),
            N=float(n_value),
            delta_ms=delta_value,
            Delta_app_ms=Delta_app_value,
        )
        fit = _select_monoexp_fit_result(
            np.asarray(b, dtype=float),
            y,
            fit_points=fit_points,
            auto_fit_points=bool(auto_fit_points),
            free_M0=bool(free_M0),
            fix_M0=float(fix_M0),
            D0_init=float(D0_init),
            auto_fit_min_points=int(auto_fit_min_points),
            auto_fit_max_points=auto_fit_max_points,
            auto_fit_rel_tol=float(auto_fit_rel_tol),
            auto_fit_err_floor=float(auto_fit_err_floor),
        )
        if not bool(fit.get("ok", False)):
            raise ValueError(f"Signal fit failed for model={model_name}: {fit.get('msg', '')}")

        def evaluate(g_common: np.ndarray) -> np.ndarray:
            b_common = b_from_common_gradient(np.asarray(g_common, dtype=float))
            return monoexp(b_common, float(fit["M0"]), float(fit["D0_mm2_s"]))

    elif model_name in {OGSE_SIGNAL_MODEL_FREE, OGSE_SIGNAL_MODEL_REST, OGSE_SIGNAL_MODEL_REST_OFFSET}:
        fit = _select_ogse_signal_fit_result(
            gradients,
            y,
            model=model_name,
            td_ms=float(td_value),
            n_value=float(n_value),
            fit_points=fit_points,
            auto_fit_points=bool(auto_fit_points),
            free_M0=bool(free_M0),
            fix_M0=float(fix_M0),
            D0_init=float(D0_init),
            auto_fit_min_points=int(auto_fit_min_points),
            auto_fit_max_points=auto_fit_max_points,
            auto_fit_rel_tol=float(auto_fit_rel_tol),
            auto_fit_err_floor=float(auto_fit_err_floor),
        )
        if not bool(fit.get("ok", False)):
            raise ValueError(f"Signal fit failed for model={model_name}: {fit.get('msg', '')}")

        if model_name == OGSE_SIGNAL_MODEL_FREE:
            def evaluate(g_common: np.ndarray) -> np.ndarray:
                return _ogse_signal_free(
                    td_ms=float(td_value),
                    G=np.asarray(g_common, dtype=float),
                    N=float(n_value),
                    M0=float(fit["M0"]),
                    D0_m2_ms=float(fit["D0_m2_ms"]),
                )
        else:
            def evaluate(g_common: np.ndarray) -> np.ndarray:
                return _ogse_signal_rest(
                    td_ms=float(td_value),
                    G=np.asarray(g_common, dtype=float),
                    N=float(n_value),
                    tc_ms=float(fit["tc_ms"]),
                    M0=float(fit["M0"]),
                    D0_m2_ms=float(fit["D0_m2_ms"]),
                    C=float(fit.get("C", 0.0)) if model_name == OGSE_SIGNAL_MODEL_REST_OFFSET else 0.0,
                )

    else:
        raise ValueError(
            f"Unsupported OGSE signal model {model_name!r}. "
            f"Allowed values: {sorted(VALID_OGSE_SIGNAL_MODELS)}."
        )

    fit_row = {
        "model": model_name,
        "ycol": str(ycol),
        "g_type": str(g_type),
        "fit_points": fit.get("fit_points", np.nan),
        "fit_strategy": fit.get("fit_strategy", "fixed"),
        "auto_fit_metric": fit.get("auto_fit_metric", np.nan),
        "auto_fit_score": fit.get("auto_fit_score", np.nan),
        "n_points": int(len(d)),
        "n_fit": int(fit.get("n_fit", 0)),
        "td_ms": float(td_value),
        "N": float(n_value),
        "delta_ms": float(delta_value) if delta_value is not None else np.nan,
        "Delta_app_ms": float(Delta_app_value) if Delta_app_value is not None else np.nan,
        "M0": float(fit.get("M0", np.nan)),
        "M0_err": fit.get("M0_err", np.nan),
        "D0_mm2_s": fit.get("D0_mm2_s", np.nan),
        "D0_err_mm2_s": fit.get("D0_err_mm2_s", np.nan),
        "D0_m2_ms": fit.get("D0_m2_ms", np.nan),
        "D0_err_m2_ms": fit.get("D0_err_m2_ms", np.nan),
        "tc_ms": fit.get("tc_ms", np.nan),
        "tc_err_ms": fit.get("tc_err_ms", np.nan),
        "C": fit.get("C", np.nan),
        "C_err": fit.get("C_err", np.nan),
        "rmse": fit.get("rmse", np.nan),
        "rmse_log": fit.get("rmse_log", np.nan),
        "chi2": fit.get("chi2", np.nan),
        "method": fit.get("method", ""),
        "ok": True,
        "msg": fit.get("msg", ""),
    }
    return _SignalCurveFit(
        model=model_name,
        row=fit_row,
        gradient_values=gradients,
        evaluate=evaluate,
        bvalue_from_gradient=b_from_common_gradient,
        n_value=float(n_value),
        td_ms=float(td_value),
        delta_ms=delta_value,
        Delta_app_ms=Delta_app_value,
    )


def _common_gradient_grid(
    g1: np.ndarray,
    g2: np.ndarray,
    *,
    n_grid: int | None,
    grid_min: float | None = None,
    grid_max: float | None = None,
) -> tuple[np.ndarray, str]:
    a = np.asarray(g1, dtype=float)
    b = np.asarray(g2, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Cannot build a common gradient grid from empty gradient arrays.")
    if grid_min is None and grid_max is None:
        lo = max(float(np.nanmin(a)), float(np.nanmin(b)))
        hi = min(float(np.nanmax(a)), float(np.nanmax(b)))
        mode = "overlap"
        if hi < lo:
            raise ValueError(f"Signal curves have no overlapping gradient range: side1={a.min()}..{a.max()}, side2={b.min()}..{b.max()}.")
    elif grid_min is not None and grid_max is not None:
        lo = float(grid_min)
        hi = float(grid_max)
        mode = "fixed"
        if hi < lo:
            raise ValueError(f"Fixed resampling grid has max < min: {lo}..{hi}.")
    else:
        raise ValueError("Specify both grid_min and grid_max, or neither.")
    if np.isclose(lo, hi):
        return np.asarray([lo], dtype=float), mode
    if n_grid is None:
        n = min(len(np.unique(a)), len(np.unique(b)))
    else:
        n = int(n_grid)
    n = max(2, n)
    return np.linspace(lo, hi, n), mode


def build_fitted_resampled_ogse_contrast(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    *,
    axes: tuple[str, ...] | None = None,
    ycol: str = "value_norm",
    signal_model: str = OGSE_SIGNAL_MODEL_MONOEXP,
    g_type: str = "g",
    grid_n: int | None = None,
    grid_min: float | None = None,
    grid_max: float | None = None,
    fit_points: int | None = 6,
    auto_fit_points: bool = False,
    auto_fit_min_points: int = AUTO_FIT_MIN_POINTS,
    auto_fit_max_points: int | None = AUTO_FIT_MAX_POINTS,
    auto_fit_rel_tol: float = AUTO_FIT_REL_TOL,
    auto_fit_err_floor: float = AUTO_FIT_ERR_FLOOR,
    free_M0: bool = False,
    fix_M0: float = 1.0,
    D0_init: float = 0.0023,
    gamma: float = 267.5221900,
    peak_D0_fix: float = 3.2e-12,
    delta_ms: float | None = None,
    Delta_app_ms: float | None = None,
    td_ms: float | None = None,
    key_cols: tuple[str, ...] = ("stat", "roi", "direction", "b_step"),
) -> FittedResampledContrastResult:
    raise_on_unrecognized_column_names(df_ref.columns, context="build_fitted_resampled_ogse_contrast(df_ref)")
    raise_on_unrecognized_column_names(df_cmp.columns, context="build_fitted_resampled_ogse_contrast(df_cmp)")

    model_name = str(signal_model)
    if model_name not in VALID_OGSE_SIGNAL_MODELS:
        raise ValueError(
            f"Unsupported OGSE signal model {model_name!r}. "
            f"Allowed values: {sorted(VALID_OGSE_SIGNAL_MODELS)}."
        )
    fit_axis = normalize_axis_base(g_type)
    if axis_uses_bvalue(fit_axis):
        common_gradient_col = gradient_column_name(fit_axis)
    else:
        common_gradient_col = gradient_base_for_axis(fit_axis)
    bvalue_col = _associated_bvalue_column(fit_axis, df_ref, df_cmp)

    a = df_ref.copy()
    b = df_cmp.copy()
    if axes is not None:
        axes = tuple(str(x) for x in axes)
        a["direction"] = a["direction"].astype(str)
        b["direction"] = b["direction"].astype(str)
        a = a[a["direction"].isin(axes)].copy()
        b = b[b["direction"].isin(axes)].copy()

    group_cols = [c for c in key_cols if c != "b_step" and c in a.columns and c in b.columns]
    if "roi" not in group_cols or "direction" not in group_cols:
        raise KeyError("Fitted/resampled OGSE contrast requires roi and direction columns on both sides.")
    if "stat" not in group_cols and "stat" in a.columns and "stat" in b.columns:
        group_cols = ["stat", *group_cols]

    rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    fit_point_rows: list[dict[str, object]] = []
    fit_curve_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    skip_cols = set(key_cols) | {"value", "value_norm", "S0", common_gradient_col, bvalue_col}

    for key, ref_group in a.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))
        cmp_mask = pd.Series(True, index=b.index)
        for col, value in key_dict.items():
            cmp_mask &= b[col].astype(str) == str(value)
        cmp_group = b[cmp_mask].copy()
        if cmp_group.empty:
            continue

        ref_fit = _fit_one_signal_curve(
            ref_group,
            model=model_name,
            ycol=ycol,
            g_type=fit_axis,
            fit_points=fit_points,
            auto_fit_points=auto_fit_points,
            auto_fit_min_points=auto_fit_min_points,
            auto_fit_max_points=auto_fit_max_points,
            auto_fit_rel_tol=auto_fit_rel_tol,
            auto_fit_err_floor=auto_fit_err_floor,
            free_M0=free_M0,
            fix_M0=fix_M0,
            D0_init=D0_init,
            gamma=gamma,
            delta_ms=delta_ms,
            Delta_app_ms=Delta_app_ms,
            td_ms=td_ms,
        )
        cmp_fit = _fit_one_signal_curve(
            cmp_group,
            model=model_name,
            ycol=ycol,
            g_type=fit_axis,
            fit_points=fit_points,
            auto_fit_points=auto_fit_points,
            auto_fit_min_points=auto_fit_min_points,
            auto_fit_max_points=auto_fit_max_points,
            auto_fit_rel_tol=auto_fit_rel_tol,
            auto_fit_err_floor=auto_fit_err_floor,
            free_M0=free_M0,
            fix_M0=fix_M0,
            D0_init=D0_init,
            gamma=gamma,
            delta_ms=delta_ms,
            Delta_app_ms=Delta_app_ms,
            td_ms=td_ms,
        )

        g_common, grid_mode = _common_gradient_grid(
            ref_fit.gradient_values,
            cmp_fit.gradient_values,
            n_grid=grid_n,
            grid_min=grid_min,
            grid_max=grid_max,
        )
        y1 = ref_fit.evaluate(g_common)
        y2 = cmp_fit.evaluate(g_common)
        contrast = np.asarray(y1, dtype=float) - np.asarray(y2, dtype=float)
        b1 = ref_fit.bvalue_from_gradient(g_common)
        b2 = cmp_fit.bvalue_from_gradient(g_common)

        s0_1 = _unique_float(ref_group, "S0", required=False)
        s0_2 = _unique_float(cmp_group, "S0", required=False)
        value = contrast.copy()
        value_norm = contrast.copy()
        if ycol == "value_norm" and s0_1 is not None and s0_2 is not None:
            value = np.asarray(y1, dtype=float) * float(s0_1) - np.asarray(y2, dtype=float) * float(s0_2)
        elif ycol == "value" and s0_1 is not None and s0_2 is not None and s0_1 != 0.0 and s0_2 != 0.0:
            value_norm = np.asarray(y1, dtype=float) / float(s0_1) - np.asarray(y2, dtype=float) / float(s0_2)

        side_meta_1 = _side_scalar_metadata(ref_group, side=1, skip_cols=skip_cols)
        side_meta_2 = _side_scalar_metadata(cmp_group, side=2, skip_cols=skip_cols)
        if s0_1 is not None:
            side_meta_1["S0_1"] = float(s0_1)
        if s0_2 is not None:
            side_meta_2["S0_2"] = float(s0_2)

        for i, g_val in enumerate(g_common, start=1):
            row: dict[str, object] = {
                **key_dict,
                "b_step": int(i),
                "value": float(value[i - 1]),
                "value_norm": float(value_norm[i - 1]),
                "contrast_source": "fitted_resampled",
                "signal_fit_model": model_name,
                "signal_fit_ycol": str(ycol),
                "signal_fit_g_type": str(fit_axis),
                "resample_grid": grid_mode,
                "resample_grid_n": int(len(g_common)),
                "resample_grid_min": float(g_common[0]) if len(g_common) else np.nan,
                "resample_grid_max": float(g_common[-1]) if len(g_common) else np.nan,
                common_gradient_col: float(g_val),
                f"{common_gradient_col}_1": float(g_val),
                f"{common_gradient_col}_2": float(g_val),
                bvalue_col: float(0.5 * (float(b1[i - 1]) + float(b2[i - 1]))),
                f"{bvalue_col}_1": float(b1[i - 1]),
                f"{bvalue_col}_2": float(b2[i - 1]),
                "signal_fit_1": float(y1[i - 1]),
                "signal_fit_2": float(y2[i - 1]),
                **side_meta_1,
                **side_meta_2,
            }
            rows.append(row)

        valid_peak = np.isfinite(g_common) & np.isfinite(contrast)
        if np.any(valid_peak):
            valid_indices = np.where(valid_peak)[0]
            peak_idx = int(valid_indices[int(np.nanargmax(contrast[valid_peak]))])
            g_peak = float(g_common[peak_idx])
            signal_peak = float(contrast[peak_idx])
            l_G, L_cf, lcf_peak_m, tc_peak_ms = _tc_peak_from_notebook_formula(
                td_ms=float(ref_fit.td_ms),
                g_peak_mTpm=g_peak,
                D0_fix_m2_ms=float(peak_D0_fix),
                gamma_rad_ms_mT=float(gamma),
            )
        else:
            peak_idx = -1
            g_peak = np.nan
            signal_peak = np.nan
            l_G = np.nan
            L_cf = np.nan
            lcf_peak_m = np.nan
            tc_peak_ms = np.nan

        peak_rows.append(
            {
                **key_dict,
                "td_ms": float(ref_fit.td_ms),
                "td_min_ms": float(ref_fit.td_ms),
                "td_max_ms": float(ref_fit.td_ms),
                "n_td": 1,
                "N_1": int(round(float(ref_fit.n_value))),
                "N_2": int(round(float(cmp_fit.n_value))),
                "delta_ms_1": float(ref_fit.delta_ms) if ref_fit.delta_ms is not None else np.nan,
                "Delta_app_ms_1": float(ref_fit.Delta_app_ms) if ref_fit.Delta_app_ms is not None else np.nan,
                "delta_ms_2": float(cmp_fit.delta_ms) if cmp_fit.delta_ms is not None else np.nan,
                "Delta_app_ms_2": float(cmp_fit.Delta_app_ms) if cmp_fit.Delta_app_ms is not None else np.nan,
                "fit_kind": "ogse_contrast",
                "model": model_name,
                "ycol": str(ycol),
                "stat": key_dict.get("stat", np.nan),
                "n_points": int(len(contrast)),
                "n_fit": int(len(contrast)),
                "rmse": np.nan,
                "chi2": np.nan,
                "gbase": str(fit_axis),
                "fit_xcol": str(fit_axis),
                "plot_xcol": common_gradient_col,
                "xplot": "resampled",
                "f_corr": np.nan,
                "f_corr_1": 1.0,
                "f_corr_2": 1.0,
                "peak_method": "signal_fit_resampled_grid",
                "peak_grid_n": int(len(g_common)),
                "g1_max_raw_mTm": float(np.nanmax(ref_fit.gradient_values)) if ref_fit.gradient_values.size else np.nan,
                "g2_max_raw_mTm": float(np.nanmax(cmp_fit.gradient_values)) if cmp_fit.gradient_values.size else np.nan,
                "g1_max_corr_mTm": float(np.nanmax(ref_fit.gradient_values)) if ref_fit.gradient_values.size else np.nan,
                "g2_max_corr_mTm": float(np.nanmax(cmp_fit.gradient_values)) if cmp_fit.gradient_values.size else np.nan,
                "peak_g_max_mTm": float(g_common[-1]) if len(g_common) else np.nan,
                "g1_search_max_raw_mTm": float(g_common[-1]) if len(g_common) else np.nan,
                "g2_search_max_raw_mTm": float(g_common[-1]) if len(g_common) else np.nan,
                "g1_search_max_corr_mTm": float(g_common[-1]) if len(g_common) else np.nan,
                "g2_search_max_corr_mTm": float(g_common[-1]) if len(g_common) else np.nan,
                "peak_resample_gradient": True,
                "peak_resample_g_max_corr_mTm": float(g_common[-1]) if len(g_common) else np.nan,
                "peak_fraction": float(peak_idx / max(1, len(g_common) - 1)) if peak_idx >= 0 else np.nan,
                "g1_peak_raw_mTm": g_peak,
                "g2_peak_raw_mTm": g_peak,
                "g1_peak_corr_mTm": g_peak,
                "g2_peak_corr_mTm": g_peak,
                "x_peak_raw_mTm": g_peak,
                "x_peak_corr_mTm": g_peak,
                "g_peak_resampled_raw_mTm": g_peak,
                "g_peak_resampled_corr_mTm": g_peak,
                "signal_peak": signal_peak,
                "l_G_peak_m": l_G,
                "L_cf_peak": L_cf,
                "lcf_peak_m": lcf_peak_m,
                "tc_peak_ms": tc_peak_ms,
                "tc_peak_resampled_ms": tc_peak_ms,
                "method": "signal_fit_1_minus_signal_fit_2_resampled_peak",
                "ok": bool(np.isfinite(signal_peak)),
                "msg": "",
            }
        )

        for side, fit in [(1, ref_fit), (2, cmp_fit)]:
            fit_row = {**key_dict, **fit.row}
            fit_row["side"] = int(side)
            fit_row["resample_grid"] = grid_mode
            fit_row["resample_grid_n"] = int(len(g_common))
            fit_row["resample_grid_min"] = float(g_common[0]) if len(g_common) else np.nan
            fit_row["resample_grid_max"] = float(g_common[-1]) if len(g_common) else np.nan
            fit_rows.append(fit_row)

        for side, group, fit in [(1, ref_group, ref_fit), (2, cmp_group, cmp_fit)]:
            d_points = group.sort_values("b_step", kind="stable").copy()
            n_fit = int(fit.row.get("n_fit", 0) or 0)
            y_points = pd.to_numeric(d_points[ycol], errors="coerce").to_numpy(dtype=float)
            b_steps = pd.to_numeric(d_points["b_step"], errors="coerce").to_numpy(dtype=float)
            for idx, (b_step, g_val, y_val) in enumerate(zip(b_steps, fit.gradient_values, y_points), start=1):
                fit_point_rows.append(
                    {
                        **key_dict,
                        "side": int(side),
                        "b_step": int(b_step) if np.isfinite(b_step) else idx,
                        common_gradient_col: float(g_val),
                        "signal_observed": float(y_val),
                        "fit_used": bool(idx <= n_fit),
                        "signal_fit_model": model_name,
                        "signal_fit_ycol": str(ycol),
                        "signal_fit_g_type": str(fit_axis),
                    }
                )

            y_curve = fit.evaluate(g_common)
            for idx, (g_val, y_val) in enumerate(zip(g_common, y_curve), start=1):
                fit_curve_rows.append(
                    {
                        **key_dict,
                        "side": int(side),
                        "resample_step": int(idx),
                        common_gradient_col: float(g_val),
                        "signal_fit": float(y_val),
                        "signal_fit_model": model_name,
                        "signal_fit_ycol": str(ycol),
                        "signal_fit_g_type": str(fit_axis),
                        "resample_grid": grid_mode,
                        "resample_grid_n": int(len(g_common)),
                        "resample_grid_min": float(g_common[0]) if len(g_common) else np.nan,
                        "resample_grid_max": float(g_common[-1]) if len(g_common) else np.nan,
                    }
                )

    return FittedResampledContrastResult(
        df=pd.DataFrame(rows),
        signal_fit_params=pd.DataFrame(fit_rows),
        signal_fit_points=pd.DataFrame(fit_point_rows),
        signal_fit_curves=pd.DataFrame(fit_curve_rows),
        contrast_peak_params=pd.DataFrame(peak_rows),
    )


__all__ = [
    "ContrastResult",
    "FittedResampledContrastResult",
    "build_fitted_resampled_ogse_contrast",
    "make_contrast",
]
