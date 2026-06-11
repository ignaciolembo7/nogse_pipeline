from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.io import fit_params_output_basename, read_table_file, write_table_outputs
from fitting.b_from_g import (
    VALID_AXIS_BASES,
    axes_share_gradient_family,
    axis_from_gradient,
    build_axis_bundle,
    normalize_axis_base,
)
from fitting.core import chi2 as fit_chi2
from fitting.core import fit_least_squares
from fitting.core import r2_score, rmse as fit_rmse
from fitting.experiments import split_all_or_values
from fitting.signal_tables import signal_table_analysis_id
from nogse_plotting.plot_nogse_signal_vs_g import plot_nogse_signal_group
from models.model_fitting import M_nogse_free, M_nogse_mixed
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.fit_params_schema import standardize_fit_params
from tools.value_formatting import scalar_or_compact_column, scalar_or_compact_series


@dataclass(frozen=True)
class FitParameterSpec:
    name: str
    p0: float
    bounds: tuple[float, float]
    log_scale: bool = False


@dataclass(frozen=True)
class SignalModelSpec:
    name: str
    evaluator: Callable[..., np.ndarray]
    fit_params: tuple[FitParameterSpec, ...]


@dataclass(frozen=True)
class FitOutputs:
    fit_params: pd.DataFrame


@dataclass(frozen=True)
class BuiltModel:
    model: Callable[..., np.ndarray]
    x_model_ms: float
    param_names: list[str]
    p0: list[float]
    bounds: tuple[list[float], list[float]]
    log_params: list[str]


SIGNAL_MODELS: dict[str, SignalModelSpec] = {
    "free_cpmg": SignalModelSpec(
        name="free_cpmg",
        evaluator=M_nogse_free,
        fit_params=(
            FitParameterSpec(name="M0", p0=1.0, bounds=(0.0, np.inf), log_scale=False),
            FitParameterSpec(name="D0_m2_ms", p0=2.3e-12, bounds=(1e-16, np.inf), log_scale=True),
        ),
    ),
    "free_hahn": SignalModelSpec(
        name="free_hahn",
        evaluator=M_nogse_free,
        fit_params=(
            FitParameterSpec(name="M0", p0=1.0, bounds=(0.0, np.inf), log_scale=False),
            FitParameterSpec(name="D0_m2_ms", p0=2.3e-12, bounds=(1e-16, np.inf), log_scale=True),
        ),
    ),
}
GLOBAL_SIGNAL_MODELS = ("mixed_global",)
VALID_MODELS = tuple(sorted((*SIGNAL_MODELS, *GLOBAL_SIGNAL_MODELS)))
GAMMA_DEFAULT = 267.5221900


def analysis_id_from_path(path: Path) -> str:
    return signal_table_analysis_id(path)


def _unique_scalar(df: pd.DataFrame, col: str, *, required: bool = False):
    if col not in df.columns:
        if required:
            raise ValueError(f"Missing required column {col!r}.")
        return None
    values = pd.Series(df[col]).dropna().unique().tolist()
    if len(values) == 0:
        if required:
            raise ValueError(f"Column {col!r} is empty.")
        return None
    if len(values) > 1:
        raise ValueError(f"Column {col!r} is not unique inside the group: {values}")
    return values[0]


def _resolve_sigma(avg_df: pd.DataFrame, std_df: pd.DataFrame | None, *, xcol: str, ycol: str) -> np.ndarray | None:
    if std_df is None or std_df.empty:
        return None

    err = std_df.copy()
    err[xcol] = pd.to_numeric(err[xcol], errors="coerce")
    err["value"] = pd.to_numeric(err["value"], errors="coerce")
    err = err.dropna(subset=[xcol, "value"]).sort_values(xcol)
    if err.empty:
        return None

    sigma = err["value"].to_numpy(dtype=float).copy()
    if ycol == "value_norm":
        s0 = pd.to_numeric(avg_df["S0"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = sigma / s0

    if sigma.shape[0] != avg_df.shape[0]:
        return None
    if not np.isfinite(sigma).any():
        return None
    sigma[~np.isfinite(sigma)] = np.nanmax(sigma[np.isfinite(sigma)])
    sigma[sigma <= 0] = np.nanmax(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
    return sigma


def validate_bounds(name: str, bounds: Sequence[float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound: {lower}, {upper}")
    return lower, upper


def _clamp_p0(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(min(max(float(value), lower), upper))


def validate_fixed_value(name: str, value: float | None, bounds: tuple[float, float]) -> None:
    if value is None:
        return
    lower, upper = bounds
    if not lower <= float(value) <= upper:
        raise ValueError(f"{name} fixed value {value} is outside bounds [{lower}, {upper}].")


def validate_log_bounds(name: str, bounds: tuple[float, float]) -> None:
    lower, _upper = bounds
    if lower <= 0:
        raise ValueError(f"{name} lower bound must be positive when fitting in log-space: {lower}")


def _resolve_plot_axis(*, fit_axis: str, plot_axis: str | None) -> str:
    resolved_fit = normalize_axis_base(fit_axis)
    if plot_axis is None:
        return resolved_fit
    resolved_plot = normalize_axis_base(plot_axis)
    if not axes_share_gradient_family(resolved_fit, resolved_plot):
        raise ValueError(
            "plot_xcol must stay in the same gradient family as xcol. "
            f"Received xcol={fit_axis!r}, plot_xcol={plot_axis!r}."
        )
    return resolved_plot


def _resolve_model_geometry(
    df: pd.DataFrame,
    *,
    n_value: int,
) -> tuple[float, float, float]:
    x_value = _unique_scalar(df, "x", required=False)
    y_value = _unique_scalar(df, "y", required=False)
    x_model_ms = float(x_value) if x_value is not None else np.nan
    y_model_ms = float(y_value) if y_value is not None else np.nan
    if np.isfinite(x_model_ms) and np.isfinite(y_model_ms):
        te_model_ms = y_model_ms + (float(n_value) - 1.0) * x_model_ms
    else:
        te_value = (
            _coerce_optional_time(_unique_scalar(df, "TN", required=False))
            or _coerce_optional_time(_unique_scalar(df, "td_ms", required=False))
            or _coerce_optional_time(_unique_scalar(df, "TE", required=False))
        )
        if te_value is None:
            raise ValueError(f"Invalid x/y geometry in signal table: x={x_value!r}, y={y_value!r}.")
        te_model_ms = float(te_value)
        x_model_ms = float(te_model_ms) / float(n_value)
        y_model_ms = float(te_model_ms) - (float(n_value) - 1.0) * x_model_ms
    if not np.isfinite(te_model_ms):
        raise ValueError(f"Invalid TE reconstructed from x/y geometry: x={x_model_ms}, y={y_model_ms}, N={n_value}.")
    return float(te_model_ms), float(x_model_ms), float(y_model_ms)


def _coerce_optional_time(value: object | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _read_table(path: str | Path) -> pd.DataFrame:
    return read_table_file(path)


def _ensure_subject_column(df: pd.DataFrame, *, analysis_id: str, source_file: str) -> pd.DataFrame:
    out = df.copy()
    if "sheet" in out.columns:
        out["sheet"] = out["sheet"].map(canonical_sheet_name)
    else:
        out["sheet"] = canonical_sheet_name(analysis_id)
    if "subj" in out.columns:
        subj = out["subj"].astype(str).str.strip()
        invalid = subj.str.lower().isin({"", "nan", "none", "<na>"})
        out["subj"] = subj.mask(invalid, out["sheet"].map(lambda sheet: infer_subj_label(sheet, source_name=source_file)))
    else:
        out["subj"] = out["sheet"].map(lambda sheet: infer_subj_label(sheet, source_name=source_file))
    return out


def _normalize_alpha_table(
    alpha_df: pd.DataFrame | None,
    *,
    alpha_col: str | None,
    alpha_td_col: str,
) -> pd.DataFrame | None:
    if alpha_df is None:
        return None
    out = alpha_df.copy()
    resolved_alpha_col = alpha_col
    if resolved_alpha_col is None:
        for candidate in ("alpha", "alpha_macro"):
            if candidate in out.columns:
                resolved_alpha_col = candidate
                break
    if resolved_alpha_col is None or resolved_alpha_col not in out.columns:
        raise ValueError(
            "Could not resolve an alpha column. Pass --alpha_col or include one of: alpha, alpha_macro."
        )
    rename_map = {resolved_alpha_col: "__alpha__"}
    if alpha_td_col in out.columns:
        rename_map[alpha_td_col] = "__alpha_td_ms__"
    out = out.rename(columns=rename_map)
    out["__alpha__"] = pd.to_numeric(out["__alpha__"], errors="coerce")
    if "__alpha_td_ms__" in out.columns:
        out["__alpha_td_ms__"] = pd.to_numeric(out["__alpha_td_ms__"], errors="coerce")
    for col in ("subj", "roi", "direction"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out.dropna(subset=["__alpha__"]).copy()


def _lookup_alpha_for_curve(
    curve: pd.DataFrame,
    alpha_df: pd.DataFrame | None,
    *,
    td_ms: float,
    alpha_td_tol_ms: float,
) -> tuple[float, str]:
    if "alpha" in curve.columns:
        value = _unique_scalar(curve, "alpha", required=False)
        if value is not None and np.isfinite(float(value)):
            return float(value), "input:alpha"

    if alpha_df is None or alpha_df.empty:
        raise ValueError("mixed_global requires alpha values in the signal table or via --alpha_table.")

    candidates = alpha_df.copy()
    for col in ("subj", "roi", "direction"):
        if col in curve.columns and col in candidates.columns:
            value = _unique_scalar(curve, col, required=False)
            if value is not None:
                candidates = candidates[candidates[col].astype(str) == str(value)].copy()

    if candidates.empty:
        label = ", ".join(
            f"{col}={_unique_scalar(curve, col, required=False)!r}"
            for col in ("subj", "roi", "direction")
            if col in curve.columns
        )
        raise ValueError(f"No alpha match for curve ({label}).")

    if "__alpha_td_ms__" in candidates.columns and candidates["__alpha_td_ms__"].notna().any():
        dt = np.abs(pd.to_numeric(candidates["__alpha_td_ms__"], errors="coerce") - float(td_ms))
        candidates = candidates[dt <= float(alpha_td_tol_ms)].copy()
        if candidates.empty:
            raise ValueError(f"No alpha match within {alpha_td_tol_ms} ms for td_ms={td_ms}.")

    values = candidates["__alpha__"].dropna().unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Alpha is not unique for td_ms={td_ms}: {values[:10]}")
    return float(values[0]), "alpha_table"


def _validate_time_consistency(label: str, value_ms: float | None, te_model_ms: float) -> None:
    if value_ms is None:
        return
    if np.isclose(float(value_ms), float(te_model_ms), rtol=0.0, atol=1e-6):
        return
    raise ValueError(
        f"{label}={value_ms} is inconsistent with x/y-derived TE={te_model_ms}. "
        "The modern NOGSE signal fit requires a single table-defined geometry."
    )


def _resolve_runtime_parameter(
    param: FitParameterSpec,
    *,
    fix_m0: float | None,
    fix_d0: float | None,
    m0_bounds: tuple[float, float],
    d0_bounds: tuple[float, float],
) -> tuple[float | None, tuple[float, float]]:
    if param.name == "M0":
        return fix_m0, m0_bounds
    if param.name == "D0_m2_ms":
        return fix_d0, d0_bounds
    return None, param.bounds


def _build_model(
    model_name: str,
    *,
    n_value: int,
    te_model_ms: float,
    x_model_ms: float,
    fix_m0: float | None,
    fix_d0: float | None,
    m0_bounds: tuple[float, float],
    d0_bounds: tuple[float, float],
) -> BuiltModel:
    if model_name not in SIGNAL_MODELS:
        raise ValueError(f"Unsupported NOGSE signal model {model_name!r}. Allowed values: {sorted(SIGNAL_MODELS)}.")

    spec = SIGNAL_MODELS[model_name]

    param_names: list[str] = []
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    log_params: list[str] = []
    fixed_params: dict[str, float] = {}

    for param in spec.fit_params:
        fixed_value, runtime_bounds = _resolve_runtime_parameter(
            param,
            fix_m0=fix_m0,
            fix_d0=fix_d0,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )
        if fixed_value is not None:
            fixed_params[param.name] = float(fixed_value)
            continue
        param_names.append(param.name)
        p0.append(_clamp_p0(float(param.p0), runtime_bounds))
        lower.append(float(runtime_bounds[0]))
        upper.append(float(runtime_bounds[1]))
        if param.log_scale:
            log_params.append(param.name)

    def model(g: np.ndarray, *values: float) -> np.ndarray:
        fitted = dict(fixed_params)
        fitted.update(zip(param_names, values))
        ordered_values = [float(fitted[param.name]) for param in spec.fit_params]
        return spec.evaluator(float(te_model_ms), g, int(n_value), x_model_ms, *ordered_values)

    return BuiltModel(
        model=model,
        x_model_ms=x_model_ms,
        param_names=param_names,
        p0=p0,
        bounds=(lower, upper),
        log_params=log_params,
    )


def fit_one_group(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    model_name: str,
    xcol: str,
    ycol: str,
    fix_m0: float | None,
    fix_d0: float | None,
    m0_bounds: tuple[float, float],
    d0_bounds: tuple[float, float],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    data = avg_df.copy()
    data[xcol] = pd.to_numeric(data[xcol], errors="coerce")
    data[ycol] = pd.to_numeric(data[ycol], errors="coerce")
    data = data.dropna(subset=[xcol, ycol]).sort_values(xcol)
    if data.empty:
        raise ValueError("No valid points remain after numeric filtering.")

    tn_value = _coerce_optional_time(_unique_scalar(data, "TN", required=False))
    td_value = _coerce_optional_time(_unique_scalar(data, "td_ms", required=False))
    n_value = int(round(float(_unique_scalar(data, "N", required=True))))
    te_model_ms, x_model_ms, _y_model_ms = _resolve_model_geometry(
        data,
        n_value=n_value,
    )
    _validate_time_consistency("TN", tn_value, te_model_ms)
    _validate_time_consistency("td_ms", td_value, te_model_ms)
    tn_ms = float(tn_value) if tn_value is not None else float(td_value) if td_value is not None else float(te_model_ms)

    built = _build_model(
        model_name,
        n_value=n_value,
        te_model_ms=te_model_ms,
        x_model_ms=x_model_ms,
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
    )

    x = data[xcol].to_numpy(dtype=float)
    y = data[ycol].to_numpy(dtype=float)
    sigma = _resolve_sigma(data, std_df, xcol=xcol, ycol=ycol)

    p0 = list(built.p0)
    if "M0" in built.param_names:
        p0[built.param_names.index("M0")] = _clamp_p0(float(np.nanmax(y)), m0_bounds)

    if built.param_names:
        fit = fit_least_squares(
            built.model,
            x,
            y,
            param_names=built.param_names,
            p0=p0,
            bounds=built.bounds,
            log_params=built.log_params,
            sigma=sigma,
            max_nfev=200000,
            method="least_squares_logD" if built.log_params else "least_squares",
        )
        yhat = fit.yhat
        values = fit.values
        errors = fit.errors
        method = fit.method
        rmse = fit.rmse
        chi2 = fit.chi2
        r2 = fit.r2
        popt = [values[name] for name in built.param_names]
    else:
        yhat = built.model(x)
        values = {}
        errors = {}
        method = "fixed"
        rmse = fit_rmse(y, yhat)
        chi2 = fit_chi2(y, yhat)
        r2 = r2_score(y, yhat)
        popt = []

    fit_row: dict[str, object] = {
        "model": model_name,
        "xcol": xcol,
        "ycol": ycol,
        "TN": tn_ms,
        "N": n_value,
        "x_model_ms": built.x_model_ms,
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "rmse": float(rmse),
        "r2": float(r2),
        "chi2": float(chi2),
        "method": method,
        "n_points": int(len(x)),
        "n_fit": int(len(x)),
        "M0_bound_min": float(m0_bounds[0]),
        "M0_bound_max": float(m0_bounds[1]),
        "D0_bound_min": float(d0_bounds[0]),
        "D0_bound_max": float(d0_bounds[1]),
    }
    if fix_m0 is not None:
        fit_row["M0"] = float(fix_m0)
        fit_row["M0_err"] = np.nan
    if fix_d0 is not None:
        fit_row["D0_m2_ms"] = float(fix_d0)
        fit_row["D0_err_m2_ms"] = np.nan
    fit_row.update(values)
    fit_row.update(errors)

    xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    yfit = built.model(xfit, *popt) if built.param_names else built.model(xfit)
    return fit_row, x, y, np.column_stack([xfit, yfit])


def plot_fit_one_group(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    xcol: str,
    ycol: str,
    fit_row: dict[str, object],
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_curve: np.ndarray,
    out_png: Path,
) -> None:
    analysis_id = str(fit_row.get("analysis_id", "")) or "nogse_signal"
    plot_nogse_signal_group(
        avg_df=avg_df,
        std_df=std_df,
        xcol=xcol,
        ycol=ycol,
        out_png=out_png,
        analysis_id=analysis_id,
        roi=str(fit_row.get("roi", "roi")),
        direction=str(fit_row.get("direction", "direction")),
        signal_type=None,
        fit_row=fit_row,
        fit_curve=fit_curve,
        x_data=np.asarray(x_data, dtype=float),
        y_data=np.asarray(y_data, dtype=float),
        data_label="signal",
        connect_data=False,
    )


def _curve_group_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["analysis_id", "source_file", "subj", "roi", "direction", "type", "TN", "td_ms", "N", "x", "y"]
    return [col for col in preferred if col in df.columns]


def _prepare_global_mixed_rows(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    xcol: str,
    plot_xcol: str,
    ycol: str,
    f_by_direction: Mapping[str, float] | None,
    alpha_df: pd.DataFrame | None,
    alpha_td_tol_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    std_rows: list[pd.DataFrame] = []

    group_cols = _curve_group_columns(avg_df)
    if not group_cols:
        raise ValueError("Could not identify curve grouping columns for mixed_global.")

    for _key, curve in avg_df.groupby(group_cols, sort=False, dropna=False):
        curve_fit = curve.copy()
        n_value = int(round(float(_unique_scalar(curve_fit, "N", required=True))))
        te_model_ms, x_model_ms, _y_model_ms = _resolve_model_geometry(curve_fit, n_value=n_value)
        tn_value = _coerce_optional_time(_unique_scalar(curve_fit, "TN", required=False))
        td_value = _coerce_optional_time(_unique_scalar(curve_fit, "td_ms", required=False))
        _validate_time_consistency("TN", tn_value, te_model_ms)
        _validate_time_consistency("td_ms", td_value, te_model_ms)
        td_ms = float(tn_value) if tn_value is not None else float(td_value) if td_value is not None else te_model_ms
        alpha_value, alpha_source = _lookup_alpha_for_curve(
            curve_fit,
            alpha_df,
            td_ms=td_ms,
            alpha_td_tol_ms=alpha_td_tol_ms,
        )
        if not np.isfinite(alpha_value) or alpha_value < 0.0 or alpha_value > 1.0:
            raise ValueError(
                f"Invalid alpha={alpha_value} for mixed_global at td_ms={td_ms}. "
                "M_nogse_mixed requires alpha in [0, 1]."
            )

        direction = str(_unique_scalar(curve_fit, "direction", required=True))
        f_corr = float(f_by_direction.get(direction, 1.0)) if f_by_direction else 1.0
        delta_ms = _unique_scalar(curve_fit, "delta_ms", required=False)
        Delta_app_ms = _unique_scalar(curve_fit, "Delta_app_ms", required=False)

        fit_bundle = build_axis_bundle(
            curve_fit,
            axis=xcol,
            correction_factor=float(f_corr),
            gamma=GAMMA_DEFAULT,
            N=n_value,
            delta_ms=None if delta_ms is None else float(delta_ms),
            Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
        )
        plot_bundle = build_axis_bundle(
            curve_fit,
            axis=plot_xcol,
            correction_factor=float(f_corr),
            gamma=GAMMA_DEFAULT,
            N=n_value,
            delta_ms=None if delta_ms is None else float(delta_ms),
            Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
        )

        curve_fit["__fit_g__"] = fit_bundle.gradient_corr
        curve_fit["__plot_x__"] = plot_bundle.axis_corr
        curve_fit["__TE_ms__"] = float(te_model_ms)
        curve_fit["__x_model_ms__"] = float(x_model_ms)
        curve_fit["__N__"] = int(n_value)
        curve_fit["__alpha__"] = float(alpha_value)
        curve_fit["__alpha_source__"] = str(alpha_source)
        curve_fit["__td_model_ms__"] = float(td_ms)
        curve_fit["__f_corr__"] = float(f_corr)
        rows.append(curve_fit)

        if std_df is not None and not std_df.empty:
            std_curve = std_df.copy()
            for col in group_cols:
                if col in std_curve.columns and col in curve_fit.columns:
                    value = _unique_scalar(curve_fit, col, required=False)
                    if value is not None:
                        std_curve = std_curve[std_curve[col].astype(str) == str(value)].copy()
            if not std_curve.empty:
                std_fit_bundle = build_axis_bundle(
                    std_curve,
                    axis=xcol,
                    correction_factor=float(f_corr),
                    gamma=GAMMA_DEFAULT,
                    N=n_value,
                    delta_ms=None if delta_ms is None else float(delta_ms),
                    Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
                )
                std_plot_bundle = build_axis_bundle(
                    std_curve,
                    axis=plot_xcol,
                    correction_factor=float(f_corr),
                    gamma=GAMMA_DEFAULT,
                    N=n_value,
                    delta_ms=None if delta_ms is None else float(delta_ms),
                    Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
                )
                std_curve["__fit_g__"] = std_fit_bundle.gradient_corr
                std_curve["__plot_x__"] = std_plot_bundle.axis_corr
                std_curve["__td_model_ms__"] = float(td_ms)
                std_rows.append(std_curve)

    prepared = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    prepared_std = pd.concat(std_rows, ignore_index=True) if std_rows else pd.DataFrame()
    return prepared, prepared_std


def _resolve_global_sigma(data: pd.DataFrame, std_data: pd.DataFrame, *, ycol: str) -> np.ndarray | None:
    if std_data.empty:
        return None
    left = data[["__fit_g__", "__td_model_ms__"]].copy()
    left["__row_order__"] = np.arange(len(left))
    right = std_data[["__fit_g__", "__td_model_ms__", "value"]].copy()
    merged = left.merge(right, on=["__fit_g__", "__td_model_ms__"], how="left", sort=False)
    merged = merged.sort_values("__row_order__", kind="stable")
    sigma = pd.to_numeric(merged["value"], errors="coerce").to_numpy(dtype=float)
    if ycol == "value_norm":
        s0 = pd.to_numeric(data.get("S0", np.nan), errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = sigma / s0
    if sigma.shape[0] != data.shape[0] or not np.isfinite(sigma).any():
        return None
    finite = np.isfinite(sigma)
    sigma[~finite] = np.nanmax(sigma[finite])
    sigma[sigma <= 0] = np.nanmax(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
    return sigma


def fit_mixed_global_group(
    data: pd.DataFrame,
    std_data: pd.DataFrame | None,
    *,
    ycol: str,
    fix_m0: float | None,
    fix_d0: float | None,
    tc_init: float,
    m0_bounds: tuple[float, float],
    d0_bounds: tuple[float, float],
    tc_bounds: tuple[float, float],
) -> tuple[dict[str, object], pd.DataFrame]:
    work = data.copy()
    for col in ("__fit_g__", ycol, "__TE_ms__", "__x_model_ms__", "__N__", "__alpha__"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["__fit_g__", ycol, "__TE_ms__", "__x_model_ms__", "__N__", "__alpha__"])
    if work.empty:
        raise ValueError("No valid points remain after numeric filtering.")

    G = work["__fit_g__"].to_numpy(dtype=float)
    y = work[ycol].to_numpy(dtype=float)
    TE = work["__TE_ms__"].to_numpy(dtype=float)
    x_model = work["__x_model_ms__"].to_numpy(dtype=float)
    n_values = work["__N__"].to_numpy(dtype=float)
    alphas = work["__alpha__"].to_numpy(dtype=float)
    sigma = _resolve_global_sigma(work, std_data if std_data is not None else pd.DataFrame(), ycol=ycol)

    fixed_params: dict[str, float] = {}
    param_names = ["tc_ms"]
    p0 = [_clamp_p0(float(tc_init), tc_bounds)]
    lower = [float(tc_bounds[0])]
    upper = [float(tc_bounds[1])]
    log_params = ["tc_ms"]

    if fix_m0 is None:
        param_names.append("M0")
        p0.append(_clamp_p0(float(np.nanmax(y)), m0_bounds))
        lower.append(float(m0_bounds[0]))
        upper.append(float(m0_bounds[1]))
    else:
        fixed_params["M0"] = float(fix_m0)

    if fix_d0 is None:
        param_names.append("D0_m2_ms")
        p0.append(_clamp_p0(2.3e-12, d0_bounds))
        lower.append(float(d0_bounds[0]))
        upper.append(float(d0_bounds[1]))
        log_params.append("D0_m2_ms")
    else:
        fixed_params["D0_m2_ms"] = float(fix_d0)

    def model(g: np.ndarray, *values: float) -> np.ndarray:
        params = dict(fixed_params)
        params.update(zip(param_names, values))
        return M_nogse_mixed(
            TE,
            np.asarray(g, dtype=float),
            n_values,
            x_model,
            float(params["tc_ms"]),
            alphas,
            float(params["M0"]),
            float(params["D0_m2_ms"]),
        )

    fit = fit_least_squares(
        model,
        G,
        y,
        param_names=param_names,
        p0=p0,
        bounds=(lower, upper),
        log_params=log_params,
        sigma=sigma,
        max_nfev=300000,
        method="least_squares_global_mixed_log",
    )
    fitted_params = dict(fixed_params)
    fitted_params.update(fit.values)
    yhat = fit.yhat
    out_data = work.copy()
    out_data["__yhat__"] = yhat

    row: dict[str, object] = {
        "model": "mixed_global",
        "ycol": ycol,
        "n_points": int(len(y)),
        "n_fit": int(len(y)),
        "n_td": int(work["__td_model_ms__"].nunique()),
        "td_min_ms": float(np.nanmin(work["__td_model_ms__"])),
        "td_max_ms": float(np.nanmax(work["__td_model_ms__"])),
        "x_min": float(np.nanmin(G)),
        "x_max": float(np.nanmax(G)),
        "G_min": float(np.nanmin(G)),
        "G_max": float(np.nanmax(G)),
        "alpha": float(np.nanmean(alphas)),
        "alpha_min": float(np.nanmin(alphas)),
        "alpha_max": float(np.nanmax(alphas)),
        "alpha_source": scalar_or_compact_series(work["__alpha_source__"], name="alpha_source", required=False),
        "tc_ms": float(fitted_params["tc_ms"]),
        "M0": float(fitted_params["M0"]),
        "D0_m2_ms": float(fitted_params["D0_m2_ms"]),
        "rmse": float(fit.rmse),
        "r2": float(fit.r2),
        "chi2": float(fit.chi2),
        "method": fit.method,
        "M0_bound_min": float(m0_bounds[0]),
        "M0_bound_max": float(m0_bounds[1]),
        "D0_bound_min": float(d0_bounds[0]),
        "D0_bound_max": float(d0_bounds[1]),
        "tc_bound_min": float(tc_bounds[0]),
        "tc_bound_max": float(tc_bounds[1]),
    }
    if fix_m0 is not None:
        row["M0_err"] = np.nan
    if fix_d0 is not None:
        row["D0_err_m2_ms"] = np.nan
    row.update(fit.errors)
    return row, out_data


def plot_mixed_global_group(
    fit_data: pd.DataFrame,
    fit_row: dict[str, object],
    *,
    out_png: Path,
    ycol: str,
) -> None:
    if fit_data.empty:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    td_values = sorted(pd.to_numeric(fit_data["__td_model_ms__"], errors="coerce").dropna().unique().tolist())
    for idx, td_ms in enumerate(td_values):
        sub = fit_data[np.isclose(fit_data["__td_model_ms__"].astype(float), float(td_ms), atol=1e-9)].copy()
        sub = sub.sort_values("__plot_x__", kind="stable")
        color = cmap(idx / max(1, len(td_values) - 1))
        ax.scatter(sub["__plot_x__"], sub[ycol], s=28, color=color, label=f"td={td_ms:g} ms")
        ax.plot(sub["__plot_x__"], sub["__yhat__"], linewidth=1.8, color=color)
    ax.set_xlabel(str(fit_row.get("plot_xcol", "g")))
    ax.set_ylabel(str(ycol))
    ax.set_title(
        " | ".join(
            [
                f"subj={fit_row.get('subj', '')}",
                f"ROI={fit_row.get('roi', '')}",
                f"dir={fit_row.get('direction', '')}",
                f"tc={float(fit_row.get('tc_ms', np.nan)):.4g} ms",
            ]
        )
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fit_nogse_signal_mixed_global_long(
    df: pd.DataFrame,
    *,
    xcol: str = "g",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    fix_m0: float | None = None,
    fix_d0: float | None = None,
    tc_init: float = 5.0,
    m0_bounds: tuple[float, float] = (0.0, np.inf),
    d0_bounds: tuple[float, float] = (1e-16, np.inf),
    tc_bounds: tuple[float, float] = (0.1, 1000.0),
    f_by_direction: Mapping[str, float] | None = None,
    alpha_df: pd.DataFrame | None = None,
    alpha_col: str | None = None,
    alpha_td_col: str = "td_ms",
    alpha_td_tol_ms: float = 1e-3,
    source_file: str | None = None,
    analysis_id: str | None = None,
    outdir_plots: Path | None = None,
) -> FitOutputs:
    avg_df = df[df["stat"].astype(str) == str(stat_keep)].copy() if "stat" in df.columns else df.copy()
    if avg_df.empty:
        raise ValueError(f"No rows found for stat={stat_keep!r}.")
    std_df = df[df["stat"].astype(str) == "std"].copy() if "stat" in df.columns else pd.DataFrame()

    rois_keep = split_all_or_values(rois)
    directions_keep = split_all_or_values(directions)
    if rois_keep is not None:
        avg_df = avg_df[avg_df["roi"].astype(str).isin(rois_keep)].copy()
        std_df = std_df[std_df["roi"].astype(str).isin(rois_keep)].copy()
    if directions_keep is not None:
        avg_df = avg_df[avg_df["direction"].astype(str).isin(directions_keep)].copy()
        std_df = std_df[std_df["direction"].astype(str).isin(directions_keep)].copy()
    if avg_df.empty:
        raise ValueError("No data remains after ROI/direction filtering.")

    fit_axis = normalize_axis_base(xcol)
    resolved_plot_axis = _resolve_plot_axis(fit_axis=fit_axis, plot_axis=plot_xcol)
    alpha_lookup = _normalize_alpha_table(alpha_df, alpha_col=alpha_col, alpha_td_col=alpha_td_col)

    prepared, prepared_std = _prepare_global_mixed_rows(
        avg_df,
        std_df,
        xcol=fit_axis,
        plot_xcol=resolved_plot_axis,
        ycol=ycol,
        f_by_direction=f_by_direction,
        alpha_df=alpha_lookup,
        alpha_td_tol_ms=float(alpha_td_tol_ms),
    )

    fit_rows: list[dict[str, object]] = []
    global_group_cols = [col for col in ("subj", "roi", "direction") if col in prepared.columns]
    for key, group in prepared.groupby(global_group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(global_group_cols, key))
        std_group = pd.DataFrame()
        if not prepared_std.empty:
            std_group = prepared_std.copy()
            for col, value in key_dict.items():
                std_group = std_group[std_group[col].astype(str) == str(value)].copy()

        fit_row, fit_data = fit_mixed_global_group(
            group,
            std_group,
            ycol=ycol,
            fix_m0=fix_m0,
            fix_d0=fix_d0,
            tc_init=tc_init,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
            tc_bounds=tc_bounds,
        )
        fit_row.update({col: str(value) for col, value in key_dict.items()})
        fit_row["fit_kind"] = "nogse_signal"
        fit_row["xcol"] = str(fit_axis)
        fit_row["plot_xcol"] = str(resolved_plot_axis)
        fit_row["stat"] = str(stat_keep)
        fit_row["source_file"] = str(source_file or scalar_or_compact_series(group["source_file"], name="source_file", required=False))
        fit_row["analysis_id"] = str(analysis_id or scalar_or_compact_series(group["analysis_id"], name="analysis_id", required=False))
        fit_row["f_corr"] = scalar_or_compact_series(group["__f_corr__"], name="f_corr", required=False)

        for col in ["sheet", "protocol", "group", "type", "sequence"]:
            if col in group.columns:
                fit_row[col] = scalar_or_compact_column(group, col, required=False)
        for col in ["N", "delta_ms", "Delta_app_ms", "Hz", "TE", "TR", "x", "y"]:
            if col in group.columns:
                fit_row[col] = scalar_or_compact_column(group, col, required=False)

        fit_rows.append(fit_row)
        if outdir_plots is not None:
            out_png = outdir_plots / (
                f"{fit_row['analysis_id']}.roi-{fit_row['roi']}.dir-{fit_row['direction']}.mixed_global.png"
            )
            plot_mixed_global_group(fit_data, fit_row, out_png=out_png, ycol=ycol)
            print("Saved:", out_png)

    fit_df = pd.DataFrame(fit_rows)
    if not fit_df.empty:
        fit_df = standardize_fit_params(fit_df, fit_kind="nogse_signal", source_file=source_file)
    return FitOutputs(fit_params=fit_df)


def fit_nogse_signal_long(
    df: pd.DataFrame,
    *,
    model: str,
    xcol: str = "g",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    fix_m0: float | None = None,
    fix_d0: float | None = None,
    m0_bounds: tuple[float, float] = (0.0, np.inf),
    d0_bounds: tuple[float, float] = (1e-16, np.inf),
    f_by_direction: Mapping[str, float] | None = None,
    source_file: str | None = None,
    analysis_id: str | None = None,
    outdir_plots: Path | None = None,
) -> FitOutputs:
    if model not in SIGNAL_MODELS:
        raise ValueError(f"Unsupported NOGSE signal model {model!r}. Allowed values: {sorted(SIGNAL_MODELS)}.")

    avg_df = df[df["stat"].astype(str) == str(stat_keep)].copy()
    if avg_df.empty:
        raise ValueError(f"No rows found for stat={stat_keep!r}.")

    std_df = df[df["stat"].astype(str) == "std"].copy()

    rois_keep = split_all_or_values(rois)
    directions_keep = split_all_or_values(directions)
    if rois_keep is not None:
        avg_df = avg_df[avg_df["roi"].astype(str).isin(rois_keep)].copy()
        std_df = std_df[std_df["roi"].astype(str).isin(rois_keep)].copy()
    if directions_keep is not None:
        avg_df = avg_df[avg_df["direction"].astype(str).isin(directions_keep)].copy()
        std_df = std_df[std_df["direction"].astype(str).isin(directions_keep)].copy()

    if avg_df.empty:
        raise ValueError("No data remains after ROI/direction filtering.")

    fit_axis = normalize_axis_base(xcol)
    plot_axis = _resolve_plot_axis(fit_axis=fit_axis, plot_axis=plot_xcol)

    fit_rows: list[dict[str, object]] = []
    for (roi, direction), group in avg_df.groupby(["roi", "direction"], sort=False):
        group_fit = group.copy()
        std_group = None
        if not std_df.empty:
            std_group = std_df[
                (std_df["roi"].astype(str) == str(roi))
                & (std_df["direction"].astype(str) == str(direction))
            ].copy()

        f_corr = float(f_by_direction.get(str(direction), 1.0)) if f_by_direction else 1.0
        n_value = float(_unique_scalar(group_fit, "N", required=True))
        delta_ms = _unique_scalar(group_fit, "delta_ms", required=False)
        Delta_app_ms = _unique_scalar(group_fit, "Delta_app_ms", required=False)

        fit_bundle = build_axis_bundle(
            group_fit,
            axis=fit_axis,
            correction_factor=float(f_corr),
            gamma=GAMMA_DEFAULT,
            N=n_value,
            delta_ms=None if delta_ms is None else float(delta_ms),
            Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
        )
        plot_bundle = build_axis_bundle(
            group_fit,
            axis=plot_axis,
            correction_factor=float(f_corr),
            gamma=GAMMA_DEFAULT,
            N=n_value,
            delta_ms=None if delta_ms is None else float(delta_ms),
            Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
        )
        group_fit["__fit_x__"] = fit_bundle.gradient_corr
        group_fit[plot_axis] = plot_bundle.axis_corr

        if std_group is not None:
            std_fit_bundle = build_axis_bundle(
                std_group,
                axis=fit_axis,
                correction_factor=float(f_corr),
                gamma=GAMMA_DEFAULT,
                N=n_value,
                delta_ms=None if delta_ms is None else float(delta_ms),
                Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
            )
            std_plot_bundle = build_axis_bundle(
                std_group,
                axis=plot_axis,
                correction_factor=float(f_corr),
                gamma=GAMMA_DEFAULT,
                N=n_value,
                delta_ms=None if delta_ms is None else float(delta_ms),
                Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
            )
            std_group["__fit_x__"] = std_fit_bundle.gradient_corr
            std_group[plot_axis] = std_plot_bundle.axis_corr

        fit_row, x_data, y_data, fit_curve = fit_one_group(
            group_fit,
            std_group,
            model_name=model,
            xcol="__fit_x__",
            ycol=ycol,
            fix_m0=fix_m0,
            fix_d0=fix_d0,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )
        fit_row["f_corr"] = float(f_corr)
        fit_row["xcol"] = str(fit_axis)
        fit_row["plot_xcol"] = str(plot_axis)

        for col in [
            "subj",
            "sheet",
            "protocol",
            "group",
            "type",
            "TN",
            "x",
            "y",
            "Hz",
            "N",
            "TE",
            "TR",
            "delta_ms",
            "Delta_app_ms",
            "td_ms",
            "tm_ms",
            "max_dur_ms",
        ]:
            if col in group.columns:
                fit_row[col] = _unique_scalar(group, col, required=False)

        if "sequence" in group.columns:
            fit_row["sequence"] = scalar_or_compact_column(group, "sequence", required=False)

        if "G" in group.columns:
            g_values = pd.to_numeric(group["G"], errors="coerce").dropna().to_numpy(dtype=float)
            if g_values.size > 0:
                fit_row["G_min"] = float(np.nanmin(g_values))
                fit_row["G_max"] = float(np.nanmax(g_values))

        fit_row["roi"] = str(roi)
        fit_row["direction"] = str(direction)
        fit_row["stat"] = str(stat_keep)
        fit_row["source_file"] = str(source_file or "")
        fit_row["analysis_id"] = str(analysis_id or "")
        fit_rows.append(fit_row)

        if outdir_plots is not None:
            fit_curve_plot = fit_curve.copy()
            fit_curve_plot[:, 0] = axis_from_gradient(
                fit_curve[:, 0],
                axis=plot_axis,
                N=n_value,
                gamma=GAMMA_DEFAULT,
                delta_ms=None if delta_ms is None else float(delta_ms),
                Delta_app_ms=None if Delta_app_ms is None else float(Delta_app_ms),
            )
            out_png = outdir_plots / f"{analysis_id}.roi-{roi}.dir-{direction}.{model}.png"
            plot_fit_one_group(
                group_fit,
                std_group,
                xcol=plot_axis,
                ycol=ycol,
                fit_row=fit_row,
                x_data=plot_bundle.axis_corr,
                y_data=y_data,
                fit_curve=fit_curve_plot,
                out_png=out_png,
            )
            print("Saved:", out_png)

    fit_df = pd.DataFrame(fit_rows)
    if not fit_df.empty:
        fit_df = standardize_fit_params(fit_df, fit_kind="nogse_signal", source_file=source_file)
    return FitOutputs(fit_params=fit_df)


def run_fit_from_parquet(
    parquet_path: str | Path,
    *,
    model: str,
    out_root: str | Path,
    xcol: str = "g",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    fix_m0: float | None = None,
    fix_d0: float | None = None,
    m0_bounds: tuple[float, float] = (0.0, np.inf),
    d0_bounds: tuple[float, float] = (1e-16, np.inf),
    f_by_direction: Mapping[str, float] | None = None,
    append_model_subdir: bool = True,
) -> tuple[FitOutputs, Path]:
    p = Path(parquet_path)
    df = pd.read_parquet(p)
    analysis_id = analysis_id_from_path(p)
    if append_model_subdir:
        out_dir = Path(out_root) / model / analysis_id
    else:
        out_dir = Path(out_root) / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    outs = fit_nogse_signal_long(
        df,
        model=model,
        xcol=xcol,
        plot_xcol=plot_xcol,
        ycol=ycol,
        stat_keep=stat_keep,
        rois=rois,
        directions=directions,
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
        f_by_direction=f_by_direction,
        source_file=p.name,
        analysis_id=analysis_id,
        outdir_plots=out_dir,
    )

    fit_params_name = fit_params_output_basename(
        model=str(model),
        axis=str(xcol),
        ycol=str(ycol),
        directions=None if directions is None else [str(v) for v in directions],
    )
    out_parquet = out_dir / f"{fit_params_name}.parquet"
    write_table_outputs(
        outs.fit_params,
        out_parquet,
        xlsx_path=out_parquet.with_suffix(".xlsx"),
        csv_path=out_dir / f"{fit_params_name}.csv",
    )
    print("Saved fit table:", out_parquet)
    return outs, out_dir


def run_global_mixed_fit_from_parquets(
    parquet_paths: Sequence[str | Path],
    *,
    out_root: str | Path,
    xcol: str = "g",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    fix_m0: float | None = None,
    fix_d0: float | None = None,
    tc_init: float = 5.0,
    m0_bounds: tuple[float, float] = (0.0, np.inf),
    d0_bounds: tuple[float, float] = (1e-16, np.inf),
    tc_bounds: tuple[float, float] = (0.1, 1000.0),
    f_by_direction: Mapping[str, float] | None = None,
    alpha_table: str | Path | None = None,
    alpha_col: str | None = None,
    alpha_td_col: str = "td_ms",
    alpha_td_tol_ms: float = 1e-3,
    no_plots: bool = False,
) -> tuple[FitOutputs, Path]:
    paths = [Path(p) for p in parquet_paths]
    if not paths:
        raise ValueError("At least one signal parquet is required.")

    frames: list[pd.DataFrame] = []
    analysis_ids: list[str] = []
    source_files: list[str] = []
    for p in paths:
        df = pd.read_parquet(p)
        analysis_id = analysis_id_from_path(p)
        analysis_ids.append(analysis_id)
        source_files.append(p.name)
        df = _ensure_subject_column(df, analysis_id=analysis_id, source_file=p.name)
        df["analysis_id"] = analysis_id
        df["source_file"] = p.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    alpha_df = _read_table(alpha_table) if alpha_table is not None else None
    analysis_id = "mixed_global"
    out_dir = Path(out_root) / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    outs = fit_nogse_signal_mixed_global_long(
        combined,
        xcol=xcol,
        plot_xcol=plot_xcol,
        ycol=ycol,
        stat_keep=stat_keep,
        rois=rois,
        directions=directions,
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        tc_init=tc_init,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
        tc_bounds=tc_bounds,
        f_by_direction=f_by_direction,
        alpha_df=alpha_df,
        alpha_col=alpha_col,
        alpha_td_col=alpha_td_col,
        alpha_td_tol_ms=alpha_td_tol_ms,
        source_file="|".join(source_files),
        analysis_id="|".join(analysis_ids),
        outdir_plots=None if no_plots else out_dir,
    )

    fit_params_name = fit_params_output_basename(
        model="mixed_global",
        axis=str(xcol),
        ycol=str(ycol),
        directions=None if directions is None else [str(v) for v in directions],
    )
    out_parquet = out_dir / f"{fit_params_name}.parquet"
    write_table_outputs(
        outs.fit_params,
        out_parquet,
        xlsx_path=out_parquet.with_suffix(".xlsx"),
        csv_path=out_dir / f"{fit_params_name}.csv",
    )
    print("Saved fit table:", out_parquet)
    return outs, out_dir
