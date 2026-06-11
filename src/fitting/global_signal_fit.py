from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from fitting.core import chi2 as _chi2
from fitting.core import rmse as _rmse
from fitting.global_signal_inputs import CurveData
from fitting.model_registry import SignalModelSpec, evaluate_signal_model
from fitting.parameter_modes import (
    FitParameterConfig,
    finite_or_none,
    fixed_parameter_value,
    from_optimization_values,
    parameter_mode_summary,
    to_optimization_values,
)


ALL_PARAM_ORDER = ("tc_ms", "alpha", "RN", "M0", "C", "D0_m2_ms")


def model_curve(curve: CurveData, *, model_spec: SignalModelSpec, params: dict[str, float]) -> np.ndarray:
    return evaluate_signal_model(
        model_spec,
        td_ms=float(curve.td_ms),
        G=np.asarray(curve.g, dtype=float),
        N=float(curve.n_value),
        params=params,
        x_ms=float(curve.x_model_ms),
    )


def param_order(model_spec: SignalModelSpec) -> tuple[str, ...]:
    return tuple(name for name in ALL_PARAM_ORDER if name in set(model_spec.param_names))


def curves_by_pair(curves: Sequence[CurveData]) -> dict[str, list[CurveData]]:
    pairs: dict[str, list[CurveData]] = {}
    for curve in curves:
        pairs.setdefault(curve.pair_key, []).append(curve)
    return pairs


def curves_y_seed(curves: Sequence[CurveData], bounds: tuple[float, float]) -> float:
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


def initial_value_for_scope(param_name: str, config: FitParameterConfig, curves: Sequence[CurveData]) -> float:
    init = finite_or_none(config.init)
    if init is not None:
        return float(np.clip(init, config.bounds[0], config.bounds[1]))
    if param_name == "M0":
        return curves_y_seed(curves, config.bounds)
    raise ValueError(f"No finite initial value provided for free parameter {param_name}.")


def scope_name(config: FitParameterConfig, curve: CurveData, pair_param_ids: dict[str, int]) -> str | None:
    """Map a physical parameter mode to the optimizer variable used by one curve."""
    if config.mode == "fixed":
        return None
    if config.mode == "global_td":
        return config.name
    if config.mode == "global_contrast":
        return f"{config.name}__pair_{pair_param_ids[curve.pair_key]}"
    if config.mode == "free":
        return f"{config.name}__curve_{curve.curve_id}"
    raise ValueError(f"Unsupported mode {config.mode!r} for {config.name}.")


def scope_members(
    curves: Sequence[CurveData],
    *,
    config: FitParameterConfig,
    pair_param_ids: dict[str, int],
) -> list[tuple[str, list[CurveData]]]:
    """Return the curves controlled by each optimizer variable."""
    if config.mode == "fixed":
        return []
    if config.mode == "global_td":
        return [(config.name, list(curves))]
    if config.mode == "global_contrast":
        pairs = curves_by_pair(curves)
        return [(f"{config.name}__pair_{pair_param_ids[key]}", members) for key, members in pairs.items()]
    if config.mode == "free":
        return [(f"{config.name}__curve_{curve.curve_id}", [curve]) for curve in curves]
    raise ValueError(f"Unsupported mode {config.mode!r} for {config.name}.")


def pack_fit_problem(
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

    for param_name in param_order(model_spec):
        config = param_configs[param_name]
        if config.mode == "fixed":
            fixed_parameter_value(config)
            continue
        for scoped_name, members in scope_members(curves, config=config, pair_param_ids=pair_param_ids):
            names.append(scoped_name)
            p0.append(initial_value_for_scope(param_name, config, members))
            lower.append(float(config.bounds[0]))
            upper.append(float(config.bounds[1]))
            if config.log:
                log_params.add(scoped_name)

    return names, np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)), log_params


def param_value(
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
    scoped_name = scope_name(config, curve, pair_param_ids)
    if scoped_name is None:
        return fixed_parameter_value(config)
    return float(values.get(scoped_name, np.nan))


def param_error(
    *,
    errors: dict[str, float],
    param_configs: dict[str, FitParameterConfig],
    param_name: str,
    curve: CurveData,
    pair_param_ids: dict[str, int],
) -> float:
    if param_name not in param_configs:
        return np.nan
    scoped_name = scope_name(param_configs[param_name], curve, pair_param_ids)
    if scoped_name is None:
        return np.nan
    return float(errors.get(scoped_name, np.nan))


def mode_summary(param_configs: dict[str, FitParameterConfig], mode: str, *, model_spec: SignalModelSpec) -> str:
    return parameter_mode_summary(param_configs, mode, ordered_names=param_order(model_spec))


def free_param_count(
    param_configs: dict[str, FitParameterConfig],
    curves: Sequence[CurveData],
    *,
    model_spec: SignalModelSpec,
) -> int:
    pair_count = len(curves_by_pair(curves))
    count = 0
    for name in param_order(model_spec):
        config = param_configs[name]
        if config.mode == "global_td":
            count += 1
        elif config.mode == "global_contrast":
            count += pair_count
        elif config.mode == "free":
            count += len(curves)
    return int(count)


def fit_global_curve_group(
    curves: Sequence[CurveData],
    *,
    model_spec: SignalModelSpec,
    param_configs: dict[str, FitParameterConfig],
    max_nfev: int,
) -> tuple[dict[str, float], dict[str, float], bool, str, str]:
    """Concatenate all curve residuals in one group and fit them with least squares."""
    pair_param_ids = {key: members[0].curve_id for key, members in curves_by_pair(curves).items()}
    names, p0, bounds, log_params = pack_fit_problem(
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
                name: param_value(
                    values=params,
                    param_configs=param_configs,
                    param_name=name,
                    curve=curve,
                    pair_param_ids=pair_param_ids,
                )
                for name in model_spec.param_names
            }
            yhat = model_curve(curve, model_spec=model_spec, params=curve_params)
            if yhat.shape != curve.y.shape or not np.all(np.isfinite(yhat)):
                return np.full(sum(len(c.y) for c in curves), 1e12, dtype=float)
            residuals.append(yhat - curve.y)
        return np.concatenate(residuals)

    mode_desc = "; ".join(
        f"{mode}={mode_summary(param_configs, mode, model_spec=model_spec) or '-'}"
        for mode in ("fixed", "global_td", "global_contrast", "free")
    )
    method = f"least_squares({model_spec.name}: {mode_desc})"
    if not names:
        try:
            for curve in curves:
                curve_params = {
                    name: param_value(
                        values={},
                        param_configs=param_configs,
                        param_name=name,
                        curve=curve,
                        pair_param_ids=pair_param_ids,
                    )
                    for name in model_spec.param_names
                }
                yhat = model_curve(curve, model_spec=model_spec, params=curve_params)
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


def build_global_signal_rows(
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
    pair_param_ids = {key: members[0].curve_id for key, members in curves_by_pair(curves).items()}

    def mode_for(param_name: str) -> str:
        config = param_configs.get(param_name)
        return config.mode if config is not None else ""

    for curve in curves:
        tc = param_value(values=values, param_configs=param_configs, param_name="tc_ms", curve=curve, pair_param_ids=pair_param_ids)
        alpha = param_value(values=values, param_configs=param_configs, param_name="alpha", curve=curve, pair_param_ids=pair_param_ids)
        RN = param_value(values=values, param_configs=param_configs, param_name="RN", curve=curve, pair_param_ids=pair_param_ids)
        M0 = param_value(values=values, param_configs=param_configs, param_name="M0", curve=curve, pair_param_ids=pair_param_ids)
        C = param_value(values=values, param_configs=param_configs, param_name="C", curve=curve, pair_param_ids=pair_param_ids)
        D0 = param_value(values=values, param_configs=param_configs, param_name="D0_m2_ms", curve=curve, pair_param_ids=pair_param_ids)
        curve_params = {
            name: param_value(
                values=values,
                param_configs=param_configs,
                param_name=name,
                curve=curve,
                pair_param_ids=pair_param_ids,
            )
            for name in model_spec.param_names
        }
        yhat = model_curve(curve, model_spec=model_spec, params=curve_params) if ok else np.full_like(curve.y, np.nan, dtype=float)
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
                "fixed_params": mode_summary(param_configs, "fixed", model_spec=model_spec),
                "global_params": mode_summary(param_configs, "global_td", model_spec=model_spec),
                "global_contrast_params": mode_summary(param_configs, "global_contrast", model_spec=model_spec),
                "local_params": mode_summary(param_configs, "free", model_spec=model_spec),
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
                "M0_err": param_error(errors=errors, param_configs=param_configs, param_name="M0", curve=curve, pair_param_ids=pair_param_ids),
                "tc_ms": tc,
                "tc_err_ms": param_error(errors=errors, param_configs=param_configs, param_name="tc_ms", curve=curve, pair_param_ids=pair_param_ids),
                "alpha": alpha,
                "alpha_err": param_error(errors=errors, param_configs=param_configs, param_name="alpha", curve=curve, pair_param_ids=pair_param_ids),
                "RN": RN,
                "RN_err": param_error(errors=errors, param_configs=param_configs, param_name="RN", curve=curve, pair_param_ids=pair_param_ids),
                "RN_fixed": mode_for("RN") == "fixed",
                "D0_m2_ms": D0,
                "D0_err_m2_ms": param_error(errors=errors, param_configs=param_configs, param_name="D0_m2_ms", curve=curve, pair_param_ids=pair_param_ids),
                "D0_mm2_s": D0 * 1e9 if np.isfinite(D0) else np.nan,
                "C": C,
                "C_err": param_error(errors=errors, param_configs=param_configs, param_name="C", curve=curve, pair_param_ids=pair_param_ids),
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
