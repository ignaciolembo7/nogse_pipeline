from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from data_processing.io import read_table_file
from fitting.b_from_g import (
    axes_share_gradient_family,
    axis_from_gradient,
    build_axis_bundle,
    default_plot_axis_for_fit,
    gradient_base_for_axis,
    normalize_axis_base,
    split_axis_side,
)
from fitting.contrast_tables import (
    analysis_id_from_source_file as _analysis_id_from_source_file,
    coerce_correction_pair as _coerce_correction_pair,
    fit_row_correction_pair as _fit_row_correction_pair,
    maybe_scale_gradient as _maybe_scale_g_thorsten,
    normalize_contrast_keys as _normalize_keys,
    require_columns as _require_cols,
    unique_scalar as _unique_scalar,
)
from fitting.core import CurveFitParameter
from fitting.core import chi2 as _chi2
from fitting.core import fit_curve_fit_parameters
from fitting.core import fit_least_squares
from fitting.core import parameter_error_column
from fitting.core import rmse as _rmse
from fitting.core import stderr_from_single_param_jacobian as _stderr_from_single_param_jacobian
from ogse_plotting.plot_ogse_contrast_vs_g import FAMILY_LABEL, plot_contrast_fit
from models.model_fitting import (
    M_ogse_free,
    M_ogse_mixed,
    M_ogse_mixed_offset,
    M_ogse_rest,
    M_ogse_rest_offset,
    OGSE_contrast_vs_g_free,
    OGSE_contrast_vs_g_mixed,
    OGSE_contrast_vs_g_rest,
    OGSE_contrast_vs_g_rest_offset,
    OGSE_contrast_vs_g_tort,
)
from ogse_fitting.contrast_parameter_plan import (
    MODEL_PARAM_NAMES,
    contrast_parameter_vary_flags as _param_vary_flags,
    fixed_contrast_parameter_values as _param_fixed_values,
    normalize_global_contrast_params as _normalize_global_params,
    seed_bounds_for_contrast_parameter as _param_seed_bounds,
)
from ogse_fitting.contrast_model_registry import ContrastFitSpec, OGSE_CONTRAST_FIT_SPECS as CONTRAST_MODEL_SPECS
from ogse_fitting.tc_peak import _tc_peak_from_notebook_formula
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.fit_params_schema import standardize_fit_params
from tools.value_formatting import scalar_or_compact_series, truthy_series


# -----------------------------
# Helpers: strict schema
# -----------------------------
KEY_COLS = ("roi", "direction", "b_step")
VALID_GBASES = {"g", "g_max", "g_lin_max", "g_thorsten"}
GAMMA_DEFAULT = 267.5221900


def _normalize_gbase(gbase: str) -> str:
    return gradient_base_for_axis(gbase)


def _resolve_plot_axis(*, fit_axis: str, plot_axis: str | None, xplot: str) -> str:
    resolved_fit = normalize_axis_base(fit_axis)
    if plot_axis is None:
        side = 2 if str(xplot) == "2" else 1
        return default_plot_axis_for_fit(resolved_fit, side=side)

    plot_base, plot_side = split_axis_side(plot_axis)
    resolved_side = 1 if plot_side is None else int(plot_side)
    if resolved_side not in (1, 2):
        raise ValueError("plot_xcol must use side 1 or 2 for ogse_contrast_vs_g.")
    if not axes_share_gradient_family(resolved_fit, plot_base):
        raise ValueError(
            "plot_xcol must stay in the same gradient family as gbase. "
            f"Received gbase={fit_axis!r}, plot_xcol={plot_axis!r}."
        )
    return f"{plot_base}_{resolved_side}"


ContrastModelSpec = ContrastFitSpec  # backward-compat alias


def _fit_contrast_model(
    spec: ContrastModelSpec,
    td_ms: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    parameters: Sequence[CurveFitParameter],
):
    def model_from_params(params: dict[str, float]) -> np.ndarray:
        return spec.evaluator(td_ms, G1, G2, n_1, n_2, params)

    return fit_curve_fit_parameters(
        model_from_params,
        y,
        parameters=parameters,
        maxfev=spec.maxfev,
    )


def _model_yhat(
    *,
    model: str,
    td_ms: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    fit_row: dict[str, Any],
) -> np.ndarray:
    spec = CONTRAST_MODEL_SPECS.get(str(model))
    if spec is None:
        raise ValueError(f"Unsupported model {model!r} for curve evaluation.")
    return spec.evaluator(td_ms, G1, G2, n_1, n_2, fit_row)


def _model_side_yhat(
    *,
    model: str,
    td_ms: float,
    G: np.ndarray,
    n_1: int,
    n_2: int,
    fit_row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    model_name = str(model)
    M0 = float(fit_row["M0"])
    D0 = float(fit_row["D0_m2_ms"])
    x1 = float(td_ms) / float(n_1)
    x2 = float(td_ms) / float(n_2)

    if model_name in {"free", "ogse_free"}:
        return (
            M_ogse_free(td_ms, G, n_1, x1, M0, D0),
            M_ogse_free(td_ms, G, n_2, x2, M0, D0),
        )
    if model_name == "tort":
        alpha = float(fit_row["alpha"])
        D0_eff = alpha * D0
        return (
            M_ogse_free(td_ms, G, n_1, x1, M0, D0_eff),
            M_ogse_free(td_ms, G, n_2, x2, M0, D0_eff),
        )
    if model_name in {"rest", "ogse_rest"}:
        tc_ms = float(fit_row["tc_ms"])
        return (
            M_ogse_rest(td_ms, G, n_1, x1, tc_ms, M0, D0),
            M_ogse_rest(td_ms, G, n_2, x2, tc_ms, M0, D0),
        )
    if model_name in {"rest_offset", "rest_offset_globC", "ogse_rest_offset"}:
        tc_ms = float(fit_row["tc_ms"])
        C = float(fit_row.get("C", 0.0))
        return (
            M_ogse_rest_offset(td_ms, G, n_1, x1, tc_ms, M0, D0, C),
            M_ogse_rest_offset(td_ms, G, n_2, x2, tc_ms, M0, D0, C),
        )
    if model_name in {"mixed", "mixed_global", "ogse_mixed"}:
        tc_ms = float(fit_row["tc_ms"])
        alpha = float(fit_row["alpha"])
        return (
            M_ogse_mixed(td_ms, G, n_1, x1, tc_ms, alpha, M0, D0),
            M_ogse_mixed(td_ms, G, n_2, x2, tc_ms, alpha, M0, D0),
        )
    if model_name == "ogse_mixed_offset":
        tc_ms = float(fit_row["tc_ms"])
        alpha = float(fit_row["alpha"])
        C = float(fit_row.get("C", 0.0))
        RN = float(fit_row.get("RN", 0.0))
        return (
            M_ogse_mixed_offset(td_ms, G, n_1, x1, tc_ms, alpha, M0, D0, C, RN),
            M_ogse_mixed_offset(td_ms, G, n_2, x2, tc_ms, alpha, M0, D0, C, RN),
        )

    raise ValueError(f"Unsupported model {model!r} for side-curve evaluation.")


def _compute_peak_metrics(
    *,
    model: str,
    td_ms: float,
    n_1: int,
    n_2: int,
    fit_row: dict[str, Any],
    g1_max_corr: float,
    g2_max_corr: float,
    f_corr_1: float,
    f_corr_2: float,
    xplot: str,
    peak_grid_n: int,
    peak_D0_fix: float,
    peak_gamma: float,
    peak_g_max_mTm: float | None = None,
    peak_resample_gradient: bool = False,
    peak_resample_g_max_corr_mTm: float | None = None,
) -> dict[str, float | str]:
    if not np.isfinite(g1_max_corr) or not np.isfinite(g2_max_corr):
        return {}
    if g1_max_corr <= 0 or g2_max_corr <= 0:
        return {}

    f1_val = f_corr_1 if np.isfinite(f_corr_1) and f_corr_1 != 0.0 else np.nan
    f2_val = f_corr_2 if np.isfinite(f_corr_2) and f_corr_2 != 0.0 else np.nan

    search_g1_max_corr = float(g1_max_corr)
    search_g2_max_corr = float(g2_max_corr)
    if peak_g_max_mTm is not None and np.isfinite(float(peak_g_max_mTm)) and float(peak_g_max_mTm) > 0:
        x_is_1 = str(xplot) == "1"
        x_corr_factor = f1_val if x_is_1 else f2_val
        x_observed_max = float(g1_max_corr if x_is_1 else g2_max_corr)
        if np.isfinite(x_corr_factor):
            x_search_max_corr = float(peak_g_max_mTm) * float(x_corr_factor)
        else:
            x_search_max_corr = float(peak_g_max_mTm)
        if np.isfinite(x_search_max_corr) and x_search_max_corr > 0 and x_observed_max > 0:
            scale = x_search_max_corr / x_observed_max
            search_g1_max_corr = float(g1_max_corr) * scale
            search_g2_max_corr = float(g2_max_corr) * scale

    n_grid = max(32, int(peak_grid_n))
    frac = np.linspace(0.0, 1.0, n_grid)
    G1 = frac * float(search_g1_max_corr)
    G2 = frac * float(search_g2_max_corr)
    y = _model_yhat(model=model, td_ms=td_ms, G1=G1, G2=G2, n_1=n_1, n_2=n_2, fit_row=fit_row)
    if y.size == 0 or not np.isfinite(y).any():
        return {}

    i_peak = int(np.nanargmax(y))
    f_peak = float(frac[i_peak])
    g1_peak_corr = float(G1[i_peak])
    g2_peak_corr = float(G2[i_peak])
    y_peak = float(y[i_peak])

    g1_peak_raw = float(g1_peak_corr / f1_val) if np.isfinite(f1_val) else np.nan
    g2_peak_raw = float(g2_peak_corr / f2_val) if np.isfinite(f2_val) else np.nan
    g1_max_raw = float(g1_max_corr / f1_val) if np.isfinite(f1_val) else np.nan
    g2_max_raw = float(g2_max_corr / f2_val) if np.isfinite(f2_val) else np.nan
    g1_search_max_raw = float(search_g1_max_corr / f1_val) if np.isfinite(f1_val) else np.nan
    g2_search_max_raw = float(search_g2_max_corr / f2_val) if np.isfinite(f2_val) else np.nan

    x_is_1 = str(xplot) == "1"
    x_peak_corr = g1_peak_corr if x_is_1 else g2_peak_corr
    x_peak_raw = g1_peak_raw if x_is_1 else g2_peak_raw

    l_G, L_cf, lcf_peak_m, tc_peak_ms = _tc_peak_from_notebook_formula(
        td_ms=float(td_ms),
        g_peak_mTpm=float(x_peak_corr),
        D0_fix_m2_ms=float(peak_D0_fix),
        gamma_rad_ms_mT=float(peak_gamma),
    )

    out = {
        "peak_method": "param_grid",
        "peak_grid_n": int(n_grid),
        "g1_max_raw_mTm": g1_max_raw,
        "g2_max_raw_mTm": g2_max_raw,
        "g1_max_corr_mTm": float(g1_max_corr),
        "g2_max_corr_mTm": float(g2_max_corr),
        "peak_g_max_mTm": float(peak_g_max_mTm) if peak_g_max_mTm is not None else np.nan,
        "g1_search_max_raw_mTm": g1_search_max_raw,
        "g2_search_max_raw_mTm": g2_search_max_raw,
        "g1_search_max_corr_mTm": float(search_g1_max_corr),
        "g2_search_max_corr_mTm": float(search_g2_max_corr),
        "peak_fraction": f_peak,
        "g1_peak_raw_mTm": g1_peak_raw,
        "g2_peak_raw_mTm": g2_peak_raw,
        "g1_peak_corr_mTm": g1_peak_corr,
        "g2_peak_corr_mTm": g2_peak_corr,
        "x_peak_raw_mTm": float(x_peak_raw),
        "x_peak_corr_mTm": float(x_peak_corr),
        "signal_peak": y_peak,
        "l_G_peak_m": l_G,
        "L_cf_peak": L_cf,
        "lcf_peak_m": lcf_peak_m,
        "tc_peak_ms": tc_peak_ms,
    }

    if peak_resample_gradient:
        if peak_resample_g_max_corr_mTm is not None and np.isfinite(float(peak_resample_g_max_corr_mTm)) and float(peak_resample_g_max_corr_mTm) > 0:
            g_common_max_corr = float(peak_resample_g_max_corr_mTm)
        else:
            g_common_max_corr = float(max(search_g1_max_corr, search_g2_max_corr))
        G_common = frac * g_common_max_corr
        y_resampled = _model_yhat(
            model=model,
            td_ms=td_ms,
            G1=G_common,
            G2=G_common,
            n_1=n_1,
            n_2=n_2,
            fit_row=fit_row,
        )
        y_resampled = np.asarray(y_resampled, dtype=float)
        if y_resampled.size > 0 and np.isfinite(y_resampled).any():
            i_resampled = int(np.nanargmax(y_resampled))
            g_peak_resampled_corr = float(G_common[i_resampled])
            g_peak_resampled_raw = (
                float(g_peak_resampled_corr / (f1_val if x_is_1 else f2_val))
                if np.isfinite(f1_val if x_is_1 else f2_val)
                else np.nan
            )
            _, _, _, tc_peak_resampled_ms = _tc_peak_from_notebook_formula(
                td_ms=float(td_ms),
                g_peak_mTpm=float(g_peak_resampled_corr),
                D0_fix_m2_ms=float(peak_D0_fix),
                gamma_rad_ms_mT=float(peak_gamma),
            )
            out.update(
                {
                    "peak_resample_gradient": True,
                    "peak_resample_g_max_corr_mTm": g_common_max_corr,
                    "g_peak_resampled_raw_mTm": g_peak_resampled_raw,
                    "g_peak_resampled_corr_mTm": g_peak_resampled_corr,
                    "tc_peak_resampled_ms": tc_peak_resampled_ms,
                }
            )
    else:
        out["peak_resample_gradient"] = False

    return out


def _fit_free(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, str, float | None, float | None]:
    # Keep D0 bounds centered around the user-provided seed.
    M0_lo, M0_hi = (0.0, 2.0) if m0_bounds is None else (float(m0_bounds[0]), float(m0_bounds[1]))
    D_lo, D_hi = (
        (float(D0_value / 10.0), float(D0_value * 10.0))
        if d0_bounds is None
        else (float(d0_bounds[0]), float(d0_bounds[1]))
    )

    if (not M0_vary) and D0_vary:
        def f_log(log_D0: float) -> float:
            D0 = float(np.exp(log_D0))
            yhat = OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, float(M0_value), D0)
            if yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
                return np.inf
            return float(np.sum((y - yhat) ** 2))

        # `curve_fit` can fall into poor local minima for short-td phantoms;
        # a 1D search in log(D0) is more stable here.
        log_lo = float(np.log(D_lo))
        log_hi = float(np.log(D_hi))
        log_grid = np.linspace(log_lo, log_hi, 64)
        losses = np.array([f_log(v) for v in log_grid], dtype=float)
        i_best = int(np.nanargmin(losses))
        best_log = float(log_grid[i_best])
        best_loss = float(losses[i_best])

        ref_lo = log_grid[max(0, i_best - 1)]
        ref_hi = log_grid[min(len(log_grid) - 1, i_best + 1)]
        if ref_hi > ref_lo:
            opt = minimize_scalar(
                f_log,
                bounds=(float(ref_lo), float(ref_hi)),
                method="bounded",
                options={"xatol": 1e-6},
            )
            if bool(opt.success) and np.isfinite(float(opt.fun)) and float(opt.fun) <= best_loss:
                best_log = float(opt.x)
                best_loss = float(opt.fun)

        D0 = float(np.exp(best_log))
        yhat = OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, float(M0_value), D0)

        # Estimate D0 uncertainty from the local Jacobian when only D0 varies.
        d0_step = max(abs(D0) * 1e-6, abs(float(D0_value)) * 1e-6, 1e-18)
        d0_lo = max(float(D_lo), float(D0 - d0_step))
        d0_hi = min(float(D_hi), float(D0 + d0_step))
        D0_err = None
        if d0_hi > d0_lo:
            y_lo = OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, float(M0_value), float(d0_lo))
            y_hi = OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, float(M0_value), float(d0_hi))
            jac = (y_hi - y_lo) / float(d0_hi - d0_lo)
            D0_err = _stderr_from_single_param_jacobian(y, yhat, jac, n_params=1)

        return float(M0_value), D0, _rmse(y, yhat), _chi2(y, yhat), "logD_scalar_search", None, D0_err

    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["free"],
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        parameters=[
            CurveFitParameter("M0", float(M0_value), M0_lo, M0_hi, bool(M0_vary)),
            CurveFitParameter("D0_m2_ms", float(D0_value), D_lo, D_hi, bool(D0_vary)),
        ],
    )
    return (
        float(fit.values["M0"]),
        float(fit.values["D0_m2_ms"]),
        fit.rmse,
        fit.chi2,
        fit.method,
        float(fit.errors.get("M0_err", np.nan)),
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
    )

def _fit_tort(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
):
    M0_lo, M0_hi = (0.0, 5.0) if m0_bounds is None else (float(m0_bounds[0]), float(m0_bounds[1]))
    D_lo, D_hi = (
        (float(D0_value / 10.0), float(D0_value * 10.0))
        if d0_bounds is None
        else (float(d0_bounds[0]), float(d0_bounds[1]))
    )
    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["tort"],
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        parameters=[
            CurveFitParameter("alpha", 0.7, 0.0, 2.0, True),
            CurveFitParameter("M0", float(M0_value), M0_lo, M0_hi, bool(M0_vary)),
            CurveFitParameter("D0_m2_ms", float(D0_value), D_lo, D_hi, bool(D0_vary)),
        ],
    )
    return (
        float(fit.values["alpha"]),
        float(fit.values["M0"]),
        float(fit.values["D0_m2_ms"]),
        fit.rmse,
        fit.chi2,
        fit.method,
        float(fit.errors.get("alpha_err", np.nan)),
        float(fit.errors.get("M0_err", np.nan)) if M0_vary else None,
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
    )


def _best_tc_seed_rest(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_guess: float,
    D0_guess: float,
    tc_default: float,
    tc_bounds: tuple[float, float] = (0.1, 1000.0),
) -> float:
    tc_lo, tc_hi = float(tc_bounds[0]), float(tc_bounds[1])
    candidates = np.unique(
        np.concatenate(
            [
                np.array([float(np.clip(float(tc_default), tc_lo, tc_hi))], dtype=float),
                np.logspace(np.log10(tc_lo), np.log10(tc_hi), 96),
            ]
        )
    )
    best_tc = float(tc_default)
    best_rmse = np.inf
    for tc in candidates:
        yhat = OGSE_contrast_vs_g_rest(td, G1, G2, n_1, n_2, float(tc), float(M0_guess), float(D0_guess))
        if not np.all(np.isfinite(yhat)):
            continue
        err = _rmse(y, yhat)
        if np.isfinite(err) and err < best_rmse:
            best_rmse = float(err)
            best_tc = float(tc)
    return float(best_tc)


def _fit_rest(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
    tc_value: float,
    tc_vary: bool = True,
    tc_bounds: tuple[float, float] | None = None,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
):
    M0_lo, M0_hi = (0.0, 5.0) if m0_bounds is None else (float(m0_bounds[0]), float(m0_bounds[1]))
    D_lo, D_hi = (
        (float(D0_value / 100.0), float(D0_value * 100.0))
        if d0_bounds is None
        else (float(d0_bounds[0]), float(d0_bounds[1]))
    )
    tc_lo, tc_hi = (0.1, 1000.0) if tc_bounds is None else (float(tc_bounds[0]), float(tc_bounds[1]))
    tc_seed = _best_tc_seed_rest(
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        M0_guess=float(M0_value),
        D0_guess=float(D0_value),
        tc_default=float(tc_value),
        tc_bounds=(tc_lo, tc_hi),
    )
    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["rest"],
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        parameters=[
            CurveFitParameter("tc_ms", float(tc_seed), tc_lo, tc_hi, bool(tc_vary)),
            CurveFitParameter("M0", float(M0_value), M0_lo, M0_hi, bool(M0_vary)),
            CurveFitParameter("D0_m2_ms", float(D0_value), D_lo, D_hi, bool(D0_vary)),
        ],
    )
    return (
        float(fit.values["tc_ms"]),
        float(fit.values["M0"]),
        float(fit.values["D0_m2_ms"]),
        fit.rmse,
        fit.chi2,
        fit.method,
        float(fit.errors.get("tc_err_ms", np.nan)) if tc_vary else None,
        float(fit.errors.get("M0_err", np.nan)) if M0_vary else None,
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
    )


def _fit_rest_offset(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    C_vary: bool,
    M0_value: float,
    D0_value: float,
    C_value: float,
    tc_value: float,
    tc_vary: bool = True,
    tc_bounds: tuple[float, float] | None = None,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
    c_bounds: tuple[float, float] | None = None,
):
    M0_lo, M0_hi = (0.0, 5.0) if m0_bounds is None else (float(m0_bounds[0]), float(m0_bounds[1]))
    D_lo, D_hi = (
        (float(D0_value / 100.0), float(D0_value * 100.0))
        if d0_bounds is None
        else (float(d0_bounds[0]), float(d0_bounds[1]))
    )
    C_lo, C_hi = (0.0, 1.0) if c_bounds is None else (float(c_bounds[0]), float(c_bounds[1]))
    tc_lo, tc_hi = (0.1, 1000.0) if tc_bounds is None else (float(tc_bounds[0]), float(tc_bounds[1]))
    tc_seed = _best_tc_seed_rest(
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        M0_guess=float(M0_value),
        D0_guess=float(D0_value),
        tc_default=float(tc_value),
        tc_bounds=(tc_lo, tc_hi),
    )
    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["rest_offset"],
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        parameters=[
            CurveFitParameter("tc_ms", float(tc_seed), tc_lo, tc_hi, bool(tc_vary)),
            CurveFitParameter("M0", float(M0_value), M0_lo, M0_hi, bool(M0_vary)),
            CurveFitParameter("D0_m2_ms", float(D0_value), D_lo, D_hi, bool(D0_vary)),
            CurveFitParameter("C", float(C_value), C_lo, C_hi, bool(C_vary)),
        ],
    )
    return (
        float(fit.values["tc_ms"]),
        float(fit.values["M0"]),
        float(fit.values["D0_m2_ms"]),
        float(fit.values["C"]),
        fit.rmse,
        fit.chi2,
        fit.method,
        float(fit.errors.get("tc_err_ms", np.nan)) if tc_vary else None,
        float(fit.errors.get("M0_err", np.nan)) if M0_vary else None,
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
        float(fit.errors.get("C_err", np.nan)) if C_vary else None,
    )


def _best_seed_mixed(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_guess: float,
    D0_guess: float,
    tc_default: float,
    alpha_default: float,
) -> tuple[float, float]:
    tc_candidates = np.unique(
        np.concatenate(
            [
                np.array([float(tc_default)], dtype=float),
                np.logspace(np.log10(0.1), np.log10(1000.0), 48),
            ]
        )
    )
    alpha_candidates = np.unique(np.clip(np.array([float(alpha_default), 0.1, 0.25, 0.5, 0.75, 0.9]), 0.0, 1.0))
    best = (float(tc_default), float(alpha_default))
    best_rmse = np.inf
    for tc in tc_candidates:
        for alpha in alpha_candidates:
            yhat = OGSE_contrast_vs_g_mixed(
                td,
                G1,
                G2,
                n_1,
                n_2,
                float(tc),
                float(alpha),
                float(M0_guess),
                float(D0_guess),
            )
            if not np.all(np.isfinite(yhat)):
                continue
            err = _rmse(y, yhat)
            if np.isfinite(err) and err < best_rmse:
                best_rmse = float(err)
                best = (float(tc), float(alpha))
    return best


def _fit_mixed(
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
    tc_value: float,
    tc_vary: bool = True,
    tc_bounds: tuple[float, float] | None = None,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
):
    M0_lo, M0_hi = (0.0, 5.0) if m0_bounds is None else (float(m0_bounds[0]), float(m0_bounds[1]))
    D_lo, D_hi = (
        (float(D0_value / 100.0), float(D0_value * 100.0))
        if d0_bounds is None
        else (float(d0_bounds[0]), float(d0_bounds[1]))
    )
    tc_lo, tc_hi = (0.1, 1000.0) if tc_bounds is None else (float(tc_bounds[0]), float(tc_bounds[1]))
    tc_seed, alpha_seed = _best_seed_mixed(
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        M0_guess=float(M0_value),
        D0_guess=float(D0_value),
        tc_default=float(tc_value),
        alpha_default=0.5,
    )
    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["mixed"],
        td,
        G1,
        G2,
        n_1,
        n_2,
        y,
        parameters=[
            CurveFitParameter("tc_ms", float(np.clip(tc_seed, tc_lo, tc_hi)), tc_lo, tc_hi, bool(tc_vary)),
            CurveFitParameter("alpha", float(alpha_seed), 0.0, 1.0, True),
            CurveFitParameter("M0", float(M0_value), M0_lo, M0_hi, bool(M0_vary)),
            CurveFitParameter("D0_m2_ms", float(D0_value), D_lo, D_hi, bool(D0_vary)),
        ],
    )
    return (
        float(fit.values["tc_ms"]),
        float(fit.values["alpha"]),
        float(fit.values["M0"]),
        float(fit.values["D0_m2_ms"]),
        fit.rmse,
        fit.chi2,
        fit.method,
        float(fit.errors.get("tc_err_ms", np.nan)) if tc_vary else None,
        float(fit.errors.get("alpha_err", np.nan)),
        float(fit.errors.get("M0_err", np.nan)) if M0_vary else None,
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
    )


def _fit_selected_contrast_model(
    model: str,
    td: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    C_vary: bool,
    M0_value: float,
    D0_value: float,
    C_value: float,
    tc_value: float,
    tc_vary: bool = True,
    tc_bounds: tuple[float, float] | None = None,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
    c_bounds: tuple[float, float] | None = None,
) -> dict[str, float | str | None]:
    if model == "free":
        M0, D0, rmse, chi2, method, M0_err, D0_err = _fit_free(
            td,
            G1,
            G2,
            n_1,
            n_2,
            y,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )
        return {
            "M0": float(M0),
            "M0_err": None if M0_err is None else float(M0_err),
            "D0_m2_ms": float(D0),
            "D0_err_m2_ms": None if D0_err is None else float(D0_err),
            "rmse": float(rmse),
            "chi2": float(chi2),
            "method": str(method),
        }

    if model == "tort":
        alpha, M0, D0, rmse, chi2, method, alpha_err, M0_err, D0_err = _fit_tort(
            td,
            G1,
            G2,
            n_1,
            n_2,
            y,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )
        return {
            "alpha": float(alpha),
            "alpha_err": None if alpha_err is None else float(alpha_err),
            "M0": float(M0),
            "M0_err": None if M0_err is None else float(M0_err),
            "D0_m2_ms": float(D0),
            "D0_err_m2_ms": None if D0_err is None else float(D0_err),
            "rmse": float(rmse),
            "chi2": float(chi2),
            "method": str(method),
        }

    if model == "rest":
        tc, M0, D0, rmse, chi2, method, tc_err, M0_err, D0_err = _fit_rest(
            td,
            G1,
            G2,
            n_1,
            n_2,
            y,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            tc_value=tc_value,
            tc_vary=tc_vary,
            tc_bounds=tc_bounds,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )
        return {
            "tc_ms": float(tc),
            "tc_err_ms": None if tc_err is None else float(tc_err),
            "M0": float(M0),
            "M0_err": None if M0_err is None else float(M0_err),
            "D0_m2_ms": float(D0),
            "D0_err_m2_ms": None if D0_err is None else float(D0_err),
            "rmse": float(rmse),
            "chi2": float(chi2),
            "method": str(method),
        }

    if model in {"rest_offset", "rest_offset_globC"}:
        tc, M0, D0, C, rmse, chi2, method, tc_err, M0_err, D0_err, C_err = _fit_rest_offset(
            td,
            G1,
            G2,
            n_1,
            n_2,
            y,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            C_vary=C_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            C_value=C_value,
            tc_value=tc_value,
            tc_vary=tc_vary,
            tc_bounds=tc_bounds,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
            c_bounds=c_bounds,
        )
        return {
            "tc_ms": float(tc),
            "tc_err_ms": None if tc_err is None else float(tc_err),
            "M0": float(M0),
            "M0_err": None if M0_err is None else float(M0_err),
            "D0_m2_ms": float(D0),
            "D0_err_m2_ms": None if D0_err is None else float(D0_err),
            "C": float(C),
            "C_err": None if C_err is None else float(C_err),
            "rmse": float(rmse),
            "chi2": float(chi2),
            "method": str(method),
        }

    if model == "mixed":
        tc, alpha, M0, D0, rmse, chi2, method, tc_err, alpha_err, M0_err, D0_err = _fit_mixed(
            td,
            G1,
            G2,
            n_1,
            n_2,
            y,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            tc_value=tc_value,
            tc_vary=tc_vary,
            tc_bounds=tc_bounds,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )
        return {
            "tc_ms": float(tc),
            "tc_err_ms": None if tc_err is None else float(tc_err),
            "alpha": float(alpha),
            "alpha_err": None if alpha_err is None else float(alpha_err),
            "M0": float(M0),
            "M0_err": None if M0_err is None else float(M0_err),
            "D0_m2_ms": float(D0),
            "D0_err_m2_ms": None if D0_err is None else float(D0_err),
            "rmse": float(rmse),
            "chi2": float(chi2),
            "method": str(method),
        }

    raise ValueError(f"Model {model!r} is not implemented.")


def _read_table(path: str | Path) -> pd.DataFrame:
    return read_table_file(path)


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
        raise ValueError("Could not resolve an alpha column. Pass --alpha_col or include alpha/alpha_macro.")
    rename_map = {resolved_alpha_col: "__alpha__"}
    if alpha_td_col in out.columns:
        rename_map[alpha_td_col] = "__alpha_td_ms__"
    out = out.rename(columns=rename_map)
    out["__alpha_source_col__"] = str(resolved_alpha_col)
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
        value = _unique_scalar(curve["alpha"], name="alpha", required=False)
        if value is not None and np.isfinite(float(value)):
            return float(value), "input:alpha"
    if alpha_df is None or alpha_df.empty:
        raise ValueError("mixed_global requires alpha values in the contrast table or via --alpha_table.")

    candidates = alpha_df.copy()
    for col in ("subj", "roi", "direction"):
        if col in curve.columns and col in candidates.columns:
            value = _unique_scalar(curve[col], name=col, required=False)
            if value is not None:
                candidates = candidates[candidates[col].astype(str) == str(value)].copy()

    if candidates.empty:
        label = ", ".join(
            f"{col}={_unique_scalar(curve[col], name=col, required=False)!r}"
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
    alpha = float(values[0])
    source_cols = candidates.get("__alpha_source_col__", pd.Series(dtype=object)).dropna().unique().tolist()
    source_col = str(source_cols[0]) if len(source_cols) == 1 else "alpha_table"
    source = f"alpha_table:{source_col}"
    if not np.isfinite(alpha):
        raise ValueError(f"Invalid alpha={alpha}. OGSE_contrast_vs_g_mixed requires alpha in [0, 1].")
    if source_col == "alpha_macro":
        alpha_fit = float(np.clip(alpha, 0.0, 1.0))
        if alpha_fit != alpha:
            source = f"{source}:clipped"
        return alpha_fit, source
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"Invalid alpha={alpha}. OGSE_contrast_vs_g_mixed requires alpha in [0, 1].")
    return alpha, source


# -----------------------------
# Public API
# -----------------------------
@dataclass(frozen=True)
class FitRow:
    analysis_id: str
    subj: str
    sheet: str | None
    roi: str
    direction: str
    stat: str | None
    td_ms: float | None

    # sequence 1 params
    max_dur_ms_1: float | None
    tm_ms_1: float | None
    td_ms_1: float | None
    N_1: int
    delta_ms_1: float | None
    Delta_app_ms_1: float | None
    Hz_1: float | None
    TE_1: float | None
    TR_1: float | None
    bmax_1: float | None
    protocol_1: str | None
    sequence_1: str | None
    one_g_per_sequence_1: bool | None
    sheet_1: str | None

    # sequence 2 params
    max_dur_ms_2: float | None
    tm_ms_2: float | None
    td_ms_2: float | None
    N_2: int
    delta_ms_2: float | None
    Delta_app_ms_2: float | None
    Hz_2: float | None
    TE_2: float | None
    TR_2: float | None
    bmax_2: float | None
    protocol_2: str | None
    sequence_2: str | None
    one_g_per_sequence_2: bool | None
    sheet_2: str | None

    # fit config
    model: str
    ycol: str
    gbase: str
    fit_xcol: str
    plot_xcol: str
    xplot: str
    n_points: int
    n_fit: int
    f_corr: float
    f_corr_1: float
    f_corr_2: float

    # peak / maxima
    peak_method: str | None = None
    peak_grid_n: int | None = None
    g1_max_raw_mTm: float | None = None
    g2_max_raw_mTm: float | None = None
    g1_max_corr_mTm: float | None = None
    g2_max_corr_mTm: float | None = None
    peak_g_max_mTm: float | None = None
    g1_search_max_raw_mTm: float | None = None
    g2_search_max_raw_mTm: float | None = None
    g1_search_max_corr_mTm: float | None = None
    g2_search_max_corr_mTm: float | None = None
    peak_resample_gradient: bool | None = None
    peak_resample_g_max_corr_mTm: float | None = None
    peak_fraction: float | None = None
    g1_peak_raw_mTm: float | None = None
    g2_peak_raw_mTm: float | None = None
    g1_peak_corr_mTm: float | None = None
    g2_peak_corr_mTm: float | None = None
    x_peak_raw_mTm: float | None = None
    x_peak_corr_mTm: float | None = None
    g_peak_resampled_raw_mTm: float | None = None
    g_peak_resampled_corr_mTm: float | None = None
    signal_peak: float | None = None
    l_G_peak_m: float | None = None
    L_cf_peak: float | None = None
    lcf_peak_m: float | None = None
    tc_peak_ms: float | None = None
    tc_peak_resampled_ms: float | None = None

    # fitted
    M0: float | None = None
    M0_err: float | None = None
    D0_m2_ms: float | None = None
    D0_err_m2_ms: float | None = None
    alpha: float | None = None
    alpha_err: float | None = None
    tc_ms: float | None = None
    tc_err_ms: float | None = None
    C: float | None = None
    C_err: float | None = None

    # metrics
    rmse: float | None = None
    chi2: float | None = None
    method: str | None = None
    ok: bool = True
    msg: str = ""


def _load_ogse_contrast_tables(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        p = Path(path)
        df = pd.read_parquet(p)
        analysis_id = _analysis_id_from_source_file(p.name)
        sheet_hint = canonical_sheet_name(analysis_id)
        if "sheet" not in df.columns:
            if "sheet_1" in df.columns:
                df["sheet"] = df["sheet_1"].map(canonical_sheet_name)
            elif "sheet_2" in df.columns:
                df["sheet"] = df["sheet_2"].map(canonical_sheet_name)
            else:
                df["sheet"] = sheet_hint
        else:
            df["sheet"] = df["sheet"].map(canonical_sheet_name)
        if "subj" not in df.columns:
            df["subj"] = [infer_subj_label(sheet, source_name=analysis_id) for sheet in df["sheet"]]
        df["subj"] = df["subj"].astype(str)
        df["analysis_id"] = analysis_id
        df["source_file"] = p.name
        frames.append(df)
    if not frames:
        raise ValueError("At least one contrast table is required.")
    return pd.concat(frames, ignore_index=True)


def _contrast_curve_metadata(gg: pd.DataFrame, *, source_file: str | None, td_override_ms: float | None, td_tol_ms: float):
    def _get_float(col: str) -> float | None:
        if col not in gg.columns:
            return None
        v = _unique_scalar(gg[col], name=col, required=False)
        if v is None:
            return None
        vf = float(v)
        return vf if np.isfinite(vf) else None

    def _get_str(col: str) -> str | None:
        if col not in gg.columns:
            return None
        v = _unique_scalar(gg[col], name=col, required=False)
        if v is None:
            return None
        return str(v)

    n_1 = int(round(float(_unique_scalar(gg["N_1"], name="N_1", required=True))))
    n_2 = int(round(float(_unique_scalar(gg["N_2"], name="N_2", required=True))))
    td_ms_1 = _get_float("td_ms_1")
    td_ms_2 = _get_float("td_ms_2")
    max_dur_ms_1 = _get_float("max_dur_ms_1")
    tm_ms_1 = _get_float("tm_ms_1")

    if td_override_ms is not None:
        td_ms = float(td_override_ms)
    elif td_ms_1 is not None and td_ms_2 is not None:
        if abs(td_ms_1 - td_ms_2) > float(td_tol_ms):
            raise ValueError(f"td_ms_1 != td_ms_2: td1={td_ms_1}, td2={td_ms_2}.")
        td_ms = float(0.5 * (td_ms_1 + td_ms_2))
    elif td_ms_1 is not None:
        td_ms = float(td_ms_1)
    elif max_dur_ms_1 is not None and tm_ms_1 is not None:
        td_ms = float(2.0 * max_dur_ms_1 + tm_ms_1)
    else:
        td_ms = None

    sheet = canonical_sheet_name(_get_str("sheet") or _get_str("sheet_1") or _get_str("sheet_2"))
    subj = _get_str("subj")
    if subj is None or not str(subj).strip():
        subj = infer_subj_label(sheet, source_name=source_file)

    return {
        "subj": str(subj),
        "sheet": sheet,
        "td_ms": td_ms,
        "td_ms_1": td_ms_1,
        "td_ms_2": td_ms_2,
        "N_1": n_1,
        "N_2": n_2,
        "delta_ms_1": _get_float("delta_ms_1"),
        "Delta_app_ms_1": _get_float("Delta_app_ms_1"),
        "delta_ms_2": _get_float("delta_ms_2"),
        "Delta_app_ms_2": _get_float("Delta_app_ms_2"),
        "max_dur_ms_1": max_dur_ms_1,
        "tm_ms_1": tm_ms_1,
        "max_dur_ms_2": _get_float("max_dur_ms_2"),
        "tm_ms_2": _get_float("tm_ms_2"),
        "Hz_1": _get_float("Hz_1"),
        "Hz_2": _get_float("Hz_2"),
        "TE_1": _get_float("TE_1"),
        "TE_2": _get_float("TE_2"),
        "TR_1": _get_float("TR_1"),
        "TR_2": _get_float("TR_2"),
        "bmax_1": _get_float("bmax_1"),
        "bmax_2": _get_float("bmax_2"),
        "protocol_1": _get_str("protocol_1"),
        "protocol_2": _get_str("protocol_2"),
        "sequence_1": scalar_or_compact_series(gg["sequence_1"], name="sequence_1", required=False)
        if "sequence_1" in gg.columns
        else None,
        "sequence_2": scalar_or_compact_series(gg["sequence_2"], name="sequence_2", required=False)
        if "sequence_2" in gg.columns
        else None,
        "sheet_1": _get_str("sheet_1"),
        "sheet_2": _get_str("sheet_2"),
    }


def _prepare_mixed_global_contrast_data(
    df: pd.DataFrame,
    *,
    gbase: str,
    plot_xcol: str | None,
    ycol: str,
    directions: list[str] | None,
    rois: list[str] | None,
    stat_keep: str | None,
    xplot: str,
    n_fit: int | None,
    sort_by_x: bool,
    f_by_direction: dict[str, float | tuple[float, float]] | None,
    td_override_ms: float | None,
    td_tol_ms: float,
    alpha_df: pd.DataFrame | None,
    alpha_td_tol_ms: float,
) -> pd.DataFrame:
    df = _normalize_keys(df, label="ogse_mixed_global")
    if ycol not in df.columns:
        raise KeyError(f"ogse_mixed_global: missing ycol {ycol!r}. Columns={list(df.columns)}")
    if directions is not None and not (len(directions) == 1 and directions[0].upper() == "ALL"):
        df = df[df["direction"].astype(str).isin([str(x) for x in directions])].copy()
    if rois is not None and not (len(rois) == 1 and rois[0].upper() == "ALL"):
        df = df[df["roi"].astype(str).isin([str(r) for r in rois])].copy()
    if stat_keep is not None and "stat" in df.columns and str(stat_keep).upper() != "ALL":
        df = df[df["stat"].astype(str) == str(stat_keep)].copy()
    if df.empty:
        raise ValueError("No data remains after filtering.")

    fit_axis = normalize_axis_base(gbase)
    plot_axis = _resolve_plot_axis(fit_axis=fit_axis, plot_axis=plot_xcol, xplot=xplot)
    _plot_axis_base, plot_side = split_axis_side(plot_axis)
    plot_side = 1 if plot_side is None else int(plot_side)
    _require_cols(df, ["N_1", "N_2"], label="ogse_mixed_global (N_1/N_2)")

    rows: list[pd.DataFrame] = []
    group_cols = ["analysis_id", "source_file", "roi", "direction"] + (["stat"] if "stat" in df.columns else [])
    for _key, gg in df.groupby(group_cols, sort=False, dropna=False):
        meta = _contrast_curve_metadata(
            gg,
            source_file=str(_unique_scalar(gg["source_file"], name="source_file", required=False)),
            td_override_ms=td_override_ms,
            td_tol_ms=td_tol_ms,
        )
        td_ms = meta["td_ms"]
        if td_ms is None or not np.isfinite(float(td_ms)):
            continue
        roi = str(_unique_scalar(gg["roi"], name="roi", required=True))
        direction = str(_unique_scalar(gg["direction"], name="direction", required=True))
        if {"grad_correction_factor_1", "grad_correction_factor_2"}.issubset(gg.columns):
            f_corr_1, f_corr_2 = _coerce_correction_pair(
                (
                    _unique_scalar(gg["grad_correction_factor_1"], name="grad_correction_factor_1", required=False),
                    _unique_scalar(gg["grad_correction_factor_2"], name="grad_correction_factor_2", required=False),
                )
            )
        else:
            f_corr_1, f_corr_2 = _coerce_correction_pair(f_by_direction.get(direction, 1.0) if f_by_direction else 1.0)

        fit_bundle_1 = build_axis_bundle(
            gg,
            axis=fit_axis,
            side=1,
            correction_factor=float(f_corr_1),
            gamma=GAMMA_DEFAULT,
            N=int(meta["N_1"]),
            delta_ms=meta["delta_ms_1"],
            Delta_app_ms=meta["Delta_app_ms_1"],
        )
        fit_bundle_2 = build_axis_bundle(
            gg,
            axis=fit_axis,
            side=2,
            correction_factor=float(f_corr_2),
            gamma=GAMMA_DEFAULT,
            N=int(meta["N_2"]),
            delta_ms=meta["delta_ms_2"],
            Delta_app_ms=meta["Delta_app_ms_2"],
        )
        plot_bundle = build_axis_bundle(
            gg,
            axis=plot_axis,
            side=plot_side,
            correction_factor=float(f_corr_1 if plot_side == 1 else f_corr_2),
            gamma=GAMMA_DEFAULT,
            N=int(meta["N_1"] if plot_side == 1 else meta["N_2"]),
            delta_ms=meta["delta_ms_1"] if plot_side == 1 else meta["delta_ms_2"],
            Delta_app_ms=meta["Delta_app_ms_1"] if plot_side == 1 else meta["Delta_app_ms_2"],
        )

        alpha, alpha_source = _lookup_alpha_for_curve(gg, alpha_df, td_ms=float(td_ms), alpha_td_tol_ms=alpha_td_tol_ms)
        y = pd.to_numeric(gg[ycol], errors="coerce").to_numpy(dtype=float)
        G1 = fit_bundle_1.gradient_corr
        G2 = fit_bundle_2.gradient_corr
        x = plot_bundle.axis_corr
        m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2) & np.isfinite(x)
        y, G1, G2, x = y[m], G1[m], G2[m], x[m]
        if sort_by_x and len(y):
            order = np.argsort(x)
            y, G1, G2, x = y[order], G1[order], G2[order], x[order]
        if n_fit is not None:
            k = int(n_fit)
            y, G1, G2, x = y[:k], G1[:k], G2[:k], x[:k]
        if len(y) == 0:
            continue

        out = pd.DataFrame(
            {
                "__y__": y,
                "__G1__": G1,
                "__G2__": G2,
                "__xplot_value__": x,
                "__td_ms__": float(td_ms),
                "__N1__": int(meta["N_1"]),
                "__N2__": int(meta["N_2"]),
                "__alpha__": float(alpha),
                "__alpha_source__": str(alpha_source),
                "__f_corr_1__": float(f_corr_1),
                "__f_corr_2__": float(f_corr_2),
                "__n_points_curve__": int(len(y)),
                "__n_fit_curve__": int(len(y)),
            }
        )
        for col in group_cols:
            out[col] = scalar_or_compact_series(gg[col], name=col, required=False)
        for key_meta, value in meta.items():
            out[key_meta] = value
        out["roi"] = roi
        out["direction"] = direction
        out["gbase"] = _normalize_gbase(fit_axis)
        out["fit_xcol"] = str(fit_axis)
        out["plot_xcol"] = str(plot_axis)
        out["xplot"] = str(plot_side)
        rows.append(out)

    if not rows:
        raise ValueError("No valid curves remained for mixed_global.")
    return pd.concat(rows, ignore_index=True)


def _prepare_global_contrast_data(
    df: pd.DataFrame,
    *,
    gbase: str,
    plot_xcol: str | None,
    ycol: str,
    directions: list[str] | None,
    rois: list[str] | None,
    stat_keep: str | None,
    xplot: str,
    n_fit: int | None,
    sort_by_x: bool,
    f_by_direction: dict[str, float | tuple[float, float]] | None,
    td_override_ms: float | None,
    td_tol_ms: float,
    oneg: bool,
) -> pd.DataFrame:
    df = _normalize_keys(df, label="ogse_global")
    if ycol not in df.columns:
        raise KeyError(f"ogse_global: missing ycol {ycol!r}. Columns={list(df.columns)}")
    if directions is not None and not (len(directions) == 1 and directions[0].upper() == "ALL"):
        df = df[df["direction"].astype(str).isin([str(x) for x in directions])].copy()
    if rois is not None and not (len(rois) == 1 and rois[0].upper() == "ALL"):
        df = df[df["roi"].astype(str).isin([str(r) for r in rois])].copy()
    if stat_keep is not None and "stat" in df.columns and str(stat_keep).upper() != "ALL":
        df = df[df["stat"].astype(str) == str(stat_keep)].copy()
    if df.empty:
        raise ValueError("No data remains after filtering.")

    fit_axis = normalize_axis_base(gbase)
    plot_axis = _resolve_plot_axis(fit_axis=fit_axis, plot_axis=plot_xcol, xplot=xplot)
    _plot_axis_base, plot_side = split_axis_side(plot_axis)
    plot_side = 1 if plot_side is None else int(plot_side)
    _require_cols(df, ["N_1", "N_2"], label="ogse_global (N_1/N_2)")

    rows: list[pd.DataFrame] = []
    group_cols = ["analysis_id", "source_file", "roi", "direction"] + (["stat"] if "stat" in df.columns else [])
    for curve_id, (_key, gg) in enumerate(df.groupby(group_cols, sort=False, dropna=False)):
        meta = _contrast_curve_metadata(
            gg,
            source_file=str(_unique_scalar(gg["source_file"], name="source_file", required=False)),
            td_override_ms=td_override_ms,
            td_tol_ms=td_tol_ms,
        )
        td_ms = meta["td_ms"]
        if td_ms is None or not np.isfinite(float(td_ms)):
            continue
        roi = str(_unique_scalar(gg["roi"], name="roi", required=True))
        direction = str(_unique_scalar(gg["direction"], name="direction", required=True))
        if {"grad_correction_factor_1", "grad_correction_factor_2"}.issubset(gg.columns):
            f_corr_1, f_corr_2 = _coerce_correction_pair(
                (
                    _unique_scalar(gg["grad_correction_factor_1"], name="grad_correction_factor_1", required=False),
                    _unique_scalar(gg["grad_correction_factor_2"], name="grad_correction_factor_2", required=False),
                )
            )
        else:
            f_corr_1, f_corr_2 = _coerce_correction_pair(f_by_direction.get(direction, 1.0) if f_by_direction else 1.0)

        fit_bundle_1 = build_axis_bundle(
            gg,
            axis=fit_axis,
            side=1,
            correction_factor=float(f_corr_1),
            gamma=GAMMA_DEFAULT,
            N=int(meta["N_1"]),
            delta_ms=meta["delta_ms_1"],
            Delta_app_ms=meta["Delta_app_ms_1"],
        )
        fit_bundle_2 = build_axis_bundle(
            gg,
            axis=fit_axis,
            side=2,
            correction_factor=float(f_corr_2),
            gamma=GAMMA_DEFAULT,
            N=int(meta["N_2"]),
            delta_ms=meta["delta_ms_2"],
            Delta_app_ms=meta["Delta_app_ms_2"],
        )
        plot_bundle = build_axis_bundle(
            gg,
            axis=plot_axis,
            side=plot_side,
            correction_factor=float(f_corr_1 if plot_side == 1 else f_corr_2),
            gamma=GAMMA_DEFAULT,
            N=int(meta["N_1"] if plot_side == 1 else meta["N_2"]),
            delta_ms=meta["delta_ms_1"] if plot_side == 1 else meta["delta_ms_2"],
            Delta_app_ms=meta["Delta_app_ms_1"] if plot_side == 1 else meta["Delta_app_ms_2"],
        )

        y = pd.to_numeric(gg[ycol], errors="coerce").to_numpy(dtype=float)
        G1 = fit_bundle_1.gradient_corr
        G2 = fit_bundle_2.gradient_corr
        x = plot_bundle.axis_corr
        m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2) & np.isfinite(x)
        y, G1, G2, x = y[m], G1[m], G2[m], x[m]
        if sort_by_x and len(y):
            order = np.argsort(x)
            y, G1, G2, x = y[order], G1[order], G2[order], x[order]
        if n_fit is not None:
            k = int(n_fit)
            y, G1, G2, x = y[:k], G1[:k], G2[:k], x[:k]
        if len(y) == 0:
            continue

        one_g_per_sequence_1 = bool(truthy_series(gg["one_g_per_sequence_1"])) if "one_g_per_sequence_1" in gg.columns else None
        one_g_per_sequence_2 = bool(truthy_series(gg["one_g_per_sequence_2"])) if "one_g_per_sequence_2" in gg.columns else None
        allow_sequence_ranges = bool(oneg or one_g_per_sequence_1 or one_g_per_sequence_2)

        out = pd.DataFrame(
            {
                "__curve_id__": int(curve_id),
                "__point_id__": np.arange(len(y), dtype=int),
                "__y__": y,
                "__G1__": G1,
                "__G2__": G2,
                "__xplot_value__": x,
                "__td_ms__": float(td_ms),
                "__N1__": int(meta["N_1"]),
                "__N2__": int(meta["N_2"]),
                "__f_corr_1__": float(f_corr_1),
                "__f_corr_2__": float(f_corr_2),
                "__n_points_curve__": int(len(y)),
                "__n_fit_curve__": int(len(y)),
                "__one_g_per_sequence_1__": one_g_per_sequence_1,
                "__one_g_per_sequence_2__": one_g_per_sequence_2,
            }
        )
        for col in group_cols:
            out[col] = scalar_or_compact_series(gg[col], name=col, required=False)
        for key_meta, value in meta.items():
            out[key_meta] = value
        if allow_sequence_ranges:
            for seq_col in ("sequence_1", "sequence_2"):
                if seq_col in gg.columns:
                    out[seq_col] = scalar_or_compact_series(gg[seq_col], name=seq_col, required=False)
        out["roi"] = roi
        out["direction"] = direction
        out["gbase"] = _normalize_gbase(fit_axis)
        out["fit_xcol"] = str(fit_axis)
        out["plot_xcol"] = str(plot_axis)
        out["xplot"] = str(plot_side)
        rows.append(out)

    if not rows:
        raise ValueError("No valid curves remained for global OGSE contrast fitting.")
    return pd.concat(rows, ignore_index=True)


def fit_ogse_contrast_global_long(
    df: pd.DataFrame,
    *,
    model: str,
    global_params: Sequence[str],
    gbase: str = "g_lin_max",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    directions: list[str] | None = None,
    rois: list[str] | None = None,
    stat_keep: str | None = "avg",
    xplot: str = "1",
    n_fit: int | None = None,
    sort_by_x: bool = True,
    f_by_direction: dict[str, float | tuple[float, float]] | None = None,
    td_override_ms: float | None = None,
    td_tol_ms: float = 1e-3,
    M0_vary: bool = True,
    D0_vary: bool = True,
    C_vary: bool = True,
    M0_value: float = 1.0,
    D0_value: float = 1e-12,
    C_value: float = 0.0,
    tc_value: float = 5.0,
    tc_vary: bool = True,
    tc_bounds: tuple[float, float] | None = None,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
    c_bounds: tuple[float, float] | None = None,
    peak_grid_n: int = 1000,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
    peak_g_max_mTm: float | None = None,
    peak_resample_gradient: bool = False,
    peak_resample_g_max_corr_mTm: float | None = None,
    source_file: str | None = None,
    oneg: bool = False,
) -> pd.DataFrame:
    if model not in MODEL_PARAM_NAMES:
        raise ValueError(f"Global OGSE contrast fitting does not support model={model!r}.")
    resolved_global = _normalize_global_params(model, global_params)
    if not resolved_global:
        raise ValueError("--global_params must include at least one parameter for global fitting.")

    vary_flags = _param_vary_flags(M0_vary=M0_vary, D0_vary=D0_vary, C_vary=C_vary, tc_vary=tc_vary)
    fixed_values = _param_fixed_values(M0_value=M0_value, D0_value=D0_value, C_value=C_value, tc_value=tc_value)
    model_params = list(MODEL_PARAM_NAMES[model])
    fixed_params = [name for name in model_params if not vary_flags.get(name, True)]
    blocked = [name for name in resolved_global if name in fixed_params]
    if blocked:
        raise ValueError(f"Parameters cannot be both fixed and global: {blocked}. Use the corresponding --free_* option.")

    prepared = _prepare_global_contrast_data(
        df,
        gbase=gbase,
        plot_xcol=plot_xcol,
        ycol=ycol,
        directions=directions,
        rois=rois,
        stat_keep=stat_keep,
        xplot=xplot,
        n_fit=n_fit,
        sort_by_x=sort_by_x,
        f_by_direction=f_by_direction,
        td_override_ms=td_override_ms,
        td_tol_ms=td_tol_ms,
        oneg=oneg,
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["subj", "roi", "direction"] + (["stat"] if "stat" in prepared.columns else [])
    for key, group in prepared.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))
        y = group["__y__"].to_numpy(dtype=float)
        G1 = group["__G1__"].to_numpy(dtype=float)
        G2 = group["__G2__"].to_numpy(dtype=float)
        td = group["__td_ms__"].to_numpy(dtype=float)
        n1 = group["__N1__"].to_numpy(dtype=float)
        n2 = group["__N2__"].to_numpy(dtype=float)
        curve_ids = group["__curve_id__"].to_numpy(dtype=int)
        unique_curve_ids = sorted(int(v) for v in pd.Series(curve_ids).unique().tolist())

        param_names: list[str] = []
        p0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        log_params: list[str] = []
        value_lookup: dict[str, dict[int | None, str]] = {}

        for param in model_params:
            value_lookup[param] = {}
            if not vary_flags.get(param, True):
                continue
            seed, lo, hi = _param_seed_bounds(
                param,
                M0_value=M0_value,
                D0_value=D0_value,
                C_value=C_value,
                tc_value=tc_value,
                tc_bounds=tc_bounds,
                m0_bounds=m0_bounds,
                d0_bounds=d0_bounds,
                c_bounds=c_bounds,
            )
            if param in resolved_global:
                fit_name = param
                param_names.append(fit_name)
                p0.append(seed)
                lower.append(lo)
                upper.append(hi)
                value_lookup[param][None] = fit_name
                if param in {"tc_ms", "D0_m2_ms"}:
                    log_params.append(fit_name)
            else:
                for curve_id in unique_curve_ids:
                    fit_name = f"{param}__curve_{curve_id}"
                    param_names.append(fit_name)
                    p0.append(seed)
                    lower.append(lo)
                    upper.append(hi)
                    value_lookup[param][curve_id] = fit_name
                    if param in {"tc_ms", "D0_m2_ms"}:
                        log_params.append(fit_name)

        def params_for_curve(values: dict[str, float], curve_id: int) -> dict[str, float]:
            params = {param: float(fixed_values[param]) for param in model_params}
            for param in model_params:
                if not vary_flags.get(param, True):
                    continue
                fit_name = value_lookup[param].get(None) or value_lookup[param][int(curve_id)]
                params[param] = float(values[fit_name])
            return params

        def model_fn(x_model: np.ndarray, *values: float) -> np.ndarray:
            del x_model
            values_by_name = dict(zip(param_names, values))
            yhat = np.empty_like(y, dtype=float)
            for curve_id in unique_curve_ids:
                mask = curve_ids == int(curve_id)
                params = params_for_curve(values_by_name, int(curve_id))
                yhat[mask] = CONTRAST_MODEL_SPECS[model].evaluator(
                    td[mask],
                    G1[mask],
                    G2[mask],
                    n1[mask],
                    n2[mask],
                    params,
                )
            return yhat

        try:
            if param_names:
                fit = fit_least_squares(
                    model_fn,
                    np.arange(len(y), dtype=float),
                    y,
                    param_names=param_names,
                    p0=p0,
                    bounds=(lower, upper),
                    log_params=log_params,
                    max_nfev=int(CONTRAST_MODEL_SPECS[model].maxfev),
                    method=f"least_squares_ogse_global_{model}",
                )
                fit_values = fit.values
                fit_errors = fit.errors
                ok = True
                msg = ""
                method = fit.method
            else:
                fit_values = {}
                fit_errors = {}
                ok = True
                msg = ""
                method = "fixed_global"
        except Exception as exc:
            fit_values = {}
            fit_errors = {}
            ok = False
            msg = str(exc)
            method = f"least_squares_ogse_global_{model}"

        for curve_id in unique_curve_ids:
            curve = group[group["__curve_id__"] == int(curve_id)].copy()
            curve_params = params_for_curve(fit_values, int(curve_id)) if ok else {
                param: np.nan if vary_flags.get(param, True) else fixed_values[param] for param in model_params
            }
            y_curve = curve["__y__"].to_numpy(dtype=float)
            G1_curve = curve["__G1__"].to_numpy(dtype=float)
            G2_curve = curve["__G2__"].to_numpy(dtype=float)
            td_curve = float(curve["__td_ms__"].iloc[0])
            n1_curve = int(curve["__N1__"].iloc[0])
            n2_curve = int(curve["__N2__"].iloc[0])
            if ok:
                yhat_curve = CONTRAST_MODEL_SPECS[model].evaluator(
                    td_curve,
                    G1_curve,
                    G2_curve,
                    n1_curve,
                    n2_curve,
                    curve_params,
                )
                rmse = _rmse(y_curve, yhat_curve)
                chi2 = _chi2(y_curve, yhat_curve)
                peak_metrics = _compute_peak_metrics(
                    model=model,
                    td_ms=td_curve,
                    n_1=n1_curve,
                    n_2=n2_curve,
                    fit_row=curve_params,
                    g1_max_corr=float(np.nanmax(G1_curve)),
                    g2_max_corr=float(np.nanmax(G2_curve)),
                    f_corr_1=float(curve["__f_corr_1__"].iloc[0]),
                    f_corr_2=float(curve["__f_corr_2__"].iloc[0]),
                    xplot=str(curve["xplot"].iloc[0]),
                    peak_grid_n=int(peak_grid_n),
                    peak_D0_fix=float(peak_D0_fix),
                    peak_gamma=float(peak_gamma),
                    peak_g_max_mTm=peak_g_max_mTm,
                    peak_resample_gradient=bool(peak_resample_gradient),
                    peak_resample_g_max_corr_mTm=peak_resample_g_max_corr_mTm,
                )
            else:
                rmse = np.nan
                chi2 = np.nan
                peak_metrics = {}

            row: dict[str, Any] = {
                **{col: str(value) for col, value in key_dict.items()},
                "source_file": scalar_or_compact_series(curve["source_file"], name="source_file", required=False),
                "analysis_id": scalar_or_compact_series(curve["analysis_id"], name="analysis_id", required=False),
                "sheet": scalar_or_compact_series(curve["sheet"], name="sheet", required=False),
                "td_ms": td_curve,
                "td_min_ms": td_curve,
                "td_max_ms": td_curve,
                "n_td": int(pd.Series(group["__td_ms__"]).nunique()),
                "N_1": n1_curve,
                "N_2": n2_curve,
                "delta_ms_1": scalar_or_compact_series(curve["delta_ms_1"], name="delta_ms_1", required=False),
                "Delta_app_ms_1": scalar_or_compact_series(curve["Delta_app_ms_1"], name="Delta_app_ms_1", required=False),
                "delta_ms_2": scalar_or_compact_series(curve["delta_ms_2"], name="delta_ms_2", required=False),
                "Delta_app_ms_2": scalar_or_compact_series(curve["Delta_app_ms_2"], name="Delta_app_ms_2", required=False),
                "max_dur_ms_1": scalar_or_compact_series(curve["max_dur_ms_1"], name="max_dur_ms_1", required=False),
                "tm_ms_1": scalar_or_compact_series(curve["tm_ms_1"], name="tm_ms_1", required=False),
                "td_ms_1": scalar_or_compact_series(curve["td_ms_1"], name="td_ms_1", required=False),
                "max_dur_ms_2": scalar_or_compact_series(curve["max_dur_ms_2"], name="max_dur_ms_2", required=False),
                "tm_ms_2": scalar_or_compact_series(curve["tm_ms_2"], name="tm_ms_2", required=False),
                "td_ms_2": scalar_or_compact_series(curve["td_ms_2"], name="td_ms_2", required=False),
                "Hz_1": scalar_or_compact_series(curve["Hz_1"], name="Hz_1", required=False),
                "Hz_2": scalar_or_compact_series(curve["Hz_2"], name="Hz_2", required=False),
                "TE_1": scalar_or_compact_series(curve["TE_1"], name="TE_1", required=False),
                "TE_2": scalar_or_compact_series(curve["TE_2"], name="TE_2", required=False),
                "TR_1": scalar_or_compact_series(curve["TR_1"], name="TR_1", required=False),
                "TR_2": scalar_or_compact_series(curve["TR_2"], name="TR_2", required=False),
                "bmax_1": scalar_or_compact_series(curve["bmax_1"], name="bmax_1", required=False),
                "bmax_2": scalar_or_compact_series(curve["bmax_2"], name="bmax_2", required=False),
                "protocol_1": scalar_or_compact_series(curve["protocol_1"], name="protocol_1", required=False),
                "protocol_2": scalar_or_compact_series(curve["protocol_2"], name="protocol_2", required=False),
                "sequence_1": scalar_or_compact_series(curve["sequence_1"], name="sequence_1", required=False),
                "sequence_2": scalar_or_compact_series(curve["sequence_2"], name="sequence_2", required=False),
                "one_g_per_sequence_1": scalar_or_compact_series(curve["__one_g_per_sequence_1__"], name="one_g_per_sequence_1", required=False),
                "one_g_per_sequence_2": scalar_or_compact_series(curve["__one_g_per_sequence_2__"], name="one_g_per_sequence_2", required=False),
                "sheet_1": scalar_or_compact_series(curve["sheet_1"], name="sheet_1", required=False),
                "sheet_2": scalar_or_compact_series(curve["sheet_2"], name="sheet_2", required=False),
                "fit_kind": "ogse_contrast",
                "model": model,
                "fit_scope": "global_td",
                "global_params": ",".join(resolved_global),
                "ycol": ycol,
                "stat": key_dict.get("stat", stat_keep),
                "n_points": int(len(y_curve)),
                "n_fit": int(len(y_curve)),
                "rmse": rmse,
                "chi2": chi2,
                "gbase": scalar_or_compact_series(curve["gbase"], name="gbase", required=False),
                "fit_xcol": scalar_or_compact_series(curve["fit_xcol"], name="fit_xcol", required=False),
                "plot_xcol": scalar_or_compact_series(curve["plot_xcol"], name="plot_xcol", required=False),
                "xplot": scalar_or_compact_series(curve["xplot"], name="xplot", required=False),
                "f_corr": np.nan,
                "f_corr_1": float(curve["__f_corr_1__"].iloc[0]),
                "f_corr_2": float(curve["__f_corr_2__"].iloc[0]),
                "method": method,
                "ok": ok,
                "msg": msg,
                **peak_metrics,
            }
            for param in model_params:
                row[param] = float(curve_params.get(param, np.nan))
                err_col = parameter_error_column(param)
                if param in fixed_params:
                    row[err_col] = np.nan
                elif param in resolved_global:
                    row[err_col] = fit_errors.get(err_col, np.nan)
                else:
                    fit_name = value_lookup[param][int(curve_id)]
                    row[err_col] = fit_errors.get(parameter_error_column(fit_name), np.nan)
            if "tc_ms" in model_params:
                bounds = tc_bounds or (0.1, 1000.0)
                row["tc_bound_min"] = float(bounds[0])
                row["tc_bound_max"] = float(bounds[1])
            if "alpha" in model_params:
                row["alpha_min"] = 0.0
                row["alpha_max"] = 1.0
                row["alpha_source"] = "fit"
            rows.append(row)

    out = pd.DataFrame(rows)
    return standardize_fit_params(out, fit_kind="ogse_contrast", source_file=source_file)


def fit_ogse_contrast_mixed_global_long(
    df: pd.DataFrame,
    *,
    gbase: str = "g_thorsten",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    directions: list[str] | None = None,
    rois: list[str] | None = None,
    stat_keep: str | None = "avg",
    xplot: str = "1",
    n_fit: int | None = None,
    sort_by_x: bool = True,
    f_by_direction: dict[str, float | tuple[float, float]] | None = None,
    td_override_ms: float | None = None,
    td_tol_ms: float = 1e-3,
    M0_vary: bool = False,
    D0_vary: bool = False,
    M0_value: float = 1.0,
    D0_value: float = 3.2e-12,
    tc_value: float = 5.0,
    tc_bounds: tuple[float, float] = (0.1, 1000.0),
    m0_bounds: tuple[float, float] = (0.0, 5.0),
    d0_bounds: tuple[float, float] = (1e-16, 1e-10),
    alpha_df: pd.DataFrame | None = None,
    alpha_col: str | None = None,
    alpha_td_col: str = "td_ms",
    alpha_td_tol_ms: float = 1e-3,
    source_file: str | None = None,
) -> pd.DataFrame:
    alpha_lookup = _normalize_alpha_table(alpha_df, alpha_col=alpha_col, alpha_td_col=alpha_td_col)
    prepared = _prepare_mixed_global_contrast_data(
        df,
        gbase=gbase,
        plot_xcol=plot_xcol,
        ycol=ycol,
        directions=directions,
        rois=rois,
        stat_keep=stat_keep,
        xplot=xplot,
        n_fit=n_fit,
        sort_by_x=sort_by_x,
        f_by_direction=f_by_direction,
        td_override_ms=td_override_ms,
        td_tol_ms=td_tol_ms,
        alpha_df=alpha_lookup,
        alpha_td_tol_ms=alpha_td_tol_ms,
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["subj", "roi", "direction"] + (["stat"] if "stat" in prepared.columns else [])
    for key, group in prepared.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))
        y = group["__y__"].to_numpy(dtype=float)
        G1 = group["__G1__"].to_numpy(dtype=float)
        G2 = group["__G2__"].to_numpy(dtype=float)
        td = group["__td_ms__"].to_numpy(dtype=float)
        n1 = group["__N1__"].to_numpy(dtype=float)
        n2 = group["__N2__"].to_numpy(dtype=float)
        alpha = group["__alpha__"].to_numpy(dtype=float)

        param_names = ["tc_ms"]
        p0 = [float(np.clip(float(tc_value), float(tc_bounds[0]), float(tc_bounds[1])))]
        lower = [float(tc_bounds[0])]
        upper = [float(tc_bounds[1])]
        log_params = ["tc_ms"]
        fixed = {"M0": float(M0_value), "D0_m2_ms": float(D0_value)}
        if M0_vary:
            param_names.append("M0")
            p0.append(float(np.clip(float(M0_value), float(m0_bounds[0]), float(m0_bounds[1]))))
            lower.append(float(m0_bounds[0]))
            upper.append(float(m0_bounds[1]))
            fixed.pop("M0", None)
        if D0_vary:
            param_names.append("D0_m2_ms")
            p0.append(float(np.clip(float(D0_value), float(d0_bounds[0]), float(d0_bounds[1]))))
            lower.append(float(d0_bounds[0]))
            upper.append(float(d0_bounds[1]))
            log_params.append("D0_m2_ms")
            fixed.pop("D0_m2_ms", None)

        def model(g: np.ndarray, *values: float) -> np.ndarray:
            params = dict(fixed)
            params.update(zip(param_names, values))
            return OGSE_contrast_vs_g_mixed(
                td,
                np.asarray(g, dtype=float),
                G2,
                n1,
                n2,
                float(params["tc_ms"]),
                alpha,
                float(params["M0"]),
                float(params["D0_m2_ms"]),
            )

        try:
            fit = fit_least_squares(
                model,
                G1,
                y,
                param_names=param_names,
                p0=p0,
                bounds=(lower, upper),
                log_params=log_params,
                max_nfev=400000,
                method="least_squares_ogse_mixed_global_log",
            )
            values = dict(fixed)
            values.update(fit.values)
            errors = fit.errors
            ok = True
            msg = ""
            rmse = fit.rmse
            chi2 = fit.chi2
            method = fit.method
        except Exception as exc:
            values = dict(fixed)
            values["tc_ms"] = np.nan
            errors = {"tc_err_ms": np.nan}
            ok = False
            msg = str(exc)
            rmse = np.nan
            chi2 = np.nan
            method = "least_squares_ogse_mixed_global_log"

        row: dict[str, Any] = {
            **{col: str(value) for col, value in key_dict.items()},
            "source_file": source_file or scalar_or_compact_series(group["source_file"], name="source_file", required=False),
            "analysis_id": scalar_or_compact_series(group["analysis_id"], name="analysis_id", required=False),
            "sheet": scalar_or_compact_series(group["sheet"], name="sheet", required=False),
            "td_ms": float(np.nanmean(td)),
            "td_min_ms": float(np.nanmin(td)),
            "td_max_ms": float(np.nanmax(td)),
            "n_td": int(pd.Series(td).nunique()),
            "N_1": scalar_or_compact_series(group["N_1"], name="N_1", required=False),
            "N_2": scalar_or_compact_series(group["N_2"], name="N_2", required=False),
            "delta_ms_1": scalar_or_compact_series(group["delta_ms_1"], name="delta_ms_1", required=False),
            "Delta_app_ms_1": scalar_or_compact_series(group["Delta_app_ms_1"], name="Delta_app_ms_1", required=False),
            "delta_ms_2": scalar_or_compact_series(group["delta_ms_2"], name="delta_ms_2", required=False),
            "Delta_app_ms_2": scalar_or_compact_series(group["Delta_app_ms_2"], name="Delta_app_ms_2", required=False),
            "fit_kind": "ogse_contrast",
            "model": "mixed_global",
            "ycol": ycol,
            "stat": key_dict.get("stat", stat_keep),
            "n_points": int(len(y)),
            "n_fit": int(len(y)),
            "M0": float(values.get("M0", np.nan)),
            "M0_err": np.nan if not M0_vary else errors.get("M0_err", np.nan),
            "D0_m2_ms": float(values.get("D0_m2_ms", np.nan)),
            "D0_err_m2_ms": np.nan if not D0_vary else errors.get("D0_err_m2_ms", np.nan),
            "tc_ms": float(values.get("tc_ms", np.nan)),
            "tc_err_ms": errors.get("tc_err_ms", np.nan),
            "alpha": float(np.nanmean(alpha)),
            "alpha_min": float(np.nanmin(alpha)),
            "alpha_max": float(np.nanmax(alpha)),
            "alpha_source": scalar_or_compact_series(group["__alpha_source__"], name="alpha_source", required=False),
            "rmse": rmse,
            "chi2": chi2,
            "gbase": scalar_or_compact_series(group["gbase"], name="gbase", required=False),
            "fit_xcol": scalar_or_compact_series(group["fit_xcol"], name="fit_xcol", required=False),
            "plot_xcol": scalar_or_compact_series(group["plot_xcol"], name="plot_xcol", required=False),
            "xplot": scalar_or_compact_series(group["xplot"], name="xplot", required=False),
            "f_corr": np.nan,
            "f_corr_1": scalar_or_compact_series(group["__f_corr_1__"], name="f_corr_1", required=False),
            "f_corr_2": scalar_or_compact_series(group["__f_corr_2__"], name="f_corr_2", required=False),
            "tc_bound_min": float(tc_bounds[0]),
            "tc_bound_max": float(tc_bounds[1]),
            "method": method,
            "ok": ok,
            "msg": msg,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    return standardize_fit_params(out, fit_kind="ogse_contrast", source_file=source_file)


def fit_ogse_contrast_long(
    df: pd.DataFrame,
    *,
    model: str = "free",
    gbase: str = "g_lin_max",
    plot_xcol: str | None = None,
    ycol: str = "value_norm",
    directions: list[str] | None = None,
    rois: list[str] | None = None,
    stat_keep: str | None = "avg",
    xplot: str = "1",
    n_fit: int | None = None,
    sort_by_x: bool = True,
    f_by_direction: dict[str, float | tuple[float, float]] | None = None,
    td_override_ms: float | None = None,
    td_tol_ms: float = 1e-3,
    M0_vary: bool = True,
    D0_vary: bool = True,
    C_vary: bool = True,
    M0_value: float = 1.0,
    D0_value: float = 1e-12,
    C_value: float = 0.0,
    source_file: str | None = None,
    analysis_id: str | None = None,
    tc_value: float = 5.0,
    tc_vary: bool = True,
    tc_bounds: tuple[float, float] | None = None,
    m0_bounds: tuple[float, float] | None = None,
    d0_bounds: tuple[float, float] | None = None,
    c_bounds: tuple[float, float] | None = None,
    peak_grid_n: int = 1000,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
    peak_g_max_mTm: float | None = None,
    peak_resample_gradient: bool = False,
    peak_resample_g_max_corr_mTm: float | None = None,
    oneg: bool = False,
) -> pd.DataFrame:
    """
    Fit a long-form contrast table.
      keys: roi, direction, b_step (and optional stat)
      y:    value or value_norm
      x:    gbase_1 and gbase_2

    Sequence-specific parameters are expected in the same dataframe with _1/_2 suffixes:
      N_1, N_2, td_ms_1/td_ms_2 (or max_dur_ms_1 + tm_ms_1), etc.
    """
    df = _normalize_keys(df, label="contrast_long")
    analysis_id = analysis_id or _analysis_id_from_source_file(source_file)

    if ycol not in df.columns:
        raise KeyError(f"contrast_long: missing ycol {ycol!r}. Columns={list(df.columns)}")
    y_eff = ycol

    if directions is not None and not (len(directions) == 1 and directions[0].upper() == "ALL"):
        directions = [str(x) for x in directions]
        df = df[df["direction"].isin(directions)].copy()

    if rois is not None and not (len(rois) == 1 and rois[0].upper() == "ALL"):
        df = df[df["roi"].astype(str).isin([str(r) for r in rois])].copy()

    if stat_keep is not None and "stat" in df.columns and str(stat_keep).upper() != "ALL":
        df = df[df["stat"].astype(str) == str(stat_keep)].copy()

    fit_axis = normalize_axis_base(gbase)
    plot_axis = _resolve_plot_axis(fit_axis=fit_axis, plot_axis=plot_xcol, xplot=xplot)
    _plot_axis_base, plot_side = split_axis_side(plot_axis)
    plot_side = 1 if plot_side is None else int(plot_side)

    _require_cols(df, ["N_1", "N_2"], label="contrast_long (N_1/N_2)")

    # group by ROI+direction (+stat if exists)
    group_cols = ["roi", "direction"] + (["stat"] if "stat" in df.columns else [])
    rows: list[FitRow] = []

    for key, gg in df.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))

        roi = str(key_dict["roi"])
        direction = str(key_dict["direction"])
        stat = str(key_dict["stat"]) if "stat" in key_dict else None

        # scalars (must be unique inside the group)
        def _get_float(col: str) -> float | None:
            if col not in gg.columns:
                return None
            v = _unique_scalar(gg[col], name=col, required=False)
            if v is None:
                return None
            vf = float(v)
            return vf if np.isfinite(vf) else None

        def _get_str(col: str) -> str | None:
            if col not in gg.columns:
                return None
            v = _unique_scalar(gg[col], name=col, required=False)
            if v is None:
                return None
            return str(v)

        def _get_bool(col: str) -> bool | None:
            if col not in gg.columns:
                return None
            if pd.Series(gg[col]).dropna().empty:
                return None
            return bool(truthy_series(gg[col]))

        one_g_per_sequence_1 = _get_bool("one_g_per_sequence_1")
        one_g_per_sequence_2 = _get_bool("one_g_per_sequence_2")
        allow_sequence_ranges = bool(oneg or one_g_per_sequence_1 or one_g_per_sequence_2)

        def _get_sequence(col: str) -> str | None:
            if col not in gg.columns:
                return None
            if allow_sequence_ranges:
                v = scalar_or_compact_series(gg[col], name=col, required=False)
            else:
                v = _unique_scalar(gg[col], name=col, required=False)
            if v is None:
                return None
            return str(v)

        n_1 = int(round(float(_unique_scalar(gg["N_1"], name="N_1", required=True))))
        n_2 = int(round(float(_unique_scalar(gg["N_2"], name="N_2", required=True))))

        max_dur_ms_1 = _get_float("max_dur_ms_1")
        tm_ms_1 = _get_float("tm_ms_1")
        td_ms_1 = _get_float("td_ms_1")
        delta_ms_1 = _get_float("delta_ms_1")
        Delta_app_ms_1 = _get_float("Delta_app_ms_1")
        Hz_1 = _get_float("Hz_1")
        TE_1 = _get_float("TE_1")
        TR_1 = _get_float("TR_1")
        bmax_1 = _get_float("bmax_1")
        protocol_1 = _get_str("protocol_1")
        sequence_1 = _get_sequence("sequence_1")
        sheet_1 = _get_str("sheet_1")

        max_dur_ms_2 = _get_float("max_dur_ms_2")
        tm_ms_2 = _get_float("tm_ms_2")
        td_ms_2 = _get_float("td_ms_2")
        delta_ms_2 = _get_float("delta_ms_2")
        Delta_app_ms_2 = _get_float("Delta_app_ms_2")
        Hz_2 = _get_float("Hz_2")
        TE_2 = _get_float("TE_2")
        TR_2 = _get_float("TR_2")
        bmax_2 = _get_float("bmax_2")
        protocol_2 = _get_str("protocol_2")
        sequence_2 = _get_sequence("sequence_2")
        sheet_2 = _get_str("sheet_2")

        # timing summary for fitting internals: prefer override > td_ms_1/2 > compute from side 1
        td_ms: float | None
        if td_override_ms is not None:
            td_ms = float(td_override_ms)
        else:
            if td_ms_1 is not None and td_ms_2 is not None:
                if abs(td_ms_1 - td_ms_2) > float(td_tol_ms):
                    raise ValueError(
                        f"td_ms_1 != td_ms_2 for roi={roi}, direction={direction}. "
                        f"td1={td_ms_1}, td2={td_ms_2}. If this is intentional, pass td_override_ms."
                    )
                td_ms = float(0.5 * (td_ms_1 + td_ms_2))
            elif td_ms_1 is not None:
                td_ms = float(td_ms_1)
            elif max_dur_ms_1 is not None and tm_ms_1 is not None:
                td_ms = float(2.0 * max_dur_ms_1 + tm_ms_1)
            else:
                td_ms = None

        sheet = canonical_sheet_name(sheet_1 or sheet_2)
        subj = _get_str("subj")
        if subj is None or not str(subj).strip():
            subj = infer_subj_label(sheet, source_name=source_file)

        f_corr_1, f_corr_2 = _coerce_correction_pair(f_by_direction.get(str(direction), 1.0) if f_by_direction else 1.0)

        fit_bundle_1 = build_axis_bundle(
            gg,
            axis=fit_axis,
            side=1,
            correction_factor=float(f_corr_1),
            gamma=GAMMA_DEFAULT,
            N=n_1,
            delta_ms=delta_ms_1,
            Delta_app_ms=Delta_app_ms_1,
        )
        fit_bundle_2 = build_axis_bundle(
            gg,
            axis=fit_axis,
            side=2,
            correction_factor=float(f_corr_2),
            gamma=GAMMA_DEFAULT,
            N=n_2,
            delta_ms=delta_ms_2,
            Delta_app_ms=Delta_app_ms_2,
        )
        plot_bundle = build_axis_bundle(
            gg,
            axis=plot_axis,
            side=plot_side,
            correction_factor=float(f_corr_1 if plot_side == 1 else f_corr_2),
            gamma=GAMMA_DEFAULT,
            N=n_1 if plot_side == 1 else n_2,
            delta_ms=delta_ms_1 if plot_side == 1 else delta_ms_2,
            Delta_app_ms=Delta_app_ms_1 if plot_side == 1 else Delta_app_ms_2,
        )

        y = pd.to_numeric(gg[y_eff], errors="coerce").to_numpy(dtype=float)
        G1 = fit_bundle_1.gradient_corr
        G2 = fit_bundle_2.gradient_corr
        x = plot_bundle.axis_corr

        m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2) & np.isfinite(x)
        y, G1, G2, x = y[m], G1[m], G2[m], x[m]
        n_points = int(len(y))

        if td_ms is None or not np.isfinite(float(td_ms)) or n_points == 0:
            rows.append(
                FitRow(
                    analysis_id=str(analysis_id),
                    subj=str(subj),
                    sheet=sheet,
                    roi=roi,
                    direction=direction,
                    stat=stat,
                    td_ms=None if td_ms is None else float(td_ms),
                    max_dur_ms_1=max_dur_ms_1,
                    tm_ms_1=tm_ms_1,
                    td_ms_1=td_ms_1,
                    N_1=n_1,
                    delta_ms_1=delta_ms_1,
                    Delta_app_ms_1=Delta_app_ms_1,
                    Hz_1=Hz_1,
                    TE_1=TE_1,
                    TR_1=TR_1,
                    bmax_1=bmax_1,
                    protocol_1=protocol_1,
                    sequence_1=sequence_1,
                    one_g_per_sequence_1=one_g_per_sequence_1,
                    sheet_1=sheet_1,
                    max_dur_ms_2=max_dur_ms_2,
                    tm_ms_2=tm_ms_2,
                    td_ms_2=td_ms_2,
                    N_2=n_2,
                    delta_ms_2=delta_ms_2,
                    Delta_app_ms_2=Delta_app_ms_2,
                    Hz_2=Hz_2,
                    TE_2=TE_2,
                    TR_2=TR_2,
                    bmax_2=bmax_2,
                    protocol_2=protocol_2,
                    sequence_2=sequence_2,
                    one_g_per_sequence_2=one_g_per_sequence_2,
                    sheet_2=sheet_2,
                    model=model,
                    ycol=ycol,
                    gbase=_normalize_gbase(fit_axis),
                    fit_xcol=str(fit_axis),
                    plot_xcol=str(plot_axis),
                    xplot=str(plot_side),
                    n_points=n_points,
                    n_fit=n_points,
                    f_corr=np.nan,
                    f_corr_1=1.0,
                    f_corr_2=1.0,
                    ok=False,
                    msg="Empty group or td_ms could not be inferred.",
                )
            )
            continue

        if sort_by_x:
            order = np.argsort(x)
            y, G1, G2, x = y[order], G1[order], G2[order], x[order]

        if n_fit is not None:
            k = int(n_fit)
            y, G1, G2, x = y[:k], G1[:k], G2[:k], x[:k]

        n_fit_used = int(len(y))
        td = float(td_ms)

        base = dict(
            analysis_id=str(analysis_id),
            subj=str(subj),
            sheet=sheet,
            roi=roi,
            direction=direction,
            stat=stat,
            td_ms=float(td),
            max_dur_ms_1=max_dur_ms_1,
            tm_ms_1=tm_ms_1,
            td_ms_1=td_ms_1,
            N_1=n_1,
            delta_ms_1=delta_ms_1,
            Delta_app_ms_1=Delta_app_ms_1,
            Hz_1=Hz_1,
            TE_1=TE_1,
            TR_1=TR_1,
            bmax_1=bmax_1,
            protocol_1=protocol_1,
            sequence_1=sequence_1,
            one_g_per_sequence_1=one_g_per_sequence_1,
            sheet_1=sheet_1,
            max_dur_ms_2=max_dur_ms_2,
            tm_ms_2=tm_ms_2,
            td_ms_2=td_ms_2,
            N_2=n_2,
            delta_ms_2=delta_ms_2,
            Delta_app_ms_2=Delta_app_ms_2,
            Hz_2=Hz_2,
            TE_2=TE_2,
            TR_2=TR_2,
            bmax_2=bmax_2,
            protocol_2=protocol_2,
            sequence_2=sequence_2,
            one_g_per_sequence_2=one_g_per_sequence_2,
            sheet_2=sheet_2,
            model=model,
            ycol=ycol,
            gbase=_normalize_gbase(fit_axis),
            fit_xcol=str(fit_axis),
            plot_xcol=str(plot_axis),
            xplot=str(plot_side),
            n_points=n_points,
            n_fit=n_fit_used,
            f_corr=np.nan,
            f_corr_1=f_corr_1,
            f_corr_2=f_corr_2,
        )

        try:
            fit_values = _fit_selected_contrast_model(
                model,
                td,
                G1,
                G2,
                n_1,
                n_2,
                y,
                M0_vary=M0_vary,
                D0_vary=D0_vary,
                C_vary=C_vary,
                M0_value=M0_value,
                D0_value=D0_value,
                C_value=C_value,
                tc_value=tc_value,
                tc_vary=tc_vary,
                tc_bounds=tc_bounds,
                m0_bounds=m0_bounds,
                d0_bounds=d0_bounds,
                c_bounds=c_bounds,
            )
            peak_metrics = _compute_peak_metrics(
                model=model,
                td_ms=td,
                n_1=n_1,
                n_2=n_2,
                fit_row=fit_values,
                g1_max_corr=float(np.nanmax(G1)),
                g2_max_corr=float(np.nanmax(G2)),
                f_corr_1=float(f_corr_1),
                f_corr_2=float(f_corr_2),
                xplot=str(plot_side),
                peak_grid_n=int(peak_grid_n),
                peak_D0_fix=float(peak_D0_fix),
                peak_gamma=float(peak_gamma),
                peak_g_max_mTm=peak_g_max_mTm,
                peak_resample_gradient=bool(peak_resample_gradient),
                peak_resample_g_max_corr_mTm=peak_resample_g_max_corr_mTm,
            )
            rows.append(FitRow(**base, **peak_metrics, **fit_values))
        except Exception as e:
            rows.append(FitRow(**base, ok=False, msg=str(e)))

    out = pd.DataFrame([r.__dict__ for r in rows])

    # Keep the standardized downstream fit-parameter schema.
    out = standardize_fit_params(out, fit_kind="ogse_contrast", source_file=source_file)
    return out


def plot_fit_one_group(
    df_group: pd.DataFrame,
    fit_row: dict,
    *,
    out_png: Path,
    gbase: str,
    ycol: str,
) -> None:
    df_group = _normalize_keys(df_group, label="plot_group")

    if ycol not in df_group.columns:
        raise KeyError(f"plot_group: missing ycol {ycol!r}. Columns={list(df_group.columns)}")
    y = pd.to_numeric(df_group[ycol], errors="coerce").to_numpy(dtype=float).copy()
    fit_axis = str(fit_row.get("fit_xcol", gbase))
    fallback_side = 2 if str(fit_row.get("xplot", "1")) == "2" else 1
    plot_axis = str(fit_row.get("plot_xcol", default_plot_axis_for_fit(fit_axis, side=fallback_side)))
    _plot_axis_base, plot_side = split_axis_side(plot_axis)
    plot_side = 1 if plot_side is None else int(plot_side)

    td_val = fit_row.get("td_ms")
    if td_val is None:
        td_val = fit_row.get("td_ms_1") if fit_row.get("td_ms_1") is not None else fit_row.get("td_ms_2")
    td = float(td_val) if td_val is not None else 0.0
    n_1 = int(fit_row.get("N_1"))
    n_2 = int(fit_row.get("N_2"))
    model = str(fit_row.get("model", "free"))
    delta_ms_1 = fit_row.get("delta_ms_1")
    Delta_app_ms_1 = fit_row.get("Delta_app_ms_1")
    delta_ms_2 = fit_row.get("delta_ms_2")
    Delta_app_ms_2 = fit_row.get("Delta_app_ms_2")
    f_corr_1, f_corr_2 = _fit_row_correction_pair(fit_row)

    fit_bundle_1 = build_axis_bundle(
        df_group,
        axis=fit_axis,
        side=1,
        correction_factor=float(f_corr_1),
        gamma=GAMMA_DEFAULT,
        N=float(n_1),
        delta_ms=None if pd.isna(delta_ms_1) else float(delta_ms_1),
        Delta_app_ms=None if pd.isna(Delta_app_ms_1) else float(Delta_app_ms_1),
    )
    fit_bundle_2 = build_axis_bundle(
        df_group,
        axis=fit_axis,
        side=2,
        correction_factor=float(f_corr_2),
        gamma=GAMMA_DEFAULT,
        N=float(n_2),
        delta_ms=None if pd.isna(delta_ms_2) else float(delta_ms_2),
        Delta_app_ms=None if pd.isna(Delta_app_ms_2) else float(Delta_app_ms_2),
    )
    plot_bundle = build_axis_bundle(
        df_group,
        axis=plot_axis,
        side=plot_side,
        correction_factor=float(f_corr_1 if plot_side == 1 else f_corr_2),
        gamma=GAMMA_DEFAULT,
        N=float(n_1 if plot_side == 1 else n_2),
        delta_ms=None if pd.isna(delta_ms_1 if plot_side == 1 else delta_ms_2) else float(delta_ms_1 if plot_side == 1 else delta_ms_2),
        Delta_app_ms=None if pd.isna(Delta_app_ms_1 if plot_side == 1 else Delta_app_ms_2) else float(Delta_app_ms_1 if plot_side == 1 else Delta_app_ms_2),
    )

    G1 = fit_bundle_1.gradient_corr
    G2 = fit_bundle_2.gradient_corr
    x = plot_bundle.axis_corr
    m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2) & np.isfinite(x)
    y, G1, G2, x = y[m], G1[m], G2[m], x[m]

    G1max = float(np.nanmax(G1)) if len(G1) else 0.0
    G2max = float(np.nanmax(G2)) if len(G2) else 0.0
    f = np.linspace(0, 1, 250)
    G1s = f * G1max
    G2s = f * G2max

    ys = None

    if model == "free" and bool(fit_row.get("ok", True)):
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_free(td, G1s, G2s, n_1, n_2, M0, D0)

    if model == "tort" and bool(fit_row.get("ok", True)):
        alpha = float(fit_row["alpha"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_tort(td, G1s, G2s, n_1, n_2, alpha, M0, D0)

    if model == "rest" and bool(fit_row.get("ok", True)):
        tc = float(fit_row["tc_ms"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_rest(td, G1s, G2s, n_1, n_2, tc, M0, D0)

    if model in {"rest_offset", "rest_offset_globC"} and bool(fit_row.get("ok", True)):
        tc = float(fit_row["tc_ms"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        C = float(fit_row.get("C", 0.0))
        ys = OGSE_contrast_vs_g_rest_offset(td, G1s, G2s, n_1, n_2, tc, M0, D0, C)

    if model in {"mixed", "mixed_global"} and bool(fit_row.get("ok", True)):
        tc = float(fit_row["tc_ms"])
        alpha = float(fit_row["alpha"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_mixed(td, G1s, G2s, n_1, n_2, tc, alpha, M0, D0)

    plot_contrast_fit(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        fit_x=(
            axis_from_gradient(
                np.asarray(G1s if plot_side == 1 else G2s, dtype=float),
                axis=plot_axis,
                N=float(n_1 if plot_side == 1 else n_2),
                gamma=GAMMA_DEFAULT,
                delta_ms=None if pd.isna(delta_ms_1 if plot_side == 1 else delta_ms_2) else float(delta_ms_1 if plot_side == 1 else delta_ms_2),
                Delta_app_ms=None if pd.isna(Delta_app_ms_1 if plot_side == 1 else Delta_app_ms_2) else float(Delta_app_ms_1 if plot_side == 1 else Delta_app_ms_2),
            )
            if ys is not None
            else None
        ),
        fit_y=np.asarray(ys, dtype=float) if ys is not None else None,
        fit_row=fit_row,
        out_png=out_png,
        x_label=str(plot_axis),
        ycol=ycol,
        family_label=FAMILY_LABEL,
    )
