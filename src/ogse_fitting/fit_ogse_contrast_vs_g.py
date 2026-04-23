from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fitting.b_from_g import (
    axes_share_gradient_family,
    axis_from_gradient,
    build_axis_bundle,
    default_plot_axis_for_fit,
    gradient_base_for_axis,
    normalize_axis_base,
    split_axis_side,
)
from fitting.core import CurveFitParameter
from fitting.core import chi2 as _chi2
from fitting.core import fit_curve_fit_parameters
from fitting.core import rmse as _rmse
from fitting.core import stderr_from_single_param_jacobian as _stderr_from_single_param_jacobian
from ogse_plotting.plot_ogse_contrast_vs_g import FAMILY_LABEL, plot_contrast_fit
from models.model_fitting import OGSE_contrast_vs_g_free, OGSE_contrast_vs_g_tort, OGSE_contrast_vs_g_rest
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.fit_params_schema import standardize_fit_params
from tools.strict_columns import raise_on_unrecognized_column_names
from tools.value_formatting import scalar_or_compact_series, truthy_series


# -----------------------------
# Helpers: strict schema
# -----------------------------
KEY_COLS = ("roi", "direction", "b_step")
VALID_GBASES = {"g", "g_max", "g_lin_max", "g_thorsten"}
GAMMA_DEFAULT = 267.5221900


def _require_cols(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns {missing}. Columns={list(df.columns)}")


def _normalize_keys(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """
    Strict schema: keep direction as string and b_step as int.
    This function does not rename columns; it only enforces stable types for grouping and plotting.
    """
    out = df.copy()
    raise_on_unrecognized_column_names(out.columns, context=label)

    _require_cols(out, KEY_COLS, label=label)

    out["direction"] = out["direction"].astype(str)

    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["roi", "direction", "b_step"]].head(10)
        raise ValueError(f"{label}: b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)

    # Keep stat as a string when present.
    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)

    return out


def _unique_scalar(series: pd.Series, *, name: str, required: bool = False) -> Any:
    u = series.dropna().unique()
    if len(u) == 0:
        if required:
            raise ValueError(f"Could not infer '{name}': column is empty or all-NaN.")
        return None
    if len(u) > 1:
        raise ValueError(f"'{name}' is not unique within the group. Values={u.tolist()[:10]}")
    return u[0]


def _normalize_gbase(gbase: str) -> str:
    return gradient_base_for_axis(gbase)


def _gcols(gbase: str) -> tuple[str, str]:
    b = _normalize_gbase(gbase)
    return f"{b}_1", f"{b}_2"


def _maybe_scale_g_thorsten(gbase: str, arr: np.ndarray) -> np.ndarray:
    """
    Keep the thorsten scaling convention: sqrt(2) * abs(g_thorsten).
    """
    b = _normalize_gbase(gbase)
    if b == "g_thorsten":
        return np.sqrt(2.0) * np.abs(arr)
    return arr


def _coerce_correction_pair(value: Any) -> tuple[float, float]:
    if value is None:
        return 1.0, 1.0

    if isinstance(value, (tuple, list, np.ndarray, pd.Series)) and len(value) >= 2:
        f1 = float(value[0])
        f2 = float(value[1])
    else:
        f1 = float(value)
        f2 = f1

    if not np.isfinite(f1) or f1 <= 0:
        f1 = 1.0
    if not np.isfinite(f2) or f2 <= 0:
        f2 = 1.0
    return f1, f2


def _fit_row_correction_pair(fit_row: dict[str, Any] | pd.Series) -> tuple[float, float]:
    f1 = fit_row.get("f_corr_1", np.nan)
    f2 = fit_row.get("f_corr_2", np.nan)
    if pd.notna(f1) and pd.notna(f2):
        return _coerce_correction_pair((f1, f2))
    return 1.0, 1.0


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


def _analysis_id_from_source_file(source_file: str | None) -> str:
    if not source_file:
        return ""
    stem = Path(str(source_file)).stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


@dataclass(frozen=True)
class ContrastModelSpec:
    name: str
    evaluator: Callable[[float, np.ndarray, np.ndarray, int, int, dict[str, Any]], np.ndarray]
    maxfev: int


def _eval_free(
    td_ms: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_free(td_ms, G1, G2, n_1, n_2, float(params["M0"]), float(params["D0_m2_ms"]))


def _eval_tort(
    td_ms: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_tort(
        td_ms,
        G1,
        G2,
        n_1,
        n_2,
        float(params["alpha"]),
        float(params["M0"]),
        float(params["D0_m2_ms"]),
    )


def _eval_rest(
    td_ms: float,
    G1: np.ndarray,
    G2: np.ndarray,
    n_1: int,
    n_2: int,
    params: dict[str, Any],
) -> np.ndarray:
    return OGSE_contrast_vs_g_rest(
        td_ms,
        G1,
        G2,
        n_1,
        n_2,
        float(params["tc_ms"]),
        float(params["M0"]),
        float(params["D0_m2_ms"]),
    )


CONTRAST_MODEL_SPECS: dict[str, ContrastModelSpec] = {
    "free": ContrastModelSpec(name="free", evaluator=_eval_free, maxfev=400000),
    "tort": ContrastModelSpec(name="tort", evaluator=_eval_tort, maxfev=600000),
    "rest": ContrastModelSpec(name="rest", evaluator=_eval_rest, maxfev=600000),
}


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


def _tc_peak_from_notebook_formula(
    *,
    td_ms: float,
    g_peak_raw_mTpm: float,
    D0_fix_m2_ms: float,
    gamma_rad_ms_mT: float,
) -> tuple[float, float, float, float]:
    g_raw = float(g_peak_raw_mTpm)
    if not np.isfinite(g_raw) or g_raw <= 0:
        return (np.nan, np.nan, np.nan, np.nan)

    D0 = float(D0_fix_m2_ms)
    gamma = float(gamma_rad_ms_mT)
    td = float(td_ms)

    l_G = ((2.0 ** (3.0 / 2.0)) * D0 / (gamma * g_raw)) ** (1.0 / 3.0)
    l_d = np.sqrt(2.0 * D0 * td)
    L_d = l_d / l_G
    L_cf = ((3.0 / 2.0) ** (1.0 / 4.0)) * (L_d ** (-1.0 / 2.0))
    lcf = L_cf * l_G
    tc_peak_ms = (lcf**2) / (2.0 * D0)
    return (float(l_G), float(L_cf), float(lcf), float(tc_peak_ms))


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
) -> dict[str, float | str]:
    if not np.isfinite(g1_max_corr) or not np.isfinite(g2_max_corr):
        return {}
    if g1_max_corr <= 0 or g2_max_corr <= 0:
        return {}

    n_grid = max(32, int(peak_grid_n))
    frac = np.linspace(0.0, 1.0, n_grid)
    G1 = frac * float(g1_max_corr)
    G2 = frac * float(g2_max_corr)
    y = _model_yhat(model=model, td_ms=td_ms, G1=G1, G2=G2, n_1=n_1, n_2=n_2, fit_row=fit_row)
    if y.size == 0 or not np.isfinite(y).any():
        return {}

    i_peak = int(np.nanargmax(y))
    f_peak = float(frac[i_peak])
    g1_peak_corr = float(G1[i_peak])
    g2_peak_corr = float(G2[i_peak])
    y_peak = float(y[i_peak])

    f1_val = f_corr_1 if np.isfinite(f_corr_1) and f_corr_1 != 0.0 else np.nan
    f2_val = f_corr_2 if np.isfinite(f_corr_2) and f_corr_2 != 0.0 else np.nan
    g1_peak_raw = float(g1_peak_corr / f1_val) if np.isfinite(f1_val) else np.nan
    g2_peak_raw = float(g2_peak_corr / f2_val) if np.isfinite(f2_val) else np.nan
    g1_max_raw = float(g1_max_corr / f1_val) if np.isfinite(f1_val) else np.nan
    g2_max_raw = float(g2_max_corr / f2_val) if np.isfinite(f2_val) else np.nan

    x_is_1 = str(xplot) == "1"
    x_peak_corr = g1_peak_corr if x_is_1 else g2_peak_corr
    x_peak_raw = g1_peak_raw if x_is_1 else g2_peak_raw

    l_G, L_cf, lcf_peak_m, tc_peak_ms = _tc_peak_from_notebook_formula(
        td_ms=float(td_ms),
        g_peak_raw_mTpm=float(x_peak_raw),
        D0_fix_m2_ms=float(peak_D0_fix),
        gamma_rad_ms_mT=float(peak_gamma),
    )

    return {
        "peak_method": "param_grid",
        "peak_grid_n": int(n_grid),
        "g1_max_raw_mTm": g1_max_raw,
        "g2_max_raw_mTm": g2_max_raw,
        "g1_max_corr_mTm": float(g1_max_corr),
        "g2_max_corr_mTm": float(g2_max_corr),
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
) -> tuple[float, float, float, float, str, float | None, float | None]:
    # Keep D0 bounds centered around the user-provided seed.
    D_lo, D_hi = float(D0_value / 10.0), float(D0_value * 10.0)

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
            CurveFitParameter("M0", float(M0_value), 0.0, 2.0, bool(M0_vary)),
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
):
    D_lo, D_hi = float(D0_value / 10.0), float(D0_value * 10.0)
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
            CurveFitParameter("M0", float(M0_value), 0.0, 5.0, bool(M0_vary)),
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
) -> float:
    candidates = np.unique(
        np.concatenate(
            [
                np.array([float(tc_default)], dtype=float),
                np.logspace(np.log10(0.1), np.log10(1000.0), 96),
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
):
    D_lo, D_hi = float(D0_value / 100.0), float(D0_value * 100.0)
    tc_lo, tc_hi = 0.1, 1000.0
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
            CurveFitParameter("tc_ms", float(tc_seed), tc_lo, tc_hi, True),
            CurveFitParameter("M0", float(M0_value), 0.0, 5.0, bool(M0_vary)),
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
        float(fit.errors.get("tc_err_ms", np.nan)),
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
    M0_value: float,
    D0_value: float,
    tc_value: float,
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

    raise ValueError(f"Model {model!r} is not implemented.")


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
    peak_fraction: float | None = None
    g1_peak_raw_mTm: float | None = None
    g2_peak_raw_mTm: float | None = None
    g1_peak_corr_mTm: float | None = None
    g2_peak_corr_mTm: float | None = None
    x_peak_raw_mTm: float | None = None
    x_peak_corr_mTm: float | None = None
    signal_peak: float | None = None
    l_G_peak_m: float | None = None
    L_cf_peak: float | None = None
    lcf_peak_m: float | None = None
    tc_peak_ms: float | None = None

    # fitted
    M0: float | None = None
    M0_err: float | None = None
    D0_m2_ms: float | None = None
    D0_err_m2_ms: float | None = None
    alpha: float | None = None
    alpha_err: float | None = None
    tc_ms: float | None = None
    tc_err_ms: float | None = None

    # metrics
    rmse: float | None = None
    chi2: float | None = None
    method: str | None = None
    ok: bool = True
    msg: str = ""


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
    M0_value: float = 1.0,
    D0_value: float = 1e-12,
    source_file: str | None = None,
    analysis_id: str | None = None,
    tc_value: float = 5.0,
    tc_vary: bool = True,
    peak_grid_n: int = 1000,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
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
                        f"td_ms_1 != td_ms_2 para roi={roi}, direction={direction}. "
                        f"td1={td_ms_1}, td2={td_ms_2}. Si es intencional, pasá td_override_ms."
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
                M0_value=M0_value,
                D0_value=D0_value,
                tc_value=tc_value,
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
