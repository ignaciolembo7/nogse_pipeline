from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fitting.core import CurveFitParameter
from fitting.core import chi2 as _chi2
from fitting.core import fit_curve_fit_parameters
from fitting.core import rmse as _rmse
from fitting.core import stderr_from_single_param_jacobian as _stderr_from_single_param_jacobian
from nogse_models.nogse_model_fitting import (
    NOGSE_contrast_vs_g_free,
    NOGSE_contrast_vs_g_rest,
    NOGSE_contrast_vs_g_tort,
)
from nogse_plotting.plot_nogse_contrast_vs_g import plot_nogse_contrast_fit
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.fit_params_schema import standardize_fit_params
from tools.strict_columns import raise_on_unrecognized_column_names
from tools.value_formatting import scalar_or_compact_series, truthy_series


KEY_COLS = ("roi", "direction", "b_step")
VALID_GBASES = {"g", "g_max", "g_lin_max", "g_thorsten"}


def _require_cols(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns {missing}. Columns={list(df.columns)}")


def _normalize_keys(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    out = df.copy()
    raise_on_unrecognized_column_names(out.columns, context=label)
    _require_cols(out, KEY_COLS, label=label)
    out["direction"] = out["direction"].astype(str)
    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)

    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["roi", "direction", "b_step"]].head(10)
        raise ValueError(f"{label}: b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)
    return out


def _unique_scalar(series: pd.Series, *, name: str, required: bool = False) -> Any:
    u = pd.Series(series).dropna().unique()
    if len(u) == 0:
        if required:
            raise ValueError(f"Could not infer '{name}': column is empty or all-NaN.")
        return None
    if len(u) > 1:
        raise ValueError(f"'{name}' is not unique within the group. Values={u.tolist()[:10]}")
    return u[0]


def _analysis_id_from_source_file(source_file: str | None) -> str:
    if not source_file:
        return ""
    stem = Path(str(source_file)).stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _normalize_gbase(gbase: str) -> str:
    b = str(gbase).strip()
    if b.endswith("_1") or b.endswith("_2"):
        b = b[:-2]
    if b not in VALID_GBASES:
        raise ValueError(f"fit_nogse_contrast_vs_g: unrecognized gbase {gbase!r}. Allowed values: {sorted(VALID_GBASES)}.")
    return b


def _gcol(df: pd.DataFrame, gbase: str, *, side: int = 1) -> str:
    b = _normalize_gbase(gbase)
    side_col = f"{b}_{int(side)}"
    if side_col in df.columns:
        return side_col
    if b in df.columns:
        return b
    raise KeyError(f"contrast_long: missing gradient column {side_col!r} or {b!r}. Columns={list(df.columns)}")


def _maybe_scale_g_thorsten(gbase: str, arr: np.ndarray) -> np.ndarray:
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


@dataclass(frozen=True)
class ContrastModelSpec:
    name: str
    evaluator: Callable[[float, np.ndarray, int, dict[str, Any]], np.ndarray]
    maxfev: int


def _eval_free(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, Any]) -> np.ndarray:
    return NOGSE_contrast_vs_g_free(td_ms, G, n_value, float(params["M0"]), float(params["D0_m2_ms"]))


def _eval_tort(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, Any]) -> np.ndarray:
    return NOGSE_contrast_vs_g_tort(
        td_ms,
        G,
        n_value,
        float(params["alpha"]),
        float(params["M0"]),
        float(params["D0_m2_ms"]),
    )


def _eval_rest(td_ms: float, G: np.ndarray, n_value: int, params: dict[str, Any]) -> np.ndarray:
    return NOGSE_contrast_vs_g_rest(
        td_ms,
        G,
        n_value,
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
    G: np.ndarray,
    n_value: int,
    y: np.ndarray,
    *,
    parameters: Sequence[CurveFitParameter],
):
    def model_from_params(params: dict[str, float]) -> np.ndarray:
        return spec.evaluator(td_ms, G, n_value, params)

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
    G: np.ndarray,
    n_value: int,
    fit_row: dict[str, Any],
) -> np.ndarray:
    spec = CONTRAST_MODEL_SPECS.get(str(model))
    if spec is None:
        raise ValueError(f"Unsupported model {model!r} for curve evaluation.")
    return spec.evaluator(td_ms, G, n_value, fit_row)


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
    n_value: int,
    fit_row: dict[str, Any],
    g_max_corr: float,
    f_corr_1: float,
    f_corr_2: float,
    peak_grid_n: int,
    peak_D0_fix: float,
    peak_gamma: float,
    g2_max_raw: float | None = None,
    g2_max_corr: float | None = None,
) -> dict[str, float | str]:
    if not np.isfinite(g_max_corr) or g_max_corr <= 0:
        return {}

    n_grid = max(32, int(peak_grid_n))
    frac = np.linspace(0.0, 1.0, n_grid)
    G = frac * float(g_max_corr)
    y = _model_yhat(model=model, td_ms=td_ms, G=G, n_value=n_value, fit_row=fit_row)
    if y.size == 0 or not np.isfinite(y).any():
        return {}

    i_peak = int(np.nanargmax(y))
    f_peak = float(frac[i_peak])
    g_peak_corr = float(G[i_peak])
    y_peak = float(y[i_peak])

    f1_val = f_corr_1 if np.isfinite(f_corr_1) and f_corr_1 != 0.0 else np.nan
    f2_val = f_corr_2 if np.isfinite(f_corr_2) and f_corr_2 != 0.0 else np.nan
    g_peak_raw = float(g_peak_corr / f1_val) if np.isfinite(f1_val) else np.nan
    g_max_raw = float(g_max_corr / f1_val) if np.isfinite(f1_val) else np.nan

    l_G, L_cf, lcf_peak_m, tc_peak_ms = _tc_peak_from_notebook_formula(
        td_ms=float(td_ms),
        g_peak_raw_mTpm=float(g_peak_raw),
        D0_fix_m2_ms=float(peak_D0_fix),
        gamma_rad_ms_mT=float(peak_gamma),
    )

    g2_raw = (
        float(g2_max_corr / f2_val)
        if g2_max_corr is not None and np.isfinite(float(g2_max_corr)) and np.isfinite(f2_val)
        else float(g2_max_raw)
        if g2_max_raw is not None and np.isfinite(float(g2_max_raw))
        else np.nan
    )
    g2_corr = float(g2_max_corr) if g2_max_corr is not None and np.isfinite(float(g2_max_corr)) else np.nan

    return {
        "peak_method": "param_grid",
        "peak_grid_n": int(n_grid),
        "g1_max_raw_mTm": g_max_raw,
        "g2_max_raw_mTm": g2_raw,
        "g1_max_corr_mTm": float(g_max_corr),
        "g2_max_corr_mTm": g2_corr,
        "peak_fraction": f_peak,
        "g1_peak_raw_mTm": g_peak_raw,
        "g2_peak_raw_mTm": np.nan,
        "g1_peak_corr_mTm": g_peak_corr,
        "g2_peak_corr_mTm": np.nan,
        "x_peak_raw_mTm": float(g_peak_raw),
        "x_peak_corr_mTm": float(g_peak_corr),
        "signal_peak": y_peak,
        "l_G_peak_m": l_G,
        "L_cf_peak": L_cf,
        "lcf_peak_m": lcf_peak_m,
        "tc_peak_ms": tc_peak_ms,
    }


def _fit_free(
    td: float,
    G: np.ndarray,
    n_value: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
) -> tuple[float, float, float, float, str, float | None, float | None]:
    D_lo, D_hi = float(D0_value / 10.0), float(D0_value * 10.0)

    if (not M0_vary) and D0_vary:
        def f_log(log_D0: float) -> float:
            D0 = float(np.exp(log_D0))
            yhat = NOGSE_contrast_vs_g_free(td, G, n_value, float(M0_value), D0)
            if yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
                return np.inf
            return float(np.sum((y - yhat) ** 2))

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

        D0 = float(np.exp(best_log))
        yhat = NOGSE_contrast_vs_g_free(td, G, n_value, float(M0_value), D0)

        d0_step = max(abs(D0) * 1e-6, abs(float(D0_value)) * 1e-6, 1e-18)
        d0_lo = max(float(D_lo), float(D0 - d0_step))
        d0_hi = min(float(D_hi), float(D0 + d0_step))
        D0_err = None
        if d0_hi > d0_lo:
            y_lo = NOGSE_contrast_vs_g_free(td, G, n_value, float(M0_value), float(d0_lo))
            y_hi = NOGSE_contrast_vs_g_free(td, G, n_value, float(M0_value), float(d0_hi))
            jac = (y_hi - y_lo) / float(d0_hi - d0_lo)
            D0_err = _stderr_from_single_param_jacobian(y, yhat, jac, n_params=1)

        return float(M0_value), D0, _rmse(y, yhat), _chi2(y, yhat), "logD_scalar_search", None, D0_err

    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["free"],
        td,
        G,
        n_value,
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
        float(fit.errors.get("M0_err", np.nan)) if M0_vary else None,
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
    )


def _fit_tort(
    td: float,
    G: np.ndarray,
    n_value: int,
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
        G,
        n_value,
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
    G: np.ndarray,
    n_value: int,
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
        yhat = NOGSE_contrast_vs_g_rest(td, G, n_value, float(tc), float(M0_guess), float(D0_guess))
        if not np.all(np.isfinite(yhat)):
            continue
        err = _rmse(y, yhat)
        if np.isfinite(err) and err < best_rmse:
            best_rmse = float(err)
            best_tc = float(tc)
    return float(best_tc)


def _fit_rest(
    td: float,
    G: np.ndarray,
    n_value: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
    tc_value: float,
    tc_vary: bool,
):
    D_lo, D_hi = float(D0_value / 100.0), float(D0_value * 100.0)
    tc_lo, tc_hi = 0.1, 1000.0
    tc_seed = _best_tc_seed_rest(
        td,
        G,
        n_value,
        y,
        M0_guess=float(M0_value),
        D0_guess=float(D0_value),
        tc_default=float(tc_value),
    )
    fit = _fit_contrast_model(
        CONTRAST_MODEL_SPECS["rest"],
        td,
        G,
        n_value,
        y,
        parameters=[
            CurveFitParameter("tc_ms", float(tc_seed), tc_lo, tc_hi, bool(tc_vary)),
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
        float(fit.errors.get("tc_err_ms", np.nan)) if tc_vary else None,
        float(fit.errors.get("M0_err", np.nan)) if M0_vary else None,
        float(fit.errors.get("D0_err_m2_ms", np.nan)) if D0_vary else None,
    )


def _fit_selected_contrast_model(
    model: str,
    td: float,
    G: np.ndarray,
    n_value: int,
    y: np.ndarray,
    *,
    M0_vary: bool,
    D0_vary: bool,
    M0_value: float,
    D0_value: float,
    tc_value: float,
    tc_vary: bool,
) -> dict[str, float | str | None]:
    if model == "free":
        M0, D0, rmse, chi2, method, M0_err, D0_err = _fit_free(
            td,
            G,
            n_value,
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
            G,
            n_value,
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
            G,
            n_value,
            y,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            tc_value=tc_value,
            tc_vary=tc_vary,
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


FitRow = dict[str, Any]


def fit_nogse_contrast_long(
    df: pd.DataFrame,
    *,
    model: str = "free",
    gbase: str = "g_lin_max",
    ycol: str = "value_norm",
    directions: list[str] | None = None,
    rois: list[str] | None = None,
    stat_keep: str | None = "avg",
    n_fit: int | None = None,
    sort_by_x: bool = True,
    f_by_direction: dict[str, float | tuple[float, float]] | None = None,
    td_override_ms: float | None = None,
    M0_vary: bool = True,
    D0_vary: bool = True,
    M0_value: float = 1.0,
    D0_value: float = 2.3e-12,
    source_file: str | None = None,
    analysis_id: str | None = None,
    tc_value: float = 5.0,
    tc_vary: bool = True,
    peak_grid_n: int = 1000,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
    oneg: bool = False,
    **_: Any,
) -> pd.DataFrame:
    """
    Fit a NOGSE contrast table produced by make_contrast.py.

    The contrast is CPMG - HAHN, so the NOGSE model uses side 1 as the
    experimental NOGSE axis: gbase_1, N_1, and td_ms_1/TN_1.
    Side 2 metadata is preserved in the output for traceability.
    """
    if model not in CONTRAST_MODEL_SPECS:
        raise ValueError(f"Unsupported NOGSE contrast model {model!r}. Allowed values: {sorted(CONTRAST_MODEL_SPECS)}.")

    df = _normalize_keys(df, label="nogse_contrast_long")
    analysis_id = analysis_id or _analysis_id_from_source_file(source_file)

    if ycol not in df.columns:
        raise KeyError(f"nogse_contrast_long: missing ycol {ycol!r}. Columns={list(df.columns)}")

    if directions is not None and not (len(directions) == 1 and str(directions[0]).upper() == "ALL"):
        directions = [str(x) for x in directions]
        df = df[df["direction"].isin(directions)].copy()

    if rois is not None and not (len(rois) == 1 and str(rois[0]).upper() == "ALL"):
        df = df[df["roi"].astype(str).isin([str(r) for r in rois])].copy()

    if stat_keep is not None and "stat" in df.columns and str(stat_keep).upper() != "ALL":
        df = df[df["stat"].astype(str) == str(stat_keep)].copy()

    gcol = _gcol(df, gbase, side=1)
    g2col: str | None = None
    try:
        g2col = _gcol(df, gbase, side=2)
    except KeyError:
        g2col = None
    _require_cols(df, [gcol, "N_1"], label=f"nogse_contrast_long (gbase={gbase})")

    group_cols = ["roi", "direction"] + (["stat"] if "stat" in df.columns else [])
    rows: list[FitRow] = []

    for key, gg in df.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))
        roi = str(key_dict["roi"])
        direction = str(key_dict["direction"])
        stat = str(key_dict["stat"]) if "stat" in key_dict else None

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
            if col not in gg.columns or pd.Series(gg[col]).dropna().empty:
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
        n_2_val = _get_float("N_2")
        n_2 = int(round(float(n_2_val))) if n_2_val is not None else n_1

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

        if td_override_ms is not None:
            td_ms = float(td_override_ms)
        elif _get_float("TN_1") is not None:
            td_ms = float(_get_float("TN_1"))
        elif td_ms_1 is not None:
            td_ms = float(td_ms_1)
        elif max_dur_ms_1 is not None and tm_ms_1 is not None:
            td_ms = float(2.0 * max_dur_ms_1 + tm_ms_1)
        elif TE_1 is not None:
            td_ms = float(TE_1)
        else:
            td_ms = None

        sheet = canonical_sheet_name(sheet_1 or sheet_2 or _get_str("sheet"))
        subj = _get_str("subj")
        if subj is None or not str(subj).strip():
            subj = infer_subj_label(sheet, source_name=source_file)

        y = pd.to_numeric(gg[ycol], errors="coerce").to_numpy(dtype=float)
        G = pd.to_numeric(gg[gcol], errors="coerce").to_numpy(dtype=float)
        G = _maybe_scale_g_thorsten(gbase, G)
        G2_raw_arr = None
        if g2col is not None:
            G2_raw_arr = _maybe_scale_g_thorsten(gbase, pd.to_numeric(gg[g2col], errors="coerce").to_numpy(dtype=float))

        m = np.isfinite(y) & np.isfinite(G)
        y, G = y[m], G[m]
        if G2_raw_arr is not None:
            G2_raw_arr = G2_raw_arr[m]
        n_points = int(len(y))

        f_corr_1, f_corr_2 = _coerce_correction_pair(f_by_direction.get(str(direction), 1.0) if f_by_direction else 1.0)
        G_corr = G * f_corr_1
        G2_corr = G2_raw_arr * f_corr_2 if G2_raw_arr is not None else None

        if sort_by_x and n_points > 0:
            order = np.argsort(G_corr)
            y, G_corr, G = y[order], G_corr[order], G[order]
            if G2_corr is not None and G2_raw_arr is not None:
                G2_corr, G2_raw_arr = G2_corr[order], G2_raw_arr[order]

        if n_fit is not None:
            k = int(n_fit)
            y, G_corr, G = y[:k], G_corr[:k], G[:k]
            if G2_corr is not None and G2_raw_arr is not None:
                G2_corr, G2_raw_arr = G2_corr[:k], G2_raw_arr[:k]

        n_fit_used = int(len(y))
        base: FitRow = dict(
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
            gbase=_normalize_gbase(gbase),
            xplot="1",
            n_points=n_points,
            n_fit=n_fit_used,
            f_corr_1=f_corr_1,
            f_corr_2=f_corr_2,
        )

        if td_ms is None or not np.isfinite(float(td_ms)) or n_fit_used == 0:
            rows.append({**base, "ok": False, "msg": "Empty group or td_ms/TN could not be inferred."})
            continue

        try:
            td = float(td_ms)
            fit_values = _fit_selected_contrast_model(
                model,
                td,
                G_corr,
                n_1,
                y,
                M0_vary=M0_vary,
                D0_vary=D0_vary,
                M0_value=M0_value,
                D0_value=D0_value,
                tc_value=tc_value,
                tc_vary=tc_vary,
            )
            g2_max_raw = float(np.nanmax(G2_raw_arr)) if G2_raw_arr is not None and len(G2_raw_arr) else None
            g2_max_corr = float(np.nanmax(G2_corr)) if G2_corr is not None and len(G2_corr) else None
            peak_metrics = _compute_peak_metrics(
                model=model,
                td_ms=td,
                n_value=n_1,
                fit_row=fit_values,
                g_max_corr=float(np.nanmax(G_corr)),
                f_corr_1=float(f_corr_1),
                f_corr_2=float(f_corr_2),
                peak_grid_n=int(peak_grid_n),
                peak_D0_fix=float(peak_D0_fix),
                peak_gamma=float(peak_gamma),
                g2_max_raw=g2_max_raw,
                g2_max_corr=g2_max_corr,
            )
            rows.append({**base, **peak_metrics, **fit_values, "ok": True, "msg": ""})
        except Exception as e:
            rows.append({**base, "ok": False, "msg": str(e)})

    out = pd.DataFrame(rows)
    return standardize_fit_params(out, fit_kind="nogse_contrast", source_file=source_file)


def plot_nogse_contrast_fit_one_group(
    df_group: pd.DataFrame,
    fit_row: dict,
    *,
    out_png: Path,
    gbase: str,
    ycol: str,
) -> None:
    df_group = _normalize_keys(df_group, label="nogse_plot_group")

    if ycol not in df_group.columns:
        raise KeyError(f"nogse_plot_group: missing ycol {ycol!r}. Columns={list(df_group.columns)}")

    gcol = _gcol(df_group, gbase, side=1)
    y = pd.to_numeric(df_group[ycol], errors="coerce").to_numpy(dtype=float).copy()
    G = pd.to_numeric(df_group[gcol], errors="coerce").to_numpy(dtype=float).copy()
    G = _maybe_scale_g_thorsten(gbase, G)

    f_corr_1, _f_corr_2 = _fit_row_correction_pair(fit_row)
    G = G * f_corr_1
    m = np.isfinite(y) & np.isfinite(G)
    y, G = y[m], G[m]

    td_val = fit_row.get("td_ms")
    td = float(td_val) if td_val is not None else 0.0
    n_value = int(fit_row.get("N_1"))
    model = str(fit_row.get("model", "free"))

    Gmax = float(np.nanmax(G)) if len(G) else 0.0
    Gs = np.linspace(0, Gmax, 250)
    ys = None

    if model == "free" and bool(fit_row.get("ok", True)):
        ys = NOGSE_contrast_vs_g_free(td, Gs, n_value, float(fit_row["M0"]), float(fit_row["D0_m2_ms"]))
    if model == "tort" and bool(fit_row.get("ok", True)):
        ys = NOGSE_contrast_vs_g_tort(
            td,
            Gs,
            n_value,
            float(fit_row["alpha"]),
            float(fit_row["M0"]),
            float(fit_row["D0_m2_ms"]),
        )
    if model == "rest" and bool(fit_row.get("ok", True)):
        ys = NOGSE_contrast_vs_g_rest(
            td,
            Gs,
            n_value,
            float(fit_row["tc_ms"]),
            float(fit_row["M0"]),
            float(fit_row["D0_m2_ms"]),
        )

    plot_nogse_contrast_fit(
        x=np.asarray(G, dtype=float),
        y=np.asarray(y, dtype=float),
        fit_x=np.asarray(Gs, dtype=float) if ys is not None else None,
        fit_y=np.asarray(ys, dtype=float) if ys is not None else None,
        fit_row=fit_row,
        out_png=out_png,
        gbase=_normalize_gbase(gbase),
        ycol=ycol,
    )


__all__ = [
    "CONTRAST_MODEL_SPECS",
    "ContrastModelSpec",
    "FitRow",
    "fit_nogse_contrast_long",
    "plot_nogse_contrast_fit_one_group",
]
