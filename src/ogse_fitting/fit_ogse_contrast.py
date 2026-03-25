from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from nogse_models.nogse_model_fitting import OGSE_contrast_vs_g_free, OGSE_contrast_vs_g_tort, OGSE_contrast_vs_g_rest
from plottings.fit_plot_style import finish_fit_figure, plot_fit_curve, plot_fit_data, start_fit_figure
from tools.fit_params_schema import standardize_fit_params


# -----------------------------
# Helpers: strict schema
# -----------------------------
KEY_COLS = ("roi", "direction", "b_step")


def _require_cols(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] faltan columnas {missing}. Columns={list(df.columns)}")


def _normalize_keys(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """
    Regla dura: direction siempre string, b_step int.
    (No renombra columnas, solo fuerza tipos para agrupar/plotear consistentemente.)
    """
    out = df.copy()

    if "axis" in out.columns:
        raise KeyError(f"[{label}] encontré columna prohibida 'axis'. Este pipeline usa SOLO 'direction'.")

    _require_cols(out, KEY_COLS, label=label)

    out["direction"] = out["direction"].astype(str)

    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["roi", "direction", "b_step"]].head(10)
        raise ValueError(f"[{label}] b_step tiene NaN/no-numérico. Ejemplos:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)

    # stat si existe, siempre str
    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)

    return out


def _unique_scalar(series: pd.Series, *, name: str, required: bool = False) -> Any:
    u = series.dropna().unique()
    if len(u) == 0:
        if required:
            raise ValueError(f"No pude inferir '{name}': columna vacía/NaN.")
        return None
    if len(u) > 1:
        raise ValueError(f"'{name}' no es único dentro del grupo. Valores={u.tolist()[:10]}")
    return u[0]


def _normalize_gbase(gbase: str) -> str:
    """
    Normaliza nombres históricos.
    - gthorsten -> g_thorsten
    """
    b = str(gbase).strip()
    if b.endswith("_1"):
        b = b[:-2]
    if b.endswith("_2"):
        b = b[:-2]
    if b == "gthorsten":
        b = "g_thorsten"
    return b


def _gcols(gbase: str) -> tuple[str, str]:
    b = _normalize_gbase(gbase)
    return f"{b}_1", f"{b}_2"


def _maybe_scale_g_thorsten(gbase: str, arr: np.ndarray) -> np.ndarray:
    """
    Mantengo el comportamiento histórico: para thorsten escalamos por sqrt(2) y abs.
    (Si no lo querés, sacamos esta función.)
    """
    b = _normalize_gbase(gbase)
    if b == "g_thorsten":
        return np.sqrt(2.0) * np.abs(arr)
    return arr


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _chi2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum((y - yhat) ** 2))


# -----------------------------
# Fitting backends
# -----------------------------
# def _fit_free_numpy_bruteforce(
#     td: float,
#     G1: np.ndarray,
#     G2: np.ndarray,
#     N1: int,
#     N2: int,
#     y: np.ndarray,
#     *,
#     M0_vary: bool,
#     D0_vary: bool,
#     M0_value: float,
#     D0_value: float,
# ) -> tuple[float, float, float, float, str, float | None, float | None]:
#     if not M0_vary and not D0_vary:
#         yhat = OGSE_contrast_vs_g_free(td, G1, G2, N1, N2, M0_value, D0_value)
#         return float(M0_value), float(D0_value), _rmse(y, yhat), _chi2(y, yhat), "fixed", None, None

#     D_grid = np.logspace(np.log10(D0_value / 100.0), np.log10(D0_value * 10.0), 240) if D0_vary else np.array([D0_value], float)
#     M_grid = np.linspace(0.0, 2.0, 240) if M0_vary else np.array([M0_value], float)

#     best = (np.inf, None, None)  # mse, M0, D0
#     for D0 in D_grid:
#         v = OGSE_contrast_vs_g_free(td, G1, G2, N1, N2, 1.0, float(D0))
#         if M0_vary:
#             denom = float(np.dot(v, v))
#             if denom <= 0:
#                 continue
#             M0_ls = float(np.dot(v, y) / denom)
#             M0_ls = float(np.clip(M0_ls, 0.0, 2.0))
#             yhat = M0_ls * v
#             mse = float(np.mean((y - yhat) ** 2))
#             if mse < best[0]:
#                 best = (mse, M0_ls, float(D0))
#         else:
#             yhat = float(M0_value) * v
#             mse = float(np.mean((y - yhat) ** 2))
#             if mse < best[0]:
#                 best = (mse, float(M0_value), float(D0))

#     if best[1] is None:
#         raise RuntimeError("No pude encontrar mínimo en la grilla (free).")

#     yhat = OGSE_contrast_vs_g_free(td, G1, G2, N1, N2, best[1], best[2])
#     return best[1], best[2], _rmse(y, yhat), _chi2(y, yhat), "numpy_bruteforce", None, None


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
    # try:
    #     from scipy.optimize import curve_fit  # type: ignore
    # except Exception:
    #     return _fit_free_numpy_bruteforce(
    #         td, G1, G2, N1, N2, y,
    #         M0_vary=M0_vary, D0_vary=D0_vary, M0_value=M0_value, D0_value=D0_value,
    #     )

    if not M0_vary and not D0_vary:
        yhat = OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, M0_value, D0_value)
        return float(M0_value), float(D0_value), _rmse(y, yhat), _chi2(y, yhat), "fixed", None, None

    # bounds razonables alrededor del guess
    D_lo, D_hi = float(D0_value / 10.0), float(D0_value * 10.0)

    if M0_vary and D0_vary:
        def f(_dummy, M0, D0):
            return OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, M0, D0)
        p0 = [float(M0_value), float(D0_value)]
        bounds = ([0.0, D_lo], [2.0, D_hi])
        popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=400000)
        yhat = f(None, *popt)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan])
        return float(popt[0]), float(popt[1]), _rmse(y, yhat), _chi2(y, yhat), "scipy_curve_fit", float(perr[0]), float(perr[1])

    if (not M0_vary) and D0_vary:
        def f(_dummy, D0):
            return OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, float(M0_value), D0)
        p0 = [float(D0_value)]
        bounds = ([D_lo], [D_hi])
        popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=400000)
        yhat = f(None, *popt)
        D0 = float(popt[0])
        D0_err = float(np.sqrt(pcov[0, 0])) if pcov is not None and np.isfinite(pcov[0, 0]) else None
        return float(M0_value), D0, _rmse(y, yhat), _chi2(y, yhat), "scipy_curve_fit", None, D0_err

    def f(_dummy, M0):
        return OGSE_contrast_vs_g_free(td, G1, G2, n_1, n_2, M0, float(D0_value))
    p0 = [float(M0_value)]
    bounds = ([0.0], [2.0])
    popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=400000)
    yhat = f(None, *popt)
    M0 = float(popt[0])
    M0_err = float(np.sqrt(pcov[0, 0])) if pcov is not None and np.isfinite(pcov[0, 0]) else None
    return M0, float(D0_value), _rmse(y, yhat), _chi2(y, yhat), "scipy_curve_fit", M0_err, None

def _fit_tort(td: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, y: np.ndarray):

    def f(_dummy, alpha, M0, D0):
        return OGSE_contrast_vs_g_tort(td, G1, G2, n_1, n_2, alpha, M0, D0)

    p0 = [0.7, 1.0, 1e-12]
    bounds = ([0.0, 0.0, 1e-14], [2.0, 5.0, 1e-9])

    popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=600000)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan, np.nan])
    yhat = f(None, *popt)
    rmse = _rmse(y, yhat)
    chi2 = _chi2(y, yhat)
    alpha, M0, D0 = map(float, popt)
    return alpha, M0, D0, rmse, chi2, "scipy_curve_fit", float(perr[0]), float(perr[1]), float(perr[2])

def _fit_rest(td: float, G1: np.ndarray, G2: np.ndarray, n_1: int, n_2: int, y: np.ndarray):

    def f(_dummy, tc, M0, D0):
        return OGSE_contrast_vs_g_rest(td, G1, G2, n_1, n_2, tc, M0, D0)

    p0 = [0.7, 1.0, 1e-12]
    bounds = ([0.0, 0.0, 1e-14], [2.0, 5.0, 1e-9])

    popt, pcov = curve_fit(f, np.zeros_like(y), y, p0=p0, bounds=bounds, maxfev=600000)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan, np.nan])
    yhat = f(None, *popt)
    rmse = _rmse(y, yhat)
    chi2 = _chi2(y, yhat)
    alpha, M0, D0 = map(float, popt)
    return alpha, M0, D0, rmse, chi2, "scipy_curve_fit", float(perr[0]), float(perr[1]), float(perr[2])

# -----------------------------
# Public API
# -----------------------------
@dataclass(frozen=True)
class FitRow:
    roi: str
    direction: str
    stat: str | None

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
    sheet_2: str | None

    # fit config
    model: str
    ycol: str
    gbase: str
    xplot: str
    n_points: int
    n_fit: int
    f_corr: float

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
    ycol: str = "value_norm",
    directions: list[str] | None = None,
    rois: list[str] | None = None,
    stat_keep: str | None = "avg",
    xplot: str = "1",
    n_fit: int | None = None,
    sort_by_x: bool = True,
    f_by_direction: dict[str, float] | None = None,
    td_override_ms: float | None = None,
    td_tol_ms: float = 1e-3,
    M0_vary: bool = True,
    D0_vary: bool = True,
    M0_value: float = 1.0,
    D0_value: float = 1e-12,
    source_file: str | None = None,
    tc_value: float = 5.0,
    tc_vary: bool = True,
) -> pd.DataFrame:
    """
    Fit de una tabla de contraste LONG (tu formato nuevo):
      keys: roi, direction, b_step (y opcional stat)
      y:    value o value_norm (contraste)
      x:    gbase_1 y gbase_2

    Espera parámetros por secuencia en el mismo DF (sufijos _1/_2):
      N_1, N_2, td_ms_1/td_ms_2 (o max_dur_ms_1 + tm_ms_1), etc.
    """
    df = _normalize_keys(df, label="contrast_long")

    # aliases backward-compat (solo por si hay tablas viejas)
    y_alias = {"value_norm": "contrast_norm", "value": "contrast"}
    y_eff = ycol if ycol in df.columns else (y_alias.get(ycol) if y_alias.get(ycol) in df.columns else None)
    if y_eff is None:
        raise KeyError(f"No encuentro ycol='{ycol}' (ni alias). Columns={list(df.columns)}")

    if directions is not None and not (len(directions) == 1 and directions[0].upper() == "ALL"):
        directions = [str(x) for x in directions]
        df = df[df["direction"].isin(directions)].copy()

    if rois is not None and not (len(rois) == 1 and rois[0].upper() == "ALL"):
        df = df[df["roi"].astype(str).isin([str(r) for r in rois])].copy()

    if stat_keep is not None and "stat" in df.columns and str(stat_keep).upper() != "ALL":
        df = df[df["stat"].astype(str) == str(stat_keep)].copy()

    # g columns
    g1c, g2c = _gcols(gbase)
    _require_cols(df, [g1c, g2c], label=f"contrast_long (gbase={gbase})")

    # required sequence params
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
        sequence_1 = _get_str("sequence_1")
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
        sequence_2 = _get_str("sequence_2")
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

        # arrays
        y = pd.to_numeric(gg[y_eff], errors="coerce").to_numpy(dtype=float)
        G1 = pd.to_numeric(gg[g1c], errors="coerce").to_numpy(dtype=float)
        G2 = pd.to_numeric(gg[g2c], errors="coerce").to_numpy(dtype=float)

        G1 = _maybe_scale_g_thorsten(gbase, G1)
        G2 = _maybe_scale_g_thorsten(gbase, G2)

        m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2)
        y, G1, G2 = y[m], G1[m], G2[m]
        n_points = int(len(y))

        if td_ms is None or not np.isfinite(float(td_ms)) or n_points == 0:
            rows.append(
                FitRow(
                    roi=roi,
                    direction=direction,
                    stat=stat,
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
                    sheet_2=sheet_2,
                    model=model,
                    ycol=ycol,
                    gbase=_normalize_gbase(gbase),
                    xplot=str(xplot),
                    n_points=n_points,
                    n_fit=n_points,
                    f_corr=1.0,
                    ok=False,
                    msg="Grupo vacío o no pude inferir td_ms.",
                )
            )
            continue

        # correction factor by direction (string key)
        f_corr = float(f_by_direction.get(str(direction), 1.0)) if f_by_direction else 1.0
        G1 = G1 * f_corr
        G2 = G2 * f_corr

        # sort by x for fitting stability
        if sort_by_x:
            x = G1 if str(xplot) == "1" else G2
            order = np.argsort(x)
            y, G1, G2 = y[order], G1[order], G2[order]

        if n_fit is not None:
            k = int(n_fit)
            y, G1, G2 = y[:k], G1[:k], G2[:k]

        n_fit_used = int(len(y))
        td = float(td_ms)

        base = dict(
            roi=roi,
            direction=direction,
            stat=stat,
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
            sheet_2=sheet_2,
            model=model,
            ycol=ycol,
            gbase=_normalize_gbase(gbase),
            xplot=str(xplot),
            n_points=n_points,
            n_fit=n_fit_used,
            f_corr=f_corr,
        )

        try:
            if model == "free":
                M0, D0, rmse, chi2, method, M0_err, D0_err = _fit_free(
                    td, G1, G2, n_1, n_2, y,
                    M0_vary=M0_vary, D0_vary=D0_vary, M0_value=M0_value, D0_value=D0_value
                )
                rows.append(
                    FitRow(
                        **base,
                        M0=float(M0),
                        M0_err=None if M0_err is None else float(M0_err),
                        D0_m2_ms=float(D0),
                        D0_err_m2_ms=None if D0_err is None else float(D0_err),
                        rmse=float(rmse),
                        chi2=float(chi2),
                        method=str(method),
                    )
                )

            elif model == "tort":
                alpha, M0, D0, rmse, chi2, method, alpha_err, M0_err, D0_err = _fit_tort(td, G1, G2, n_1, n_2, y)
                rows.append(
                    FitRow(
                        **base,
                        alpha=float(alpha),
                        alpha_err=None if alpha_err is None else float(alpha_err),
                        M0=float(M0),
                        M0_err=None if M0_err is None else float(M0_err),
                        D0_m2_ms=float(D0),
                        D0_err_m2_ms=None if D0_err is None else float(D0_err),
                        rmse=float(rmse),
                        chi2=float(chi2),
                        method=str(method),
                    )
                )

            elif model == "rest":
                tc, M0, D0, rmse, chi2, method, tc_err, M0_err, D0_err = _fit_rest(td, G1, G2, n_1, n_2, y)
                rows.append(
                    FitRow(
                        **base,
                        tc_ms=float(tc),
                        tc_err_ms=None if tc_err is None else float(tc_err),
                        M0=float(M0),
                        M0_err=None if M0_err is None else float(M0_err),
                        D0_m2_ms=float(D0),
                        D0_err_m2_ms=None if D0_err is None else float(D0_err),
                        rmse=float(rmse),
                        chi2=float(chi2),
                        method=str(method),
                    )
                )
            
            else:
                rows.append(FitRow(**base, ok=False, msg=f"Modelo '{model}' no implementado."))
        except Exception as e:
            rows.append(FitRow(**base, ok=False, msg=str(e)))

    out = pd.DataFrame([r.__dict__ for r in rows])

    # Mantengo esto porque tu pipeline ya lo usa downstream.
    out = standardize_fit_params(out, fit_kind="nogse_contrast", source_file=source_file)
    return out


def plot_fit_one_group(
    df_group: pd.DataFrame,
    fit_row: dict,
    *,
    out_png: Path,
    gbase: str,
    ycol: str,
) -> None:
    """
    Plot simple: y vs gbase_1, curva del modelo usando parámetros del fit_row.
    """
    df_group = _normalize_keys(df_group, label="plot_group")

    # y alias for legacy
    y_alias = {"value_norm": "contrast_norm", "value": "contrast"}
    y_eff = ycol if ycol in df_group.columns else (y_alias.get(ycol) if y_alias.get(ycol) in df_group.columns else None)
    if y_eff is None:
        raise KeyError(f"No encuentro ycol='{ycol}' en df_group.")

    g1c, g2c = _gcols(gbase)
    _require_cols(df_group, [g1c, g2c], label="plot_group")

    y = pd.to_numeric(df_group[y_eff], errors="coerce").to_numpy(dtype=float).copy()
    G1 = pd.to_numeric(df_group[g1c], errors="coerce").to_numpy(dtype=float).copy()
    G2 = pd.to_numeric(df_group[g2c], errors="coerce").to_numpy(dtype=float).copy()

    G1 = _maybe_scale_g_thorsten(gbase, G1)
    G2 = _maybe_scale_g_thorsten(gbase, G2)

    f_corr = float(fit_row.get("f_corr", 1.0) or 1.0)
    G1 = G1 * f_corr
    G2 = G2 * f_corr
    m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2)
    y, G1, G2 = y[m], G1[m], G2[m]

    td = float(fit_row.get("td_ms_1") if fit_row.get("td_ms_1") is not None else fit_row.get("td_ms_2") if fit_row.get("td_ms_2") is not None else 0.0)
    n_1 = int(fit_row.get("N_1"))
    n_2 = int(fit_row.get("N_2"))
    model = str(fit_row.get("model", "free"))

    # curva suave
    G1max = float(np.nanmax(G1)) if len(G1) else 0.0
    G2max = float(np.nanmax(G2)) if len(G2) else 0.0
    f = np.linspace(0, 1, 250)
    G1s = f * G1max
    G2s = f * G2max

    ys = None
    label = model

    if model == "free" and bool(fit_row.get("ok", True)):
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_free(td, G1s, G2s, n_1, n_2, M0, D0)
        label = f"free: M0={M0:.3g}, D0={D0:.3g} m²/ms"

    if model == "tort" and bool(fit_row.get("ok", True)):
        alpha = float(fit_row["alpha"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_tort(td, G1s, G2s, n_1, n_2, alpha, M0, D0)
        label = f"tort: a={alpha:.3g}, M0={M0:.3g}, D0={D0:.3g} m²/ms"

    if model == "rest" and bool(fit_row.get("ok", True)):
        tc = float(fit_row["tc_ms"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        ys = OGSE_contrast_vs_g_rest(td, G1s, G2s, n_1, n_2, tc, M0, D0)
        label = f"rest: tc={tc:.3g} ms, M0={M0:.3g}, D0={D0:.3g} m²/ms"

    start_fit_figure()
    plot_fit_data(G1, y, label="data")
    if ys is not None:
        plot_fit_curve(G1s, ys, label=label)

    roi = fit_row.get("roi", "roi")
    direction = fit_row.get("direction", "direction")
    finish_fit_figure(
        title=f"OGSE contrast fit | ROI={roi} | direction={direction} | $t_d$={td:.1f} ms | N{n_1}-N{n_2}",
        xlabel=f"{_normalize_gbase(gbase)}_1 (mT/m)",
        ylabel=ycol,
        out_png=out_png,
    )
