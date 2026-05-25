from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from data_processing.io import write_table_outputs, write_xlsx_sheets
from ogse_fitting.contrast_fit_panels import (
    _fit_row_correction_pair,
    _gcols,
    _load_contrast_table_cached,
    _maybe_scale_g_thorsten,
    _resolve_contrast_parquet,
    _sanitize_token,
    _subset_group,
)
from ogse_fitting.contrast_tc_peak_panels import _derived_axes_from_g
from ogse_fitting.fit_ogse_contrast_vs_g import _model_side_yhat
from models.model_fitting import M_ogse_free, M_ogse_mixed, M_ogse_rest, M_ogse_rest_offset
from tc_fittings.contrast_fit_table import load_contrast_fit_params


def _fmt_num(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return _sanitize_token(str(value))
    if not np.isfinite(x):
        return "NA"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _folder_name(row: pd.Series) -> str:
    subj = _sanitize_token(str(row.get("sheet", row.get("subj", "NA"))))
    direction = _sanitize_token(str(row.get("direction", "NA")))
    td = _fmt_num(row.get("td_ms", np.nan))
    gbase = _sanitize_token(str(row.get("gbase", "g")))
    n1 = _fmt_num(row.get("N_1", np.nan))
    n2 = _fmt_num(row.get("N_2", np.nan))
    model = _sanitize_token(str(row.get("model", "model")))
    return f"{subj}_dir{direction}_td{td}_{gbase}_N{n1}-N{n2}_model-{model}"


def _side_points(df_group: pd.DataFrame, row: pd.Series, *, side: int) -> tuple[np.ndarray, np.ndarray]:
    gbase = str(row.get("gbase", "g"))
    ycol = str(row.get("ycol", "value_norm"))
    g1c, g2c = _gcols(gbase)
    gcol = g1c if side == 1 else g2c
    y_candidates = [f"{ycol}_{side}", f"value_norm_{side}", f"value_{side}"]
    ycol_eff = next((c for c in y_candidates if c in df_group.columns), None)
    if ycol_eff is None or gcol not in df_group.columns:
        return np.array([]), np.array([])

    x = pd.to_numeric(df_group[gcol], errors="coerce").to_numpy(dtype=float)
    x = _maybe_scale_g_thorsten(gbase, x)
    f1, f2 = _fit_row_correction_pair(row)
    x = x * (f1 if side == 1 else f2)
    y = pd.to_numeric(df_group[ycol_eff], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    return x[order], y[order]


def _experimental_points_table(df_group: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    gbase = str(row.get("gbase", "g"))
    ycol = str(row.get("ycol", "value_norm"))
    g1c, g2c = _gcols(gbase)
    f1, f2 = _fit_row_correction_pair(row)
    records: list[dict[str, object]] = []

    for side, gcol, factor, n_col in [(1, g1c, f1, "N_1"), (2, g2c, f2, "N_2")]:
        y_candidates = [f"{ycol}_{side}", f"value_norm_{side}", f"value_{side}"]
        ycol_eff = next((c for c in y_candidates if c in df_group.columns), None)
        if ycol_eff is None or gcol not in df_group.columns:
            continue
        raw_g = pd.to_numeric(df_group[gcol], errors="coerce").to_numpy(dtype=float)
        raw_g = _maybe_scale_g_thorsten(gbase, raw_g)
        corr_g = raw_g * float(factor)
        y = pd.to_numeric(df_group[ycol_eff], errors="coerce").to_numpy(dtype=float)
        b_step = pd.to_numeric(df_group.get("b_step", pd.Series(np.arange(1, len(df_group) + 1))), errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(corr_g) & np.isfinite(y)
        for idx in np.where(m)[0]:
            records.append(
                {
                    "analysis_id": row.get("analysis_id", ""),
                    "subj": row.get("subj", ""),
                    "sheet": row.get("sheet", ""),
                    "roi": row.get("roi", ""),
                    "direction": row.get("direction", ""),
                    "stat": row.get("stat", ""),
                    "side": int(side),
                    "N": row.get(n_col, np.nan),
                    "b_step": int(b_step[idx]) if np.isfinite(b_step[idx]) else int(idx + 1),
                    "gbase": gbase,
                    f"{gbase}_raw_mTm": float(raw_g[idx]),
                    f"{gbase}_corr_mTm": float(corr_g[idx]),
                    "f_corr": float(factor),
                    "ycol": ycol,
                    "signal_observed": float(y[idx]),
                }
            )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    return out.sort_values(["side", f"{gbase}_corr_mTm", "b_step"], kind="stable").reset_index(drop=True)


def _resampled_rows(row: pd.Series, g_common: np.ndarray, *, peak_D0_fix: float, peak_gamma: float) -> pd.DataFrame:
    curves, _ = _resampled_curves_table(row, g_common, experimental_points=pd.DataFrame())
    return _resampled_rows_from_curves(row, g_common, curves, peak_D0_fix=peak_D0_fix, peak_gamma=peak_gamma)


def _resampled_rows_from_curves(
    row: pd.Series,
    g_common: np.ndarray,
    curves: pd.DataFrame,
    *,
    peak_D0_fix: float,
    peak_gamma: float,
) -> pd.DataFrame:
    td = float(row["td_ms"])
    n1 = int(row["N_1"])
    n2 = int(row["N_2"])
    s1 = pd.to_numeric(curves["signal_fit_1"], errors="coerce").to_numpy(dtype=float)
    s2 = pd.to_numeric(curves["signal_fit_2"], errors="coerce").to_numpy(dtype=float)
    contrast = pd.to_numeric(curves["contrast_fit_resampled"], errors="coerce").to_numpy(dtype=float)
    Ld, lcf_um, Lcf, tc_ms = _derived_axes_from_g(
        g_common,
        td_ms=td,
        peak_D0_fix=float(peak_D0_fix),
        peak_gamma=float(peak_gamma),
    )
    gbase = str(row.get("gbase", "g"))
    ycol = str(row.get("ycol", "value_norm"))
    records: list[dict[str, object]] = []
    for i, g_val in enumerate(g_common, start=1):
        records.append(
            {
                "analysis_id": row.get("analysis_id", ""),
                "subj": row.get("subj", ""),
                "sheet": row.get("sheet", ""),
                "roi": row.get("roi", ""),
                "direction": row.get("direction", ""),
                "stat": row.get("stat", ""),
                "b_step": int(i),
                "td_ms": float(td),
                "N_1": int(n1),
                "N_2": int(n2),
                "model": row.get("model", ""),
                "ycol": ycol,
                "gbase": gbase,
                "resample_grid": "fixed_corrected",
                "resample_grid_n": int(len(g_common)),
                "resample_grid_min_mTm": float(g_common[0]),
                "resample_grid_max_mTm": float(g_common[-1]),
                "gradient_unit": "mT/m",
                gbase: float(g_val),
                f"{gbase}_1": float(g_val),
                f"{gbase}_2": float(g_val),
                "signal_fit_1": float(s1[i - 1]),
                "signal_fit_2": float(s2[i - 1]),
                "value": float(contrast[i - 1]),
                "value_norm": float(contrast[i - 1]),
                "Ld": float(Ld[i - 1]),
                "lcf": float(lcf_um[i - 1]),
                "Lcf": float(Lcf[i - 1]),
                "lcf_a": float(Lcf[i - 1]),
                "tc": float(tc_ms[i - 1]),
            }
        )
    return pd.DataFrame.from_records(records)


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _bounds_from_row(row: pd.Series, prefix: str, default: tuple[float, float]) -> tuple[float, float]:
    lo = _finite_float(row.get(f"{prefix}_min", np.nan), default[0])
    hi = _finite_float(row.get(f"{prefix}_max", np.nan), default[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return default
    return float(lo), float(hi)


def _tc_bounds_from_row(row: pd.Series) -> tuple[float, float]:
    lo = _finite_float(row.get("tc_bound_min", np.nan), 0.1)
    hi = _finite_float(row.get("tc_bound_max", np.nan), 1000.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return 0.1, 1000.0
    return float(lo), float(hi)


def _signal_model_yhat(
    *,
    model: str,
    td_ms: float,
    G: np.ndarray,
    N: int,
    M0: float,
    D0: float,
    tc_ms: float,
    alpha: float,
    C: float = 0.0,
) -> np.ndarray:
    x = float(td_ms) / float(N)
    model_name = str(model)
    if model_name == "free":
        return M_ogse_free(td_ms, G, N, x, M0, D0)
    if model_name == "tort":
        return M_ogse_free(td_ms, G, N, x, M0, alpha * D0)
    if model_name == "rest":
        return M_ogse_rest(td_ms, G, N, x, tc_ms, M0, D0)
    if model_name == "rest_offset":
        return M_ogse_rest_offset(td_ms, G, N, x, tc_ms, M0, D0, C)
    if model_name in {"mixed", "mixed_global"}:
        return M_ogse_mixed(td_ms, G, N, x, tc_ms, alpha, M0, D0)
    raise ValueError(f"Unsupported model {model!r} for independent signal fitting.")


def _observed_side_arrays(
    experimental_points: pd.DataFrame,
    row: pd.Series,
    *,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if experimental_points.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    gbase = str(row.get("gbase", "g"))
    gcol = f"{gbase}_corr_mTm"
    sub = experimental_points[experimental_points["side"].astype(int) == int(side)]
    if sub.empty or gcol not in sub.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = pd.to_numeric(sub[gcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub["signal_observed"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    return x[order], y[order]


def _fit_independent_signal_curve(
    row: pd.Series,
    experimental_points: pd.DataFrame,
    g_common: np.ndarray,
    *,
    side: int,
) -> tuple[np.ndarray, dict[str, object]]:
    model = str(row.get("model", "rest"))
    td_ms = float(row["td_ms"])
    n = int(row[f"N_{side}"])
    obs_g, obs_y = _observed_side_arrays(experimental_points, row, side=side)

    M0 = _finite_float(row.get("M0", 1.0), 1.0)
    D0 = _finite_float(row.get("D0_m2_ms", np.nan), _finite_float(row.get("D0_mm2_s", 3.2e-3), 3.2e-3) * 1e-9)
    tc_ms = _finite_float(row.get("tc_ms", 5.0), 5.0)
    alpha = _finite_float(row.get("alpha", 0.5), 0.5)
    C = _finite_float(row.get("C", 0.0), 0.0)
    method = "independent_signal_fit"
    ok = True
    msg = "ok"

    def record(yhat_grid: np.ndarray, yhat_obs: np.ndarray) -> dict[str, object]:
        resid = np.asarray(yhat_obs, dtype=float) - np.asarray(obs_y, dtype=float)
        finite = np.isfinite(resid)
        rmse = float(np.sqrt(np.mean(resid[finite] ** 2))) if finite.any() else np.nan
        chi2 = float(np.sum(resid[finite] ** 2)) if finite.any() else np.nan
        return {
            "side": int(side),
            "N": int(n),
            "n_points": int(len(obs_y)),
            "signal_fit_model": model,
            "signal_fit_method": method,
            "signal_fit_M0": float(M0),
            "signal_fit_D0_m2_ms": float(D0),
            "signal_fit_tc_ms": float(tc_ms),
            "signal_fit_alpha": float(alpha),
            "signal_fit_C": float(C),
            "signal_fit_rmse": rmse,
            "signal_fit_chi2": chi2,
            "signal_fit_ok": bool(ok),
            "signal_fit_msg": msg,
        }

    if len(obs_y) < 2:
        method = "contrast_fit_fallback"
        ok = False
        msg = "not enough side-specific experimental points"
        s1, s2 = _model_side_yhat(
            model=model,
            td_ms=td_ms,
            G=g_common,
            n_1=int(row["N_1"]),
            n_2=int(row["N_2"]),
            fit_row=row.to_dict(),
        )
        yhat_grid = np.asarray(s1 if side == 1 else s2, dtype=float)
        yhat_obs = np.interp(obs_g, g_common, yhat_grid) if len(obs_g) else np.array([], dtype=float)
        return yhat_grid, record(yhat_grid, yhat_obs)

    tc_lo, tc_hi = _tc_bounds_from_row(row)
    d0_lo, d0_hi = 2.3e-14, 2.3e-10
    c_lo, c_hi = -1.0, 1.0
    alpha_lo, alpha_hi = _bounds_from_row(row, "alpha", (0.0, 1.0))
    alpha_lo = max(float(alpha_lo), 0.0)
    alpha_hi = min(float(alpha_hi), 1.0)
    if alpha_lo >= alpha_hi:
        alpha_lo, alpha_hi = 0.0, 1.0

    if model == "free":
        p0 = np.array([np.log10(np.clip(D0, d0_lo, d0_hi))], dtype=float)
        bounds = (np.array([np.log10(d0_lo)]), np.array([np.log10(d0_hi)]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return 10.0 ** float(p[0]), tc_ms, alpha, C

    elif model == "tort":
        p0 = np.array([np.clip(alpha, alpha_lo, alpha_hi)], dtype=float)
        bounds = (np.array([alpha_lo]), np.array([alpha_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, tc_ms, float(p[0]), C

    elif model == "rest":
        p0 = np.array([np.clip(tc_ms, tc_lo, tc_hi)], dtype=float)
        bounds = (np.array([tc_lo]), np.array([tc_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, float(p[0]), alpha, C

    elif model == "rest_offset":
        p0 = np.array([np.clip(tc_ms, tc_lo, tc_hi), np.clip(C, c_lo, c_hi)], dtype=float)
        bounds = (np.array([tc_lo, c_lo]), np.array([tc_hi, c_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, float(p[0]), alpha, float(p[1])

    elif model in {"mixed", "mixed_global"}:
        p0 = np.array([np.clip(tc_ms, tc_lo, tc_hi), np.clip(alpha, alpha_lo, alpha_hi)], dtype=float)
        bounds = (np.array([tc_lo, alpha_lo]), np.array([tc_hi, alpha_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, float(p[0]), float(p[1]), C

    else:
        raise ValueError(f"Unsupported model {model!r} for independent signal fitting.")

    def residuals(p: np.ndarray) -> np.ndarray:
        d0_fit, tc_fit, alpha_fit, c_fit = unpack(p)
        with np.errstate(all="ignore"):
            yhat = _signal_model_yhat(
                model=model,
                td_ms=td_ms,
                G=obs_g,
                N=n,
                M0=M0,
                D0=d0_fit,
                tc_ms=tc_fit,
                alpha=alpha_fit,
                C=c_fit,
            )
        resid = np.asarray(yhat, dtype=float) - obs_y
        return np.where(np.isfinite(resid), resid, 1e6)

    try:
        fit = least_squares(residuals, p0, bounds=bounds, max_nfev=5000)
        ok = bool(fit.success)
        msg = str(fit.message)
        D0, tc_ms, alpha, C = unpack(fit.x)
    except Exception as exc:
        ok = False
        msg = str(exc)

    yhat_grid = _signal_model_yhat(
        model=model,
        td_ms=td_ms,
        G=g_common,
        N=n,
        M0=M0,
        D0=D0,
        tc_ms=tc_ms,
        alpha=alpha,
        C=C,
    )
    yhat_obs = _signal_model_yhat(
        model=model,
        td_ms=td_ms,
        G=obs_g,
        N=n,
        M0=M0,
        D0=D0,
        tc_ms=tc_ms,
        alpha=alpha,
        C=C,
    )
    return np.asarray(yhat_grid, dtype=float), record(yhat_grid, yhat_obs)


def _resampled_curves_table(
    row: pd.Series,
    g_common: np.ndarray,
    *,
    experimental_points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if experimental_points.empty:
        s1, s2 = _model_side_yhat(
            model=str(row["model"]),
            td_ms=float(row["td_ms"]),
            G=g_common,
            n_1=int(row["N_1"]),
            n_2=int(row["N_2"]),
            fit_row=row.to_dict(),
        )
        fit_params = pd.DataFrame(
            [
                {
                    "side": 1,
                    "N": int(row["N_1"]),
                    "n_points": 0,
                    "signal_fit_model": row.get("model", ""),
                    "signal_fit_method": "contrast_fit_fallback",
                    "signal_fit_M0": row.get("M0", np.nan),
                    "signal_fit_D0_m2_ms": row.get("D0_m2_ms", np.nan),
                    "signal_fit_tc_ms": row.get("tc_ms", np.nan),
                    "signal_fit_alpha": row.get("alpha", np.nan),
                    "signal_fit_C": row.get("C", np.nan),
                    "signal_fit_rmse": np.nan,
                    "signal_fit_chi2": np.nan,
                    "signal_fit_ok": False,
                    "signal_fit_msg": "missing experimental side points",
                },
                {
                    "side": 2,
                    "N": int(row["N_2"]),
                    "n_points": 0,
                    "signal_fit_model": row.get("model", ""),
                    "signal_fit_method": "contrast_fit_fallback",
                    "signal_fit_M0": row.get("M0", np.nan),
                    "signal_fit_D0_m2_ms": row.get("D0_m2_ms", np.nan),
                    "signal_fit_tc_ms": row.get("tc_ms", np.nan),
                    "signal_fit_alpha": row.get("alpha", np.nan),
                    "signal_fit_C": row.get("C", np.nan),
                    "signal_fit_rmse": np.nan,
                    "signal_fit_chi2": np.nan,
                    "signal_fit_ok": False,
                    "signal_fit_msg": "missing experimental side points",
                },
            ]
        )
    else:
        s1, fit1 = _fit_independent_signal_curve(row, experimental_points, g_common, side=1)
        s2, fit2 = _fit_independent_signal_curve(row, experimental_points, g_common, side=2)
        fit_params = pd.DataFrame([fit1, fit2])

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    contrast = s1 - s2
    gbase = str(row.get("gbase", "g"))
    curves = pd.DataFrame(
        {
            "resample_step": np.arange(1, len(g_common) + 1, dtype=int),
            "gbase": gbase,
            f"{gbase}_corr_mTm": np.asarray(g_common, dtype=float),
            "N_1": int(row["N_1"]),
            "N_2": int(row["N_2"]),
            "signal_fit_1": s1,
            "signal_fit_2": s2,
            "contrast_fit_resampled": contrast,
        }
    )
    return curves, fit_params


def _fit_params_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict()])


def _metadata_table(row: pd.Series, g_common: np.ndarray) -> pd.DataFrame:
    keys = {
        "analysis_id": row.get("analysis_id", ""),
        "subj": row.get("subj", ""),
        "sheet": row.get("sheet", ""),
        "roi": row.get("roi", ""),
        "direction": row.get("direction", ""),
        "td_ms": row.get("td_ms", np.nan),
        "model": row.get("model", ""),
        "gbase": row.get("gbase", ""),
        "N_1": row.get("N_1", np.nan),
        "N_2": row.get("N_2", np.nan),
        "f_corr_1": row.get("f_corr_1", np.nan),
        "f_corr_2": row.get("f_corr_2", np.nan),
        "resample_grid": "fixed_corrected",
        "resample_grid_n": int(len(g_common)),
        "resample_grid_min_mTm": float(g_common[0]),
        "resample_grid_max_mTm": float(g_common[-1]),
        "gradient_unit": "mT/m",
        "contrast_definition": "independent_signal_fit_1(g_common) - independent_signal_fit_2(g_common)",
        "experimental_points_used_for_contrast": False,
    }
    return pd.DataFrame([keys])


def _plot_fit_and_contrast(
    *,
    out_png: Path,
    row: pd.Series,
    experimental_points: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    gbase = str(row.get("gbase", "g"))
    gcol = f"{gbase}_corr_mTm"

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    side1 = experimental_points[experimental_points["side"].astype(int) == 1] if not experimental_points.empty else pd.DataFrame()
    side2 = experimental_points[experimental_points["side"].astype(int) == 2] if not experimental_points.empty else pd.DataFrame()
    if not side1.empty:
        ax.plot(side1[gcol], side1["signal_observed"], "o", color="#1f77b4", markersize=4, alpha=0.75, label=f"N={int(row['N_1'])} data")
    if not side2.empty:
        ax.plot(side2[gcol], side2["signal_observed"], "o", color="#d62728", markersize=4, alpha=0.75, label=f"N={int(row['N_2'])} data")

    ax.plot(curves[gcol], curves["signal_fit_1"], "-", color="#1f77b4", linewidth=2.0, label=f"N={int(row['N_1'])} fit")
    ax.plot(curves[gcol], curves["signal_fit_2"], "-", color="#d62728", linewidth=2.0, label=f"N={int(row['N_2'])} fit")
    ax.plot(curves[gcol], curves["contrast_fit_resampled"], "-", color="#2ca02c", linewidth=2.1, label=f"contrast N={int(row['N_1'])}-N={int(row['N_2'])}")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel(f"{gbase} corrected [mT/m]")
    ax.set_ylabel(str(row.get("ycol", "value_norm")))
    ax.set_title(
        f"{row.get('sheet', row.get('subj', ''))} | {row.get('roi', '')} | "
        f"dir={row.get('direction', '')} | td={float(row['td_ms']):g} ms | model={row.get('model', '')}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _write_roi_workbook(
    *,
    out_xlsx: Path,
    row: pd.Series,
    experimental_points: pd.DataFrame,
    curves: pd.DataFrame,
    signal_fit_params: pd.DataFrame,
    contrast_table: pd.DataFrame,
    g_common: np.ndarray,
) -> Path:
    return write_xlsx_sheets(
        {
            "metadata": _metadata_table(row, g_common),
            "fit_params": _fit_params_table(row),
            "signal_fit_params": signal_fit_params,
            "experimental_points": experimental_points,
            "resampled_curves": curves,
            "resampled_contrast": contrast_table,
        },
        out_xlsx,
    )


def export_resampled_contrasts_from_fits(
    *,
    fits_root: str | Path,
    contrast_root: str | Path,
    out_dir: str | Path,
    pattern: str = "**/fit_params.*",
    grid_min_mTm: float = 0.0,
    grid_max_mTm: float = 90.0,
    grid_n: int = 1000,
    models: list[str] | None = None,
    subjs: list[str] | None = None,
    rois: list[str] | None = None,
    directions: list[str] | None = None,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
    ok_only: bool = True,
) -> list[Path]:
    df = load_contrast_fit_params(
        [fits_root],
        pattern=pattern,
        models=models,
        subjs=subjs,
        rois=rois,
        directions=directions,
        ok_only=ok_only,
    )
    if df.empty:
        raise ValueError("No valid fits remained after filtering.")

    g_common = np.linspace(float(grid_min_mTm), float(grid_max_mTm), int(grid_n))
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, pd.DataFrame] = {}
    written: list[Path] = []

    for folder_key, sub in df.groupby(["sheet", "subj", "direction", "td_ms", "gbase", "N_1", "N_2", "model"], sort=True, dropna=False):
        folder = out_base / _folder_name(sub.iloc[0])
        folder.mkdir(parents=True, exist_ok=True)
        roi_tables: list[pd.DataFrame] = []

        for _, row in sub.sort_values("roi", kind="stable").iterrows():
            contrast_path = _resolve_contrast_parquet(
                analysis_id=str(row["analysis_id"]),
                sheet=row.get("sheet", None),
                contrast_root=contrast_root,
            )
            contrast_df = _load_contrast_table_cached(contrast_path, cache)
            df_group = _subset_group(contrast_df, row)
            experimental_points = _experimental_points_table(df_group, row)
            curves, signal_fit_params = _resampled_curves_table(row, g_common, experimental_points=experimental_points)
            table = _resampled_rows_from_curves(row, g_common, curves, peak_D0_fix=peak_D0_fix, peak_gamma=peak_gamma)
            roi_tables.append(table)

            roi_token = _sanitize_token(str(row["roi"]))
            for stale in (
                folder / f"roi-{roi_token}.signal_fits.png",
                folder / f"roi-{roi_token}.resampled_contrast.png",
                folder / f"roi-{roi_token}.resampled_contrast.xlsx",
            ):
                if stale.exists():
                    stale.unlink()

            contrast_out = folder / f"roi-{roi_token}.resampled_contrast.parquet"
            write_table_outputs(table, contrast_out)
            _plot_fit_and_contrast(
                out_png=folder / f"roi-{roi_token}.fit_and_resampled_contrast.png",
                row=row,
                experimental_points=experimental_points,
                curves=curves,
            )
            workbook = _write_roi_workbook(
                out_xlsx=folder / f"roi-{roi_token}.fit_and_resampled_contrast.xlsx",
                row=row,
                experimental_points=experimental_points,
                curves=curves,
                contrast_table=table,
                signal_fit_params=signal_fit_params,
                g_common=g_common,
            )
            written.append(contrast_out)
            written.append(workbook)

        if roi_tables:
            combined = pd.concat(roi_tables, ignore_index=True)
            combined_out = folder / "resampled_contrasts.long.parquet"
            write_table_outputs(combined, combined_out, xlsx_path=combined_out.with_suffix(".xlsx"))
            written.append(combined_out)

    return written
