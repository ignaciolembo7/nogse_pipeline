from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from data_processing.io import write_table_outputs, write_xlsx_sheets
from fitting.b_from_g import b_from_g
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
from models.model_fitting import (
    M_ogse_free, M_ogse_mixed, M_ogse_mixed_offset, M_ogse_rest, M_ogse_rest_offset,
    M_nogse_free, M_nogse_rest, M_nogse_mixed,
)
from tc_fittings.contrast_fit_table import load_contrast_fit_params

VALID_RESAMPLED_CONTRAST_MODES = ("signal_fit", "rest_only")
VALID_GRID_MAX_MODES = ("fixed", "observed_pair_max")


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


def _describe_fit_row(row: pd.Series) -> str:
    return (
        f"analysis_id={row.get('analysis_id', '')!r}, "
        f"roi={row.get('roi', '')!r}, "
        f"direction={row.get('direction', '')!r}, "
        f"stat={row.get('stat', '')!r}, "
        f"model={row.get('model', '')!r}"
    )


def _require_experimental_points(
    *,
    experimental_points: pd.DataFrame,
    df_group: pd.DataFrame,
    row: pd.Series,
    contrast_path: Path,
) -> None:
    if not experimental_points.empty:
        return

    gbase = str(row.get("gbase", "g"))
    ycol = str(row.get("ycol", "value_norm"))
    expected = [f"{gbase}_1", f"{gbase}_2", f"{ycol}_1", f"{ycol}_2"]
    missing = [col for col in expected if col not in df_group.columns]
    available = ", ".join(str(c) for c in df_group.columns[:40])
    raise ValueError(
        "Could not extract experimental N1/N2 signal points for resampled contrast export. "
        f"{_describe_fit_row(row)}. contrast_table={contrast_path}. "
        f"matched_rows={len(df_group)}. missing_columns={missing}. "
        f"available_columns_first40=[{available}]"
    )


def _resampled_grid_for_row(
    experimental_points: pd.DataFrame,
    row: pd.Series,
    *,
    grid_min_mTm: float,
    grid_max_mTm: float,
    grid_n: int,
    grid_max_mode: str,
) -> np.ndarray:
    mode = str(grid_max_mode)
    if mode not in VALID_GRID_MAX_MODES:
        raise ValueError(f"Invalid resampled grid max mode {mode!r}. Expected one of {VALID_GRID_MAX_MODES}.")

    lo = float(grid_min_mTm)
    if mode == "fixed":
        hi = float(grid_max_mTm)
    else:
        gbase = str(row.get("gbase", "g"))
        gcol = f"{gbase}_corr_mTm"
        if experimental_points.empty or gcol not in experimental_points.columns:
            raise ValueError(
                "Cannot use observed_pair_max resampling without corrected experimental gradients. "
                f"{_describe_fit_row(row)}. missing_column={gcol!r}"
            )
        values = pd.to_numeric(experimental_points[gcol], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                "Cannot use observed_pair_max resampling because no finite corrected gradients were found. "
                f"{_describe_fit_row(row)}."
            )
        hi = float(np.max(values))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(
            "Invalid resampled gradient grid limits. "
            f"{_describe_fit_row(row)}. grid_min_mTm={lo!r}, grid_max_mTm={hi!r}, mode={mode!r}"
        )
    if int(grid_n) < 2:
        raise ValueError(f"grid_n must be at least 2; got {grid_n!r}.")
    return np.linspace(lo, hi, int(grid_n))


_SIDE_META_SKIP_PREFIXES = ("value", "bvalue", "g", "Ld", "lcf", "Lcf", "tc", "contrast")


def _scalar_side_meta(df_group: pd.DataFrame) -> dict[str, object]:
    """Return scalar _1/_2 metadata from the first row of df_group, excluding per-point columns."""
    if df_group.empty:
        return {}
    row0 = df_group.iloc[0]
    return {
        col: row0[col]
        for col in df_group.columns
        if (col.endswith("_1") or col.endswith("_2"))
        and not any(col.startswith(p) for p in _SIDE_META_SKIP_PREFIXES)
    }


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
    contrast_mode: str = "signal_fit",
    grid_max_mode: str = "fixed",
    g_min_derived: float = 0.0,
    side_meta: dict[str, object] | None = None,
    bvalue_arrs: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    td = float(row["td_ms"])
    n1 = int(row["N_1"])
    n2 = int(row["N_2"])
    s1 = pd.to_numeric(curves["signal_fit_1"], errors="coerce").to_numpy(dtype=float)
    s2 = pd.to_numeric(curves["signal_fit_2"], errors="coerce").to_numpy(dtype=float)
    c1 = pd.to_numeric(curves.get("contrast_signal_fit_1", curves["signal_fit_1"]), errors="coerce").to_numpy(dtype=float)
    c2 = pd.to_numeric(curves.get("contrast_signal_fit_2", curves["signal_fit_2"]), errors="coerce").to_numpy(dtype=float)
    contrast = pd.to_numeric(curves["contrast_fit_resampled"], errors="coerce").to_numpy(dtype=float)
    Ld, lcf_um, Lcf, tc_ms = _derived_axes_from_g(
        g_common,
        td_ms=td,
        peak_D0_fix=float(peak_D0_fix),
        peak_gamma=float(peak_gamma),
    )
    below_min = g_common < float(g_min_derived)
    Ld[below_min] = np.nan
    lcf_um[below_min] = np.nan
    Lcf[below_min] = np.nan
    tc_ms[below_min] = np.nan
    gbase = str(row.get("gbase", "g"))
    ycol = str(row.get("ycol", "value_norm"))
    c_global_values = pd.to_numeric(curves.get("C_global", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    c_global_err_values = pd.to_numeric(curves.get("C_global_err", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    c_global = float(c_global_values[0]) if len(c_global_values) == 1 else np.nan
    c_global_err = float(c_global_err_values[0]) if len(c_global_err_values) == 1 else np.nan
    contrast_definition_values = curves.get("contrast_definition", pd.Series(dtype=object)).astype(str).dropna().unique().tolist()
    contrast_definition = contrast_definition_values[0] if len(contrast_definition_values) == 1 else ""
    _side_meta = side_meta or {}
    _bvalue_arrs = bvalue_arrs or {}
    records: list[dict[str, object]] = []
    for i, g_val in enumerate(g_common, start=1):
        record: dict[str, object] = {
            "row_kind": "contrast",
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
            "resample_grid": "corrected",
            "resample_grid_max_mode": grid_max_mode,
            "resample_grid_n": int(len(g_common)),
            "resample_grid_min_mTm": float(g_common[0]),
            "resample_grid_max_mTm": float(g_common[-1]),
            "gradient_unit": "mT/m",
            "resampled_contrast_mode": contrast_mode,
            gbase: float(g_val),
            f"{gbase}_1": float(g_val),
            f"{gbase}_2": float(g_val),
            "signal_fit_1": float(s1[i - 1]),
            "signal_fit_2": float(s2[i - 1]),
            "value_1": float(s1[i - 1]),
            "value_norm_1": float(s1[i - 1]),
            "value_2": float(s2[i - 1]),
            "value_norm_2": float(s2[i - 1]),
            "contrast_signal_fit_1": float(c1[i - 1]),
            "contrast_signal_fit_2": float(c2[i - 1]),
            "contrast_definition": contrast_definition,
            "contrast_value": float(contrast[i - 1]),
            "contrast_value_norm": float(contrast[i - 1]),
            "Ld": float(Ld[i - 1]),
            "lcf": float(lcf_um[i - 1]),
            "Lcf": float(Lcf[i - 1]),
            "lcf_a": float(Lcf[i - 1]),
            "tc": float(tc_ms[i - 1]),
        }
        for bv_col, bv_arr in _bvalue_arrs.items():
            record[bv_col] = float(bv_arr[i - 1])
        record.update(_side_meta)
        if np.isfinite(c_global):
            record["C_global"] = c_global
            record["C_global_err"] = c_global_err
        records.append(record)
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
    RN: float = 0.0,
    x_ms: float = np.nan,  # explicit lobe duration for NOGSE; if NaN falls back to td_ms/N
) -> np.ndarray:
    _x_ms = float(x_ms)
    x = _x_ms if np.isfinite(_x_ms) else float(td_ms) / float(N)
    model_name = str(model)
    if model_name in {"free", "ogse_free"}:
        return M_ogse_free(td_ms, G, N, x, M0, D0)
    if model_name == "tort":
        return M_ogse_free(td_ms, G, N, x, M0, alpha * D0)
    if model_name in {"rest", "ogse_rest"}:
        return M_ogse_rest(td_ms, G, N, x, tc_ms, M0, D0)
    if model_name in {"rest_offset", "rest_offset_globC", "ogse_rest_offset"}:
        return M_ogse_rest_offset(td_ms, G, N, x, tc_ms, M0, D0, C)
    if model_name in {"mixed", "ogse_mixed_global", "ogse_mixed"}:
        return M_ogse_mixed(td_ms, G, N, x, tc_ms, alpha, M0, D0)
    if model_name == "ogse_mixed_offset":
        return M_ogse_mixed_offset(td_ms, G, N, x, tc_ms, alpha, M0, D0, C, RN)
    # --- NOGSE models ---
    if model_name in {"nogse_free", "nogse_free_cpmg", "nogse_free_hahn", "nogse_free_offset"}:
        return M_nogse_free(TE=td_ms, G=G, N=N, x=x, M0=M0, D0=D0)
    if model_name in {"nogse_rest", "nogse_rest_offset"}:
        return M_nogse_rest(TE=td_ms, G=G, N=N, x=x, tc=tc_ms, M0=M0, D0=D0)
    if model_name in {"nogse_mixed", "nogse_mixed_global"}:
        return M_nogse_mixed(TE=td_ms, G=G, N=N, x=x, tc=tc_ms, alpha=alpha, M0=M0, D0=D0)
    raise ValueError(f"Unsupported model {model!r} for independent signal fitting.")


def _uses_signal_alpha(model: str) -> bool:
    return str(model) in {"tort", "mixed", "ogse_mixed_global", "ogse_mixed", "ogse_mixed_offset", "nogse_mixed", "nogse_mixed_global"}


def _uses_independent_signal_C(model: str) -> bool:
    return str(model) in {"rest_offset", "ogse_rest_offset", "ogse_mixed_offset", "nogse_free_offset", "nogse_rest_offset"}


def _uses_signal_RN(model: str) -> bool:
    return str(model) == "ogse_mixed_offset"


def _clean_signal_fit_params(df: pd.DataFrame, model: str) -> pd.DataFrame:
    out = df.copy()
    if not _uses_signal_alpha(model) and "signal_fit_alpha" in out.columns:
        out = out.drop(columns=["signal_fit_alpha"])
    if not _uses_independent_signal_C(model) and "signal_fit_C" in out.columns:
        out = out.drop(columns=["signal_fit_C"])
    if not _uses_signal_RN(model) and "signal_fit_RN" in out.columns:
        out = out.drop(columns=["signal_fit_RN"])
    return out


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
    n = int(row.get(f"N_{side}", row.get("N", 1)))
    x_ms = float(row.get("x_model_ms", np.nan))  # explicit lobe duration for NOGSE
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
        out = {
            "side": int(side),
            "N": int(n),
            "n_points": int(len(obs_y)),
            "signal_fit_model": model,
            "signal_fit_method": method,
            "signal_fit_M0": float(M0),
            "signal_fit_D0_m2_ms": float(D0),
            "signal_fit_tc_ms": float(tc_ms),
            "signal_fit_rmse": rmse,
            "signal_fit_chi2": chi2,
            "signal_fit_ok": bool(ok),
            "signal_fit_msg": msg,
        }
        if _uses_signal_alpha(model):
            out["signal_fit_alpha"] = float(alpha)
        if _uses_independent_signal_C(model):
            out["signal_fit_C"] = float(C)
        return out

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

    elif model in {"rest_offset", "rest_offset_globC"}:
        p0 = np.array([np.clip(tc_ms, tc_lo, tc_hi), np.clip(C, c_lo, c_hi)], dtype=float)
        bounds = (np.array([tc_lo, c_lo]), np.array([tc_hi, c_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, float(p[0]), alpha, float(p[1])

    elif model in {"mixed", "ogse_mixed_global"}:
        p0 = np.array([np.clip(tc_ms, tc_lo, tc_hi), np.clip(alpha, alpha_lo, alpha_hi)], dtype=float)
        bounds = (np.array([tc_lo, alpha_lo]), np.array([tc_hi, alpha_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, float(p[0]), float(p[1]), C

    elif model in {"nogse_free", "nogse_free_cpmg", "nogse_free_hahn"}:
        p0 = np.array([np.log10(np.clip(D0, d0_lo, d0_hi))], dtype=float)
        bounds = (np.array([np.log10(d0_lo)]), np.array([np.log10(d0_hi)]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return 10.0 ** float(p[0]), tc_ms, alpha, C

    elif model == "nogse_rest":
        p0 = np.array([np.clip(tc_ms, tc_lo, tc_hi)], dtype=float)
        bounds = (np.array([tc_lo]), np.array([tc_hi]))

        def unpack(p: np.ndarray) -> tuple[float, float, float, float]:
            return D0, float(p[0]), alpha, C

    elif model == "nogse_mixed":
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
                x_ms=x_ms,
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
        x_ms=x_ms,
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
        x_ms=x_ms,
    )
    return np.asarray(yhat_grid, dtype=float), record(yhat_grid, yhat_obs)


def _parameter_stderr_from_least_squares(fit, residual_count: int, parameter_count: int) -> np.ndarray:
    if residual_count <= parameter_count:
        return np.full(parameter_count, np.nan, dtype=float)
    try:
        jac = np.asarray(fit.jac, dtype=float)
        _, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
        if singular_values.size == 0:
            return np.full(parameter_count, np.nan, dtype=float)
        threshold = np.finfo(float).eps * max(jac.shape) * singular_values[0]
        valid = singular_values > threshold
        if not np.any(valid):
            return np.full(parameter_count, np.nan, dtype=float)
        cov = (vt[valid].T / singular_values[valid] ** 2) @ vt[valid]
        resid = np.asarray(fit.fun, dtype=float)
        finite = np.isfinite(resid)
        if not finite.any():
            return np.full(parameter_count, np.nan, dtype=float)
        variance = float(np.sum(resid[finite] ** 2) / max(1, residual_count - parameter_count))
        diag = np.diag(cov) * variance
        return np.sqrt(np.where(diag >= 0.0, diag, np.nan))
    except Exception:
        return np.full(parameter_count, np.nan, dtype=float)


def _fit_global_c_signal_curves(
    row: pd.Series,
    experimental_points: pd.DataFrame,
    g_common: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model = str(row.get("model", ""))
    if model != "rest_offset_globC":
        raise ValueError(f"_fit_global_c_signal_curves only supports rest_offset_globC, got {model!r}.")

    td_ms = float(row["td_ms"])
    n1 = int(row["N_1"])
    n2 = int(row["N_2"])
    obs_g1, obs_y1 = _observed_side_arrays(experimental_points, row, side=1)
    obs_g2, obs_y2 = _observed_side_arrays(experimental_points, row, side=2)
    if len(obs_y1) < 2 or len(obs_y2) < 2:
        raise ValueError(
            "rest_offset_globC requires side-specific signal points for both OGSE curves. "
            f"{_describe_fit_row(row)}. N pair=({n1}, {n2}); "
            f"side1_points={len(obs_y1)}, side2_points={len(obs_y2)}."
        )

    M0 = _finite_float(row.get("M0", 1.0), 1.0)
    D0 = _finite_float(row.get("D0_m2_ms", np.nan), _finite_float(row.get("D0_mm2_s", 3.2e-3), 3.2e-3) * 1e-9)
    tc_ms = _finite_float(row.get("tc_ms", 5.0), 5.0)
    C = _finite_float(row.get("C", 0.0), 0.0)
    tc_lo, tc_hi = _tc_bounds_from_row(row)
    c_lo, c_hi = _bounds_from_row(row, "C", (0.0, 1.0))
    p0 = np.array(
        [
            np.clip(tc_ms, tc_lo, tc_hi),
            np.clip(tc_ms, tc_lo, tc_hi),
            np.clip(C, c_lo, c_hi),
        ],
        dtype=float,
    )
    bounds = (np.array([tc_lo, tc_lo, c_lo], dtype=float), np.array([tc_hi, tc_hi, c_hi], dtype=float))

    def residuals(p: np.ndarray) -> np.ndarray:
        tc1, tc2, c_global = map(float, p)
        with np.errstate(all="ignore"):
            yhat1 = _signal_model_yhat(
                model="rest_offset_globC",
                td_ms=td_ms,
                G=obs_g1,
                N=n1,
                M0=M0,
                D0=D0,
                tc_ms=tc1,
                alpha=0.5,
                C=c_global,
            )
            yhat2 = _signal_model_yhat(
                model="rest_offset_globC",
                td_ms=td_ms,
                G=obs_g2,
                N=n2,
                M0=M0,
                D0=D0,
                tc_ms=tc2,
                alpha=0.5,
                C=c_global,
            )
        resid = np.concatenate([np.asarray(yhat1, dtype=float) - obs_y1, np.asarray(yhat2, dtype=float) - obs_y2])
        return np.where(np.isfinite(resid), resid, 1e6)

    method = "joint_signal_fit_global_C"
    ok = True
    msg = "ok"
    stderr = np.full(3, np.nan, dtype=float)
    try:
        fit = least_squares(residuals, p0, bounds=bounds, max_nfev=10000)
        ok = bool(fit.success)
        msg = str(fit.message)
        tc1, tc2, C_global = map(float, fit.x)
        stderr = _parameter_stderr_from_least_squares(fit, len(obs_y1) + len(obs_y2), 3)
    except Exception as exc:
        ok = False
        msg = str(exc)
        tc1, tc2, C_global = map(float, p0)

    s1_grid = _signal_model_yhat(
        model="rest_offset_globC",
        td_ms=td_ms,
        G=g_common,
        N=n1,
        M0=M0,
        D0=D0,
        tc_ms=tc1,
        alpha=0.5,
        C=C_global,
    )
    s2_grid = _signal_model_yhat(
        model="rest_offset_globC",
        td_ms=td_ms,
        G=g_common,
        N=n2,
        M0=M0,
        D0=D0,
        tc_ms=tc2,
        alpha=0.5,
        C=C_global,
    )
    yhat1 = _signal_model_yhat(
        model="rest_offset_globC",
        td_ms=td_ms,
        G=obs_g1,
        N=n1,
        M0=M0,
        D0=D0,
        tc_ms=tc1,
        alpha=0.5,
        C=C_global,
    )
    yhat2 = _signal_model_yhat(
        model="rest_offset_globC",
        td_ms=td_ms,
        G=obs_g2,
        N=n2,
        M0=M0,
        D0=D0,
        tc_ms=tc2,
        alpha=0.5,
        C=C_global,
    )

    def row_record(side: int, n: int, obs_y: np.ndarray, yhat_obs: np.ndarray, tc_fit: float, tc_err: float) -> dict[str, object]:
        resid = np.asarray(yhat_obs, dtype=float) - np.asarray(obs_y, dtype=float)
        finite = np.isfinite(resid)
        return {
            "side": int(side),
            "N": int(n),
            "n_points": int(len(obs_y)),
            "signal_fit_model": "rest_offset_globC",
            "signal_fit_method": method,
            "signal_fit_M0": float(M0),
            "signal_fit_D0_m2_ms": float(D0),
            "signal_fit_tc_ms": float(tc_fit),
            "signal_fit_tc_err_ms": float(tc_err) if np.isfinite(tc_err) else np.nan,
            "signal_fit_C_global": float(C_global),
            "signal_fit_C_global_err": float(stderr[2]) if np.isfinite(stderr[2]) else np.nan,
            "signal_fit_shared_parameters": "C",
            "signal_fit_rmse": float(np.sqrt(np.mean(resid[finite] ** 2))) if finite.any() else np.nan,
            "signal_fit_chi2": float(np.sum(resid[finite] ** 2)) if finite.any() else np.nan,
            "signal_fit_ok": bool(ok),
            "signal_fit_msg": msg,
        }

    fit_params = pd.DataFrame(
        [
            row_record(1, n1, obs_y1, yhat1, tc1, stderr[0]),
            row_record(2, n2, obs_y2, yhat2, tc2, stderr[1]),
        ]
    )
    return np.asarray(s1_grid, dtype=float), np.asarray(s2_grid, dtype=float), fit_params


def _signal_fit_param_row(fit_params: pd.DataFrame, *, side: int) -> pd.Series:
    if fit_params.empty or "side" not in fit_params.columns:
        raise ValueError(f"Could not find signal-fit parameters for side={side}.")
    sub = fit_params[pd.to_numeric(fit_params["side"], errors="coerce") == int(side)]
    if len(sub) != 1:
        raise ValueError(f"Expected exactly one signal-fit parameter row for side={side}; found {len(sub)}.")
    return sub.iloc[0]


def _rest_signal_from_signal_fit_params(
    row: pd.Series,
    fit_params: pd.DataFrame,
    g_common: np.ndarray,
    *,
    side: int,
) -> np.ndarray:
    params = _signal_fit_param_row(fit_params, side=side)
    td_ms = float(row["td_ms"])
    n = int(row[f"N_{side}"])
    x = td_ms / float(n)
    M0 = _finite_float(params.get("signal_fit_M0", row.get("M0", 1.0)), 1.0)
    D0 = _finite_float(params.get("signal_fit_D0_m2_ms", row.get("D0_m2_ms", np.nan)), 3.2e-12)
    tc_ms = _finite_float(params.get("signal_fit_tc_ms", row.get("tc_ms", 5.0)), 5.0)
    return np.asarray(M_ogse_rest(td_ms, g_common, n, x, tc_ms, M0, D0), dtype=float)


def _contrast_component_curves(
    row: pd.Series,
    fit_params: pd.DataFrame,
    g_common: np.ndarray,
    *,
    signal_fit_1: np.ndarray,
    signal_fit_2: np.ndarray,
    contrast_mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    model = str(row.get("model", ""))
    mode = str(contrast_mode)
    if mode not in VALID_RESAMPLED_CONTRAST_MODES:
        raise ValueError(f"Invalid resampled contrast mode {mode!r}. Expected one of {VALID_RESAMPLED_CONTRAST_MODES}.")
    if mode == "rest_only" and model in {"rest_offset", "rest_offset_globC"}:
        rest1 = _rest_signal_from_signal_fit_params(row, fit_params, g_common, side=1)
        rest2 = _rest_signal_from_signal_fit_params(row, fit_params, g_common, side=2)
        return rest1, rest2, "M_ogse_rest_N1_from_offset_fit(g_common) - M_ogse_rest_N2_from_offset_fit(g_common)"
    return (
        np.asarray(signal_fit_1, dtype=float),
        np.asarray(signal_fit_2, dtype=float),
        "signal_fit_1(g_common) - signal_fit_2(g_common)",
    )


def _resampled_curves_table(
    row: pd.Series,
    g_common: np.ndarray,
    *,
    experimental_points: pd.DataFrame,
    contrast_mode: str = "signal_fit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = str(row.get("model", ""))
    if str(row.get("fit_kind", "")) == "ogse_contrast_from_global_signal_fit":
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
                    "n_points": int((experimental_points["side"].astype(int) == 1).sum()) if not experimental_points.empty else 0,
                    "signal_fit_model": model,
                    "signal_fit_method": "global_signal_fit_reused",
                    "signal_fit_M0": row.get("M0", np.nan),
                    "signal_fit_D0_m2_ms": row.get("D0_m2_ms", np.nan),
                    "signal_fit_tc_ms": row.get("tc_ms", np.nan),
                    "signal_fit_alpha": row.get("alpha", np.nan),
                    "signal_fit_C": row.get("C", np.nan),
                    "signal_fit_RN": row.get("RN", np.nan),
                    "signal_fit_shared_parameters": row.get("global_params", ""),
                    "signal_fit_pair_parameters": row.get("global_contrast_params", ""),
                    "signal_fit_ok": bool(row.get("ok", True)),
                    "signal_fit_msg": "global signal fit reused",
                },
                {
                    "side": 2,
                    "N": int(row["N_2"]),
                    "n_points": int((experimental_points["side"].astype(int) == 2).sum()) if not experimental_points.empty else 0,
                    "signal_fit_model": model,
                    "signal_fit_method": "global_signal_fit_reused",
                    "signal_fit_M0": row.get("M0", np.nan),
                    "signal_fit_D0_m2_ms": row.get("D0_m2_ms", np.nan),
                    "signal_fit_tc_ms": row.get("tc_ms", np.nan),
                    "signal_fit_alpha": row.get("alpha", np.nan),
                    "signal_fit_C": row.get("C", np.nan),
                    "signal_fit_RN": row.get("RN", np.nan),
                    "signal_fit_shared_parameters": row.get("global_params", ""),
                    "signal_fit_pair_parameters": row.get("global_contrast_params", ""),
                    "signal_fit_ok": bool(row.get("ok", True)),
                    "signal_fit_msg": "global signal fit reused",
                },
            ]
        )
        fit_params = _clean_signal_fit_params(fit_params, model)
    elif experimental_points.empty:
        if model == "rest_offset_globC":
            raise ValueError(
                "rest_offset_globC requires side-specific experimental signal points and cannot use "
                f"the contrast-fit fallback. {_describe_fit_row(row)}."
            )
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
        fit_params = _clean_signal_fit_params(fit_params, model)
    elif model == "rest_offset_globC":
        s1, s2, fit_params = _fit_global_c_signal_curves(row, experimental_points, g_common)
    else:
        s1, fit1 = _fit_independent_signal_curve(row, experimental_points, g_common, side=1)
        s2, fit2 = _fit_independent_signal_curve(row, experimental_points, g_common, side=2)
        fit_params = pd.DataFrame([fit1, fit2])
        fit_params = _clean_signal_fit_params(fit_params, model)

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    c1, c2, contrast_definition = _contrast_component_curves(
        row,
        fit_params,
        g_common,
        signal_fit_1=s1,
        signal_fit_2=s2,
        contrast_mode=contrast_mode,
    )
    contrast = c1 - c2
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
            "contrast_signal_fit_1": c1,
            "contrast_signal_fit_2": c2,
            "contrast_fit_resampled": contrast,
            "contrast_definition": contrast_definition,
        }
    )
    if model == "rest_offset_globC" and "signal_fit_C_global" in fit_params.columns:
        c_values = pd.to_numeric(fit_params["signal_fit_C_global"], errors="coerce").dropna().unique()
        c_err_values = pd.to_numeric(fit_params.get("signal_fit_C_global_err", pd.Series(dtype=float)), errors="coerce").dropna().unique()
        if len(c_values) != 1:
            raise ValueError(f"rest_offset_globC produced ambiguous C_global values: {c_values.tolist()}")
        curves["C_global"] = float(c_values[0])
        curves["C_global_err"] = float(c_err_values[0]) if len(c_err_values) == 1 else np.nan
    return curves, fit_params


def _fit_params_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict()])


def _metadata_table(
    row: pd.Series,
    g_common: np.ndarray,
    *,
    contrast_definition: str | None = None,
    contrast_mode: str = "signal_fit",
    grid_max_mode: str = "fixed",
) -> pd.DataFrame:
    if contrast_definition is None:
        contrast_definition = "signal_fit_1(g_common) - signal_fit_2(g_common)"
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
        "resample_grid": "corrected",
        "resample_grid_max_mode": grid_max_mode,
        "resample_grid_n": int(len(g_common)),
        "resample_grid_min_mTm": float(g_common[0]),
        "resample_grid_max_mTm": float(g_common[-1]),
        "gradient_unit": "mT/m",
        "resampled_contrast_mode": contrast_mode,
        "contrast_definition": contrast_definition,
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
    contrast_definition = str(curves.get("contrast_definition", pd.Series([""])).iloc[0])
    contrast_note = " rest-only" if contrast_definition.startswith("M_ogse_rest_") else ""
    ax.plot(
        curves[gcol],
        curves["contrast_fit_resampled"],
        "-",
        color="#2ca02c",
        linewidth=2.1,
        label=f"contrast N={int(row['N_1'])}-N={int(row['N_2'])}{contrast_note}",
    )
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
    contrast_mode: str = "signal_fit",
    grid_max_mode: str = "fixed",
) -> Path:
    contrast_definition = None
    if "contrast_definition" in curves.columns:
        definitions = curves["contrast_definition"].astype(str).dropna().unique().tolist()
        contrast_definition = definitions[0] if len(definitions) == 1 else None
    return write_xlsx_sheets(
        {
            "metadata": _metadata_table(
                row,
                g_common,
                contrast_definition=contrast_definition,
                contrast_mode=contrast_mode,
                grid_max_mode=grid_max_mode,
            ),
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
    contrast_mode: str = "signal_fit",
    grid_max_mode: str = "fixed",
    g_min_derived: float = 0.0,
) -> list[Path]:
    if contrast_mode not in VALID_RESAMPLED_CONTRAST_MODES:
        raise ValueError(f"Invalid resampled contrast mode {contrast_mode!r}. Expected one of {VALID_RESAMPLED_CONTRAST_MODES}.")
    if grid_max_mode not in VALID_GRID_MAX_MODES:
        raise ValueError(f"Invalid resampled grid max mode {grid_max_mode!r}. Expected one of {VALID_GRID_MAX_MODES}.")

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
    if "analysis_id" in df.columns:
        is_nested_fitresamp = df["analysis_id"].astype(str).str.contains("_fitresamp-", regex=False)
        if is_nested_fitresamp.any():
            skipped = sorted(df.loc[is_nested_fitresamp, "analysis_id"].astype(str).unique().tolist())
            print(
                "Skipping fit_params from already fitted/resampled contrast tables:",
                len(skipped),
            )
            for analysis_id in skipped[:20]:
                print(f"  - {analysis_id}")
            if len(skipped) > 20:
                print(f"  ... {len(skipped) - 20} more")
            df = df.loc[~is_nested_fitresamp].copy()
    if df.empty:
        raise ValueError("No valid non-fitresampled fits remained after filtering.")

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, pd.DataFrame] = {}
    written: list[Path] = []

    for folder_key, sub in df.groupby(["sheet", "subj", "direction", "td_ms", "gbase", "N_1", "N_2", "model"], sort=True, dropna=False):
        folder = out_base / _folder_name(sub.iloc[0])
        folder.mkdir(parents=True, exist_ok=True)
        roi_tables: list[pd.DataFrame] = []

        for _, row in sub.sort_values("roi", kind="stable").iterrows():
            # For rows produced by fit_global_signal (fit_kind="ogse_contrast_from_global_signal_fit"),
            # the model parameters are already stored in the fit row and curves are reconstructed
            # analytically by _resampled_curves_table — no contrast parquet lookup is needed.
            df_group: pd.DataFrame = pd.DataFrame()
            if str(row.get("fit_kind", "")) == "ogse_contrast_from_global_signal_fit":
                experimental_points = pd.DataFrame()
                grid_max_mode_eff = "fixed"
            else:
                contrast_path = _resolve_contrast_parquet(
                    analysis_id=str(row["analysis_id"]),
                    sheet=row.get("sheet", None),
                    contrast_root=contrast_root,
                )
                contrast_df = _load_contrast_table_cached(contrast_path, cache)
                df_group = _subset_group(contrast_df, row)
                experimental_points = _experimental_points_table(df_group, row)
                _require_experimental_points(
                    experimental_points=experimental_points,
                    df_group=df_group,
                    row=row,
                    contrast_path=contrast_path,
                )
                grid_max_mode_eff = grid_max_mode
            g_common = _resampled_grid_for_row(
                experimental_points,
                row,
                grid_min_mTm=grid_min_mTm,
                grid_max_mTm=grid_max_mTm,
                grid_n=grid_n,
                grid_max_mode=grid_max_mode_eff,
            )
            curves, signal_fit_params = _resampled_curves_table(
                row,
                g_common,
                experimental_points=experimental_points,
                contrast_mode=contrast_mode,
            )
            side_meta = _scalar_side_meta(df_group)
            n1_val = int(row["N_1"])
            n2_val = int(row["N_2"])
            bvalue_arrs: dict[str, np.ndarray] = {}
            for side, n_val in ((1, n1_val), (2, n2_val)):
                delta_ms_val = float(side_meta.get(f"delta_ms_{side}", np.nan))
                dapp_ms_val = float(side_meta.get(f"Delta_app_ms_{side}", np.nan))
                if np.isfinite(delta_ms_val) and np.isfinite(dapp_ms_val):
                    # OGSE bvalue
                    bvalue_arrs[f"bvalue_{side}"] = b_from_g(
                        g_common,
                        N=float(n_val),
                        gamma=267.5221900,
                        delta_ms=delta_ms_val,
                        delta_app_ms=dapp_ms_val,
                        g_type="g",
                    )
                else:
                    # NOGSE bvalue: b = (1/12) * gamma² * G² * ((N-1)*x³ + y³) / 1e9
                    # where y = TN - (N-1)*x, TN = total refocusing time, x = lobe duration
                    TN_v = float(side_meta.get(f"TN_{side}", np.nan))
                    x_v = float(side_meta.get(f"x_model_ms_{side}", np.nan))
                    if np.isfinite(TN_v) and np.isfinite(x_v):
                        Nf = float(n_val)
                        y_ms = TN_v - (Nf - 1) * x_v
                        bvalue_arrs[f"bvalue_{side}"] = (
                            (1.0 / 12) * 267.5221900 ** 2 * g_common ** 2
                            * ((Nf - 1) * x_v ** 3 + y_ms ** 3) / 1e9
                        )
            table = _resampled_rows_from_curves(
                row,
                g_common,
                curves,
                peak_D0_fix=peak_D0_fix,
                peak_gamma=peak_gamma,
                contrast_mode=contrast_mode,
                grid_max_mode=grid_max_mode_eff,
                g_min_derived=g_min_derived,
                side_meta=side_meta,
                bvalue_arrs=bvalue_arrs,
            )
            roi_tables.append(table)

            roi_token = _sanitize_token(str(row["roi"]))
            for stale in (
                folder / f"roi-{roi_token}.signal_fits.png",
                folder / f"roi-{roi_token}.resampled_contrast.png",
                folder / f"roi-{roi_token}.resampled_contrast.xlsx",
                folder / f"roi-{roi_token}.fit_and_resampled_contrast.png",
                folder / f"roi-{roi_token}.fit_and_resampled_contrast.xlsx",
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
                contrast_mode=contrast_mode,
                grid_max_mode=grid_max_mode_eff,
            )
            written.append(contrast_out)
            written.append(workbook)

        if roi_tables:
            combined = pd.concat(roi_tables, ignore_index=True)
            combined_out = folder / "resampled_contrasts.long.parquet"
            write_table_outputs(combined, combined_out, xlsx_path=combined_out.with_suffix(".xlsx"))
            written.append(combined_out)

    return written
