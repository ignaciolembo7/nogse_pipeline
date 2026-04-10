from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.schema import finalize_clean_dproj_long, finalize_clean_signal_long
from ogse_fitting.b_from_g import b_from_g
from signal_rotation.dirs import load_default_dirs, load_dirs_csv


ROTATED_DIRECTION_ORDER = ["eig1", "eig2", "eig3", "x", "y", "z", "tra", "long"]


def _unique_scalar_from_aliases(df: pd.DataFrame, aliases: list[str], cast=float):
    for col in aliases:
        if col not in df.columns:
            continue
        u = pd.Series(df[col]).dropna().unique()
        if len(u) != 1:
            continue
        try:
            return cast(u[0])
        except Exception:
            continue
    return None


def _unique_str_from_aliases(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col not in df.columns:
            continue
        u = pd.Series(df[col]).dropna().astype(str).str.lower().unique().tolist()
        if len(u) == 1:
            return str(u[0])
    return None


def _first_finite(series: pd.Series | None) -> float:
    if series is None:
        return np.nan
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return np.nan
    return float(vals.iloc[0])


def _build_signal_row(
    template: pd.Series,
    *,
    direction: str,
    b_step: int,
    bvalue: float,
    value: float,
    S0: float,
    g: float,
    g_max: float,
    g_lin_max: float,
    g_thorsten: float,
    bvalue_g: float,
    bvalue_g_lin_max: float,
    bvalue_thorsten: float,
) -> dict:
    row = template.to_dict()
    row["direction"] = direction
    row["b_step"] = int(b_step)
    row["bvalue"] = float(bvalue)
    row["value"] = float(value)
    row["S0"] = float(S0)
    row["value_norm"] = float(value) / float(S0) if float(S0) != 0.0 else np.nan
    row["g"] = g
    row["g_max"] = g_max
    row["g_lin_max"] = g_lin_max
    row["g_thorsten"] = g_thorsten
    row["bvalue_g"] = bvalue_g
    row["bvalue_g_lin_max"] = bvalue_g_lin_max
    row["bvalue_thorsten"] = bvalue_thorsten
    return row


def _build_dproj_row(
    template: pd.Series,
    *,
    direction: str,
    b_step: int,
    bvalue: float,
    D_proj: float,
    g: float,
    g_max: float,
    g_lin_max: float,
    g_thorsten: float,
    bvalue_g: float,
    bvalue_g_lin_max: float,
    bvalue_thorsten: float,
) -> dict:
    row = template.to_dict()
    row["direction"] = direction
    row["b_step"] = int(b_step)
    row["bvalue"] = float(bvalue)
    row["D_proj"] = float(D_proj)
    row["g"] = g
    row["g_max"] = g_max
    row["g_lin_max"] = g_lin_max
    row["g_thorsten"] = g_thorsten
    row["bvalue_g"] = bvalue_g
    row["bvalue_g_lin_max"] = bvalue_g_lin_max
    row["bvalue_thorsten"] = bvalue_thorsten
    return row


def design_matrix(n_dirs: np.ndarray) -> np.ndarray:
    """A @ d = y, con d = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]."""
    nx, ny, nz = n_dirs[:, 0], n_dirs[:, 1], n_dirs[:, 2]
    A = np.stack([nx * nx, ny * ny, nz * nz, 2 * nx * ny, 2 * nx * nz, 2 * ny * nz], axis=1)
    return A


def vec6_to_tensor(d: np.ndarray) -> np.ndarray:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = d
    return np.array(
        [
            [Dxx, Dxy, Dxz],
            [Dxy, Dyy, Dyz],
            [Dxz, Dyz, Dzz],
        ],
        dtype=float,
    )


def fit_tensor_from_signals(
    b: float,
    s_norm: np.ndarray,
    n_dirs: np.ndarray,
    *,
    solver: str = "lstsq",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Ajusta tensor D desde señales normalizadas (S/S0) por direction:
      -log(S/S0)/b = n^T D n
    """
    s_norm = np.asarray(s_norm, dtype=float)
    if np.any(~np.isfinite(s_norm)):
        raise ValueError("s_norm contiene NaN/inf")

    s_clip = np.clip(s_norm, eps, None)
    y = -np.log(s_clip) / float(b)

    A = design_matrix(n_dirs)

    if solver == "solve" and A.shape[0] == A.shape[1]:
        d = np.linalg.solve(A, y)
    else:
        d, *_ = np.linalg.lstsq(A, y, rcond=None)

    return vec6_to_tensor(d)


def D_proj(D: np.ndarray, n: np.ndarray) -> float:
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-15)
    return float(n.T @ D @ n)


@dataclass(frozen=True)
class RotResult:
    rotated_signal_long: pd.DataFrame
    dproj_long: pd.DataFrame


def rotate_signals_tensor(
    df_long: pd.DataFrame,
    *,
    stat_avg: str = "avg",
    s0_mode: str = "dir1",
    solver: str = "lstsq",
    b_col: str = "bvalue",
    gamma: float = 267.5221900,
    g_type: str = "g_lin_max",
    dirs_csv: str | Path | None = None,
) -> RotResult:
    """
    Toma el long DF de 1 archivo y produce:
      1) señales rotadas con el mismo formato limpio que process_one_results
      2) una tabla D_proj por eje rotado
    """
    if b_col != "bvalue":
        raise ValueError("rotate_signals_tensor espera tablas limpias y usa b_col='bvalue'.")

    clean = finalize_clean_signal_long(df_long)
    gradient_axis_kind = _unique_str_from_aliases(clean, ["gradient_axis_kind"])
    if gradient_axis_kind == "g":
        derived_cols = ["bvalue", "bvalue_g", "bvalue_g_lin_max", "bvalue_thorsten"]
        has_any_b_axis = any(
            col in clean.columns and pd.to_numeric(clean[col], errors="coerce").notna().any()
            for col in derived_cols
        )
        if not has_any_b_axis:
            raise ValueError(
                "rotate_signals_tensor does not support direct g-only inputs yet. "
                "This step needs a real b-value axis or precomputed bvalue_* columns."
            )

    dfa = clean[clean["stat"] == stat_avg].copy()
    if dfa.empty:
        raise ValueError(f"No hay filas con stat='{stat_avg}'.")

    Nval = _unique_scalar_from_aliases(dfa, ["N"], cast=int)
    delta_ms = _unique_scalar_from_aliases(dfa, ["delta_ms"], cast=float)
    delta_app_ms = _unique_scalar_from_aliases(dfa, ["Delta_app_ms"], cast=float)
    can_compute_b_from_g = (Nval is not None) and (delta_ms is not None) and (delta_app_ms is not None)

    ndirs = int(pd.Series(dfa["direction"]).dropna().nunique())
    if dirs_csv is not None:
        n_dirs = load_dirs_csv(dirs_csv)
        if n_dirs.shape[0] != ndirs:
            raise ValueError(f"dirs_csv tiene {n_dirs.shape[0]} filas, pero el dataset tiene ndirs={ndirs}.")
    else:
        n_dirs = load_default_dirs(ndirs)

    rotated_rows: list[dict] = []
    dproj_rows: list[dict] = []

    for roi, d_roi in dfa.groupby("roi", sort=False):
        d_roi = d_roi.sort_values(["direction", "b_step"], kind="stable")
        d_b0 = d_roi[pd.to_numeric(d_roi["b_step"], errors="coerce") == 0].copy()
        if d_b0.empty:
            raise ValueError(f"ROI={roi}: no encontré b_step==0 (S0).")

        dir1_b0 = d_b0[d_b0["direction"] == 1]
        if s0_mode == "dir1":
            if dir1_b0.empty:
                raise ValueError(f"ROI={roi}: no encontré direction==1 en b0 para s0_mode='dir1'.")
            b0_template = dir1_b0.iloc[0].copy()
            S0 = float(pd.to_numeric(dir1_b0["value"], errors="coerce").iloc[0])
        elif s0_mode == "mean":
            b0_template = d_b0.iloc[0].copy()
            S0 = float(pd.to_numeric(d_b0["value"], errors="coerce").mean())
        else:
            raise ValueError("s0_mode debe ser 'dir1' o 'mean'.")

        d_dir1_nz = d_roi[(d_roi["direction"] == 1) & (pd.to_numeric(d_roi["b_step"], errors="coerce") > 0)].copy()
        if d_dir1_nz.empty:
            raise ValueError(f"ROI={roi}: no hay datos dir1 con b_step>0 para calcular rotación.")

        max_step = int(pd.to_numeric(d_dir1_nz["b_step"], errors="coerce").max())
        idx_maxb = pd.to_numeric(d_dir1_nz["bvalue"], errors="coerce").idxmax()
        max_row = d_dir1_nz.loc[idx_maxb]
        g_max_for_lin = _first_finite(pd.Series([max_row.get("g_max", np.nan), max_row.get("g", np.nan)]))
        if not np.isfinite(g_max_for_lin):
            raise ValueError(f"ROI={roi}: no pude inferir g_max para g_lin_max.")

        for direction_name in ROTATED_DIRECTION_ORDER:
            rotated_rows.append(
                _build_signal_row(
                    b0_template,
                    direction=direction_name,
                    b_step=0,
                    bvalue=0.0,
                    value=S0,
                    S0=S0,
                    g=0.0,
                    g_max=0.0,
                    g_lin_max=0.0,
                    g_thorsten=0.0,
                    bvalue_g=0.0,
                    bvalue_g_lin_max=0.0,
                    bvalue_thorsten=0.0,
                )
            )

        for b_step, d_bs in d_roi[pd.to_numeric(d_roi["b_step"], errors="coerce") > 0].groupby("b_step", sort=False):
            d_bs = d_bs.sort_values("direction", kind="stable")
            if len(d_bs) != ndirs:
                raise ValueError(f"ROI={roi}, b_step={b_step}: esperaba {ndirs} dirs, tengo {len(d_bs)}.")

            dir1_rows = d_bs[d_bs["direction"] == 1]
            if dir1_rows.empty:
                raise ValueError(f"ROI={roi}, b_step={b_step}: no encontré direction==1.")
            template = dir1_rows.iloc[0].copy()

            g_dir1 = _first_finite(dir1_rows["g"]) if "g" in dir1_rows.columns else np.nan
            g_th_dir1 = _first_finite(dir1_rows["g_thorsten"]) if "g_thorsten" in dir1_rows.columns else np.nan
            g_lin = float(g_max_for_lin) * (float(b_step) / float(max_step))

            bvalue_g = _first_finite(dir1_rows["bvalue_g"]) if "bvalue_g" in dir1_rows.columns else np.nan
            bvalue_g_lin_max = _first_finite(dir1_rows["bvalue_g_lin_max"]) if "bvalue_g_lin_max" in dir1_rows.columns else np.nan
            bvalue_thorsten = _first_finite(dir1_rows["bvalue_thorsten"]) if "bvalue_thorsten" in dir1_rows.columns else np.nan

            if can_compute_b_from_g:
                if np.isfinite(g_dir1):
                    bvalue_g = float(
                        b_from_g(
                            np.array([g_dir1]),
                            N=float(Nval),
                            gamma=float(gamma),
                            delta_ms=float(delta_ms),
                            delta_app_ms=float(delta_app_ms),
                            g_type="g",
                        )[0]
                    )
                bvalue_g_lin_max = float(
                    b_from_g(
                        np.array([g_lin]),
                        N=float(Nval),
                        gamma=float(gamma),
                        delta_ms=float(delta_ms),
                        delta_app_ms=float(delta_app_ms),
                        g_type="g_lin_max",
                    )[0]
                )
                if np.isfinite(g_th_dir1):
                    bvalue_thorsten = float(
                        b_from_g(
                            np.array([g_th_dir1]),
                            N=float(Nval),
                            gamma=float(gamma),
                            delta_ms=float(delta_ms),
                            delta_app_ms=float(delta_app_ms),
                            g_type="g_thorsten",
                        )[0]
                    )

            if g_type == "g":
                if not np.isfinite(bvalue_g):
                    raise ValueError(f"ROI={roi}, b_step={b_step}: g_type='g' pero falta bvalue_g.")
                b_fit = bvalue_g
            elif g_type == "g_thorsten":
                if not np.isfinite(bvalue_thorsten):
                    raise ValueError(f"ROI={roi}, b_step={b_step}: g_type='g_thorsten' pero falta bvalue_thorsten.")
                b_fit = bvalue_thorsten
            else:
                if not np.isfinite(bvalue_g_lin_max):
                    raise ValueError(f"ROI={roi}, b_step={b_step}: g_type='g_lin_max' pero falta bvalue_g_lin_max.")
                b_fit = bvalue_g_lin_max

            s = pd.to_numeric(d_bs["value"], errors="coerce").to_numpy(dtype=float)
            s_norm = s / S0
            D = fit_tensor_from_signals(b=b_fit, s_norm=s_norm, n_dirs=n_dirs, solver=solver)

            e_vals, e_vecs = np.linalg.eigh(D)
            idx = np.argsort(e_vals)[::-1]
            v1 = e_vecs[:, idx[0]]
            v2 = e_vecs[:, idx[1]]
            v3 = e_vecs[:, idx[2]]

            axes_full = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
                "eig1": v1,
                "eig2": v2,
                "eig3": v3,
                "long": np.array([1.0, 0.0, 0.0]),
            }

            signal_cache: dict[str, float] = {}
            for axis_name, axis_vec in axes_full.items():
                Dp = D_proj(D, axis_vec)
                S = S0 * np.exp(-b_fit * Dp)
                signal_cache[axis_name] = S

                rotated_rows.append(
                    _build_signal_row(
                        template,
                        direction=axis_name,
                        b_step=int(b_step),
                        bvalue=b_fit,
                        value=S,
                        S0=S0,
                        g=g_dir1,
                        g_max=g_max_for_lin,
                        g_lin_max=g_lin,
                        g_thorsten=g_th_dir1,
                        bvalue_g=bvalue_g,
                        bvalue_g_lin_max=bvalue_g_lin_max,
                        bvalue_thorsten=bvalue_thorsten,
                    )
                )

                if axis_name in {"x", "y", "z", "eig1", "eig2", "eig3"}:
                    dproj_rows.append(
                        _build_dproj_row(
                            template,
                            direction=axis_name,
                            b_step=int(b_step),
                            bvalue=b_fit,
                            D_proj=Dp,
                            g=g_dir1,
                            g_max=g_max_for_lin,
                            g_lin_max=g_lin,
                            g_thorsten=g_th_dir1,
                            bvalue_g=bvalue_g,
                            bvalue_g_lin_max=bvalue_g_lin_max,
                            bvalue_thorsten=bvalue_thorsten,
                        )
                    )

            for k in range(ndirs):
                dproj_rows.append(
                    _build_dproj_row(
                        template,
                        direction=f"dir{k+1}",
                        b_step=int(b_step),
                        bvalue=b_fit,
                        D_proj=D_proj(D, n_dirs[k]),
                        g=g_dir1,
                        g_max=g_max_for_lin,
                        g_lin_max=g_lin,
                        g_thorsten=g_th_dir1,
                        bvalue_g=bvalue_g,
                        bvalue_g_lin_max=bvalue_g_lin_max,
                        bvalue_thorsten=bvalue_thorsten,
                    )
                )

            rotated_rows.append(
                _build_signal_row(
                    template,
                    direction="tra",
                    b_step=int(b_step),
                    bvalue=b_fit,
                    value=0.5 * (signal_cache["y"] + signal_cache["z"]),
                    S0=S0,
                    g=g_dir1,
                    g_max=g_max_for_lin,
                    g_lin_max=g_lin,
                    g_thorsten=g_th_dir1,
                    bvalue_g=bvalue_g,
                    bvalue_g_lin_max=bvalue_g_lin_max,
                    bvalue_thorsten=bvalue_thorsten,
                )
            )

    df_rot = pd.DataFrame(rotated_rows)
    df_rot["direction"] = pd.Categorical(df_rot["direction"], categories=ROTATED_DIRECTION_ORDER, ordered=True)
    df_rot = df_rot.sort_values(["stat", "roi", "direction", "b_step"], kind="stable").reset_index(drop=True)
    df_rot["direction"] = df_rot["direction"].astype(str)
    df_rot = finalize_clean_signal_long(df_rot)

    df_dproj = pd.DataFrame(dproj_rows).sort_values(["roi", "direction", "b_step"], kind="stable").reset_index(drop=True)
    df_dproj = finalize_clean_dproj_long(df_dproj)

    return RotResult(rotated_signal_long=df_rot, dproj_long=df_dproj)
