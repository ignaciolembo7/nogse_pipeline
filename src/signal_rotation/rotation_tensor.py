from __future__ import annotations

from signal_rotation.dirs import load_dirs_csv, load_default_dirs
from dataclasses import dataclass
from data_processing.schema import ensure_signal_long_schema, ensure_dproj_long_schema
import numpy as np
import pandas as pd
from pathlib import Path
from ogse_fitting.b_from_g import b_from_g

def design_matrix(n_dirs: np.ndarray) -> np.ndarray:
    """A @ d = y, con d = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]."""
    nx, ny, nz = n_dirs[:, 0], n_dirs[:, 1], n_dirs[:, 2]
    A = np.stack([nx*nx, ny*ny, nz*nz, 2*nx*ny, 2*nx*nz, 2*ny*nz], axis=1)
    return A

def vec6_to_tensor(d: np.ndarray) -> np.ndarray:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = d
    return np.array([
        [Dxx, Dxy, Dxz],
        [Dxy, Dyy, Dyz],
        [Dxz, Dyz, Dzz],
    ], dtype=float)

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
    gamma: float = 267.5221900, # 1/ms.mT 
    g_type: str = "g_lin_max",   # "g", "g_lin_max", "gthorsten"
    dirs_csv: str | Path | None = None, 
) -> RotResult:

    """
    Toma el long DF de 1 archivo (1 N) y produce:
      1) df_rot: señales rotadas en formato long (axis = x/y/z/longitudinal/transversal_1/transversal_2)
      2) df_dproj: D_proj por axis (útil para QC)
    """
    Nval = None
    if "param_N" in df_long.columns:
        u = pd.Series(df_long["param_N"]).dropna().unique()
        if len(u) == 1:
            Nval = int(u[0])

    delta_ms = None
    delta_app_ms = None

    if "param_delta_ms" in df_long.columns:
        u = pd.Series(df_long["param_delta_ms"]).dropna().unique()
        if len(u) == 1:
            delta_ms = float(u[0])

    if "param_delta_app_ms" in df_long.columns:
        u = pd.Series(df_long["param_delta_app_ms"]).dropna().unique()
        if len(u) == 1:
            delta_app_ms = float(u[0])

    if Nval is None or delta_ms is None or delta_app_ms is None:
        raise ValueError("Faltan parámetros para b_from_g: necesito param_N, param_delta_ms y param_delta_app_ms.")

    req = {"stat", "direction", "b_step", "roi", "value", b_col}
    missing = req - set(df_long.columns)
    if missing:
        raise ValueError(f"Faltan columnas en df_long: {missing}")

    # Filtramos a avg (como notebook: lee sheet 'avg')
    dfa = df_long[df_long["stat"] == stat_avg].copy()
    if dfa.empty:
        raise ValueError(f"No hay filas con stat='{stat_avg}'.")
    param_cols = [c for c in dfa.columns if c.startswith("param_")]
    meta_cols = [c for c in dfa.columns if c.startswith("meta_")]
    if "source_file" in dfa.columns:
        meta_cols.append("source_file")

    ndirs = int(pd.Series(dfa["direction"]).dropna().nunique())

    if dirs_csv is not None:
        n_dirs = load_dirs_csv(dirs_csv)
        if n_dirs.shape[0] != ndirs:
            raise ValueError(f"dirs_csv tiene {n_dirs.shape[0]} filas, pero el dataset tiene ndirs={ndirs}.")
    else:
        n_dirs = load_default_dirs(ndirs)

    # Ejes destino
    axes = {
        "x": np.array([1, 0, 0], dtype=float),
        "y": np.array([0, 1, 0], dtype=float),
        "z": np.array([0, 0, 1], dtype=float),
    }

    # Asegurar numéricos
    dfa["value"] = pd.to_numeric(dfa["value"], errors="coerce")
    # Convertimos columnas de b si existen
    if "bvalue" in dfa.columns:
        dfa["bvalue"] = pd.to_numeric(dfa["bvalue"], errors="coerce")
    if "bvalue_thorsten" in dfa.columns:
        dfa["bvalue_thorsten"] = pd.to_numeric(dfa["bvalue_thorsten"], errors="coerce")

    out_rows = []
    dproj_rows = []

    # Procesar por ROI
    for roi, d_roi in dfa.groupby("roi", sort=False):
        # S0
        d_b0 = d_roi[d_roi["b_step"] == 0].copy()
        if d_b0.empty:
            raise ValueError(f"ROI={roi}: no encontré b_step==0 (S0).")

        if s0_mode == "dir1":
            s0_row = d_b0[d_b0["direction"] == 1]
            if s0_row.empty:
                raise ValueError(f"ROI={roi}: no encontré direction==1 en b0 para s0_mode='dir1'.")
            S0 = float(s0_row["value"].iloc[0])
        elif s0_mode == "mean":
            S0 = float(d_b0["value"].mean())
        else:
            raise ValueError("s0_mode debe ser 'dir1' o 'mean'.")
        
        # --- g_lin_max correcto: usar g en el punto de bvalue_orig máximo (dir1) y escalar por b_step
        d_dir1_nz = d_roi[(d_roi["direction"] == 1) & (d_roi["b_step"] > 0)].copy()
        if d_dir1_nz.empty:
            raise ValueError(f"ROI={roi}: no hay datos dir1 con b_step>0 para calcular g_lin_max.")

        max_step = int(pd.to_numeric(d_dir1_nz["b_step"], errors="coerce").max())

        b_for_max = pd.to_numeric(d_dir1_nz[b_col], errors="coerce")
        idx_maxb = b_for_max.idxmax()

        g_base_col = "g" if "g" in d_dir1_nz.columns else ("g_max" if "g_max" in d_dir1_nz.columns else None)
        if g_base_col is None:
            raise ValueError(f"ROI={roi}: no encontré columna 'g' ni 'g_max' para construir g_lin_max.")

        g_max_for_lin = float(pd.to_numeric(d_dir1_nz.loc[idx_maxb, g_base_col], errors="coerce"))
        if not np.isfinite(g_max_for_lin):
            raise ValueError(f"ROI={roi}: g_max_for_lin es NaN/inf.")

        roi_meta = {}
        for c in meta_cols:
            if c in d_roi.columns:
                v = d_roi[c].dropna()
                if not v.empty:
                    roi_meta[c] = v.iloc[0]

        # Para cada b_step > 0, fit tensor con señales por direction
        for b_step, d_bs in d_roi[d_roi["b_step"] > 0].groupby("b_step", sort=False):
  
            # señales por direction, ordenadas 1..ndirs
            d_bs = d_bs.sort_values("direction", kind="stable")
            if len(d_bs) != ndirs:
                raise ValueError(f"ROI={roi}, b_step={b_step}: esperaba {ndirs} dirs, tengo {len(d_bs)}.")

            # --- elegimos la columna de gradiente según g_type
            g_col = None
            if g_type == "g" and "g" in d_bs.columns:
                g_col = "g"
            elif g_type == "g_max" and "g_max" in d_bs.columns:
                g_col = "g_max"
            elif g_type == "gthorsten":
                if "gthorsten" in d_bs.columns:
                    g_col = "gthorsten"
                elif "gthorsten_mTm" in d_bs.columns:
                    g_col = "gthorsten_mTm"
            elif g_type == "g_lin_max" and "g_lin_max" in d_bs.columns:
                g_col = "g_lin_max"

          
            # --- bvalue original (del archivo)
            b_orig_series = pd.to_numeric(d_bs.loc[d_bs["direction"] == 1, b_col], errors="coerce")
            bvalue_orig = float(b_orig_series.iloc[0]) if (not b_orig_series.empty and pd.notna(b_orig_series.iloc[0])) else np.nan

            # --- g (dir1) para este b_step
            g_dir1 = np.nan
            if "g" in d_bs.columns:
                g_dir1_series = pd.to_numeric(d_bs.loc[d_bs["direction"] == 1, "g"], errors="coerce")
                g_dir1 = float(g_dir1_series.iloc[0]) if (not g_dir1_series.empty and pd.notna(g_dir1_series.iloc[0])) else np.nan

            # --- gthorsten (dir1) para este b_step (acepta gthorsten o gthorsten_mTm)
            th_col = "gthorsten" if "gthorsten" in d_bs.columns else ("gthorsten_mTm" if "gthorsten_mTm" in d_bs.columns else None)
            g_th_dir1 = np.nan
            if th_col is not None:
                g_th_series = pd.to_numeric(d_bs.loc[d_bs["direction"] == 1, th_col], errors="coerce")
                g_th_dir1 = float(g_th_series.iloc[0]) if (not g_th_series.empty and pd.notna(g_th_series.iloc[0])) else np.nan

            # --- g_lin_max construido: g_max_for_lin * (b_step / max_step)
            frac = float(b_step) / float(max_step)
            g_lin = g_max_for_lin * frac

            # --- bvalues calculados para cada tipo de g
            bvalue_g = (
                float(b_from_g(np.array([g_dir1]), N=float(Nval), gamma=float(gamma),
                            delta_ms=float(delta_ms), delta_app_ms=float(delta_app_ms), g_type="g")[0])
                if np.isfinite(g_dir1) else np.nan
            )

            bvalue_g_lin_max = float(
                b_from_g(np.array([g_lin]), N=float(Nval), gamma=float(gamma),
                        delta_ms=float(delta_ms), delta_app_ms=float(delta_app_ms), g_type="g_lin_max")[0]
            )

            bvalue_gthorsten = (
                float(b_from_g(np.array([g_th_dir1]), N=float(Nval), gamma=float(gamma),
                            delta_ms=float(delta_ms), delta_app_ms=float(delta_app_ms), g_type="gthorsten")[0])
                if np.isfinite(g_th_dir1) else np.nan
            )

            # --- b usado para el fit / predicción (compatibilidad con tu parámetro g_type)
            if g_type == "g":
                if not np.isfinite(bvalue_g):
                    raise ValueError(f"ROI={roi}, b_step={b_step}: g_type='g' pero falta g (dir1).")
                b = bvalue_g
            elif g_type == "gthorsten":
                if not np.isfinite(bvalue_gthorsten):
                    raise ValueError(f"ROI={roi}, b_step={b_step}: g_type='gthorsten' pero falta gthorsten (dir1).")
                b = bvalue_gthorsten
            else:  # "g_lin_max"
                b = bvalue_g_lin_max

            b_report = b

            s = d_bs["value"].to_numpy(dtype=float)
            s_norm = s / S0

            D = fit_tensor_from_signals(b=b, s_norm=s_norm, n_dirs=n_dirs, solver=solver)

            # eigen-decomp
            e_vals, e_vecs = np.linalg.eigh(D)
            idx = np.argsort(e_vals)[::-1] 
            v1 = e_vecs[:, idx[0]]
            v2 = e_vecs[:, idx[1]]
            v3 = e_vecs[:, idx[2]]

            # ejes canónicos
            axes_full = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
                # eigen-directions (autovalores)
                "eig1": v1,
                "eig2": v2,
                "eig3": v3,
                # definiciones tuyas
                "long": np.array([1.0, 0.0, 0.0]),
            }

            # --- generar filas por eje
            S_y = None
            S_z = None

            # --- extras por b_step (se copian a todas las filas de salida)
            extras = dict(roi_meta)
            extras["meta_gamma"] = float(gamma)

            # 1) arrastrar todos los param_*
            for c in param_cols:
                vals = d_bs[c].dropna()
                if vals.empty:
                    continue
                uniq = vals.astype(str).unique()
                if len(uniq) == 1:
                    extras[c] = vals.iloc[0]
                else:
                    nums = pd.to_numeric(vals, errors="coerce")
                    extras[c] = float(nums.median()) if nums.notna().any() else vals.iloc[0]

            # 2) g explícitos por b_step (NO medianas)
            if np.isfinite(g_dir1):
                extras["g"] = g_dir1
            extras["g_max"] = g_max_for_lin
            extras["g_lin_max"] = g_lin
            if np.isfinite(g_th_dir1):
                extras["gthorsten"] = g_th_dir1



            for axis_name, axis_vec in axes_full.items():
                Dp = D_proj(D, axis_vec)
                S  = S0 * np.exp(-b * Dp)

                if axis_name == "y":
                    S_y = S
                elif axis_name == "z":
                    S_z = S

                row_dict = {
                    "roi": roi,
                    "axis": axis_name,
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "signal": S,
                    "signal_norm": S / S0,
                    "S0": S0,
                }
                row_dict.update(extras)
                row_dict["bvalue_g"] = bvalue_g
                row_dict["bvalue_g_lin_max"] = bvalue_g_lin_max
                row_dict["bvalue_gthorsten"] = bvalue_gthorsten
                row_dict["bvalue_orig"] = bvalue_orig
                out_rows.append(row_dict)

                dproj_dict = {
                    "roi": roi,
                    "axis": axis_name,
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "D_proj": Dp,
                }
                dproj_dict.update(extras)
                dproj_dict["bvalue_g"] = bvalue_g
                dproj_dict["bvalue_g_lin_max"] = bvalue_g_lin_max
                dproj_dict["bvalue_gthorsten"] = bvalue_gthorsten
                dproj_dict["bvalue_orig"] = bvalue_orig
                if axis_name in {"x", "y", "z", "eig1", "eig2", "eig3"}:
                    dproj_rows.append(dproj_dict)

            # --- Dproj también en las direcciones medidas (dir1..dirN)
            for k in range(ndirs):
                Dp_dir = D_proj(D, n_dirs[k])
                dproj_dict = {
                    "roi": roi,
                    "axis": f"dir{k+1}",
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "D_proj": Dp_dir,
                }
                dproj_dict.update(extras)
                dproj_rows.append(dproj_dict)

            # --- tra = promedio de señales y y z (NO vector, NO D_proj)
            if (S_y is not None) and (S_z is not None):
                S_tra = 0.5 * (S_y + S_z)
                row_dict = {
                    "roi": roi,
                    "axis": "tra",
                    "b_step": int(b_step),
                    "bvalue": b_report,
                    "signal": S_tra,
                    "signal_norm": S_tra / S0,
                    "S0": S0,
                }
                row_dict.update(extras)

                out_rows.append(row_dict)

    df_rot = pd.DataFrame(out_rows).sort_values(["roi", "param_N", "axis", "b_step"], kind="stable")
    keep_cols = (
    ["roi", "axis", "S0"]
    + [c for c in df_rot.columns if c.startswith("param_")]
    + [c for c in df_rot.columns if c.startswith("meta_")]
    + (["source_file"] if "source_file" in df_rot.columns else [])
    + [c for c in ["g","g_max","g_lin_max","gthorsten","bvalue_orig"] if c in df_rot.columns]
    )

    b0 = (
        df_rot.groupby(["roi", "axis"], as_index=False)
            .first()[keep_cols]
    )

    b0["b_step"] = 0
    b0["bvalue"] = 0.0
    b0["signal"] = b0["S0"]
    b0["signal_norm"] = 1.0
    for c in ["g", "g_max", "g_lin_max", "gthorsten"]:
        if c in b0.columns:
            b0[c] = 0.0
    if "bvalue_orig" in b0.columns:
        b0["bvalue_orig"] = 0.0
    for c in ["bvalue_g", "bvalue_g_lin_max", "bvalue_gthorsten", "bvalue_orig"]:
        if c in b0.columns:
            b0[c] = 0.0


    df_rot = pd.concat([b0, df_rot], ignore_index=True)

    # --- misma estructura que el long original
    df_rot = df_rot.rename(columns={"axis": "direction", "signal": "value"})
    df_rot["stat"] = stat_avg  # típicamente "avg"

    # --- ordenar direcciones como pediste
    dir_order = ["eig1", "eig2", "eig3", "x", "y", "z", "tra", "long"]
    df_rot["direction"] = pd.Categorical(df_rot["direction"], categories=dir_order, ordered=True)

    # --- orden final: por roi+direction, y dentro de cada curva b_step (0 primero)
    sort_cols = [c for c in ["stat", "roi", "direction", "b_step"] if c in df_rot.columns]
    df_rot = df_rot.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # (opcional) dejar direction como string “normal” en el output
    df_rot["direction"] = df_rot["direction"].astype(str)

    # --- (opcional) ordenar columnas “como siempre”
    first_cols = ["stat", "roi", "direction", "b_step", "bvalue", "value"]
    rest = [c for c in df_rot.columns if c not in first_cols]
    df_rot = df_rot[first_cols + rest]

    df_dproj = pd.DataFrame(dproj_rows).sort_values(["roi", "axis", "b_step"], kind="stable")
    df_rot = ensure_signal_long_schema(df_rot)
    df_dproj = ensure_dproj_long_schema(df_dproj)

    return RotResult(rotated_signal_long=df_rot, dproj_long=df_dproj)