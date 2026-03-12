from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# Conversión:
# 1 mm^2/s = 1e-9 m^2/ms
_MM2_S_TO_M2_MS = 1e-9
_M2_MS_TO_MM2_S = 1e9


# Schema único: mismas columnas para monoexp y nogse-contrast
SCHEMA_COLS = [
    # IDs / matching
    "source",          # ID del experimento (string), clave para joins
    "exp_name",        # alias opcional (si ya lo usás)
    "roi",
    # "axis",            # canonical
    "direction",       # alias = axis (para compatibilidad)

    # Timing
    "max_dur_ms",
    "tm_ms",
    "td_ms",
    "delta_ms",
    "delta_app_ms",

    # Fit config
    "fit_kind",        # "monoexp" / "nogse_contrast"
    "model",           # free/tort/monoexp
    "gbase",           # para contraste
    "g_type",          # para monoexp
    "ycol",
    "stat",
    "fit_points",
    "N",               # monoexp (si aplica)
    "N1",              # contraste
    "N2",              # contraste
    "n_points",
    "n_fit",

    # Corrección gradientes
    "f_corr",

    # Parámetros comunes
    "M0",
    "M0_err",
    "D0_m2_ms",
    "D0_err_m2_ms",
    "D0_mm2_s",
    "D0_err_mm2_s",
    # Parámetro específico (restricted)
    "tc_ms",
    "tc_err_ms",

    # Parámetros específicos (tortuoso)
    "alpha",
    "alpha_err",

    # Métricas
    "rmse",
    "chi2",
    "method",
    "ok",
    "msg",
]


def standardize_fit_params(
    df: pd.DataFrame,
    *,
    fit_kind: str,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Normaliza nombres, agrega columnas faltantes, y ordena según SCHEMA_COLS."""
    out = df.copy()

    # --- source / exp_name ---
    if "source" not in out.columns:
        if source is not None:
            out["source"] = str(source)
        elif "exp_name" in out.columns:
            out["source"] = out["exp_name"].astype(str)
        else:
            out["source"] = ""

    if "exp_name" not in out.columns:
        out["exp_name"] = out["source"].astype(str)

    # --- axis/direction: siempre ambos ---
    # if "axis" not in out.columns and "direction" in out.columns:
    #     out["axis"] = out["direction"].astype(str)
    if "direction" not in out.columns and "axis" in out.columns:
        out["direction"] = out["axis"].astype(str)
    # if "axis" in out.columns:
    #     out["axis"] = out["axis"].astype(str)
    #     out["direction"] = out["axis"].astype(str)

    # --- Unidades D0: asegurar ambas ---
    # Si viene D0 (como en nogse-contrast) lo tratamos como m^2/ms
    if "D0_m2_ms" not in out.columns and "D0" in out.columns:
        out["D0_m2_ms"] = pd.to_numeric(out["D0"], errors="coerce")
    if "D0_err_m2_ms" not in out.columns and "D0_err" in out.columns:
        out["D0_err_m2_ms"] = pd.to_numeric(out["D0_err"], errors="coerce")

    if "D0_mm2_s" not in out.columns and "D0_mm2_s" in out.columns:
        pass  # ya está

    # monoexp trae D0_mm2_s -> generamos D0_m2_ms
    if "D0_m2_ms" not in out.columns and "D0_mm2_s" in out.columns:
        out["D0_m2_ms"] = pd.to_numeric(out["D0_mm2_s"], errors="coerce") * _MM2_S_TO_M2_MS
    if "D0_err_m2_ms" not in out.columns and "D0_err_mm2_s" in out.columns:
        out["D0_err_m2_ms"] = pd.to_numeric(out["D0_err_mm2_s"], errors="coerce") * _MM2_S_TO_M2_MS

    # nogse trae D0_m2_ms -> generamos D0_mm2_s
    if "D0_mm2_s" not in out.columns and "D0_m2_ms" in out.columns:
        out["D0_mm2_s"] = pd.to_numeric(out["D0_m2_ms"], errors="coerce") * _M2_MS_TO_MM2_S
    if "D0_err_mm2_s" not in out.columns and "D0_err_m2_ms" in out.columns:
        out["D0_err_mm2_s"] = pd.to_numeric(out["D0_err_m2_ms"], errors="coerce") * _M2_MS_TO_MM2_S

    # --- defaults comunes ---
    out["fit_kind"] = fit_kind
    if "ok" not in out.columns:
        out["ok"] = True
    if "msg" not in out.columns:
        out["msg"] = ""

    # --- agregar columnas faltantes ---
    for c in SCHEMA_COLS:
        if c not in out.columns:
            out[c] = np.nan

    # --- ordenar ---
    out = out[SCHEMA_COLS].copy()
    return out
