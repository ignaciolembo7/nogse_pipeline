from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_MM2_S_TO_M2_MS = 1e-9
_M2_MS_TO_MM2_S = 1e9


COMMON_COLS = [
    "source_file",
    "roi",
    "direction",
    "max_dur_ms",
    "tm_ms",
    "td_ms",
    "fit_kind",
    "model",
    "ycol",
    "stat",
    "n_points",
    "n_fit",
    "M0",
    "M0_err",
    "D0_m2_ms",
    "D0_err_m2_ms",
    "D0_mm2_s",
    "D0_err_mm2_s",
    "rmse",
    "chi2",
    "method",
    "ok",
    "msg",
]

MONOEXP_COLS = [
    "g_type",
    "fit_points",
    "N",
]

NOGSE_CONTRAST_COLS = [
    "gbase",
    "N1",
    "N2",
    "f_corr",
    "tc_ms",
    "tc_err_ms",
    "alpha",
    "alpha_err",
]


def _schema_cols_for_kind(fit_kind: str) -> list[str]:
    if fit_kind == "monoexp":
        return COMMON_COLS + MONOEXP_COLS
    if fit_kind == "nogse_contrast":
        return COMMON_COLS + NOGSE_CONTRAST_COLS
    raise ValueError(f"fit_kind desconocido: {fit_kind}")


def standardize_fit_params(
    df: pd.DataFrame,
    *,
    fit_kind: str,
    source_file: Optional[str] = None,
) -> pd.DataFrame:
    """Normaliza nombres, agrega columnas faltantes y deja solo el schema útil para cada tipo de fit."""
    out = df.copy()
    target_cols = _schema_cols_for_kind(fit_kind)

    if "source_file" not in out.columns:
        out["source_file"] = str(source_file) if source_file is not None else ""
    elif source_file is not None:
        out["source_file"] = out["source_file"].fillna(str(source_file)).replace("", str(source_file))

    if "direction" not in out.columns and "axis" in out.columns:
        out["direction"] = out["axis"].astype(str)
    if "axis" in out.columns:
        out = out.drop(columns=["axis"])

    if "D0_m2_ms" not in out.columns and "D0" in out.columns:
        out["D0_m2_ms"] = pd.to_numeric(out["D0"], errors="coerce")
    if "D0_err_m2_ms" not in out.columns and "D0_err" in out.columns:
        out["D0_err_m2_ms"] = pd.to_numeric(out["D0_err"], errors="coerce")

    if "D0_m2_ms" not in out.columns and "D0_mm2_s" in out.columns:
        out["D0_m2_ms"] = pd.to_numeric(out["D0_mm2_s"], errors="coerce") * _MM2_S_TO_M2_MS
    if "D0_err_m2_ms" not in out.columns and "D0_err_mm2_s" in out.columns:
        out["D0_err_m2_ms"] = pd.to_numeric(out["D0_err_mm2_s"], errors="coerce") * _MM2_S_TO_M2_MS

    if "D0_mm2_s" not in out.columns and "D0_m2_ms" in out.columns:
        out["D0_mm2_s"] = pd.to_numeric(out["D0_m2_ms"], errors="coerce") * _M2_MS_TO_MM2_S
    if "D0_err_mm2_s" not in out.columns and "D0_err_m2_ms" in out.columns:
        out["D0_err_mm2_s"] = pd.to_numeric(out["D0_err_m2_ms"], errors="coerce") * _M2_MS_TO_MM2_S

    out["fit_kind"] = fit_kind
    if "ok" not in out.columns:
        out["ok"] = True
    if "msg" not in out.columns:
        out["msg"] = ""

    for c in target_cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[target_cols].copy()
