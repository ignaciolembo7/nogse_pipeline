from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nogse_models.nogse_model_fitting import OGSE_contrast_vs_g_free, OGSE_contrast_vs_g_tort, OGSE_contrast_vs_g_rest
from ogse_fitting.fit_ogse_contrast import _gcols, _maybe_scale_g_thorsten

@dataclass(frozen=True)
class PeakResult:
    method: str                 # "param" | "nonparam"
    xplot: str                  # "1" o "2"
    f_peak: float
    g1_peak: float
    g2_peak: float
    y_peak: float


def peak_from_fit_row(fit_row: dict, *, n_grid: int = 2048) -> PeakResult:
    """
    Pico del modelo ajustado evaluado en una grilla f∈[0,1] escalando con g1_max/g2_max.
    Requiere fit_row.ok == True y params válidos.
    """
    if not fit_row.get("ok", True):
        raise ValueError("fit_row no OK; no puedo sacar pico paramétrico.")

    model = str(fit_row["model"])
    td_ms = float(fit_row["td_ms"])
    n_1 = int(fit_row["N_1"])
    n_2 = int(fit_row["N_2"])
    xplot = str(fit_row.get("xplot", "1"))

    g1_max = float(fit_row["g1_max"])
    g2_max = float(fit_row["g2_max"])

    f = np.linspace(0.0, 1.0, int(n_grid))
    G1 = f * g1_max
    G2 = f * g2_max

    if model == "free":
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        y = OGSE_contrast_vs_g_free(td_ms, G1, G2, n_1, n_2, M0, D0)
    elif model == "tort":
        alpha = float(fit_row["alpha"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        y = OGSE_contrast_vs_g_tort(td_ms, G1, G2, n_1, n_2, alpha, M0, D0)
    elif model == "rest":
        tc_ms = float(fit_row["tc_ms"])
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_m2_ms"])
        y = OGSE_contrast_vs_g_rest(td_ms, G1, G2, n_1, n_2, tc_ms, M0, D0)
    else:
        raise ValueError(f"peak_from_fit_row: modelo '{model}' no soportado.")

    i = int(np.nanargmax(y))
    return PeakResult(
        method="param",
        xplot=xplot,
        f_peak=float(f[i]),
        g1_peak=float(G1[i]),
        g2_peak=float(G2[i]),
        y_peak=float(y[i]),
    )


def peak_from_data_group(
    df_group: pd.DataFrame,
    *,
    gbase: str,
    ycol: str,
    xplot: str = "1",
    f_corr: float = 1.0,
    smooth: bool = True,
    smooth_window: int = 7,
    smooth_poly: int = 2,
) -> PeakResult:
    """
    Pico NO paramétrico: máximo de y(g) en los datos, con opcional suavizado Savitzky-Golay.
    """
    from scipy.signal import savgol_filter  # type: ignore

    g1c, g2c = _gcols(gbase)

    y = pd.to_numeric(df_group[ycol], errors="coerce").to_numpy(float)
    G1 = pd.to_numeric(df_group[g1c], errors="coerce").to_numpy(float)
    G2 = pd.to_numeric(df_group[g2c], errors="coerce").to_numpy(float)

    G1 = _maybe_scale_g_thorsten(gbase, G1) * float(f_corr)
    G2 = _maybe_scale_g_thorsten(gbase, G2) * float(f_corr)

    m = np.isfinite(y) & np.isfinite(G1) & np.isfinite(G2)
    y, G1, G2 = y[m], G1[m], G2[m]

    if len(y) < 3:
        raise ValueError("Muy pocos puntos para pico no paramétrico.")

    x = G1 if str(xplot) == "1" else G2
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    G1 = G1[order]
    G2 = G2[order]

    if smooth:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        w = min(w, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if w >= 5:
            y_s = savgol_filter(y, window_length=w, polyorder=int(smooth_poly))
        else:
            y_s = y
    else:
        y_s = y

    i = int(np.nanargmax(y_s))

    # f_peak lo definimos por fracción relativa a máximos para mantener compatibilidad
    g1_max = float(np.nanmax(G1))
    g2_max = float(np.nanmax(G2))
    f_peak = float(G1[i] / g1_max) if str(xplot) == "1" and g1_max > 0 else float(G2[i] / g2_max) if g2_max > 0 else np.nan

    return PeakResult(
        method="nonparam",
        xplot=str(xplot),
        f_peak=float(f_peak),
        g1_peak=float(G1[i]),
        g2_peak=float(G2[i]),
        y_peak=float(y_s[i]),
    )


def tc_from_peak(
    *,
    g_peak_mTpm: float,
    delta_ms: float,
    D0_m2_per_ms: float,
    tc_factor: float = 2.0,
    gamma_rad_per_s_T: float = 2.0 * np.pi * 42.577478518e6,  # rad/s/T
) -> float:
    """
    Conversión “genérica” para tu pipeline:
      q = gamma * G * delta
      l = 1/q
      tc = l^2 / (tc_factor * D0)

    Unidades asumidas:
      - G en mT/m  -> se convierte a T/m con 1e-3
      - delta en ms
      - D0 en m^2/ms  (coherente con usar Td en ms en tus modelos)
      - gamma en rad/s/T (se convierte a rad/ms/T multiplicando 1e-3)
    """
    G_Tpm = float(g_peak_mTpm) * 1e-3
    gamma_rad_per_ms_T = float(gamma_rad_per_s_T) * 1e-3
    q = gamma_rad_per_ms_T * G_Tpm * float(delta_ms)  # rad/m
    if q <= 0:
        return float("nan")
    l = 1.0 / q  # m
    tc_ms = (l * l) / (float(tc_factor) * float(D0_m2_per_ms))
    return float(tc_ms)
