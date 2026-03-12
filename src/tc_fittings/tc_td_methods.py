# src/ogse_contrast/tc_td_methods.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import least_squares

@dataclass(frozen=True)
class TcTdMethod:
    name: str
    model: str                 # "linear" (por ahora)
    loss: str                  # "linear" | "huber" | "soft_l1" | "cauchy" | "arctan"
    k_last: Optional[int]      # None => usar todos los puntos
    f_scale: str | float = "auto"  # "auto" o número
    description: str = ""

def _auto_f_scale(y: np.ndarray) -> float:
    # escala robusta (MAD) del y como valor razonable por defecto
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    s = 1.4826 * mad
    return float(s) if s > 0 else 1.0

def fit_linear(x: np.ndarray, y: np.ndarray, loss: str = "linear", f_scale: str | float = "auto") -> Tuple[float, float, float, float, float]:
    """
    Devuelve: slope, intercept, slope_se, intercept_se, r2
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # OLS rápido con stderr exacto
    if loss in (None, "linear"):
        res = linregress(x, y)
        slope = float(res.slope)
        intercept = float(res.intercept)
        slope_se = float(res.stderr) if res.stderr is not None else np.nan
        intercept_se = float(getattr(res, "intercept_stderr", np.nan))
        r2 = float(res.rvalue ** 2)
        return slope, intercept, slope_se, intercept_se, r2

    # Robust (Huber/soft_l1/etc) usando least_squares
    p0 = np.polyfit(x, y, 1)  # inicialización
    fs = _auto_f_scale(y) if f_scale == "auto" else float(f_scale)

    def fun(p):
        return (p[0] * x + p[1] - y)

    ls = least_squares(fun, x0=p0, loss=loss, f_scale=fs)
    slope, intercept = map(float, ls.x)

    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # stderr aproximado (Gauss-Newton)
    slope_se = intercept_se = np.nan
    if ls.jac is not None and len(x) > 2:
        J = ls.jac
        dof = max(0, len(y) - 2)
        if dof > 0:
            s2 = float(ls.cost * 2 / dof)  # cost = 0.5*sum(res^2)
            try:
                cov = np.linalg.inv(J.T @ J) * s2
                se = np.sqrt(np.diag(cov))
                slope_se = float(se[0])
                intercept_se = float(se[1])
            except np.linalg.LinAlgError:
                pass

    return slope, intercept, slope_se, intercept_se, r2

def _shade(color: str, factor: float) -> Tuple[float, float, float]:
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    white = (1, 1, 1)
    return tuple(white[i] + factor * (rgb[i] - white[i]) for i in range(3))

def run_tc_td_fit(
    df_params: pd.DataFrame,
    out_dir: Path,
    method: TcTdMethod,
    region2color: Dict[str, str],
    brains: list[str],
    y_col: str,
    y_label: str,
    fig_prefix: str,
) -> pd.DataFrame:
    """
    Ajusta y = f(Td) por (roi, brain, direction). Guarda:
    - figuras por roi (2x3)
    - tabla xlsx con slope/intercept (+ errores, r2)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ajustes = []

    for dir_actual in ["long", "tra"]:
        df_dir = df_params[df_params["direction"] == dir_actual]
        regiones = sorted(df_dir["roi"].unique())

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
        for ax, region in zip(axes.flat, regiones):
            base_color = region2color[region]
            for i, brain in enumerate(brains):
                sub = df_dir[(df_dir["roi"] == region) & (df_dir["brain"] == brain)].sort_values("td_ms")
                if sub.empty:
                    continue

                x = sub["td_ms"].to_numpy()
                y = sub[y_col].to_numpy()

                if method.k_last is not None and len(x) > method.k_last:
                    x_fit = x[-method.k_last :]
                    y_fit = y[-method.k_last :]
                else:
                    x_fit = x
                    y_fit = y

                slope, intercept, slope_se, intercept_se, r2 = fit_linear(
                    x_fit, y_fit, loss=method.loss, f_scale=method.f_scale
                )

                ajustes.append({
                    "roi": region,
                    "brain": brain,
                    "direction": dir_actual,
                    "method": method.name,
                    "loss": method.loss,
                    "k_last": method.k_last,
                    "f_scale": method.f_scale,
                    "pendiente": slope,
                    "error_pendiente": slope_se,
                    "ordenada": intercept,
                    "error_ordenada": intercept_se,
                    "r2": r2,
                })

                # plot
                ax.plot(x, y, "o", color=_shade(base_color, [0.25, 0.5, 1.0][i % 3]), label=brain)
                if np.isfinite(slope) and np.isfinite(intercept):
                    xx = np.linspace(x_fit.min(), x_fit.max(), 50)
                    ax.plot(xx, slope * xx + intercept, "-", color=_shade(base_color, [0.25, 0.5, 1.0][i % 3]), linewidth=2)

            ax.set_title(region, fontsize=14)
            ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
            ax.set_ylabel(y_label, fontsize=18)
            ax.grid(True)
            ax.legend(fontsize=10)

        plt.suptitle(f"{fig_prefix} | method={method.name} | dir={dir_actual}", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_dir / f"{fig_prefix}_dir={dir_actual}_{method.name}.png", dpi=300)
        plt.close()

    df_aj = pd.DataFrame(ajustes)
    df_aj.to_excel(out_dir / f"ajustes_{fig_prefix}_{method.name}.xlsx", index=False)
    return df_aj

# Registry: agregar métodos acá (1 línea por método)
METHODS: Dict[str, TcTdMethod] = {
    "ols_last3": TcTdMethod(
        name="ols_last3", model="linear", loss="linear", k_last=3, f_scale="auto",
        description="OLS lineal usando últimos 3 puntos (rápido, como linregress)."
    ),
    "huber_last3": TcTdMethod(
        name="huber_last3", model="linear", loss="huber", k_last=3, f_scale="auto",
        description="Huber lineal usando últimos 3 puntos (robusto a outliers)."
    ),
    "ols_all": TcTdMethod(
        name="ols_all", model="linear", loss="linear", k_last=None, f_scale="auto",
        description="OLS lineal usando todos los puntos."
    ),
    "huber_all": TcTdMethod(
        name="huber_all", model="linear", loss="huber", k_last=None, f_scale="auto",
        description="Huber lineal usando todos los puntos."
    ),
}
