from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nogse_models import nogse_model_fitting

from .config import OGSEFitConfig
from .data_loading import get_f_value, td_ms

def compute_region_colors(cfg: OGSEFitConfig, all_curves: dict[str, pd.DataFrame]) -> dict[str, str]:
    regiones_unicas = sorted({key.split("|")[1] for key in all_curves})
    return {region: cfg.palette[i % len(cfg.palette)] for i, region in enumerate(regiones_unicas)}

def augment_params_and_plot_individual(cfg: OGSEFitConfig, all_curves: dict[str, pd.DataFrame], df_params: pd.DataFrame, tabla_f: pd.DataFrame) -> pd.DataFrame:
    region2color = compute_region_colors(cfg, all_curves)
    out_ind = cfg.plot_dir_out / "individual_plots"
    out_ind.mkdir(parents=True, exist_ok=True)

    for direction in ["long", "tra"]:
        keys = [k for k in all_curves if k.endswith(f"|{direction}")]
        for key in keys:
            name, region, d_str, _ = key.split("|")
            d = float(d_str)
            td = td_ms(d, cfg.TM)

            f_valor = get_f_value(tabla_f, name, td, direction)

            df = all_curves[key]
            g1 = df["g1"].values * f_valor
            g2 = df["g2"].values * f_valor
            y = df["signal"].values

            row_param = df_params[
                (df_params["Archivo_origen"] == name) &
                (df_params["roi"] == region) &
                (df_params["d"] == float(d)) &
                (df_params["direction"] == direction)
            ]
            if row_param.empty:
                print(f"[WARN] No se encontraron parámetros para {key}")
                continue

            tc = float(row_param["tc_fit"].values[0])
            D0 = float(row_param["D0_fit"].values[0])
            M0 = float(row_param["M0_fit"].values[0])

            g1_fit = np.linspace(0, g1.max(), 1000)
            g2_fit = np.linspace(0, g2.max(), 1000)
            y_fit = nogse_model_fitting.OGSE_contrast_vs_g_rest(td, g1_fit, g2_fit, cfg.N1, cfg.N2, tc, M0, D0)

            signal_max = float(np.max(y_fit))
            g_at_max = float(g1_fit[np.argmax(y_fit)])

            row_idx = row_param.index[0]
            df_params.loc[row_idx, "signal_max"] = signal_max
            df_params.loc[row_idx, "g_at_max"] = g_at_max / f_valor

            l_G = ((2**(3/2)) * cfg.D0_fix / (cfg.gamma * (g_at_max / f_valor)))**(1/3)
            l_d = np.sqrt(2 * cfg.D0_fix * td)
            L_d = l_d / l_G
            L_cf = ((3/2)**(1/4)) * (L_d**(-1/2))
            lcf_at_max = float(L_cf * l_G)
            df_params.loc[row_idx, "lcf_at_max"] = lcf_at_max

            tc_at_max = (lcf_at_max**2)/(2*cfg.D0_fix)
            df_params.loc[row_idx, "tc_at_max"] = float(tc_at_max)

            # plot
            color = region2color[region]
            plt.figure(figsize=(8, 6))
            plt.scatter(g_at_max / f_valor, signal_max, s=100, color="black", marker="o", label="máximo")
            plt.plot(g1 / f_valor, y, "o" if direction == "long" else "^", label="Datos", markersize=7, color=color)
            plt.plot(
                g1_fit / f_valor, y_fit, "-" if direction == "long" else "--",
                label=f"Ajuste\n$tc$={tc:.4f} ms\n$D_0$={D0:.2e} m$^2$/ms\n$M_0$={M0:.2e}",
                linewidth=2, color=color
            )
            plt.xlabel("Gradient strength $G$ [mT/m]", fontsize=18)
            plt.ylabel(f"OGSE contrast $\Delta M_{{N{cfg.N1}-N{cfg.N2}}}$", fontsize=18)
            plt.title(f"{name} | {cfg.method} | {region} | dir {direction} | d = {d} ms", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(title_fontsize=15, fontsize=12, loc="best")
            plt.grid(True)
            plt.tight_layout()
            fname_fig = f"{name}_{region}_d={d}_{direction}.png"
            plt.savefig(out_ind / fname_fig, dpi=300)
            plt.close()

    out_xlsx = cfg.plot_dir_out / "globalfits" / f"globalfit_{cfg.fit_model}_{cfg.method}.xlsx"
    df_params.to_excel(out_xlsx, index=False)
    print("✅ Resultados y gráficos guardados.")
    return df_params
