from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ..nogse_models.nogse_model_fitting import nogse_model_fitting

from ..ogse_fitting.config import OGSEFitConfig
from ..ogse_fitting.data_loading import get_f_value, td_ms
from ..ogse_fitting.metrics import compute_region_colors

def shade_color(color: str, factor: float) -> tuple[float, float, float]:
    rgb = mcolors.to_rgb(color)
    white = (0.9, 0.9, 0.9)
    return tuple(white[i] + factor * (rgb[i] - white[i]) for i in range(3))

def plot_individual_by_dirs(cfg: OGSEFitConfig, all_curves: dict[str, pd.DataFrame], df_params: pd.DataFrame, tabla_f: pd.DataFrame) -> None:
    region2color = compute_region_colors(cfg, all_curves)
    out_dir = cfg.plot_dir_out / "individual_plots_by_dirs"
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = sorted({
        (name, region, d)
        for name, region, d, _ in (key.split("|") for key in all_curves.keys())
        if f"{name}|{region}|{d}|long" in all_curves and f"{name}|{region}|{d}|tra" in all_curves
    })

    for name, region, d in combos:
        td = td_ms(float(d), cfg.TM)
        color = region2color[region]
        plt.figure(figsize=(8, 6))

        for direction in ["long", "tra"]:
            key = f"{name}|{region}|{d}|{direction}"
            df = all_curves.get(key)
            if df is None:
                continue

            f_valor = get_f_value(tabla_f, name, td, direction)

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

            label = f"{direction} Data"
            ajuste_label = f"{direction} Fit\n$D_0$={D0:.2e} m$^2$/ms"
            alpha_val = 1.0 if direction == "long" else 0.5

            plt.plot(g1 / f_valor, y, "o" if direction == "long" else "^", label=label, markersize=7, color=color, alpha=alpha_val)
            plt.plot(g1_fit / f_valor, y_fit, "-" if direction == "long" else "--", label=ajuste_label, linewidth=2, color=color, alpha=alpha_val)

        plt.xlabel("Gradient strength $G$ [mT/m]", fontsize=18)
        plt.ylabel(f"OGSE contrast $\Delta M_{{N{cfg.N1}-N{cfg.N2}}}$", fontsize=18)
        plt.title(f"{name} | {region} | d = {d} ms", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title_fontsize=14, fontsize=12, loc="best")
        plt.grid(True)
        plt.tight_layout()

        fname_fig = f"{name}_{region}_d={d}.png"
        plt.savefig(out_dir / fname_fig, dpi=300)
        plt.close()

    print("✅ Gráficos comparativos entre direcciones generados correctamente.")

def plot_globalfits_grid_G(cfg: OGSEFitConfig, all_curves: dict[str, pd.DataFrame], df_params: pd.DataFrame, tabla_f: pd.DataFrame) -> None:
    region2color = compute_region_colors(cfg, all_curves)
    regiones_unicas = sorted({key.split("|")[1] for key in all_curves})
    regiones = regiones_unicas
    brains = ["BRAIN", "MBBL", "LUDG"]

    for brain in brains:
        fig, axes = plt.subplots(len(regiones), 2, figsize=(12, 3.5 * len(regiones)), sharex=False, sharey=False)

        for i, region in enumerate(regiones):
            for j, direction in enumerate(["long", "tra"]):
                ax = axes[i, j] if len(regiones) > 1 else axes[j]

                keys = sorted(
                    [k for k in all_curves if k.endswith(f"|{direction}") and f"|{region}|" in k and brain in k],
                    key=lambda k: td_ms(float(k.split("|")[2]), cfg.TM),
                )
                if not keys:
                    continue

                td_values = [td_ms(float(k.split("|")[2]), cfg.TM) for k in keys]
                td_min, td_max = min(td_values), max(td_values)

                for key in keys:
                    name, reg, d_str, _ = key.split("|")
                    d = float(d_str)
                    td = td_ms(d, cfg.TM)
                    df = all_curves[key]

                    f_valor = get_f_value(tabla_f, name, td, direction)

                    g1 = df["g1"].values * f_valor
                    g2 = df["g2"].values * f_valor
                    y = df["signal"].values

                    row_params = df_params[
                        (df_params["Archivo_origen"] == name) &
                        (df_params["roi"] == reg) &
                        (df_params["td_ms"] == td) &
                        (df_params["direction"] == direction)
                    ]
                    if row_params.empty:
                        print(f"⚠️ No se encontraron parámetros para: {name}, {reg}, Td={td}, dir={direction}")
                        continue

                    tc = float(row_params["tc_fit"].values[0])
                    D0 = float(row_params["D0_fit"].values[0])
                    M0 = float(row_params["M0_fit"].values[0])

                    g1_fit = np.linspace(0, g1.max(), 1000)
                    g2_fit = np.linspace(0, g2.max(), 1000)
                    y_fit = nogse_model_fitting.OGSE_contrast_vs_g_rest(td, g1_fit, g2_fit, cfg.N1, cfg.N2, tc, M0, D0)

                    td_norm = (td - td_min) / (td_max - td_min) if td_max != td_min else 1.0
                    color = shade_color(region2color[region], td_norm)

                    label = f"{name} | Td={td:.1f} ms"
                    ax.plot(g1 / f_valor, y, "o-" if direction == "long" else "^--", label=label, color=color, alpha=0.8)
                    ax.plot(g1_fit / f_valor, y_fit, "-", color=color, alpha=0.8)

                ax.set_title(f"{region} - direction: {direction}")
                ax.set_xlabel("Gradient strength $G$ [mT/m]", fontsize=18)
                ax.set_ylabel(f"OGSE contrast $\Delta M_{{N{cfg.N1}-N{cfg.N2}}}$", fontsize=18)
                ax.grid(True)
                ax.legend(fontsize=7, loc="best")

        plt.suptitle(f"OGSE contrast vs gradient strength $G$ — {brain}", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = cfg.plot_dir_out / "globalfits" / f"globalfits_{cfg.fit_model}_regions_y_dirs_{brain}_{cfg.method}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        plt.close()

    print("✅ Gráficos globales (G) guardados.")

def plot_globalfits_grid_xvars(cfg: OGSEFitConfig, all_curves: dict[str, pd.DataFrame], df_params: pd.DataFrame, tabla_f: pd.DataFrame) -> None:
    region2color = compute_region_colors(cfg, all_curves)
    regiones_unicas = sorted({key.split("|")[1] for key in all_curves})
    regiones = regiones_unicas
    brains = ["BRAIN", "MBBL", "LUDG"]

    x_vars = ["lcf_a", "lcf", "ld_a"]
    for x_var in x_vars:
        for brain in brains:
            fig, axes = plt.subplots(len(regiones), 2, figsize=(12, 3.5 * len(regiones)), sharex=False, sharey=False)
            for i, region in enumerate(regiones):
                for j, direction in enumerate(["long", "tra"]):
                    ax = axes[i, j] if len(regiones) > 1 else axes[j]

                    keys = sorted(
                        [k for k in all_curves if k.endswith(f"|{direction}") and f"|{region}|" in k and brain in k],
                        key=lambda k: td_ms(float(k.split("|")[2]), cfg.TM),
                    )
                    if not keys:
                        continue

                    td_values = [td_ms(float(k.split("|")[2]), cfg.TM) for k in keys]
                    td_min, td_max = min(td_values), max(td_values)

                    for key in keys:
                        name, reg, d_str, _ = key.split("|")
                        d = float(d_str)
                        td = td_ms(d, cfg.TM)
                        df = all_curves[key]

                        f_valor = get_f_value(tabla_f, name, td, direction)

                        g1 = df["g1"].values * f_valor
                        g2 = df["g2"].values * f_valor
                        y = df["signal"].values

                        row_params = df_params[
                            (df_params["Archivo_origen"] == name) &
                            (df_params["roi"] == reg) &
                            (df_params["td_ms"] == td) &
                            (df_params["direction"] == direction)
                        ]
                        if row_params.empty:
                            print(f"⚠️ No se encontraron parámetros para: {name}, {reg}, Td={td}, dir={direction}")
                            continue

                        D0 = float(row_params["D0_fit"].values[0])
                        tc = float(row_params["tc_fit"].values[0])
                        signal_max = float(row_params.get("signal_max", pd.Series([np.nan])).values[0])
                        lcf_at_max = float(row_params.get("lcf_at_max", pd.Series([np.nan])).values[0])

                        g1_fit = np.linspace(0, g1.max(), 1000)
                        g2_fit = np.linspace(0, g2.max(), 1000)
                        y_fit = nogse_model_fitting.OGSE_contrast_vs_g_rest(td, g1_fit, g2_fit, cfg.N1, cfg.N2, tc, cfg.M0_value, D0)

                        td_norm = (td - td_min) / (td_max - td_min) if td_max != td_min else 1.0
                        color = shade_color(region2color[region], td_norm)
                        label = f"{name} | Td={td:.1f} ms"

                        # quitar el primer punto
                        g1 = g1[1:]
                        g1_fit = g1_fit[1:]
                        y = y[1:]
                        y_fit = y_fit[1:]

                        lg1 = ((2**(3/2)) * cfg.D0_fix / (cfg.gamma * (g1/f_valor)))**(1/3)
                        ld1 = np.sqrt(2 * cfg.D0_fix * td)
                        ld1_a = ld1 / lg1
                        lcf1_a = ((3/2)**(1/4)) * (ld1_a**(-1/2))
                        lcf1 = lcf1_a * lg1

                        lg1_fit = ((2**(3/2)) * cfg.D0_fix / (cfg.gamma * (g1_fit/f_valor)))**(1/3)
                        ld1_fit = np.sqrt(2 * cfg.D0_fix * td)
                        ld1_a_fit = ld1_fit / lg1_fit
                        lcf1_a_fit = ((3/2)**(1/4)) * (ld1_a_fit**(-1/2))
                        lcf1_fit = lcf1_a_fit * lg1_fit

                        if x_var == "lcf_a":
                            ax.plot(lcf1_a, y, "o-" if direction=="long" else "^--", label=label, color=color, alpha=0.8)
                            ax.plot(lcf1_a_fit, y_fit, "-", color=color, alpha=0.8)
                            ax.set_xlim(0.45, 1.0)
                            ax.set_xlabel("Center filter lenght $L_{c,f}$", fontsize=18)
                        elif x_var == "lcf":
                            ax.plot(lcf1*1e6, y, "o-" if direction=="long" else "^--", label=label, color=color, alpha=0.8)
                            ax.plot(lcf1_fit*1e6, y_fit, "-", color=color, alpha=0.8)
                            if np.isfinite(lcf_at_max):
                                ax.scatter(lcf_at_max*1e6, signal_max, s=20, marker="o", color="k", alpha=0.8)
                            ax.set_xlim(0, 20)
                            ax.set_xlabel("Center filter length $l_{c,f} ~[\mu m]$", fontsize=18)
                        elif x_var == "ld_a":
                            ax.plot(ld1_a, y, "o-" if direction=="long" else "^--", label=label, color=color, alpha=0.8)
                            ax.plot(ld1_a_fit, y_fit, "-", color=color, alpha=0.8)
                            ax.set_xlim(1.0, 5.0)
                            ax.set_xlabel("Diffusion lenght $L_d$", fontsize=18)

                        ax.set_ylabel(f"OGSE contrast $\Delta M_{{N{cfg.N1}-N{cfg.N2}}}$", fontsize=18)
                        ax.set_title(f"{region} - direction: {direction}")
                        ax.grid(True)
                        ax.legend(fontsize=7, loc="best")

            if x_var == "lcf_a":
                title = f"OGSE contrast vs Center filter lenght $L_{{c,f}}$ - {brain}"
            elif x_var == "lcf":
                title = f"OGSE contrast vs Center filter length $l_{{c,f}}$ - {brain}"
            else:
                title = f"OGSE contrast vs Diffusion lenght $L_d$ - {brain}"

            plt.suptitle(title, fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out = cfg.plot_dir_out / "globalfits" / f"globalfits_{cfg.fit_model}_regions_y_dirs_{brain}_{cfg.method}_{x_var}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300)
            plt.close()

    print("✅ Gráficos adimensionales guardados.")

def plot_param_comparisons(cfg: OGSEFitConfig, df_params: pd.DataFrame, all_curves: dict[str, pd.DataFrame]) -> None:
    region2color = compute_region_colors(cfg, all_curves)
    regiones_unicas = sorted({key.split("|")[1] for key in all_curves})
    regiones = regiones_unicas
    brains = ["LUDG", "MBBL", "BRAIN"]

    def shade_color_local(color, factor):
        rgb = mcolors.to_rgb(color)
        white = (1, 1, 1)
        return tuple(white[i] + factor * (rgb[i] - white[i]) for i in range(3))

    factors = [0.25, 0.5, 1.0]

    out_dir = cfg.plot_dir_out / "globalfits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dir_actual in ["long", "tra"]:
        df_dir = df_params[df_params["direction"] == dir_actual]

        for what, col in [("tc_at_max", "$t_c$ [ms]")]:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
            for ax, region in zip(axes.flat, regiones):
                base_color = region2color[region]
                for i, brain in enumerate(brains):
                    sub = df_dir[(df_dir["roi"] == region) & (df_dir["brain"] == brain)].sort_values("td_ms")
                    if not sub.empty:
                        ax.plot(sub["td_ms"], sub[what], "o-" if dir_actual=="long" else "^--",
                                markersize=7, linewidth=2, label=brain,
                                color=shade_color_local(base_color, factors[i]))
                ax.set_title(region, fontsize=14)
                ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
                ax.set_ylabel("$t_{c,max}$ [ms]", fontsize=18)
                ax.grid(True)
                ax.legend(fontsize=10)
            plt.suptitle(f"Non parametric | {col} | Direction: {dir_actual} | Model: {cfg.fit_model} | Method: {cfg.method} | Volunteers comparsion", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(out_dir / f"fit_{cfg.fit_model}_{what}_vs_Td_dir={dir_actual}_{cfg.method}_Vol-comp.png", dpi=300)
            plt.close()

        for what, col in [("tc_at_max", "$t_c$ [ms]")]:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
            for ax, brain in zip(axes, brains[::-1]):
                for region in regiones:
                    sub = df_dir[(df_dir["brain"] == brain) & (df_dir["roi"] == region)].sort_values("td_ms")
                    if not sub.empty:
                        ax.plot(sub["td_ms"], sub[what], "o-" if dir_actual=="long" else "^--",
                                markersize=7, linewidth=2, label=region, color=region2color[region])
                ax.set_title(brain, fontsize=14)
                ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
                ax.legend(fontsize=10)
                if ax is axes[0]:
                    ax.set_ylabel("$t_{c,max}$ [ms]", fontsize=18)
                ax.grid(True)
            plt.suptitle(f"Non parametric | {col} | Direction: {dir_actual} | Model: {cfg.fit_model} | Method: {cfg.method} | Region comparison", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(out_dir / f"fit_{cfg.fit_model}_{what}_vs_Td_dir={dir_actual}_{cfg.method}_Reg-comp.png", dpi=300)
            plt.close()

    for dir_actual in ["long", "tra"]:
        df_dir = df_params[df_params["direction"] == dir_actual]
        for what, col in [("signal_max", "Amplitud")]:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
            for ax, region in zip(axes.flat, regiones):
                base_color = region2color[region]
                for i, brain in enumerate(brains):
                    sub = df_dir[(df_dir["roi"] == region) & (df_dir["brain"] == brain)].sort_values("td_ms")
                    if not sub.empty:
                        ax.plot(sub["td_ms"], sub[what], "o-" if dir_actual=="long" else "^--",
                                markersize=7, linewidth=2, label=brain,
                                color=shade_color_local(base_color, factors[i]))
                ax.set_title(region, fontsize=14)
                ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
                ax.set_ylabel("Amplitud", fontsize=18)
                ax.grid(True)
                ax.legend(fontsize=10)
            plt.suptitle(f"Non parametric | {col} | Direction: {dir_actual} | Model: {cfg.fit_model} | Method: {cfg.method} | Volunteers comparsion", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(out_dir / f"fit_{cfg.fit_model}_{what}_vs_Td_dir={dir_actual}_{cfg.method}_Vol-comp.png", dpi=300)
            plt.close()

        for what, col in [("signal_max", "Amplitud")]:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
            for ax, brain in zip(axes, brains[::-1]):
                for region in regiones:
                    sub = df_dir[(df_dir["brain"] == brain) & (df_dir["roi"] == region)].sort_values("td_ms")
                    if not sub.empty:
                        ax.plot(sub["td_ms"], sub[what], "o-" if dir_actual=="long" else "^--",
                                markersize=7, linewidth=2, label=region, color=region2color[region])
                ax.set_title(brain, fontsize=14)
                ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
                ax.legend(fontsize=10)
                if ax is axes[0]:
                    ax.set_ylabel("Amplitud", fontsize=18)
                ax.grid(True)
            plt.suptitle(f"Non parametric | {col} | Direction: {dir_actual} | Model: {cfg.fit_model} | Method: {cfg.method} | Region comparison", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(out_dir / f"fit_{cfg.fit_model}_{what}_vs_Td_dir={dir_actual}_{cfg.method}_Reg-comp.png", dpi=300)
            plt.close()

    print("✅ Gráficos por roi y brain generados y guardados.")
