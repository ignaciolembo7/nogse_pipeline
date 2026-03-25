from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit, least_squares
from matplotlib.colors import to_rgba
from .tc_td_methods import METHODS, run_tc_td_fit

from ogse_fitting.config import OGSEFitConfig

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def linear_fits_last3(cfg: OGSEFitConfig, df_params: pd.DataFrame, regiones: list[str], brains: list[str], region2color: dict[str,str]) -> str:
    """Replicates notebook cells 7-10 (fit_type = lineal_tc_vs_Td_allpoints, but using last 3 points)."""
    fit_type = "lineal_tc_vs_Td_allpoints"
    out = cfg.plot_dir_out / fit_type
    ensure_dir(out)

    # tc_at_max vs Td
    ajustes_tc = []
    factors = [0.25, 0.5, 1.0]

    def shade_color(color, factor):
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        white = (1,1,1)
        return tuple(white[i] + factor*(rgb[i]-white[i]) for i in range(3))

    for dir_actual in ["long","tra"]:
        df_dir = df_params[df_params["direction"] == dir_actual]
        fig, axes = plt.subplots(2,3, figsize=(18,10), sharex=True)
        for ax, region in zip(axes.flat, regiones):
            base_color = region2color[region]
            for i, brain in enumerate(brains):
                sub = df_dir[(df_dir["roi"]==region) & (df_dir["brain"]==brain)].sort_values("td_ms")
                if len(sub) >= 2:
                    x = sub["td_ms"].values
                    y = sub["tc_at_max"].values
                    res = linregress(x[-3:], y[-3:])
                    ajustes_tc.append({
                        "roi": region, "brain": brain, "direction": dir_actual,
                        "pendiente": res.slope, "error_pendiente": res.stderr,
                        "ordenada": res.intercept, "error_ordenada": getattr(res,"intercept_stderr", np.nan),
                        "r2": res.rvalue**2
                    })
                    ax.plot(x, y, "o", color=shade_color(base_color, factors[i]), label=f"{brain}")
                    ax.plot(x[-3:], res.slope*x[-3:] + res.intercept, "-", color=shade_color(base_color, factors[i]), linewidth=2)
            ax.set_title(region, fontsize=14)
            ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
            ax.set_ylabel("$t_c$ [ms]", fontsize=18)
            ax.grid(True); ax.legend(fontsize=10)
        plt.suptitle(f"Ajuste lineal: $t_c$ vs $T_d$ | direction: {dir_actual}", fontsize=18)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out / f"ajuste_tc_vs_Td_dir={dir_actual}_{cfg.method}.png", dpi=300)
        plt.close()

    df_aj_tc = pd.DataFrame(ajustes_tc)
    df_aj_tc.to_excel(out / "ajustes_lineales_tc_vs_Td.xlsx", index=False)

    # amplitude vs Td
    ajustes_amp = []
    for dir_actual in ["long","tra"]:
        df_dir = df_params[df_params["direction"] == dir_actual]
        fig, axes = plt.subplots(2,3, figsize=(18,10), sharex=True)
        for ax, region in zip(axes.flat, regiones):
            base_color = region2color[region]
            for i, brain in enumerate(brains):
                sub = df_dir[(df_dir["roi"]==region) & (df_dir["brain"]==brain)].sort_values("td_ms")
                if len(sub) >= 2:
                    x = sub["td_ms"].values
                    y = sub["signal_max"].values
                    res = linregress(x[-3:], y[-3:])
                    ajustes_amp.append({
                        "roi": region, "brain": brain, "direction": dir_actual,
                        "pendiente": res.slope, "error_pendiente": res.stderr,
                        "ordenada": res.intercept, "error_ordenada": getattr(res,"intercept_stderr", np.nan),
                        "r2": res.rvalue**2
                    })
                    ax.plot(x, y, "o", color=shade_color(base_color, factors[i]), label=f"{brain}")
                    ax.plot(x[-4:], res.slope*x[-4:] + res.intercept, "-", color=shade_color(base_color, factors[i]), linewidth=2)
            ax.set_title(region, fontsize=14)
            ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=18)
            ax.set_ylabel("Amplitud", fontsize=18)
            ax.grid(True); ax.legend(fontsize=10)
        plt.suptitle(f"Ajuste lineal: Amplitud vs $T_d$ | direction: {dir_actual}", fontsize=18)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out / f"ajuste_amp_vs_Td_dir={dir_actual}_{cfg.method}.png", dpi=300)
        plt.close()

    df_aj_amp = pd.DataFrame(ajustes_amp)
    df_aj_amp.to_excel(out / "ajustes_lineales_amp_vs_Td.xlsx", index=False)

    # coeff plots
    _plot_linear_coeffs(out / "ajustes_lineales_tc_vs_Td.xlsx", out / "ajuste_tc_vs_Td", regiones, brains)
    _plot_linear_coeffs(out / "ajustes_lineales_amp_vs_Td.xlsx", out / "ajuste_amp_vs_Td", regiones, brains, labels=[("pendiente","Slope"),("ordenada","Intercept")])
    print("✅ Ajustes lineales guardados.")
    return fit_type

def _plot_linear_coeffs(xlsx: Path, prefix: Path, region_order: list[str], brains: list[str], labels=None):
    df_aj = pd.read_excel(xlsx)
    colors = {"BRAIN":"#377eb8", "LUDG":"#ff7f00", "MBBL":"#4daf4a"}
    df_aj["region"] = df_aj["roi"]
    df_aj["direccion"] = df_aj["direction"]
    if labels is None:
        labels = [("pendiente", "$\\alpha_{micro}$"), ("ordenada", "Intercept [ms]")]
    for var, label in labels:
        fig, axes = plt.subplots(1,2, figsize=(16,6), sharey=True)
        for i, dir_actual in enumerate(["long","tra"]):
            ax = axes[i]
            df_dir = df_aj[df_aj["direccion"]==dir_actual]
            for brain in brains:
                df_brain = df_dir[df_dir["brain"]==brain]
                x = np.arange(len(region_order))
                vals, errs = [], []
                for r in region_order:
                    row = df_brain[df_brain["region"]==r]
                    if not row.empty:
                        vals.append(row[var].values[0])
                        errs.append(row.get(f"error_{var}", pd.Series([0])).values[0])
                    else:
                        vals.append(np.nan); errs.append(0)
                y = np.array(vals); yerr = np.array(errs)
                ax.plot(x, y, "o-", color=colors.get(brain,"black"), label=brain, linewidth=2, markersize=7)
                ax.fill_between(x, y-yerr, y+yerr, color=colors.get(brain,"black"), alpha=0.2)
            ax.set_title(f"{label} vs Region — Direction: {dir_actual}", fontsize=15)
            ax.set_xticks(np.arange(len(region_order)))
            ax.set_xticklabels(region_order, rotation=45)
            ax.set_xlabel("Region", fontsize=18)
            if i==0:
                ax.set_ylabel(label, fontsize=18)
            ax.grid(True); ax.legend()
        plt.tight_layout()
        plt.savefig(str(prefix) + f"_{var}_vs_region.png", dpi=600)
        plt.close()

def alpha_macro_scatter_if_available(cfg: OGSEFitConfig, fit_type_dir: Path, method_name: str | None = None) -> None:
    """Replicates the alpha_macro vs alpha_micro scatter if summary_alpha_values.xlsx exists."""
    summary_path = cfg.plot_dir_out / "summary_alpha_values.xlsx"
    if not summary_path.exists():
        print(f"[INFO] {summary_path} no existe -> salto plots alpha_macro vs alpha_micro.")
        return

    df_summary = pd.read_excel(summary_path, decimal=",")
    # --- elegir automáticamente el xlsx de ajustes tc(Td) ---
    legacy = fit_type_dir / "ajustes_lineales_tc_vs_Td.xlsx"
    candidates = []

    if legacy.exists():
        candidates = [legacy]
    else:
        # nuevo naming modular
        candidates = list(fit_type_dir.glob("ajustes_tc_at_max_vs_Td_*.xlsx"))
        if not candidates:
            # fallback por si cambiaste prefijos
            candidates = list(fit_type_dir.glob("ajustes_*tc*Td*.xlsx"))

    if not candidates:
        print(f"[INFO] No encontré ningún archivo de ajustes tc(Td) en {fit_type_dir}. Salto alpha_macro plot.")
        return
    
    if method_name:
        filt = [p for p in candidates if method_name in p.stem]
        if filt:
            candidates = filt

    # si hay varios, elegí el más reciente
    ajustes_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Usando ajustes tc(Td): {ajustes_path.name}")
    df_ajustes = pd.read_excel(ajustes_path)


    df_summary = df_summary.rename(columns={"region":"roi", "direccion":"direction"})
    if "alpha" not in df_summary.columns and "alpha_macro" in df_summary.columns:
        df_summary["alpha"] = df_summary["alpha_macro"]
    if "alpha_error" not in df_summary.columns:
        if "alpha_macro_error" in df_summary.columns:
            df_summary["alpha_error"] = df_summary["alpha_macro_error"]
        else:
            df_summary["alpha_error"] = np.nan
    df_ajustes["roi"] = df_ajustes["roi"].str.replace("_norm","", regex=False)
    df_summary["roi"] = df_summary["roi"].str.replace("_norm","", regex=False)

    df_x = df_summary[df_summary["direction"]=="x"].copy()
    df_yz = df_summary[df_summary["direction"].isin(["y","z"])].copy()

    df_tra = df_yz.groupby(["brain","roi"]).agg({"alpha":"mean","alpha_error":"mean"}).reset_index()
    df_tra["direccion"]="tra"
    df_tra = df_tra.rename(columns={"alpha":"alpha_tra","alpha_error":"alpha_error_tra"})

    df_x = df_x.rename(columns={"alpha":"alpha_long","alpha_error":"alpha_error_long"})
    df_x = df_x[["brain","roi","alpha_long","alpha_error_long"]]

    df_merged = pd.merge(df_x, df_tra, on=["brain","roi"])

    palette = ["#377eb8","#984ea3","#e41a1c","#ff7f00","#a65628"]
    regiones_unicas = df_merged["roi"].unique()
    region_colors = {r: palette[i % len(palette)] for i,r in enumerate(regiones_unicas)}

    fig, axes = plt.subplots(1,2, figsize=(12,6))
    direcciones = ["long","tra"]
    for i, direccion in enumerate(direcciones):
        ax = axes[i]
        df_d = df_ajustes[df_ajustes["direction"]==direccion]
        brains = sorted(df_d["brain"].unique())
        brain_pos = {b:i for i,b in enumerate(brains)}
        n = len(brains)
        marker_list = ["o","s","^","D","v","P","X","*","<",">"]
        brain_marker = {b: marker_list[j % len(marker_list)] for j,b in enumerate(brains)}

        for j,b in enumerate(brains):
            fade = j/(n-1) if n>1 else 0.5
            rgba = to_rgba("black", alpha=0.3+0.7*(1-fade))
            ax.plot([],[], linestyle="None", marker=brain_marker[b], markersize=8,
                    markerfacecolor=rgba, markeredgecolor=rgba, label=f"{j+1} = {b}")

        for _, row in df_d.iterrows():
            brain = row["brain"]; slope = row["pendiente"]; region = row["roi"]
            fade = brain_pos[brain]/(n-1) if n>1 else 0.5
            rgba = to_rgba(region_colors.get(region,"#000000"), alpha=0.3+0.7*(1-fade))
            alpha_col = "alpha_long" if direccion=="long" else "alpha_tra"
            match = df_merged[(df_merged["brain"]==brain) & (df_merged["roi"]==region)]
            if match.empty: 
                continue
            alpha = match[alpha_col].values[0]
            ax.plot(alpha, slope, linestyle="None", marker=brain_marker[brain], markersize=8,
                    markerfacecolor=rgba, markeredgecolor=rgba)
            ax.text(alpha, slope, region, fontsize=8, color=rgba, ha="left", va="bottom")

        ax.set_xlabel("$\\alpha_{macro}$", fontsize=18)
        if i==0: ax.set_ylabel("$\\alpha_{micro}$", fontsize=18)
        ax.set_title(f"direction {direccion}", fontsize=18)
        ax.grid(True); ax.legend(fontsize=10, title_fontsize=11, loc="lower right")
    plt.tight_layout()
    out_png = fit_type_dir / (f"alpha_macro_vs_alpha_micro_{method_name}.png" if method_name else "alpha_macro_vs_alpha_micro.png")
    plt.savefig(out_png, dpi=600)
    plt.close()

def run_posthoc(cfg: OGSEFitConfig, df_params: pd.DataFrame, all_curves: dict[str, pd.DataFrame], region2color: dict[str,str]) -> None:
    regiones = sorted({k.split("|")[1] for k in all_curves})
    brains = ["LUDG","MBBL","BRAIN"]

    fit_type = linear_fits_last3(cfg, df_params, regiones, brains, region2color)
    alpha_macro_scatter_if_available(cfg, cfg.plot_dir_out / fit_type)

def run_tc_vs_td_only(cfg: OGSEFitConfig, df_params: pd.DataFrame, method_name: str = "ols_last3") -> None:
    if method_name not in METHODS:
        raise ValueError(f"method '{method_name}' no existe. Opciones: {list(METHODS.keys())}")

    method = METHODS[method_name]

    out_dir = cfg.plot_dir_out / "tc_vs_td"
    out_dir.mkdir(parents=True, exist_ok=True)

    regiones = sorted(df_params["roi"].unique())
    region2color = {r: cfg.palette[i % len(cfg.palette)] for i, r in enumerate(regiones)}
    brains = ["LUDG", "MBBL", "BRAIN"]

    # tc_at_max vs Td
    run_tc_td_fit(
        df_params=df_params,
        out_dir=out_dir,
        method=method,
        region2color=region2color,
        brains=brains,
        y_col="tc_at_max",
        y_label="$t_{c,max}$ [ms]",
        fig_prefix="tc_at_max_vs_Td",
    )

    # signal_max vs Td
    run_tc_td_fit(
        df_params=df_params,
        out_dir=out_dir,
        method=method,
        region2color=region2color,
        brains=brains,
        y_col="signal_max",
        y_label="Amplitude",
        fig_prefix="signal_max_vs_Td",
    )

    # alpha macro scatter (si existe summary_alpha_values.xlsx en cfg.plot_dir_out)
    alpha_macro_scatter_if_available(cfg, out_dir, method_name=method_name)

    print(f"✅ tc vs Td terminado. method={method_name} -> {out_dir}")
