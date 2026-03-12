from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_long_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_G_from_gthorsten(df: pd.DataFrame, use_sqrt2: bool = True) -> np.ndarray:
    """
    Replica tu notebook: G = sqrt(2)*|gthorsten|.
    Si use_sqrt2=False, usa |gthorsten| directo.
    """
    if "gthorsten" not in df.columns:
        raise ValueError("No existe columna 'gthorsten' en el DataFrame.")
    g = df["gthorsten"].to_numpy(dtype=float)
    g = np.abs(g)
    return (np.sqrt(2.0) * g) if use_sqrt2 else g


def plot_ogse_vs_G(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    y_col: str = "value",   # o "value"
    stat: str = "avg",
    use_sqrt2: bool = True,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> None:
    """
    Genera 3 familias de plots (como tu notebook):
      A) por (dir, roi)
      B) por roi (todas las dirs)
      C) por dir (todas las rois)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # ----------------------------
    # 1) Construir tabla avg + std
    # ----------------------------
    # En tu formato long:
    # - stat == "avg" contiene el promedio en columna "value" (o "value_norm" si existe)
    # - stat == "std" contiene la desviación estándar en columna "value"
    keys = ["direction", "b_step", "roi"]

    avg = df[df["stat"] == "avg"].copy()
    std = df[df["stat"] == "std"].copy()

    if avg.empty:
        raise ValueError("No hay filas con stat='avg' en el parquet.")
    if std.empty:
        raise ValueError("No hay filas con stat='std' en el parquet (necesario para barras de error).")

    # Validar columnas necesarias
    needed_avg = set(keys + ["gthorsten", y_col])
    missing_avg = needed_avg - set(avg.columns)
    if missing_avg:
        raise ValueError(f"En avg faltan columnas: {missing_avg}")

    needed_std = set(keys + ["value"])
    missing_std = needed_std - set(std.columns)
    if missing_std:
        raise ValueError(f"En std faltan columnas: {missing_std}")

    # Numéricos
    avg[y_col] = pd.to_numeric(avg[y_col], errors="coerce")
    std["value"] = pd.to_numeric(std["value"], errors="coerce")

    # Reducir columnas y renombrar
    avg = avg[keys + ["gthorsten", y_col]].rename(columns={y_col: "y_mean"})
    std = std[keys + ["value"]].rename(columns={"value": "y_std"})

    # Merge avg + std (misma key: direction/b_step/roi)
    d = avg.merge(std, on=keys, how="left")

    # Orden estable
    d = d.sort_values(["direction", "roi", "b_step"], kind="stable")

    # ----------------------------
    # 2) Eje X: G = sqrt(2)*|gthorsten|
    # ----------------------------
    d["_G"] = compute_G_from_gthorsten(d, use_sqrt2=use_sqrt2)

    directions = sorted(d["direction"].dropna().unique())
    rois = sorted(d["roi"].dropna().unique())

    # ----------------------------
    # 3) Y-limits: si y_col="value" NO usar (0,1)
    # ----------------------------
    if y_col == "value":
        ylim_to_use = None
    else:
        ylim_to_use = ylim

    # --- A) 1 plot por (dir, roi)
    for dir_ in directions:
        dd = d[d["direction"] == dir_]
        for roi in rois:
            dr = dd[dd["roi"] == roi]
            if dr.empty:
                continue
            plt.figure(figsize=(8, 6))
            plt.errorbar(
                dr["_G"],
                dr["y_mean"],
                yerr=dr["y_std"],
                fmt="o-",
                linewidth=2,
                markersize=6,
                capsize=3,
            )
            plt.xlabel("Modulation gradient strength G [mT/m]")
            plt.ylabel(f"OGSE signal ({y_col})")
            plt.title(f"dir={dir_} | roi={roi} | stat={stat}")
            if ylim_to_use is not None:
                plt.ylim(*ylim_to_use)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"OGSE_vs_G_roi={roi}_dir={dir_}.png", dpi=300)
            plt.close()

    # --- B) 1 plot por roi (todas las dirs)
    for roi in rois:
        plt.figure(figsize=(8, 6))
        for dir_ in directions:
            dr = d[(d["roi"] == roi) & (d["direction"] == dir_)]
            if dr.empty:
                continue
            plt.errorbar(
            dr["_G"],
            dr["y_mean"],
            yerr=dr["y_std"],
            fmt="o-",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=str(dir_),
            )
        plt.xlabel("Modulation gradient strength G [mT/m]")
        plt.ylabel(f"OGSE signal ({y_col})")
        plt.title(f"roi={roi} | stat={stat}")
        if ylim_to_use is not None:
            plt.ylim(*ylim_to_use)

        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(title="direction")
        plt.tight_layout()
        plt.savefig(out_dir / f"OGSE_vs_G_roi={roi}_all_dirs.png", dpi=300)
        plt.close()

    # --- C) 1 plot por dir (todas las rois)
    for dir_ in directions:
        plt.figure(figsize=(8, 6))
        dd = d[d["direction"] == dir_]
        for roi in rois:
            dr = dd[dd["roi"] == roi]
            if dr.empty:
                continue
            plt.errorbar(
            dr["_G"],
            dr["y_mean"],
            yerr=dr["y_std"],
            fmt="o-",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=str(roi),
            )
        plt.xlabel("Modulation gradient strength G [mT/m]")
        plt.ylabel(f"OGSE signal ({y_col})")
        plt.title(f"dir={dir_} | stat={stat}")
        if ylim_to_use is not None:
            plt.ylim(*ylim_to_use)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(title="roi", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"OGSE_vs_G_dir={dir_}_all_rois.png", dpi=300)
        plt.close()
