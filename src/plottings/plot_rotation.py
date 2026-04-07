from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tools.strict_columns import raise_on_unrecognized_column_names


DEFAULT_AXES6 = ["x", "y", "z", "longitudinal", "transversal_1", "transversal_2"]


def load_dproj_parquet(paths: list[str | Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        raise_on_unrecognized_column_names(df.columns, context=f"load_dproj_parquet({p})")
        if "direction" not in df.columns:
            raise ValueError(f"load_dproj_parquet({p}): missing required column ['direction'].")
        df["source"] = Path(p).name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out


def plot_dproj_subplots_by_N(
    df_all: pd.DataFrame,
    out_png: str | Path,
    *,
    roi: str,
    Ns: list[int],
    axes: list[str] = DEFAULT_AXES6,
    title_prefix: str = "",
):
    """
    Replica el plot tipo 'subplots 1 x len(Ns)' del notebook (cell 10).
    Requiere que df_all tenga columnas: roi, direction, bvalue, D_proj, N
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    d = df_all[df_all["roi"] == roi].copy()
    if d.empty:
        raise ValueError(f"No hay datos para roi={roi}")

    fig, axs = plt.subplots(1, len(Ns), figsize=(6 * len(Ns), 5), sharey=True)
    if len(Ns) == 1:
        axs = [axs]

    for i, N in enumerate(Ns):
        ax = axs[i]
        dn = d[d["N"] == N]
        ax.set_title(f"N={N}", fontsize=14)
        ax.set_xlabel(r"$b_{\mathrm{value}}$ [s/mm$^2$]", fontsize=12)
        if i == 0:
            ax.set_ylabel(r"$D_n$ proyectado [mm$^2$/s]", fontsize=12)
        ax.grid(True, alpha=0.3)

        for lab in axes:
            dd = dn[dn["direction"] == lab]
            if dd.empty:
                continue
            ax.plot(dd["bvalue"], dd["D_proj"], marker="o", linewidth=2, label=lab)

        ax.legend(fontsize=9)

    fig.suptitle(f"{title_prefix} {roi}".strip(), fontsize=16)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_dproj_gradient_xyz(
    df_all: pd.DataFrame,
    out_png: str | Path,
    *,
    roi: str,
    Ns: list[int],
    logy: bool = False,
    title_prefix: str = "",
):
    """
    Replica los plots tipo 'x,y,z' con degradado por N (cell 11 y 12).
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    d = df_all[(df_all["roi"] == roi) & (df_all["direction"].isin(["x", "y", "z"]))].copy()
    if d.empty:
        raise ValueError(f"No hay datos xyz para roi={roi}")

    markers = {"x": "^", "y": "s", "z": "o"}
    base_colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}

    plt.figure(figsize=(10, 7))
    plt.title(f"{title_prefix} {roi}".strip(), fontsize=16)
    plt.xlabel(r"$b_{\mathrm{value}}$ [s/mm$^2$]", fontsize=14)
    plt.ylabel(r"$D_n$ proyectado [mm$^2$/s]", fontsize=14)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)

    for axname in ["x", "y", "z"]:
        base = base_colors[axname]
        cmap = mcolors.LinearSegmentedColormap.from_list(f"{axname}_cmap", ["white", base])
        start = 0.3
        colors = [cmap(start + (1 - start) * i / max(1, (len(Ns) - 1))) for i in range(len(Ns))]

        for i, N in enumerate(Ns):
            dn = d[(d["N"] == N) & (d["direction"] == axname)]
            if dn.empty:
                continue
            plt.plot(
                dn["bvalue"], dn["D_proj"],
                marker=markers[axname], linewidth=2,
                color=colors[i],
                label=f"{axname} | N={N}",
            )

    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
