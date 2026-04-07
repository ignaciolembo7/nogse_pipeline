from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DELTA_RE = re.compile(r"_Delta(?P<Delta>\d+(?:\.\d+)?)", re.IGNORECASE)

def parse_delta_from_name(name: str) -> float:
    m = DELTA_RE.search(name)
    if not m:
        raise ValueError(f"No pude extraer Delta de: {name}")
    return float(m.group("Delta"))

def infer_exp_id_from_parquet(p: Path) -> str:
    name = p.name
    for suf in [".rot_tensor.long.parquet", ".long.parquet", ".parquet"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return p.stem

def _score_fit_file(p: Path) -> int:
    # preferir fit_params.csv, luego fit_params_k=... con k chico
    stem = p.stem.lower()
    if stem == "fit_params":
        return 0
    m = re.match(r"fit_params[_-]k=(\d+)$", stem)
    if m:
        return 1 + int(m.group(1))
    if stem.startswith("fit_params"):
        return 999
    return 10_000

def find_fit_params_file(exp_dir: Path) -> Path:
    cands = list(exp_dir.glob("fit_params*.csv")) + list(exp_dir.glob("fit_params*.xlsx"))
    if not cands:
        raise FileNotFoundError(f"No encontré fit_params* en: {exp_dir}")
    cands = sorted(cands, key=_score_fit_file)
    return cands[0]

def load_fit_params(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # xlsx
    try:
        return pd.read_excel(path, sheet_name="fit_params", engine="openpyxl")
    except ValueError:
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")

def plot_grid_all_deltas(
    *,
    data_root: str | Path,
    fits_root: str | Path,
    out_png: str | Path,
    dirs: Sequence[str],
    rois: Optional[Sequence[str]] = None,
    dir_col: str = "direction",
    roi_col: str = "roi",
    bcol: str = "bvalue",
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    stat_col: str = "stat",
    logy: bool = True,
    title: str = "",
) -> Path:
    data_root = Path(data_root)
    fits_root = Path(fits_root)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    parqs = sorted(data_root.glob("**/*.long.parquet"))
    if not parqs:
        raise FileNotFoundError(f"No encontré .long.parquet en: {data_root}")

    # cargar todo (datos + fits) en una tabla larga con Delta
    blocks = []
    for pq in parqs:
        exp_id = infer_exp_id_from_parquet(pq)
        Delta = parse_delta_from_name(exp_id)

        df = pd.read_parquet(pq)
        if stat_col in df.columns:
            df = df[df[stat_col].isin([stat_keep, "Mean"])].copy()

        # filtros de columnas
        for c in [dir_col, roi_col, bcol, ycol]:
            if c not in df.columns:
                raise ValueError(f"Falta columna {c!r} en {pq}. Columnas: {df.columns.tolist()}")

        df = df[[dir_col, roi_col, bcol, ycol]].copy()
        df[dir_col] = df[dir_col].astype(str)
        df[roi_col] = df[roi_col].astype(str)
        df[bcol] = pd.to_numeric(df[bcol], errors="coerce")
        df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
        df["Delta_ms"] = Delta
        df["exp_id"] = exp_id
        blocks.append(df)

    data = pd.concat(blocks, ignore_index=True).dropna(subset=[bcol, ycol])

    # rois
    if rois is None:
        rois = sorted(data[roi_col].unique().tolist())
    else:
        rois = [str(r) for r in rois]
        data = data[data[roi_col].isin(rois)].copy()

    # dirs
    dirs = [str(d) for d in dirs]
    data = data[data[dir_col].isin(dirs)].copy()

    deltas = sorted(data["Delta_ms"].unique().tolist())

    # preparar figura
    nrows, ncols = len(rois), len(dirs)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    # colores consistentes por Delta (usa ciclo default de matplotlib)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    # legend global sin duplicados
    used_labels = set()

    for i, roi in enumerate(rois):
        for j, d in enumerate(dirs):
            ax = axes[i, j]
            sub = data[(data[roi_col] == roi) & (data[dir_col] == d)].copy()
            if sub.empty:
                ax.axis("off")
                continue

            # title subplot
            ax.set_title(f"{roi}  G{d}")

            for k, Delta in enumerate(deltas):
                s = sub[sub["Delta_ms"] == Delta].sort_values(bcol)
                if s.empty:
                    continue

                color = colors[k % len(colors)]
                lbl_pts = f"$\\Delta={Delta:g}$"
                if lbl_pts in used_labels:
                    lbl_pts = "_nolegend_"
                else:
                    used_labels.add(lbl_pts)

                ax.plot(s[bcol].to_numpy(), s[ycol].to_numpy(), marker="o", linestyle="None",
                        markersize=3.5, label=lbl_pts, color=color)

                # curva fit: leer D0/M0 desde fit_params del experimento correspondiente
                exp_id = s["exp_id"].iloc[0]
                exp_dir = fits_root / exp_id  # si tu fits_root apunta a la carpeta del grupo, esto existe
                if not exp_dir.exists():
                    # fallback: si fits_root es el root global monoexp_fits/<grupo>, exp_dir está bien; si no, probá buscar recursivo
                    pass

                try:
                    fit_file = find_fit_params_file(exp_dir)
                    fp = load_fit_params(fit_file)
                    # buscar fila ROI+dir
                    fp["direction"] = fp["direction"].astype(str)
                    fp["roi"] = fp["roi"].astype(str)
                    row = fp[(fp["direction"] == d) & (fp["roi"] == roi)]
                    if not row.empty:
                        M0 = float(row["M0"].iloc[0])
                        D0 = float(row["D0_mm2_s"].iloc[0])
                        bmax = float(np.nanmax(s[bcol].to_numpy()))
                        b_dense = np.linspace(0.0, bmax, 300)
                        y_dense = M0 * np.exp(-b_dense * D0)

                        lbl_fit = f"fit Δ={Delta:g}"
                        if lbl_fit in used_labels:
                            lbl_fit = "_nolegend_"
                        else:
                            used_labels.add(lbl_fit)

                        ax.plot(b_dense, y_dense, linestyle="-", linewidth=2, label=lbl_fit, color=color)
                except Exception:
                    # si falta fit o algo no matchea, simplemente no dibuja la curva
                    pass

            if logy:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.25)

    # labels comunes
    for ax in axes[-1, :]:
        ax.set_xlabel("b [s/mm$^2$]", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("Signal (normalized)", fontsize=14)

    if title:
        fig.suptitle(title, y=0.98)

    # legend global a la derecha
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll != "_nolegend_":
                handles.append(hh)
                labels.append(ll)

    # quitar duplicados manteniendo orden
    seen = set()
    H, L = [], []
    for hh, ll in zip(handles, labels):
        if ll in seen:
            continue
        seen.add(ll)
        H.append(hh)
        L.append(ll)

    fig.legend(H, L, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.82, 1))  # deja espacio para legend
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png
