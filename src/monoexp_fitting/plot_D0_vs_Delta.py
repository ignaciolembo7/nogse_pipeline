from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_DELTA_RE = re.compile(r"_Delta(?P<Delta>\d+(?:\.\d+)?)", re.IGNORECASE)
_K_RE = re.compile(r"fit_params(?:_k=(\d+))?$", re.IGNORECASE)


def parse_delta_from_exp_id(exp_id: str) -> float:
    m = _DELTA_RE.search(exp_id)
    if not m:
        raise ValueError(f"No pude extraer Delta de exp_id={exp_id!r}. Esperaba '_DeltaXX.X' en el nombre.")
    return float(m.group("Delta"))

def parse_k_from_fit_params_path(p: Path) -> int | None:
    m = re.search(r"k=(\d+)", p.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

def _score_fit_params_file(p: Path) -> int:
    """
    Preferencias:
      0) fit_params.csv / fit_params.xlsx (sin sufijo)
      1) fit_params_k=<menor>.*
      2) cualquier otro fit_params*.*
    """
    stem = p.stem  # sin extensión
    m = _K_RE.match(stem)
    if m:
        k = m.group(1)
        return 0 if k is None else 1 + int(k)
    return 999


def discover_fit_params(fits_root: str | Path) -> list[Path]:
    """
    Busca 1 archivo de fit params por experimento (por carpeta):
      - fit_params.csv / fit_params.xlsx
      - fit_params_k=8.csv / fit_params_k=8.xlsx
      - etc.
    """
    root = Path(fits_root)
    files = sorted(set(root.glob("**/fit_params*.csv")) | set(root.glob("**/fit_params*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No encontré 'fit_params*.(csv|xlsx)' dentro de: {root}")
    return files



def exp_id_from_fit_params_path(p: Path) -> str:
    # nuevo layout: .../<exp_id>/fit_params*.csv|xlsx
    if p.name.lower().startswith("fit_params"):
        return p.parent.name

    # legacy layout: <exp_id>.fit_params.csv|xlsx
    name = p.name
    if name.endswith(".fit_params.csv"):
        return name[: -len(".fit_params.csv")]
    if name.endswith(".fit_params.xlsx"):
        return name[: -len(".fit_params.xlsx")]

    return p.stem


def load_all_fits(
    fits_root: str | Path,
    *,
    dir_col: str = "direction",
    roi_col: str = "roi",
    d0_col: str = "D0_mm2_s",
    dirs: Optional[Sequence[str]] = None,
    rois: Optional[Sequence[str]] = None,
    agg: str = "mean",  # mean|median|none
) -> pd.DataFrame:
    rows = []

    for f in discover_fit_params(fits_root):
        exp_id = exp_id_from_fit_params_path(f)
        Delta = parse_delta_from_exp_id(exp_id)

        if f.suffix.lower() == ".csv":
            df = pd.read_csv(f)
        else:
            # xlsx: leemos sheet fit_params (si no existe, caemos al primero)
            try:
                df = pd.read_excel(f, sheet_name="fit_params", engine="openpyxl")
            except ValueError:
                df = pd.read_excel(f, sheet_name=0, engine="openpyxl")

        # asegurar fit_points para distinguir k=8, k=15, etc.
        if "fit_points" not in df.columns:
            k = parse_k_from_fit_params_path(f)
            df["fit_points"] = k if k is not None else np.nan

        for c in [dir_col, roi_col, d0_col]:
            if c not in df.columns:
                raise ValueError(f"Falta columna {c!r} en {f}. Columnas: {df.columns.tolist()}")
            
        keep_cols = [dir_col, roi_col, d0_col] + (["fit_points"] if "fit_points" in df.columns else [])
        df = df[keep_cols].copy()

        df = df[[dir_col, roi_col, d0_col]].copy()
        df["Delta_ms"] = Delta
        df["exp_id"] = exp_id

        df[dir_col] = df[dir_col].astype(str)
        df[roi_col] = df[roi_col].astype(str)
        df[d0_col] = pd.to_numeric(df[d0_col], errors="coerce")

        if dirs is not None:
            df = df[df[dir_col].isin([str(x) for x in dirs])]
        if rois is not None:
            df = df[df[roi_col].isin([str(x) for x in rois])]

        df = df.dropna(subset=[d0_col])
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    if agg != "none":
        fn = np.mean if agg == "mean" else np.median
        group_cols = ["Delta_ms", roi_col, dir_col]
        if "fit_points" in out.columns:
            group_cols.append("fit_points")

        out = out.groupby(group_cols, as_index=False)[d0_col].agg(fn)

    return out


def _dirs_order(df: pd.DataFrame, dir_col: str, dirs: Optional[Sequence[str]]) -> list[str]:
    if dirs is not None:
        return [str(x) for x in dirs]
    return sorted(df[dir_col].astype(str).unique().tolist())


def plot_per_roi(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    roi: str,
    dir_col: str = "direction",
    roi_col: str = "roi",
    d0_col: str = "D0_mm2_s",
    dirs: Optional[Sequence[str]] = None,
    ylims: Optional[Tuple[float, float]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[df[roi_col] == str(roi)].copy()
    if sub.empty:
        raise ValueError(f"No hay datos para roi={roi!r}")

    dirs_list = _dirs_order(sub, dir_col, dirs)

    plt.figure(figsize=(8, 6))
    has_k = "fit_points" in sub.columns and sub["fit_points"].notna().any()
    ks = sorted(sub["fit_points"].dropna().unique().tolist()) if has_k else [None]

    for d in dirs_list:
        for k in ks:
            s = sub[sub[dir_col].astype(str) == str(d)]
            if k is not None:
                s = s[s["fit_points"] == k]
            s = s.sort_values("Delta_ms")
            if s.empty:
                continue
            label = f"{d} (k={int(k)})" if k is not None else str(d)
            plt.plot(s["Delta_ms"], s[d0_col], marker="o", linewidth=2, label=label)


    plt.xlabel(f"$\\Delta_{{app}}$ [ms]", fontsize=14)
    plt.ylabel(f"Diffusion coefficient $D$ [mm$^2$/s]", fontsize=14)

    plt.title(f"ROI={roi}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    plt.legend(title="direction", title_fontsize=15, fontsize=12, loc='best')
    plt.ticklabel_format(axis="y", style="sci", useOffset=False)

    if ylims is not None:
        plt.ylim(*ylims)

    plt.tight_layout()
    out_path = out_dir / f"D0_vs_Delta.ROI-{roi}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_all_rois_same_scale(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    dir_col: str = "direction",
    roi_col: str = "roi",
    d0_col: str = "D0_mm2_s",
    dirs: Optional[Sequence[str]] = None,
    rois: Optional[Sequence[str]] = None,
    ncols: int = 3,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rois_list = ([str(r) for r in rois] if rois is not None else sorted(df[roi_col].unique().tolist()))
    if not rois_list:
        raise ValueError("No hay ROIs para plotear.")

    dirs_list = _dirs_order(df, dir_col, dirs)

    y = pd.to_numeric(df[d0_col], errors="coerce").dropna()
    y_min, y_max = float(y.min()), float(y.max())
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1 * max(y_max, 1e-6)
    ylims = (y_min - pad, y_max + pad)

    n = len(rois_list)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, roi in zip(axes, rois_list):
        sub = df[df[roi_col] == roi].copy()
        for d in dirs_list:
            s = sub[sub[dir_col].astype(str) == str(d)].sort_values("Delta_ms")
            if s.empty:
                continue
            ax.plot(s["Delta_ms"], s[d0_col], marker="o", linewidth=2, label=str(d))
        ax.set_title(f"ROI={roi}")
        ax.set_xlabel(f"$\\Delta_{{app}}$ [ms]", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(*ylims)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
        ax.ticklabel_format(axis="y", style="sci", useOffset=False)


    for ax in axes[len(rois_list):]:
        ax.axis("off")

    axes[0].set_ylabel(f"Diffusion coefficient $D$ [mm$^2$/s]", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Direction", loc="upper right", fontsize=15)

    fig.tight_layout()
    out_path = out_dir / "D0_vs_Delta.ALL_ROIS.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
