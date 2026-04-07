from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.strict_columns import raise_on_unrecognized_column_names


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_long_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_G_from_g_thorsten(df: pd.DataFrame, use_sqrt2: bool = True) -> np.ndarray:
    """
    Match the notebook convention: G = sqrt(2) * |g_thorsten|.
    If use_sqrt2=False, use |g_thorsten| directly.
    """
    raise_on_unrecognized_column_names(df.columns, context="compute_G_from_g_thorsten")
    if "g_thorsten" not in df.columns:
        raise ValueError("compute_G_from_g_thorsten: missing required column ['g_thorsten'].")
    g = df["g_thorsten"].to_numpy(dtype=float)
    g = np.abs(g)
    return (np.sqrt(2.0) * g) if use_sqrt2 else g


def plot_ogse_vs_G(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    y_col: str = "value_norm",
    stat: str = "avg",
    use_sqrt2: bool = True,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> None:
    """
    Generate three plot families:
      A) by (direction, roi)
      B) by roi (all directions)
      C) by direction (all rois)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # ----------------------------
    # 1) Build the avg + std table
    # ----------------------------
    # In this long format:
    # - stat == "avg" stores the mean in "value" (or "value_norm" if requested)
    # - stat == "std" stores the standard deviation in "value"
    keys = ["direction", "b_step", "roi"]

    avg = df[df["stat"] == "avg"].copy()
    std = df[df["stat"] == "std"].copy()

    if avg.empty:
        raise ValueError("No rows with stat='avg' were found in the parquet file.")
    if std.empty:
        raise ValueError("No rows with stat='std' were found in the parquet file; they are required for error bars.")

    # Validate required columns
    needed_avg = set(keys + ["g_thorsten", y_col])
    missing_avg = needed_avg - set(avg.columns)
    if missing_avg:
        raise ValueError(f"Missing required columns in avg rows: {sorted(missing_avg)}")

    needed_std = set(keys + ["value"])
    missing_std = needed_std - set(std.columns)
    if missing_std:
        raise ValueError(f"Missing required columns in std rows: {sorted(missing_std)}")

    # Numeric columns
    avg[y_col] = pd.to_numeric(avg[y_col], errors="coerce")
    std["value"] = pd.to_numeric(std["value"], errors="coerce")

    # Keep only the needed columns and rename them
    avg = avg[keys + ["g_thorsten", y_col]].rename(columns={y_col: "y_mean"})
    std = std[keys + ["value"]].rename(columns={"value": "y_std"})

    # Merge avg + std on the shared key: direction/b_step/roi
    d = avg.merge(std, on=keys, how="left")

    # Stable ordering
    d = d.sort_values(["direction", "roi", "b_step"], kind="stable")

    # ----------------------------
    # 2) X axis: G = sqrt(2) * |g_thorsten|
    # ----------------------------
    d["_G"] = compute_G_from_g_thorsten(d, use_sqrt2=use_sqrt2)

    directions = sorted(d["direction"].dropna().unique())
    rois = sorted(d["roi"].dropna().unique())

    # ----------------------------
    # 3) Y limits: do not force (0, 1) when y_col == "value"
    # ----------------------------
    if y_col == "value":
        ylim_to_use = None
    else:
        ylim_to_use = ylim

    # --- A) One plot per (direction, roi)
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

    # --- B) One plot per roi with all directions
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

    # --- C) One plot per direction with all rois
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
