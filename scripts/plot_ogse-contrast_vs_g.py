from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

VALID_YCOLS = {"value", "value_norm"}


def _canonical_xcol(xcol: str) -> str:
    return xcol


def _canonical_ycol(ycol: str) -> str:
    if ycol not in VALID_YCOLS:
        raise ValueError(f"plot_ogse_contrast_vs_g: unrecognized y value {ycol!r}. Allowed values: {sorted(VALID_YCOLS)}.")
    return ycol


def _plot_x_series(df: pd.DataFrame, xcol: str) -> pd.Series:
    x = pd.to_numeric(df[xcol], errors="coerce")
    if xcol.startswith("g_thorsten_"):
        x = np.sqrt(2.0) * x.abs()
    return x


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"plot_ogse_contrast_vs_g: missing required columns {missing}. Columns={sorted(df.columns)}")


def _filter_stat(df: pd.DataFrame, stat: str | None) -> pd.DataFrame:
    if "stat" not in df.columns or stat is None or str(stat).upper() == "ALL":
        return df.copy()

    out = df[df["stat"].astype(str) == str(stat)].copy()
    if out.empty:
        available = sorted(df["stat"].dropna().astype(str).unique().tolist())
        raise SystemExit(f"No data remains after filtering stat={stat!r}. Available stats: {available}")
    return out


def _unique_int_or_none(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(vals) != 1:
        return None
    return int(vals[0])


def _distinct_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    golden_ratio = 0.618033988749895
    hue = 0.11
    colors: list[str] = []
    for _ in range(n):
        hue = (hue + golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.9)
        colors.append(mcolors.to_hex(rgb))
    return colors


def _selected_rois(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    available = sorted(df["roi"].dropna().astype(str).unique().tolist())
    if not requested:
        return available
    chosen = [roi for roi in requested if roi in available]
    missing = [roi for roi in requested if roi not in available]
    if missing:
        raise SystemExit(f"ROIs not found: {missing}. Available ROIs: {available}")
    return chosen


def _plot_rois_together(
    df: pd.DataFrame,
    *,
    rois: list[str],
    direction_name: str,
    xcol: str,
    ycol: str,
    title_suffix: str,
    ylabel: str,
    outpath: Path,
    legend_title: str,
) -> None:
    d = df[df["direction"].astype(str) == str(direction_name)].copy()
    if d.empty:
        return

    colors = _distinct_colors(len(rois))
    plt.figure(figsize=(8, 6))
    plt.title(f"direction={direction_name} | {title_suffix}", fontsize=14)
    plt.xlabel(f"Gradient strength ({xcol}) [mT/m]", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.3)

    for roi, color in zip(rois, colors):
        dr = d[d["roi"] == roi].copy()
        if dr.empty:
            continue
        dr[xcol] = _plot_x_series(dr, xcol)
        dr[ycol] = pd.to_numeric(dr[ycol], errors="coerce")
        dr = dr.dropna(subset=[xcol, ycol]).sort_values(xcol)
        if dr.empty:
            continue
        plt.plot(dr[xcol], dr[ycol], "o-", label=roi, color=color)

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=14, title=legend_title)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(direction="in", top=True, right=True, left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("Saved:", outpath)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet")
    ap.add_argument("--xcol", default="g_thorsten_1", help="Example: g_lin_max_1 | g_max_1 | g_thorsten_1")
    ap.add_argument("--y", dest="ycol", default="value_norm", help="Example: value | value_norm")
    ap.add_argument("--out_root", default="plots")
    ap.add_argument("--exp", default=None, help="Optional experiment folder name")
    ap.add_argument("--directions", nargs="+", default=None, help="Directions to plot (e.g. long tra or 1 2 3).")
    ap.add_argument("--dirs", nargs="+", dest="directions", help="Alias for --directions.")
    ap.add_argument("--axes", nargs="+", dest="directions", help="Alias for --directions.")
    ap.add_argument("--stat", default="avg", help="Statistic to plot. Use ALL to skip filtering.")
    ap.add_argument("--rois", nargs="+", default=None, help="Optional ROI subset for an additional combined plot.")
    args = ap.parse_args()

    p = Path(args.contrast_parquet)
    df = pd.read_parquet(p)
    df = _filter_stat(df, args.stat)

    if "direction" not in df.columns:
        raise KeyError(f"plot_ogse_contrast_vs_g: missing required column 'direction'. Columns={sorted(df.columns)}")

    xcol = _canonical_xcol(args.xcol)
    ycol = _canonical_ycol(args.ycol)
    _require_columns(df, ["roi", "direction", xcol, ycol])

    directions = [str(x) for x in (args.directions or [])]
    if not directions:
        directions = sorted(pd.Series(df["direction"]).dropna().astype(str).unique().tolist())
    if not directions:
        raise SystemExit("No valid directions were found for plotting.")

    exp_id = args.exp or (p.stem[:-len(".long")] if p.stem.endswith(".long") else p.stem)
    outdir = Path(args.out_root) / exp_id
    outdir.mkdir(parents=True, exist_ok=True)

    rois = _selected_rois(df, None)
    selected_rois = _selected_rois(df, args.rois)
    n_1 = _unique_int_or_none(df, "N_1")
    n_2 = _unique_int_or_none(df, "N_2")
    title_suffix = f"N{n_1}-N{n_2}" if (n_1 is not None and n_2 is not None) else exp_id
    ylabel = rf"OGSE contrast $\Delta M_{{N{n_1}-N{n_2}}}$" if (n_1 is not None and n_2 is not None) else ycol

    for direction_name in directions:
        _plot_rois_together(
            df,
            rois=rois,
            direction_name=direction_name,
            xcol=xcol,
            ycol=ycol,
            title_suffix=title_suffix,
            ylabel=ylabel,
            outpath=outdir / f"{exp_id}.{ycol}_vs_{xcol}.direction-{direction_name}.allROIs.png",
            legend_title="ROI",
        )

        if args.rois:
            roi_tag = "-".join(selected_rois)
            _plot_rois_together(
                df,
                rois=selected_rois,
                direction_name=direction_name,
                xcol=xcol,
                ycol=ycol,
                title_suffix=title_suffix,
                ylabel=ylabel,
                outpath=outdir / f"{exp_id}.{ycol}_vs_{xcol}.direction-{direction_name}.subsetROIs-{roi_tag}.png",
                legend_title="ROI subset",
            )

    for roi in rois:
        d = df[(df["roi"] == roi) & (df["direction"].astype(str).isin(directions))].copy()
        if d.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.title(f"ROI={roi} | {title_suffix}", fontsize=14)
        plt.xlabel(f"Gradient strength ({xcol}) [mT/m]", fontsize=18)
        plt.ylabel(rf"OGSE contrast $\Delta M_{{N{n_1}-N{n_2}}}$" if (n_1 is not None and n_2 is not None) else ycol, fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.3)

        for direction_name in directions:
            da = d[d["direction"].astype(str) == str(direction_name)].copy()
            if da.empty:
                continue
            da[xcol] = _plot_x_series(da, xcol)
            da[ycol] = pd.to_numeric(da[ycol], errors="coerce")
            da = da.dropna(subset=[xcol, ycol]).sort_values(xcol)
            if da.empty:
                continue
            plt.plot(da[xcol], da[ycol], "o-", label=str(direction_name))

        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(fontsize=14, title="direction")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(direction="in", top=True, right=True, left=True, bottom=True)
        plt.tight_layout()
        outpath = outdir / f"{exp_id}.{ycol}_vs_{xcol}.ROI-{roi}.directions.png"
        plt.savefig(outpath, dpi=300)
        plt.close()
        print("Saved:", outpath)


if __name__ == "__main__":
    main()
