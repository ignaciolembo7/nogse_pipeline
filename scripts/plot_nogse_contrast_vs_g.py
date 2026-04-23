from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from plottings.contrast_vs_g import filter_stat
from nogse_plotting.plot_nogse_contrast_vs_g import plot_nogse_contrast_summary

VALID_GBASES = {"g", "g_max", "g_lin_max", "g_thorsten"}


def _analysis_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _validate_xcol_vs_g(xcol: str) -> str:
    raw = str(xcol).strip()
    base = raw
    if raw.endswith("_1") or raw.endswith("_2"):
        base = raw[:-2]
    if base not in VALID_GBASES:
        raise ValueError(
            f"plot_nogse_contrast_vs_g only supports gradient axes {sorted(VALID_GBASES)} "
            f"with optional _1/_2 suffix. Received xcol={xcol!r}."
        )
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot NOGSE contrast-vs-g curves from long parquet tables.")
    ap.add_argument("contrast_parquet", type=Path)
    ap.add_argument("--xcol", default="g_thorsten_1", help="Example: g_lin_max_1, g_max_1, g_thorsten_1")
    ap.add_argument("--y", "--ycol", dest="ycol", default="value_norm", help="Example: value, value_norm")
    ap.add_argument("--out_root", default="plots")
    ap.add_argument("--exp", default=None, help="Optional experiment folder name")
    ap.add_argument("--directions", nargs="+", default=None, help="Directions to plot")
    ap.add_argument("--dirs", nargs="+", dest="directions", help="Alias for --directions")
    ap.add_argument("--axes", nargs="+", dest="directions", help="Alias for --directions")
    ap.add_argument("--stat", default="avg", help="Statistic to plot. Use ALL to skip filtering")
    ap.add_argument("--rois", nargs="+", default=None, help="Optional ROI subset for extra subset plots")
    args = ap.parse_args()

    path = Path(args.contrast_parquet)
    df = pd.read_parquet(path)
    df = filter_stat(df, args.stat)

    if "direction" not in df.columns:
        raise KeyError(f"Missing required column 'direction'. Columns={sorted(df.columns)}")

    directions = [str(x) for x in (args.directions or [])]
    if not directions:
        directions = sorted(pd.Series(df["direction"]).dropna().astype(str).unique().tolist())
    if not directions:
        raise ValueError("No valid directions were found for plotting.")

    exp_id = str(args.exp) if args.exp else _analysis_id_from_path(path)
    xcol = _validate_xcol_vs_g(str(args.xcol))
    outputs = plot_nogse_contrast_summary(
        df,
        out_root=args.out_root,
        exp_id=exp_id,
        xcol=xcol,
        ycol=str(args.ycol),
        directions=directions,
        rois_requested=None if args.rois is None else [str(x) for x in args.rois],
        stat=str(args.stat),
    )

    for out_path in outputs:
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
