from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from data_processing.master_table import build_analysis_id_from_columns, load_master_table, select_contrast, split_selector_values
from plottings.contrast_vs_g import filter_stat
from ogse_plotting.plot_ogse_contrast_vs_g import plot_ogse_contrast_summary

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
            f"plot_ogse_contrast_vs_g only supports gradient axes {sorted(VALID_GBASES)} "
            f"with optional _1/_2 suffix. Received xcol={xcol!r}."
        )
    return raw


def _master_selectors(args: argparse.Namespace) -> dict[str, object]:
    selectors: dict[str, object] = {}
    for arg_name, col_name in [
        ("analysis_id", "analysis_id"),
        ("subj", "subj"),
        ("sheet", "sheet"),
        ("roi", "roi"),
        ("direction", "direction"),
    ]:
        values = split_selector_values(getattr(args, arg_name, None))
        if values is not None:
            selectors[col_name] = values
    for col in ("td_ms", "N_1", "N_2", "Hz_1", "Hz_2"):
        value = getattr(args, col, None)
        if value is not None:
            selectors[col] = float(value)
    return selectors


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot OGSE contrast-vs-g curves from long parquet tables.")
    ap.add_argument("contrast_parquet", type=Path, nargs="?", default=None)
    ap.add_argument("--master-parquet", type=Path, default=None, help="Read contrast rows from the master table.")
    ap.add_argument("--analysis-id", action="append", default=None)
    ap.add_argument("--subj", action="append", default=None)
    ap.add_argument("--sheet", action="append", default=None)
    ap.add_argument("--roi", action="append", default=None)
    ap.add_argument("--direction", action="append", default=None)
    ap.add_argument("--td_ms", type=float, default=None)
    ap.add_argument("--N_1", type=float, default=None)
    ap.add_argument("--N_2", type=float, default=None)
    ap.add_argument("--Hz_1", type=float, default=None)
    ap.add_argument("--Hz_2", type=float, default=None)
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

    if args.master_parquet is not None:
        df = select_contrast(load_master_table(args.master_parquet), **_master_selectors(args))
        if df.empty:
            raise ValueError("No master contrast rows matched the requested selectors.")
        path = None
    else:
        if args.contrast_parquet is None:
            raise ValueError("Pass contrast_parquet or --master-parquet with selectors.")
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

    if args.exp:
        exp_id = str(args.exp)
    elif path is not None:
        exp_id = _analysis_id_from_path(path)
    else:
        try:
            exp_id = build_analysis_id_from_columns(
                df,
                columns=("subj", "sheet", "td_ms", "N_1", "N_2", "Hz_1", "Hz_2"),
                prefix="contrast",
            )
        except ValueError:
            exp_id = "contrast_master_selection"
    xcol = _validate_xcol_vs_g(str(args.xcol))
    outputs = plot_ogse_contrast_summary(
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
