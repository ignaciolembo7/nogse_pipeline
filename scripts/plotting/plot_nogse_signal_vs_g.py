from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from data_processing.master_table import build_analysis_id_from_columns, load_master_table, select_plot_signal, split_selector_values
from nogse_plotting.plot_nogse_signal_vs_g import analysis_id_from_path, plot_nogse_signal_table, split_all_or_values


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
    for col in ("td_ms", "N", "Hz"):
        value = getattr(args, col, None)
        if value is not None:
            selectors[col] = float(value)
    return selectors


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot NOGSE signal curves from long parquet tables.")
    ap.add_argument("signal_parquet", type=Path, nargs="?", default=None)
    ap.add_argument("--master-parquet", type=Path, default=None, help="Read signal rows from the master table.")
    ap.add_argument("--row-kind", choices=["signal", "signal_rotated"], default="signal_rotated")
    ap.add_argument("--analysis-id", action="append", default=None)
    ap.add_argument("--subj", action="append", default=None)
    ap.add_argument("--sheet", action="append", default=None)
    ap.add_argument("--roi", action="append", default=None)
    ap.add_argument("--direction", action="append", default=None)
    ap.add_argument("--td_ms", type=float, default=None)
    ap.add_argument("--N", type=float, default=None)
    ap.add_argument("--Hz", type=float, default=None)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g")
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    args = ap.parse_args()

    if args.master_parquet is not None:
        master = load_master_table(args.master_parquet)
        df = select_plot_signal(
            master,
            rotated=args.row_kind == "signal_rotated",
            **_master_selectors(args),
        )
        if df.empty:
            raise ValueError("No master signal rows matched the requested selectors.")
        try:
            analysis_id = build_analysis_id_from_columns(
                df,
                columns=("subj", "sheet", "td_ms", "N", "Hz"),
                prefix=args.row_kind,
            )
        except ValueError:
            analysis_id = f"{args.row_kind}_master_selection"
    else:
        if args.signal_parquet is None:
            raise ValueError("Pass signal_parquet or --master-parquet with selectors.")
        df = pd.read_parquet(args.signal_parquet)
        analysis_id = analysis_id_from_path(args.signal_parquet)

    out_paths = plot_nogse_signal_table(
        df,
        out_root=args.out_root,
        analysis_id=analysis_id,
        xcol=str(args.xcol),
        ycol=str(args.ycol),
        stat=str(args.stat),
        rois=split_all_or_values(args.rois),
        directions=split_all_or_values(args.directions),
    )

    for path in out_paths:
        print("Saved:", path)


if __name__ == "__main__":
    main()
