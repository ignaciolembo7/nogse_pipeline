from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from nogse_plotting.plot_nogse_signal_vs_g import analysis_id_from_path, plot_nogse_signal_table, split_all_or_values


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot NOGSE signal curves from long parquet tables.")
    ap.add_argument("signal_parquet", type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g")
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    args = ap.parse_args()

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
