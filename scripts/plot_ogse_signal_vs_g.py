from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from ogse_plotting.plot_ogse_signal_vs_g import load_long_parquet, plot_ogse_signal_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot OGSE signal curves from long parquet tables.")
    ap.add_argument("long_parquet", type=Path)
    ap.add_argument("--out_root", "--out_dir", dest="out_root", type=Path, default=Path("plots/ogse_vs_g"))
    ap.add_argument("--ycol", "--y_col", dest="ycol", default="value_norm")
    ap.add_argument("--xcol", default="g_thorsten")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--no_sqrt2", action="store_true")
    ap.add_argument("--no_ylim", action="store_true")
    args = ap.parse_args()

    df = load_long_parquet(args.long_parquet)
    out_dir = args.out_root / args.long_parquet.stem
    ylim = None if args.no_ylim else (0.0, 1.0)

    outputs = plot_ogse_signal_summary(
        df,
        out_dir,
        xcol=str(args.xcol),
        ycol=str(args.ycol),
        stat=str(args.stat),
        use_sqrt2=not bool(args.no_sqrt2),
        ylim=ylim,
    )

    print("Saved plots to:", out_dir)
    print("Generated:", len(outputs))


if __name__ == "__main__":
    main()
