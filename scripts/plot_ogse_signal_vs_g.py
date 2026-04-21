from __future__ import annotations

import repo_bootstrap  # noqa: F401
from pathlib import Path
import argparse

from plottings.plotting import load_long_parquet, plot_ogse_vs_G


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("long_parquet", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("plots/ogse_vs_g"))
    ap.add_argument("--y_col", type=str, default="value_norm")  # o value
    ap.add_argument("--stat", type=str, default="avg")
    ap.add_argument("--no_sqrt2", action="store_true")
    ap.add_argument("--no_ylim", action="store_true")
    args = ap.parse_args()

    df = load_long_parquet(args.long_parquet)

    out_dir = args.out_dir / args.long_parquet.stem
    ylim = None if args.no_ylim else (0.0, 1.0)

    plot_ogse_vs_G(
        df,
        out_dir,
        y_col=args.y_col,
        stat=args.stat,
        use_sqrt2=not args.no_sqrt2,
        ylim=ylim,
    )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
