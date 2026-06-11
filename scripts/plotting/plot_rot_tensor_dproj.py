from __future__ import annotations

import repo_bootstrap  # noqa: F401
from pathlib import Path
import argparse
import pandas as pd

from plottings.plot_rotation import (
    load_dproj_parquet,
    plot_dproj_subplots_by_N,
    plot_dproj_gradient_xyz,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dproj_parquets", nargs="+", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("plots/rotation_tensor"))
    ap.add_argument("--roi", type=str, required=True)
    ap.add_argument("--Ns", type=int, nargs="+", required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()

    df = load_dproj_parquet(args.dproj_parquets)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_dproj_subplots_by_N(
        df,
        args.out_dir / f"Dproj_subplots_{args.roi}.png",
        roi=args.roi,
        Ns=args.Ns,
        title_prefix=args.title,
    )

    plot_dproj_gradient_xyz(
        df,
        args.out_dir / f"Dproj_xyz_gradient_{args.roi}{'_log' if args.logy else ''}.png",
        roi=args.roi,
        Ns=args.Ns,
        logy=args.logy,
        title_prefix=args.title,
    )

    print("Saved plots to:", args.out_dir)


if __name__ == "__main__":
    main()
