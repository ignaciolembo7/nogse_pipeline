from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from ogse_fitting.fit_nogse_signal import (
    VALID_MODELS,
    run_fit_from_parquet,
    split_all_or_values,
    validate_bounds,
    validate_fixed_value,
    validate_log_bounds,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("signal_parquet", type=Path)
    ap.add_argument("--model", required=True, choices=sorted(VALID_MODELS))
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g")
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)

    m0_group = ap.add_mutually_exclusive_group()
    m0_group.add_argument("--fix_M0", type=float, default=None)
    m0_group.add_argument("--free_M0", action="store_true")

    d0_group = ap.add_mutually_exclusive_group()
    d0_group.add_argument("--fix_D0", type=float, default=None)
    d0_group.add_argument("--free_D0", action="store_true")

    ap.add_argument("--M0_bounds", "--M0-bounds", nargs=2, type=float, default=(0.0, float("inf")), metavar=("MIN", "MAX"))
    ap.add_argument("--D0_bounds", "--D0-bounds", nargs=2, type=float, default=(1e-16, float("inf")), metavar=("MIN", "MAX"))
    args = ap.parse_args()

    fix_m0 = args.fix_M0
    if fix_m0 is None and not args.free_M0 and args.ycol == "value_norm":
        fix_m0 = 1.0
    fix_d0 = args.fix_D0

    m0_bounds = validate_bounds("M0", args.M0_bounds)
    d0_bounds = validate_bounds("D0", args.D0_bounds)
    validate_fixed_value("M0", fix_m0, m0_bounds)
    validate_fixed_value("D0", fix_d0, d0_bounds)
    if fix_d0 is None:
        validate_log_bounds("D0", d0_bounds)

    run_fit_from_parquet(
        args.signal_parquet,
        model=args.model,
        out_root=args.out_root,
        xcol=args.xcol,
        ycol=args.ycol,
        stat_keep=args.stat,
        rois=split_all_or_values(args.rois),
        directions=split_all_or_values(args.directions),
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
    )


if __name__ == "__main__":
    main()
