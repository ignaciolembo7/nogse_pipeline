from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.master_table import (
    filter_master_rows,
    load_fit_params_table,
    load_master_table,
    select_fit_params,
)
from data_processing.io import write_table_outputs, write_xlsx_csv_outputs
from ogse_fitting.contrast_tc_peak_panels import add_resampled_data_peak_columns
from tc_fittings.contrast_fit_table import canonicalize_contrast_fit_params
from tc_fittings.contrast_fit_table import load_contrast_fit_params


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine fit_params from the contrast family fit scripts into a coherent groupfits table for the tc-vs-td stage."
    )
    ap.add_argument(
        "fits",
        nargs="*",
        help="One or more contrast-fit roots, or fit_params.(parquet|xlsx|csv) files.",
    )
    ap.add_argument("--master-fit-params", type=Path, default=None, help="Read contrast fit params from a cumulative master_fit_params table.")
    ap.add_argument("--master-parquet", type=Path, default=None, help="Read row_kind='fit_params' rows from the master table.")
    ap.add_argument("--pattern", default="**/fit_params.*", help="Relative glob used to discover fit_params inside each root.")
    ap.add_argument("--models", nargs="+", default=None, help="Filter contrast models, for example: rest tort free.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Filter subjects/phantoms.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filter directions.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filter ROIs.")
    ap.add_argument("--include-failed", action="store_true", help="Include rows with ok=False.")
    ap.add_argument(
        "--exclude-fitresamp",
        action="store_true",
        help="Ignore fit_params whose analysis_id comes from an already fitted/resampled contrast table.",
    )
    ap.add_argument(
        "--only-fitresamp",
        action="store_true",
        help="Use only fit_params whose analysis_id comes from a fitted/resampled contrast table.",
    )
    ap.add_argument(
        "--add-resampled-data-peaks",
        action="store_true",
        help="Add tc_peak_resampled_data_ms from the resampled contrast tables, matching the tc peak panels.",
    )
    ap.add_argument("--contrast-root", type=Path, default=None, help="Root with resampled contrast tables for --add-resampled-data-peaks.")
    ap.add_argument("--peak-D0-fix", type=float, default=3.2e-12, help="Fixed D0 used to convert resampled gradient to tc.")
    ap.add_argument("--peak-gamma", type=float, default=267.5221900, help="Gamma used to convert resampled gradient to tc.")
    ap.add_argument("--out-xlsx", type=Path, required=True, help="Combined xlsx output.")
    ap.add_argument("--out-parquet", type=Path, default=None, help="Additional parquet output.")
    args = ap.parse_args()

    if args.master_fit_params is not None:
        raw = load_fit_params_table(args.master_fit_params)
        df = select_fit_params(
            raw,
            fit_kind=["ogse_contrast", "nogse_contrast"],
            model=args.models,
            subj=args.subjs,
            direction=args.directions,
            roi=args.rois,
            ok_only=not bool(args.include_failed),
        )
        df = canonicalize_contrast_fit_params(df)
    elif args.master_parquet is not None:
        master = load_master_table(args.master_parquet)
        raw = filter_master_rows(master, row_kind="fit_params")
        df = select_fit_params(
            raw,
            fit_kind=["ogse_contrast", "nogse_contrast"],
            model=args.models,
            subj=args.subjs,
            direction=args.directions,
            roi=args.rois,
            ok_only=not bool(args.include_failed),
        )
        df = canonicalize_contrast_fit_params(df)
    else:
        if not args.fits:
            raise ValueError("Pass --master-fit-params, --master-parquet, or one or more legacy fit roots/files.")
        df = load_contrast_fit_params(
            args.fits,
            pattern=args.pattern,
            models=args.models,
            subjs=args.subjs,
            directions=args.directions,
            rois=args.rois,
            exclude_fitresamp=bool(args.exclude_fitresamp),
            only_fitresamp=bool(args.only_fitresamp),
            ok_only=not bool(args.include_failed),
        )
    if args.add_resampled_data_peaks:
        if args.contrast_root is None:
            raise ValueError("--add-resampled-data-peaks requires --contrast-root.")
        df = add_resampled_data_peak_columns(
            df,
            contrast_root=args.contrast_root,
            peak_D0_fix=float(args.peak_D0_fix),
            peak_gamma=float(args.peak_gamma),
        )

    write_xlsx_csv_outputs(df, args.out_xlsx, csv_path=args.out_xlsx.with_suffix(".csv"))
    if args.out_parquet is not None:
        write_table_outputs(df, args.out_parquet)

    print(f"[OK] groupfits table: {args.out_xlsx}")
    if args.out_parquet is not None:
        print(f"[OK] Parquet:      {args.out_parquet}")


if __name__ == "__main__":
    main()
