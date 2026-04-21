from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.io import write_table_outputs, write_xlsx_csv_outputs
from tc_fittings.contrast_fit_table import load_contrast_fit_params


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine fit_params from fit_ogse-contrast_vs_g.py into a coherent groupfits table for the tc-vs-td stage."
    )
    ap.add_argument(
        "fits",
        nargs="+",
        help="One or more contrast-fit roots, or fit_params.(parquet|xlsx|csv) files.",
    )
    ap.add_argument("--pattern", default="**/fit_params.*", help="Relative glob used to discover fit_params inside each root.")
    ap.add_argument("--models", nargs="+", default=None, help="Filter contrast models, for example: rest tort free.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Filter subjects/phantoms.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filter directions.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filter ROIs.")
    ap.add_argument("--include-failed", action="store_true", help="Include rows with ok=False.")
    ap.add_argument("--out-xlsx", type=Path, required=True, help="Combined xlsx output.")
    ap.add_argument("--out-parquet", type=Path, default=None, help="Additional parquet output.")
    args = ap.parse_args()

    df = load_contrast_fit_params(
        args.fits,
        pattern=args.pattern,
        models=args.models,
        subjs=args.subjs,
        directions=args.directions,
        rois=args.rois,
        ok_only=not bool(args.include_failed),
    )

    write_xlsx_csv_outputs(df, args.out_xlsx, csv_path=args.out_xlsx.with_suffix(".csv"))
    if args.out_parquet is not None:
        write_table_outputs(df, args.out_parquet)

    print(f"[OK] groupfits table: {args.out_xlsx}")
    if args.out_parquet is not None:
        print(f"[OK] Parquet:      {args.out_parquet}")


if __name__ == "__main__":
    main()
