from __future__ import annotations

import argparse
from pathlib import Path

import repo_bootstrap  # noqa: F401

from dicom_params.correlation import correlate_asconv_with_gradient


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Correlate numeric DICOM ASCCONV parameters with the gradient encoded for each series."
    )
    ap.add_argument("key_values", type=Path, help="dicom_asconv_key_values.long.csv or .long.parquet")
    ap.add_argument(
        "--nifti-table",
        type=Path,
        required=True,
        help="sequence_parameters_by_nifti_from_dicom.csv or .parquet",
    )
    ap.add_argument("--out-csv", type=Path, default=None, help="Output CSV path.")
    ap.add_argument("--out-xlsx", type=Path, default=None, help="Output Excel workbook path.")
    ap.add_argument("--source", default="ASCCONV", help="Source value to include. Default: ASCCONV")
    ap.add_argument("--chunksize", type=int, default=250_000, help="Rows per processing chunk. Default: 250000")
    ap.add_argument("--min-observations", type=int, default=3, help="Minimum numeric observations per key.")
    ap.add_argument(
        "--sort-by",
        choices=["abs_correlation", "correlation"],
        default="abs_correlation",
        help="Sort output by signed or absolute Pearson correlation. Default: abs_correlation",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print scan progress every N chunks. Default: 0 disables chunk progress messages.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs = correlate_asconv_with_gradient(
        key_values=args.key_values.resolve(),
        nifti_table=args.nifti_table.resolve(),
        out_csv=args.out_csv,
        out_xlsx=args.out_xlsx,
        source=str(args.source),
        chunksize=int(args.chunksize),
        min_observations=int(args.min_observations),
        sort_by=str(args.sort_by),
        progress_every=int(args.progress_every),
    )

    print(f"[INFO] Series with gradients: {outputs.series_count}")
    print("[OK] Wrote:")
    print(f"  {outputs.csv}")
    print(f"  {outputs.xlsx}")
    print(f"[INFO] Correlation rows: {outputs.rows}")


if __name__ == "__main__":
    main()
