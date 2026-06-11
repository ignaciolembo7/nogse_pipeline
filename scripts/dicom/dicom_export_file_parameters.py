from __future__ import annotations

import argparse
from pathlib import Path

import repo_bootstrap  # noqa: F401

from dicom_params.single_file import export_one_dicom_parameters


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export all ASCCONV parameters for one selected DICOM file.")
    ap.add_argument("key_values", type=Path, help="dicom_asconv_key_values.long.csv or .long.parquet")
    ap.add_argument("--dicom-file", required=True, help="DICOM path, basename, stem, or unique substring to export.")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional output long CSV path.")
    ap.add_argument("--out-parquet", type=Path, default=None, help="Output long Parquet path.")
    ap.add_argument("--out-xlsx", type=Path, default=None, help="Output Excel workbook path.")
    ap.add_argument("--chunksize", type=int, default=250_000, help="Rows per processing chunk. Default: 250000")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print scan progress every N chunks. Default: 0 disables chunk progress messages.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_one_dicom_parameters(
        key_values=args.key_values.resolve(),
        dicom_file=args.dicom_file,
        out_parquet=args.out_parquet,
        out_xlsx=args.out_xlsx,
        out_csv=args.out_csv,
        chunksize=int(args.chunksize),
        progress_every=int(args.progress_every),
    )

    print("[OK] Wrote:")
    if outputs.csv is not None:
        print(f"  {outputs.csv}")
    print(f"  {outputs.parquet}")
    print(f"  {outputs.xlsx}")
    print(f"[INFO] Matched DICOM files: {outputs.matched_files}")
    print(f"[INFO] Parameter rows: {outputs.rows}")


if __name__ == "__main__":
    main()
