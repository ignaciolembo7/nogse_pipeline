from __future__ import annotations

import argparse
from pathlib import Path

import repo_bootstrap  # noqa: F401

from dicom_params.extraction import extract_metadata_tables


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Extract Siemens/NOGSE DICOM metadata into per-DICOM audit tables and "
            "sequence-parameter-style summaries."
        )
    )
    ap.add_argument("dicom_root", type=Path, help="Folder containing DICOM/IMA files.")
    ap.add_argument("--out-root", type=Path, required=True, help="Output folder for extracted tables.")
    ap.add_argument("--glob", action="append", default=None, help="DICOM glob. Repeatable. Default: *.IMA")
    ap.add_argument("--recursive", action="store_true", help="Search recursively under dicom_root.")
    ap.add_argument("--nifti-root", type=Path, default=None, help="Optional folder with converted NIfTI files.")
    ap.add_argument("--nifti-glob", action="append", default=None, help="NIfTI glob. Repeatable. Default: *.nii.gz")
    ap.add_argument("--nifti-recursive", action="store_true", help="Search recursively under nifti_root.")
    ap.add_argument("--scanner-grad-max-mtm", type=float, default=80.0, help="Scanner maximum gradient in mT/m.")
    ap.add_argument(
        "--write-strings",
        action="store_true",
        help="Also write a long table with every printable string extracted from every DICOM.",
    )
    ap.add_argument("--out-xlsx", type=Path, default=None, help="Optional Excel workbook path.")
    parquet_group = ap.add_mutually_exclusive_group()
    parquet_group.add_argument("--write-parquet", dest="write_parquet", action="store_true", default=True)
    parquet_group.add_argument("--no-parquet", dest="write_parquet", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dicom_root = args.dicom_root.resolve()
    out_root = args.out_root.resolve()
    nifti_root = args.nifti_root.resolve() if args.nifti_root is not None else None

    print(f"[INFO] DICOM root: {dicom_root}")
    if nifti_root is not None:
        print(f"[INFO] NIfTI root: {nifti_root}")
    print(f"[INFO] Scanner max gradient: {args.scanner_grad_max_mtm:g} mT/m")
    print(f"[INFO] Write Parquet: {bool(args.write_parquet)}")

    outputs = extract_metadata_tables(
        dicom_root=dicom_root,
        out_root=out_root,
        dicom_patterns=args.glob or ["*.IMA"],
        dicom_recursive=bool(args.recursive),
        scanner_grad_max_mtm=float(args.scanner_grad_max_mtm),
        nifti_root=nifti_root,
        nifti_patterns=args.nifti_glob or ["*.nii.gz"],
        nifti_recursive=bool(args.nifti_recursive),
        write_strings=bool(args.write_strings),
        out_xlsx=args.out_xlsx,
        write_parquet_output=bool(args.write_parquet),
    )

    print("[OK] Wrote:")
    for path in [
        outputs.sequence_csv,
        outputs.sequence_parquet,
        outputs.nifti_csv,
        outputs.nifti_parquet,
        outputs.dicom_csv,
        outputs.dicom_parquet,
        outputs.key_values_csv,
        outputs.key_values_parquet,
        outputs.strings_csv,
        outputs.strings_parquet,
        outputs.xlsx,
    ]:
        if path is not None:
            print(f"  {path}")


if __name__ == "__main__":
    main()
