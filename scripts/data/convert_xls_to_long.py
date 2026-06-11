from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.io import infer_layout_from_filename, read_result_xls, write_table_outputs
from data_processing.reshape import to_long


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a result workbook to the canonical long parquet table.")
    ap.add_argument("xls_path", type=Path, help="Input .xls/.xlsx result workbook.")
    ap.add_argument("--out-path", type=Path, default=None, help="Output parquet path. Defaults to <input>.long.parquet.")
    ap.add_argument("--ndirs", type=int, default=None, help="Number of directions. Defaults to filename inference.")
    ap.add_argument("--nbvals", type=int, default=None, help="Number of b-values. Defaults to filename inference.")
    args = ap.parse_args()

    xls_path = args.xls_path
    layout = infer_layout_from_filename(xls_path)
    ndirs = args.ndirs if args.ndirs is not None else layout.ndirs
    nbvals = args.nbvals if args.nbvals is not None else layout.nbvals
    if nbvals is None or ndirs is None:
        raise SystemExit("Could not infer nbvals/ndirs from the filename. Pass --nbvals and --ndirs.")

    stats = read_result_xls(xls_path)
    df_long = to_long(
        stats,
        ndirs=ndirs,
        nbvals=nbvals,
        source_file=xls_path.name,
    )

    out_path = args.out_path or xls_path.with_suffix(".long.parquet")
    write_table_outputs(df_long, out_path)
    print("Saved:", out_path)
    print(df_long.head())


if __name__ == "__main__":
    main()
