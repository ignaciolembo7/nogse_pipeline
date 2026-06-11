from __future__ import annotations

import repo_bootstrap  # noqa: F401

from pathlib import Path
import argparse
import pandas as pd

from data_processing.io import write_table_outputs
from data_processing.master_table import (
    append_master_rows,
    build_analysis_id_from_columns,
)
from fitting.cli_common import add_master_source_args
from pipeline.recipe import selected_rows_or_legacy_dataframe
from signal_rotation.rotation_tensor import rotate_signals_tensor


def _infer_exp_dir(df: pd.DataFrame, long_parquet: Path) -> str:
    if "sheet" in df.columns:
        vals = pd.Series(df["sheet"]).dropna().astype(str).str.strip().unique().tolist()
        if len(vals) == 1 and vals[0]:
            return vals[0]

    parent = long_parquet.parent.name
    if parent and parent != ".":
        return parent

    stem = long_parquet.stem.replace(".long", "")
    if "_ep2d" in stem:
        return stem.split("_ep2d")[0]
    return stem


def _analysis_id_from_columns(df: pd.DataFrame, *, row_kind: str) -> str:
    preferred = (
        "subj",
        "sheet",
        "type",
        "protocol",
        "group",
        "TN",
        "N",
        "td_ms",
        "Hz",
        "G",
        "sequence",
    )
    unique_cols = []
    for col in preferred:
        if col not in df.columns:
            continue
        values = pd.Series(df[col]).dropna().unique()
        if len(values) == 1:
            unique_cols.append(col)
    return build_analysis_id_from_columns(df, columns=unique_cols, prefix=row_kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("long_parquet", type=Path, nargs="?", help="Legacy input clean signal .long.parquet.")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/ogse_experiments/data-rotated/tables"))
    add_master_source_args(ap, default_row_kind="signal")
    ap.add_argument(
        "--no-legacy-output",
        action="store_true",
        help="Do not write legacy *.rot_tensor.long.parquet and *.Dproj.long.parquet outputs.",
    )
    ap.add_argument("--solver", type=str, default="lstsq", choices=["lstsq", "solve"])
    ap.add_argument("--s0_mode", type=str, default="dir1", choices=["dir1", "mean"])
    ap.add_argument("--b_col", type=str, default="bvalue")
    ap.add_argument(
        "--dirs_txt",
        type=Path,
        default=None,
        help="Nx3 TXT with directions and no header. Defaults to assets/dirs/dirs_{ndirs}.txt.",
    )
    ap.add_argument("--dirs_csv", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.dirs_txt is not None and args.dirs_csv is not None:
        raise SystemExit("Use only one of --dirs_txt or --dirs_csv.")

    selected = selected_rows_or_legacy_dataframe(
        args,
        legacy_path=args.long_parquet,
        default_row_kind=str(args.row_kind or "signal"),
        signal_rotated=False,
    )
    df = selected.df
    if df is None:
        raise SystemExit("No input rows were selected.")
    input_kind = selected.source
    dirs_file = args.dirs_txt if args.dirs_txt is not None else args.dirs_csv

    res = rotate_signals_tensor(
        df,
        solver=args.solver,
        s0_mode=args.s0_mode,
        b_col=args.b_col,
        dirs_file=dirs_file,
    )

    if args.master_parquet is not None:
        rotated_analysis_id = _analysis_id_from_columns(res.rotated_signal_long, row_kind="signal_rotated")
        append_master_rows(
            args.master_parquet if args.master_parquet.exists() else None,
            res.rotated_signal_long,
            row_kind="signal_rotated",
            analysis_id=rotated_analysis_id,
            out_path=args.master_parquet,
        )
        print("Appended rotated signals to master:", args.master_parquet)

    if not args.no_legacy_output:
        if args.long_parquet is not None:
            exp_dir = args.out_dir / _infer_exp_dir(df, args.long_parquet)
            stem = args.long_parquet.stem.replace(".long", "")
        else:
            exp_label = _analysis_id_from_columns(df, row_kind=input_kind)
            exp_dir = args.out_dir / "master"
            stem = exp_label
        exp_dir.mkdir(parents=True, exist_ok=True)

        out_rot = exp_dir / f"{stem}.rot_tensor.long.parquet"
        out_dpr = exp_dir / f"{stem}.rot_tensor.Dproj.long.parquet"

        write_table_outputs(res.rotated_signal_long, out_rot, xlsx_path=out_rot.with_suffix(".xlsx"))
        write_table_outputs(res.dproj_long, out_dpr, xlsx_path=out_dpr.with_suffix(".xlsx"))

        print("Saved rotated signals:", out_rot)
        print("Saved Dproj:", out_dpr)


if __name__ == "__main__":
    main()
