from __future__ import annotations

import repo_bootstrap  # noqa: F401

from pathlib import Path
import argparse
import pandas as pd

from data_processing.io import write_table_outputs
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("long_parquet", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/ogse_experiments/data-rotated"))
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

    df = pd.read_parquet(args.long_parquet)
    dirs_file = args.dirs_txt if args.dirs_txt is not None else args.dirs_csv

    res = rotate_signals_tensor(
        df,
        solver=args.solver,
        s0_mode=args.s0_mode,
        b_col=args.b_col,
        dirs_file=dirs_file,
    )

    exp_dir = args.out_dir / _infer_exp_dir(df, args.long_parquet)
    exp_dir.mkdir(parents=True, exist_ok=True)
    stem = args.long_parquet.stem.replace(".long", "")

    out_rot = exp_dir / f"{stem}.rot_tensor.long.parquet"
    out_dpr = exp_dir / f"{stem}.rot_tensor.Dproj.long.parquet"

    write_table_outputs(res.rotated_signal_long, out_rot, xlsx_path=out_rot.with_suffix(".xlsx"))
    write_table_outputs(res.dproj_long, out_dpr, xlsx_path=out_dpr.with_suffix(".xlsx"))
    
    print("Saved rotated signals:", out_rot)
    print("Saved Dproj:", out_dpr)


if __name__ == "__main__":
    main()
