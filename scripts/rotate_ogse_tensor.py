from __future__ import annotations

import repo_bootstrap  # noqa: F401

from pathlib import Path
import argparse
import pandas as pd

from data_processing.io import write_table_outputs
from data_processing.master_table import (
    append_master_rows,
    build_analysis_id_from_columns,
    load_master_table,
    select_signal,
)
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


def _split_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    return out or None


def _master_selectors(args: argparse.Namespace) -> dict[str, object]:
    selectors: dict[str, object] = {}
    for arg_name, col_name in [
        ("analysis_id", "analysis_id"),
        ("subj", "subj"),
        ("sheet", "sheet"),
        ("roi", "roi"),
        ("stat", "stat"),
        ("source_file", "source_file"),
    ]:
        values = _split_values(getattr(args, arg_name))
        if values is not None:
            selectors[col_name] = values

    for arg_name, col_name in [("td_ms", "td_ms"), ("N", "N"), ("Hz", "Hz")]:
        value = getattr(args, arg_name)
        if value is not None:
            selectors[col_name] = float(value)
    return selectors


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


def _load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.master_parquet is not None and args.long_parquet is None:
        master = load_master_table(args.master_parquet)
        selectors = _master_selectors(args)
        df = select_signal(master, rotated=False, **selectors)
        if df.empty:
            raise SystemExit(f"No unrotated signal rows matched selectors: {selectors}")
        return df, "master"

    if args.long_parquet is None:
        raise SystemExit("Provide long_parquet or use --master-parquet with selectors.")
    return pd.read_parquet(args.long_parquet), "parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("long_parquet", type=Path, nargs="?", help="Legacy input clean signal .long.parquet.")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/ogse_experiments/data-rotated/tables"))
    ap.add_argument(
        "--master-parquet",
        type=Path,
        default=None,
        help="Optional master table. Without long_parquet, input rows are selected from this table; rotated rows are appended back.",
    )
    ap.add_argument("--analysis-id", action="append", default=None, help="Master-table analysis_id selector.")
    ap.add_argument("--subj", action="append", default=None, help="Master-table subj selector.")
    ap.add_argument("--sheet", action="append", default=None, help="Master-table sheet selector.")
    ap.add_argument("--roi", action="append", default=None, help="Master-table ROI selector. Can be repeated.")
    ap.add_argument("--stat", action="append", default=None, help="Master-table stat selector.")
    ap.add_argument("--source-file", action="append", default=None, help="Master-table source_file selector.")
    ap.add_argument("--td_ms", type=float, default=None, help="Master-table td_ms selector.")
    ap.add_argument("--N", type=float, default=None, help="Master-table N selector.")
    ap.add_argument("--Hz", type=float, default=None, help="Master-table Hz selector.")
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

    df, input_kind = _load_input(args)
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
