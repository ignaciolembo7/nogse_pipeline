from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from nogse_plotting.plot_nogse_signal_vs_g import (
    analysis_id_from_path,
    plot_nogse_signal_table,
    split_all_or_values,
)


YCOL_DIRS = {
    "value": "raw",
    "value_norm": "normalized",
}


def _signal_table_paths(root: Path, pattern: str) -> list[Path]:
    paths = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if ".Dproj." in path.name:
            continue
        paths.append(path)
    return paths


def _has_required_columns(df: pd.DataFrame, *, xcol: str, ycol: str) -> bool:
    required = {"stat", "roi", "direction", xcol, ycol}
    return required.issubset(set(df.columns))


def _relative_parent(path: Path, root: Path) -> Path:
    try:
        rel = path.parent.relative_to(root)
    except ValueError:
        return Path()
    return Path() if str(rel) == "." else rel


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot raw and normalized signal tables versus a gradient column.")
    ap.add_argument("tables_root", type=Path, help="Root containing clean signal *.long.parquet tables.")
    ap.add_argument("--out_root", type=Path, required=True, help="Root folder for signal plots.")
    ap.add_argument("--pattern", default="**/*.long.parquet", help="Relative glob inside tables_root.")
    ap.add_argument("--xcol", default="g_thorsten")
    ap.add_argument("--ycols", nargs="+", default=["value", "value_norm"])
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    args = ap.parse_args()

    tables_root = args.tables_root
    if not tables_root.is_dir():
        raise SystemExit(f"Tables root not found: {tables_root}")

    paths = _signal_table_paths(tables_root, str(args.pattern))
    if not paths:
        print(f"No signal tables found in: {tables_root}")
        return

    total_outputs = 0
    for table_path in paths:
        df = pd.read_parquet(table_path)
        analysis_id = analysis_id_from_path(table_path)
        rel_parent = _relative_parent(table_path, tables_root)

        for ycol in args.ycols:
            ycol = str(ycol)
            if not _has_required_columns(df, xcol=str(args.xcol), ycol=ycol):
                print(f"Skipping {table_path}: missing required columns for x={args.xcol}, y={ycol}")
                continue

            out_group = YCOL_DIRS.get(ycol, ycol)
            out_root = args.out_root / out_group / rel_parent
            out_paths = plot_nogse_signal_table(
                df,
                out_root=out_root,
                analysis_id=analysis_id,
                xcol=str(args.xcol),
                ycol=ycol,
                stat=str(args.stat),
                rois=split_all_or_values(args.rois),
                directions=split_all_or_values(args.directions),
            )
            total_outputs += len(out_paths)
            for path in out_paths:
                print("Saved:", path)

    print("Generated signal plots:", total_outputs)


if __name__ == "__main__":
    main()
