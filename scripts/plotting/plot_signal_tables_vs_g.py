from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.master_table import build_analysis_id_from_columns, load_master_table, select_plot_signal, split_selector_values
from nogse_plotting.plot_nogse_signal_vs_g import (
    plot_nogse_signal_table,
    split_all_or_values,
)


YCOL_DIRS = {
    "value": "raw",
    "value_norm": "normalized",
}


def _has_required_columns(df: pd.DataFrame, *, xcol: str, ycol: str) -> bool:
    required = {"stat", "roi", "direction", xcol, ycol}
    return required.issubset(set(df.columns))


def _master_selectors(args: argparse.Namespace) -> dict[str, object]:
    selectors: dict[str, object] = {}
    for arg_name, col_name in [
        ("analysis_id", "analysis_id"),
        ("subj", "subj"),
        ("sheet", "sheet"),
        ("roi", "roi"),
        ("direction", "direction"),
        ("stat", "stat"),
        ("source_file", "source_file"),
    ]:
        values = split_selector_values(getattr(args, arg_name, None))
        if values is not None:
            selectors[col_name] = values
    for col_name in ("td_ms", "N", "Hz"):
        value = getattr(args, col_name, None)
        if value is not None:
            selectors[col_name] = float(value)
    return selectors


def _plot_one_table(
    df: pd.DataFrame,
    *,
    analysis_id: str,
    out_root: Path,
    xcol: str,
    ycols: list[str],
    stat: str,
    rois: list[str] | None,
    directions: list[str] | None,
) -> int:
    total_outputs = 0
    for ycol in ycols:
        ycol = str(ycol)
        if not _has_required_columns(df, xcol=xcol, ycol=ycol):
            print(f"Skipping {analysis_id}: missing required columns for x={xcol}, y={ycol}")
            continue
        out_group = YCOL_DIRS.get(ycol, ycol)
        out_paths = plot_nogse_signal_table(
            df,
            out_root=out_root / out_group,
            analysis_id=analysis_id,
            xcol=xcol,
            ycol=ycol,
            stat=stat,
            rois=rois,
            directions=directions,
        )
        total_outputs += len(out_paths)
        for path in out_paths:
            print("Saved:", path)
    return total_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot signal rows versus a gradient column.")
    ap.add_argument("--master-parquet", type=Path, required=True, help="Read signal rows from the master table.")
    ap.add_argument("--row-kind", default="signal_rotated", help="Master row_kind to plot. Defaults to signal_rotated.")
    ap.add_argument("--analysis-id", action="append", default=None)
    ap.add_argument("--subj", action="append", default=None)
    ap.add_argument("--sheet", action="append", default=None)
    ap.add_argument("--roi", action="append", default=None)
    ap.add_argument("--direction", action="append", default=None)
    ap.add_argument("--source-file", action="append", default=None)
    ap.add_argument("--td_ms", type=float, default=None)
    ap.add_argument("--N", type=float, default=None)
    ap.add_argument("--Hz", type=float, default=None)
    ap.add_argument("--out_root", type=Path, required=True, help="Root folder for signal plots.")
    ap.add_argument("--xcol", default="g_thorsten")
    ap.add_argument("--ycols", nargs="+", default=["value", "value_norm"])
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    args = ap.parse_args()

    rois = split_all_or_values(args.rois)
    directions = split_all_or_values(args.directions)

    master = load_master_table(args.master_parquet)
    selectors = _master_selectors(args)
    df = select_plot_signal(master, rotated=str(args.row_kind) == "signal_rotated", **selectors)
    if df.empty:
        raise SystemExit(f"No master signal rows matched selectors: {selectors}")
    try:
        analysis_id = build_analysis_id_from_columns(
            df,
            columns=[c for c in ("subj", "sheet", "type", "td_ms", "N", "Hz", "direction") if c in df.columns],
            prefix=str(args.row_kind),
        )
    except ValueError:
        analysis_id = f"{args.row_kind}_master_selection"
    total_outputs = _plot_one_table(
        df,
        analysis_id=analysis_id,
        out_root=args.out_root,
        xcol=str(args.xcol),
        ycols=[str(y) for y in args.ycols],
        stat=str(args.stat),
        rois=rois,
        directions=directions,
    )

    print("Generated signal plots:", total_outputs)
    if total_outputs == 0:
        raise SystemExit("No signal plots were generated from the selected master rows.")


if __name__ == "__main__":
    main()
