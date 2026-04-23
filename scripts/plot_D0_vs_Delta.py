from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.io import write_xlsx_csv_outputs
from monoexp_fitting.plot_D0_vs_Delta import (
    load_all_measurements,
    load_selected_bstep_map,
    plot_all_groups,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build D vs Delta_app_ms curves from *.Dproj.long.parquet tables."
    )
    ap.add_argument("--dproj-root", required=True, help="Root folder with *.Dproj.long.parquet tables.")
    ap.add_argument("--pattern", default="**/*.Dproj.long.parquet", help="Relative glob inside dproj-root.")
    ap.add_argument("--out-dir", required=True, help="Output folder for plots and the combined table.")
    subj_group = ap.add_mutually_exclusive_group()
    subj_group.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms to include, for example: BRAIN-3 LUDG-2 PHANTOM3.")
    subj_group.add_argument("--brains", nargs="+", dest="subjs", help="Legacy alias for --subjs.")
    ap.add_argument("--rois", nargs="+", default=None, help="ROIs to include.")
    ap.add_argument("--dirs", nargs="+", default=["x", "y", "z"], help="Directions to include.")

    selector = ap.add_mutually_exclusive_group()
    selector.add_argument("--N", type=float, default=None, help="Filter by N.")
    selector.add_argument("--Hz", type=float, default=None, help="Filter by Hz.")

    ap.add_argument("--bvalue-decimals", type=int, default=1, help="Decimals used to round bvalue before grouping.")
    ap.add_argument(
        "--bvalmax",
        type=int,
        default=None,
        help=(
            "Bstep (1-based) to use for the horizontal line and plot alpha. "
            "If omitted, the highest bvalue is used."
        ),
    )
    ap.add_argument("--reference-D0", type=float, default=0.0032, help="Reference value used to annotate alpha in the plot.")
    ap.add_argument("--reference-D0-error", type=float, default=0.0000283512, help="Reference value error.")
    ap.add_argument(
        "--summary-alpha",
        default=None,
        help=(
            "Optional path to summary_alpha_values (xlsx/csv/parquet). "
            "If omitted and <out-dir>/summary_alpha_values.xlsx exists, it is used automatically."
        ),
    )
    args = ap.parse_args()

    N = 1.0 if args.N is None and args.Hz is None else args.N

    df = load_all_measurements(
        args.dproj_root,
        pattern=args.pattern,
        dirs=args.dirs,
        rois=args.rois,
        subjs=args.subjs,
        N=N,
        Hz=args.Hz,
        bvalue_decimals=int(args.bvalue_decimals),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_summary = out_dir / "summary_alpha_values.xlsx"
    summary_alpha_path = Path(args.summary_alpha) if args.summary_alpha is not None else (default_summary if default_summary.exists() else None)

    selected_bstep_by_group = None
    if summary_alpha_path is not None:
        selected_bstep_by_group = load_selected_bstep_map(summary_alpha_path)
        print(
            f"[INFO] Loaded selected_bstep map from {summary_alpha_path} "
            f"({len(selected_bstep_by_group)} group entries)."
        )
        if args.bvalmax is not None:
            print(
                "[INFO] --bvalmax is used only as fallback for groups missing in summary_alpha."
            )

    plot_all_groups(
        df,
        out_dir=out_dir,
        selected_bstep=args.bvalmax,
        selected_bstep_by_group=selected_bstep_by_group,
        reference_D0=float(args.reference_D0),
        reference_D0_error=float(args.reference_D0_error),
    )

    write_xlsx_csv_outputs(
        df,
        out_dir / "D_vs_delta_app.combined.xlsx",
        csv_path=out_dir / "D_vs_delta_app.combined.csv",
    )
    print(f"[OK] Plots + tables in: {out_dir}")


if __name__ == "__main__":
    main()
