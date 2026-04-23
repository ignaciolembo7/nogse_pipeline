from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from fitting.experiments import experiment_models, split_all_or_values, validate_experiment_model
from fitting.gradient_correction import (
    CorrectionLookupSpec,
    build_direction_factors,
    infer_td_ms,
    read_correction_table,
    unique_int,
)
from nogse_fitting.fit_nogse_signal_vs_g import (
    analysis_id_from_path,
    run_fit_from_parquet,
    validate_bounds,
    validate_fixed_value,
    validate_log_bounds,
)


EXPERIMENT = "nogse_signal_vs_g"


def _resolve_direction_factors(
    *,
    signal_parquet: Path,
    apply_grad_corr: bool,
    corr_xlsx: Path | None,
    corr_roi: str,
    corr_td_ms: float | None,
    corr_tol_ms: float,
    corr_sheet: str | None,
    corr_n1: int | None,
    corr_n2: int | None,
) -> dict[str, float] | None:
    if not apply_grad_corr:
        return None
    if corr_xlsx is None:
        raise ValueError("--apply_grad_corr requires --corr_xlsx.")

    df = pd.read_parquet(signal_parquet)
    analysis_id = analysis_id_from_path(signal_parquet)
    td_ms = infer_td_ms(df, analysis_id=analysis_id, override=corr_td_ms)
    if td_ms is None:
        raise ValueError("Could not infer td_ms for correction lookup. Pass --corr_td_ms or add td_ms to the input table.")

    n1 = int(corr_n1) if corr_n1 is not None else unique_int(df, "N_1", "N")
    n2 = int(corr_n2) if corr_n2 is not None else unique_int(df, "N_2", "N")
    corr = read_correction_table(corr_xlsx)
    return build_direction_factors(
        corr,
        spec=CorrectionLookupSpec(
            roi_ref=str(corr_roi),
            td_ms=float(td_ms),
            tol_ms=float(corr_tol_ms),
            sheet=(corr_sheet or analysis_id),
            n1=n1,
            n2=n2,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("signal_parquet", type=Path)
    ap.add_argument("--model", required=True, choices=sorted(experiment_models(EXPERIMENT)))
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g")
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)

    m0_group = ap.add_mutually_exclusive_group()
    m0_group.add_argument("--fix_M0", type=float, default=None)
    m0_group.add_argument("--free_M0", action="store_true")

    d0_group = ap.add_mutually_exclusive_group()
    d0_group.add_argument("--fix_D0", type=float, default=None)
    d0_group.add_argument("--free_D0", action="store_true")

    ap.add_argument("--M0_bounds", "--M0-bounds", nargs=2, type=float, default=(0.0, float("inf")), metavar=("MIN", "MAX"))
    ap.add_argument("--D0_bounds", "--D0-bounds", nargs=2, type=float, default=(1e-16, float("inf")), metavar=("MIN", "MAX"))

    corr_group = ap.add_mutually_exclusive_group()
    corr_group.add_argument("--apply_grad_corr", action="store_true")
    corr_group.add_argument("--no_grad_corr", action="store_true")
    ap.add_argument("--corr_xlsx", type=Path, default=None)
    ap.add_argument("--corr_roi", default="water")
    ap.add_argument("--corr_td_ms", type=float, default=None)
    ap.add_argument("--corr_tol_ms", type=float, default=1e-3)
    ap.add_argument("--corr_sheet", default=None)
    ap.add_argument("--corr_n1", type=int, default=None)
    ap.add_argument("--corr_n2", type=int, default=None)
    ap.add_argument("--grad_corr_power", type=float, default=1.0)
    args = ap.parse_args()

    validate_experiment_model(EXPERIMENT, args.model)

    fix_m0 = args.fix_M0
    if fix_m0 is None and not args.free_M0 and args.ycol == "value_norm":
        fix_m0 = 1.0
    fix_d0 = args.fix_D0

    m0_bounds = validate_bounds("M0", args.M0_bounds)
    d0_bounds = validate_bounds("D0", args.D0_bounds)
    validate_fixed_value("M0", fix_m0, m0_bounds)
    validate_fixed_value("D0", fix_d0, d0_bounds)
    if fix_d0 is None:
        validate_log_bounds("D0", d0_bounds)

    apply_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    f_by_direction = _resolve_direction_factors(
        signal_parquet=args.signal_parquet,
        apply_grad_corr=apply_corr,
        corr_xlsx=args.corr_xlsx,
        corr_roi=args.corr_roi,
        corr_td_ms=args.corr_td_ms,
        corr_tol_ms=args.corr_tol_ms,
        corr_sheet=args.corr_sheet,
        corr_n1=args.corr_n1,
        corr_n2=args.corr_n2,
    )

    run_fit_from_parquet(
        args.signal_parquet,
        model=args.model,
        out_root=args.out_root,
        xcol=args.xcol,
        ycol=args.ycol,
        stat_keep=args.stat,
        rois=split_all_or_values(args.rois),
        directions=split_all_or_values(args.directions),
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
        f_by_direction=f_by_direction,
        grad_corr_power=float(args.grad_corr_power),
        append_model_subdir=False,
    )


if __name__ == "__main__":
    main()
