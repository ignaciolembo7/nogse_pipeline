from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from data_processing.io import fit_params_output_basename
from fitting.b_from_g import VALID_AXIS_BASES
from fitting.cli_common import (
    add_fit_master_output_args,
    add_master_source_args,
    add_parameter_mode_args,
    append_fit_params_outputs,
    build_common_parameter_plan,
    signal_correction_factors_from_rows,
    update_master_signal_correction_factors,
)
from fitting.experiments import experiment_models, split_all_or_values, validate_experiment_model
from fitting.model_registry import canonical_signal_model_name, get_signal_model
from nogse_fitting.fit_nogse_signal_vs_g import (
    analysis_id_from_path,
    run_global_mixed_fit_from_parquets,
    run_fit_from_parquet,
    validate_bounds,
    validate_fixed_value,
    validate_log_bounds,
)
from pipeline.recipe import selected_rows_or_legacy_paths


EXPERIMENT = "nogse_signal_vs_g"


def _infer_preferred_side(df: pd.DataFrame, *, model: str) -> int:
    if "type" in df.columns:
        values = sorted({str(v).strip().upper() for v in df["type"].dropna().tolist() if str(v).strip()})
        if len(values) == 1:
            if values[0] == "CPMG":
                return 1
            if values[0] == "HAHN":
                return 2
            raise ValueError(f"Unsupported NOGSE sequence type for gradient correction: {values[0]!r}.")
        if len(values) > 1:
            raise ValueError(f"Expected a single NOGSE sequence type, found: {values}.")

    if str(model).endswith("_cpmg"):
        return 1
    if str(model).endswith("_hahn"):
        return 2
    raise ValueError("Could not infer the preferred correction side from the signal table or model name.")


def _resolve_direction_factors(
    *,
    signal_parquet: Path,
    apply_grad_corr: bool,
    model: str,
) -> dict[str, float] | None:
    if not apply_grad_corr:
        return None

    df = pd.read_parquet(signal_parquet)
    _infer_preferred_side(df, model=model)
    return signal_correction_factors_from_rows(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("signal_parquet", type=Path, nargs="*")
    ap.add_argument("--model", required=True, choices=sorted(experiment_models(EXPERIMENT)))
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g", choices=sorted(VALID_AXIS_BASES))
    ap.add_argument("--plot_xcol", default=None, choices=sorted(VALID_AXIS_BASES))
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
    ap.add_argument("--tc_init", type=float, default=5.0, help="Initial tc seed in ms for model=mixed_global.")
    ap.add_argument("--tc_bounds", "--tc-bounds", nargs=2, type=float, default=(0.1, 1000.0), metavar=("MIN", "MAX"))
    ap.add_argument("--alpha_table", type=Path, default=None, help="Optional table with fixed alpha per subj/roi/direction/td.")
    ap.add_argument("--alpha_col", default=None, help="Alpha column in --alpha_table. Defaults to alpha, then alpha_macro.")
    ap.add_argument("--alpha_td_col", default="td_ms", help="td column in --alpha_table used for per-td matching.")
    ap.add_argument("--alpha_td_tol_ms", type=float, default=1e-3, help="Tolerance in ms for matching alpha by td.")
    ap.add_argument("--no_plots", action="store_true", help="Skip fit plots.")

    corr_group = ap.add_mutually_exclusive_group()
    corr_group.add_argument("--apply_grad_corr", action="store_true")
    corr_group.add_argument("--no_grad_corr", action="store_true")
    add_master_source_args(ap, default_row_kind="signal", include_stat=False)
    add_parameter_mode_args(ap)
    add_fit_master_output_args(ap)
    args = ap.parse_args()

    validate_experiment_model(EXPERIMENT, args.model)

    has_param_plan = any(getattr(args, name) for name in ("param_mode", "param_init", "param_fixed", "param_bounds"))
    plan = None
    if has_param_plan:
        spec = get_signal_model(canonical_signal_model_name(args.model, family="nogse"))
        plan = build_common_parameter_plan(
            args,
            param_names=spec.param_names,
            default_modes=spec.default_modes,
            default_inits=spec.default_inits,
            default_bounds=spec.default_bounds,
            log_params=spec.log_params,
        )

    fix_m0 = args.fix_M0
    if fix_m0 is None and not args.free_M0 and args.ycol == "value_norm":
        fix_m0 = 1.0
    fix_d0 = args.fix_D0
    if plan is not None and "M0" in plan.configs:
        if plan.mode("M0") == "fixed":
            fix_m0 = float(plan.fixed("M0", fix_m0 if fix_m0 is not None else 1.0))
            args.free_M0 = False
        elif plan.mode("M0") in {"free", "global_td", "global_contrast"}:
            fix_m0 = None
            args.free_M0 = True
    if plan is not None and "D0_m2_ms" in plan.configs:
        if plan.mode("D0_m2_ms") == "fixed":
            fix_d0 = float(plan.fixed("D0_m2_ms", fix_d0))
            args.free_D0 = False
        elif plan.mode("D0_m2_ms") in {"free", "global_td", "global_contrast"}:
            fix_d0 = None
            args.free_D0 = True

    m0_bounds = validate_bounds("M0", args.M0_bounds)
    d0_bounds = validate_bounds("D0", args.D0_bounds)
    tc_bounds = validate_bounds("tc", args.tc_bounds)
    validate_fixed_value("M0", fix_m0, m0_bounds)
    validate_fixed_value("D0", fix_d0, d0_bounds)
    if fix_d0 is None:
        validate_log_bounds("D0", d0_bounds)
    validate_log_bounds("tc", tc_bounds)

    selected = selected_rows_or_legacy_paths(
        args,
        legacy_paths=args.signal_parquet,
        default_row_kind=str(args.row_kind or "signal"),
        signal_rotated=str(args.row_kind or "signal") == "signal_rotated",
        temp_prefix="nogse_master_nogse_signal_",
    )
    signal_paths = selected.paths

    backend_model = args.model
    if backend_model == "nogse_free":
        df_for_model = pd.read_parquet(signal_paths[0])
        preferred_side = _infer_preferred_side(df_for_model, model="free_cpmg")
        backend_model = "free_cpmg" if int(preferred_side) == 1 else "free_hahn"

    if args.model != "mixed_global" and len(signal_paths) != 1:
        raise ValueError("Multiple signal parquet inputs are only supported with --model mixed_global.")

    apply_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    f_by_direction = _resolve_direction_factors(
        signal_parquet=signal_paths[0],
        apply_grad_corr=apply_corr,
        model=backend_model,
    )

    try:
        if backend_model == "mixed_global":
            outs, _fit_dir = run_global_mixed_fit_from_parquets(
                signal_paths,
                out_root=args.out_root,
                xcol=args.xcol,
                plot_xcol=args.plot_xcol,
                ycol=args.ycol,
                stat_keep=args.stat,
                rois=split_all_or_values(args.rois),
                directions=split_all_or_values(args.directions),
                fix_m0=fix_m0,
                fix_d0=fix_d0,
                tc_init=float(args.tc_init),
                m0_bounds=m0_bounds,
                d0_bounds=d0_bounds,
                tc_bounds=tc_bounds,
                f_by_direction=f_by_direction,
                alpha_table=args.alpha_table,
                alpha_col=args.alpha_col,
                alpha_td_col=args.alpha_td_col,
                alpha_td_tol_ms=float(args.alpha_td_tol_ms),
                no_plots=bool(args.no_plots),
            )
            append_fit_params_outputs(
                outs.fit_params,
                args,
                fit_kind="nogse_signal",
                model=str(backend_model),
                source=selected.source,
                out_basename=fit_params_output_basename(model=str(backend_model), axis=str(args.xcol), ycol=str(args.ycol), directions=None),
            )
            update_master_signal_correction_factors(args, f_by_direction)
            return

        outs, _fit_dir = run_fit_from_parquet(
            signal_paths[0],
            model=backend_model,
            out_root=args.out_root,
            xcol=args.xcol,
            plot_xcol=args.plot_xcol,
            ycol=args.ycol,
            stat_keep=args.stat,
            rois=split_all_or_values(args.rois),
            directions=split_all_or_values(args.directions),
            fix_m0=fix_m0,
            fix_d0=fix_d0,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
            f_by_direction=f_by_direction,
            append_model_subdir=False,
        )
        append_fit_params_outputs(
            outs.fit_params,
            args,
            fit_kind="nogse_signal",
            model=str(backend_model),
            source=selected.source,
            out_basename=fit_params_output_basename(model=str(backend_model), axis=str(args.xcol), ycol=str(args.ycol), directions=None),
        )
        update_master_signal_correction_factors(args, f_by_direction)
    finally:
        selected.cleanup()


if __name__ == "__main__":
    main()
