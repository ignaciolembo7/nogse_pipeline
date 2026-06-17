from __future__ import annotations

"""Fit global signal models from master-table signal curves.

This script is intentionally a thin orchestration layer:

1. read signal rows from the master table;
2. build one curve per subject/ROI/direction/td/N selection;
3. fit each subject/ROI/direction group with one concatenated least-squares problem;
4. write the standard fit tables and optional plots.

The physical fitting logic lives in ``fitting.global_signal_fit``. The input
selection logic lives in ``fitting.global_signal_inputs``.
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import repo_bootstrap  # noqa: F401

from data_processing.io import write_table_outputs
from fitting.b_from_g import normalize_axis_base
from fitting.global_signal_fit import build_global_signal_rows, fit_global_curve_group, param_order
from fitting.global_signal_inputs import (
    group_curves_by_subject_roi_direction,
    load_master_signal_input,
    prepare_signal_curves,
    split_values,
)
from fitting.global_signal_outputs import build_contrast_fit_rows, write_per_analysis_outputs
from fitting.model_registry import get_signal_model, signal_model_names
from fitting.parameter_modes import (
    FitParameterConfig,
    VALID_PARAMETER_MODES,
    make_parameter_config,
    normalize_parameter_mode,
)


def parse_bounds(values: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values.")
    lo, hi = float(values[0]), float(values[1])
    if not lo < hi:
        raise ValueError(f"{name} must be increasing. Received {values}.")
    return lo, hi


def build_param_configs(args: argparse.Namespace, *, model_spec) -> dict[str, FitParameterConfig]:
    mode_args = {
        "tc_ms": args.tc_mode,
        "alpha": args.alpha_mode,
        "RN": args.RN_mode,
        "M0": args.M0_mode,
        "C": args.C_mode,
        "D0_m2_ms": args.D0_mode,
    }
    init_args = {
        "tc_ms": args.tc_init,
        "alpha": args.alpha_init,
        "RN": args.RN_init,
        "M0": args.M0_init,
        "C": args.C_init,
        "D0_m2_ms": args.D0_fixed,
    }
    fixed_args = {
        "tc_ms": args.tc_fixed,
        "alpha": args.alpha_fixed,
        "RN": args.RN_fixed,
        "M0": args.M0_fixed,
        "C": args.C_fixed,
        "D0_m2_ms": args.D0_fixed if normalize_parameter_mode(str(args.D0_mode), param_name="D0_m2_ms") == "fixed" else None,
    }
    bounds_args = {
        "tc_ms": (args.tc_bounds, "tc_bounds"),
        "alpha": (args.alpha_bounds, "alpha_bounds"),
        "RN": (args.RN_bounds, "RN_bounds"),
        "M0": (args.M0_bounds, "M0_bounds"),
        "C": (args.C_bounds, "C_bounds"),
        "D0_m2_ms": (args.D0_bounds, "D0_bounds"),
    }

    configs: dict[str, FitParameterConfig] = {}
    for name in param_order(model_spec):
        bounds_values, bounds_name = bounds_args[name]
        configs[name] = make_parameter_config(
            name=name,
            mode=str(mode_args[name]),
            init=float(init_args[name]),
            fixed=fixed_args[name],
            bounds=parse_bounds(bounds_values, name=bounds_name),
            log=name in set(model_spec.log_params),
        )
    return configs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Global signal fit from master-table signal rows."
        ),
    )
    ap.add_argument("--master-parquet", type=Path, required=True, help="Master table containing the signal rows to fit.")
    ap.add_argument("--row-kind", default="signal_rotated", help="Master row_kind to fit. Defaults to signal_rotated.")
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--type-seq", "--type_seq", "--family", dest="type_seq", choices=["ogse", "nogse"], default="ogse")
    ap.add_argument("--model", default=None, choices=signal_model_names())
    ap.add_argument("--ycol", default="value", help="Signal column to fit. Use value or value_norm.")
    ap.add_argument("--g_type", default="g_thorsten", help="Gradient column used as G in mT/m.")
    ap.add_argument("--subjs", nargs="*", default=None)
    ap.add_argument("--sheets", nargs="*", default=None)
    ap.add_argument("--td_ms", type=float, default=None)
    ap.add_argument("--N", type=float, default=None)
    ap.add_argument("--Hz", type=float, default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--n_fit", type=int, default=None, help="Use first n points after sorting by G. Default: all valid points.")
    ap.add_argument("--min_points", type=int, default=4)

    mode_choices = list(VALID_PARAMETER_MODES)
    ap.add_argument("--tc_mode", default="global_td", choices=mode_choices)
    ap.add_argument("--alpha_mode", default="global_td", choices=mode_choices)
    ap.add_argument("--RN_mode", default="global_td", choices=mode_choices)
    ap.add_argument("--M0_mode", default="global_contrast", choices=mode_choices)
    ap.add_argument("--C_mode", default="global_contrast", choices=mode_choices)
    ap.add_argument("--D0_mode", choices=mode_choices, default="fixed")
    ap.add_argument("--D0_fixed", type=float, default=2.3e-12, help="D0 in m^2/ms; used as fixed value or initial seed.")
    ap.add_argument("--tc_init", type=float, default=5.0)
    ap.add_argument("--tc_fixed", type=float, default=None)
    ap.add_argument("--alpha_init", type=float, default=0.5)
    ap.add_argument("--alpha_fixed", type=float, default=None)
    ap.add_argument("--M0_init", type=float, default=np.nan)
    ap.add_argument("--M0_fixed", type=float, default=None)
    ap.add_argument("--C_init", type=float, default=0.0)
    ap.add_argument("--C_fixed", type=float, default=None)
    ap.add_argument("--RN_init", type=float, default=0.0)
    ap.add_argument("--RN_fixed", type=float, default=None)
    ap.add_argument("--tc_bounds", nargs=2, type=float, default=(0.1, 1000.0))
    ap.add_argument("--alpha_bounds", nargs=2, type=float, default=(0.0, 1.0))
    ap.add_argument("--M0_bounds", nargs=2, type=float, default=(0.0, 1e9))
    ap.add_argument("--C_bounds", nargs=2, type=float, default=(-1e9, 1e9))
    ap.add_argument("--RN_bounds", nargs=2, type=float, default=(0.0, 1e9))
    ap.add_argument("--D0_bounds", nargs=2, type=float, default=(1e-16, 1e-10))
    ap.add_argument("--max_nfev", type=int, default=400000)

    corr_group = ap.add_mutually_exclusive_group()
    corr_group.add_argument("--apply_grad_corr", action="store_true")
    corr_group.add_argument("--no_grad_corr", action="store_true")
    ap.add_argument(
        "--corr_missing",
        choices=["error", "identity", "skip"],
        default="error",
        help="Policy when a requested gradient correction factor is missing.",
    )
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--write_csv", action="store_true", help="Write optional CSV siblings. Default: parquet/xlsx only.")
    ap.add_argument(
        "--write_signal_tables",
        action="store_true",
        help="Write curve-level signal_fit_params and signal_fit_points audit tables. Default: only contrast-level fit_params.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    g_type = normalize_axis_base(str(args.g_type))
    model_name = str(args.model or f"{args.type_seq}_mixed_offset")
    model_spec = get_signal_model(model_name)
    if model_spec.family != str(args.type_seq):
        raise ValueError(f"Model {model_spec.name!r} belongs to type_seq {model_spec.family!r}, not {args.type_seq!r}.")

    param_configs = build_param_configs(args, model_spec=model_spec)
    signal_df = load_master_signal_input(args)

    apply_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)

    curves = prepare_signal_curves(
        signal_df,
        ycol=str(args.ycol),
        g_type=g_type,
        stat=None if str(args.stat).upper() == "ALL" else str(args.stat),
        rois=split_values(args.rois),
        directions=split_values(args.directions),
        n_fit=args.n_fit,
        min_points=int(args.min_points),
        apply_corr=apply_corr,
        corr_missing=str(args.corr_missing),
    )

    fit_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for group_key, group_curves in sorted(group_curves_by_subject_roi_direction(curves).items()):
        # One optimizer sees all curves in this group. Parameter modes decide
        # whether a value is shared across td, shared across a contrast pair,
        # fixed, or local to a single curve.
        values, errors, ok, msg, method = fit_global_curve_group(
            group_curves,
            model_spec=model_spec,
            param_configs=param_configs,
            max_nfev=int(args.max_nfev),
        )
        rows, points = build_global_signal_rows(
            group_curves,
            model_spec=model_spec,
            values=values,
            errors=errors,
            param_configs=param_configs,
            ok=ok,
            msg=msg,
            method=method,
            ycol=str(args.ycol),
            g_type=g_type,
        )
        fit_rows.extend(rows)
        point_rows.extend(points)
        print(
            "Fitted group:",
            f"subj={group_key[0]}",
            f"roi={group_key[1]}",
            f"direction={group_key[2]}",
            f"stat={group_key[3]}",
            f"curves={len(group_curves)}",
            f"ok={ok}",
        )

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    signal_fit_df = pd.DataFrame(fit_rows)
    pair_fit_df = pd.DataFrame(build_contrast_fit_rows(fit_rows, model_spec=model_spec))
    fit_df = pair_fit_df if not pair_fit_df.empty else signal_fit_df
    points_df = pd.DataFrame(point_rows)

    fit_path = out_root / f"fit_params.{model_spec.name}.{g_type}.{args.ycol}.parquet"
    signal_path = out_root / f"signal_fit_params.{model_spec.name}.{g_type}.{args.ycol}.parquet"
    points_path = out_root / f"signal_fit_points.{model_spec.name}.{g_type}.{args.ycol}.parquet"
    write_table_outputs(
        fit_df,
        fit_path,
        xlsx_path=fit_path.with_suffix(".xlsx"),
        csv_path=fit_path.with_suffix(".csv") if args.write_csv else None,
    )
    if args.write_signal_tables:
        write_table_outputs(
            signal_fit_df,
            signal_path,
            xlsx_path=signal_path.with_suffix(".xlsx"),
            csv_path=signal_path.with_suffix(".csv") if args.write_csv else None,
        )
        write_table_outputs(
            points_df,
            points_path,
            csv_path=points_path.with_suffix(".csv") if args.write_csv else None,
        )
    if not args.no_plots:
        write_per_analysis_outputs(
            fit_df=fit_df,
            signal_df=signal_fit_df,
            points_df=points_df,
            out_root=out_root,
            model_name=model_spec.name,
            ycol=str(args.ycol),
            g_type=g_type,
            write_csv=bool(args.write_csv),
            write_signal_tables=bool(args.write_signal_tables),
        )
    print("Saved fit table:", fit_path)
    if args.write_signal_tables:
        print("Saved signal-fit table:", signal_path)
        print("Saved point table:", points_path)
    if not args.no_plots:
        print("Saved per-analysis tables and plots in:", out_root)


if __name__ == "__main__":
    main()
