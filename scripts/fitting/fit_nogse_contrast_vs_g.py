from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from fitting.b_from_g import VALID_AXIS_BASES
from fitting.cli_common import (
    add_fit_master_output_args,
    add_master_source_args,
    add_parameter_mode_args,
    append_fit_params_outputs,
    build_common_parameter_plan,
)
from fitting.experiments import experiment_models, validate_experiment_model
from fitting.model_registry import canonical_contrast_model_name, get_contrast_model
from fitting.gradient_correction import (
    CorrectionLookupSpec,
    build_direction_factors,
    infer_td_ms,
    read_correction_table,
    unique_int,
)
from nogse_fitting.fit_nogse_contrast_vs_g import fit_nogse_contrast_long, plot_nogse_contrast_fit_one_group
from pipeline.recipe import selected_rows_or_legacy_table
from data_processing.io import fit_params_output_basename, write_table_outputs
from tools.brain_labels import canonical_sheet_name, infer_subj_label


def _analysis_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _validate_bounds(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound: {lower}, {upper}")
    return lower, upper


def _validate_fixed_value(name: str, value: float | None, bounds: tuple[float, float] | None) -> None:
    if value is None or bounds is None:
        return
    lower, upper = bounds
    if not lower <= float(value) <= upper:
        raise ValueError(f"{name} fixed value {value} is outside bounds [{lower}, {upper}].")


def _validate_log_bounds(name: str, bounds: tuple[float, float] | None) -> None:
    if bounds is None:
        return
    lower, _upper = bounds
    if lower <= 0:
        raise ValueError(f"{name} lower bound must be positive when fitting in log-space: {lower}")


def main() -> None:
    plot_axis_choices = sorted({*VALID_AXIS_BASES, *[f"{axis}_1" for axis in VALID_AXIS_BASES]})
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet", type=Path, nargs="?", help="Input long-form contrast parquet produced by make_contrast.py")

    ap.add_argument("--model", required=True, choices=sorted(experiment_models("nogse_contrast_vs_g")))
    ap.add_argument("--gbase", default="g_lin_max", choices=sorted(VALID_AXIS_BASES))
    ap.add_argument("--plot_xcol", default=None, choices=plot_axis_choices)
    ap.add_argument("--ycol", default="value_norm", help="Signal column: value or value_norm")

    ap.add_argument("--directions", nargs="*", default=None, help="Filter by direction values, for example: 1 2 3 or long tra")
    ap.add_argument("--direction", nargs="*", dest="directions", help="Alias for --directions.")

    ap.add_argument("--subjs", nargs="*", default=None, help="Filter subjects/phantoms. Use ALL to keep all of them.")
    ap.add_argument("--rois", nargs="*", default=None, help="Filter ROIs. Use ALL to keep all of them.")
    ap.add_argument("--stat", default="avg", help="Filter the stat column. Use ALL to skip this filter.")
    ap.add_argument("--oneg", action="store_true", help="Allow one-g-per-sequence contrast tables with sequence ranges.")

    ap.add_argument("--out_root", required=True)
    ap.add_argument("--no_plots", action="store_true")

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--apply_grad_corr", action="store_true")
    grp.add_argument("--no_grad_corr", action="store_true")

    ap.add_argument("--corr_xlsx", type=Path, default=None)
    ap.add_argument("--corr_roi", default="Agua")
    ap.add_argument("--corr_td_ms", type=float, default=None)
    ap.add_argument("--corr_tol_ms", type=float, default=1e-3)
    ap.add_argument("--corr_sheet", default=None, help="Optional sheet name to use inside the correction table. Defaults to the analysis_id prefix.")

    grp_m0 = ap.add_mutually_exclusive_group()
    grp_m0.add_argument("--fix_M0", type=float, default=None, help="Fix M0 to a specific value.")
    grp_m0.add_argument(
        "--free_M0",
        nargs="?",
        const=1.0,
        type=float,
        default=None,
        help="Keep M0 free. Optional value is the initial seed. Default seed: 1.0.",
    )
    grp_d0 = ap.add_mutually_exclusive_group()
    grp_d0.add_argument("--fix_D0", type=float, default=None, help="Fix D0 in m^2/ms. Example: 3.2e-12 for 0.0032 mm^2/s.")
    grp_d0.add_argument(
        "--free_D0",
        nargs="?",
        const=2.3e-12,
        type=float,
        default=None,
        help="Keep D0 free. Optional value is the initial seed. Default seed: 2.3e-12.",
    )
    grp_tc = ap.add_mutually_exclusive_group()
    grp_tc.add_argument("--fix_tc", type=float, default=None, help="Fix tc in ms. Used only for model=rest.")
    grp_tc.add_argument(
        "--free_tc",
        nargs="?",
        const=5.0,
        type=float,
        default=None,
        help="Keep tc free. Optional value is the initial seed. Default seed: 5.0. Used only for model=rest.",
    )
    grp_g0 = ap.add_mutually_exclusive_group()
    grp_g0.add_argument(
        "--fix_g0",
        type=float,
        default=None,
        help="Fix g0 in mT/m. Used only for model=nogse_free_grad_offset.",
    )
    grp_g0.add_argument(
        "--free_g0",
        nargs="?",
        const=0.0,
        type=float,
        default=None,
        help=(
            "Keep g0 free in mT/m. Optional value is the initial seed. "
            "Default seed: 0.0. Used only for model=nogse_free_grad_offset."
        ),
    )
    ap.add_argument("--M0_bounds", "--M0-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--D0_bounds", "--D0-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--tc_bounds", "--tc-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument(
        "--g0_bounds",
        "--g0-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="g0 bounds in mT/m. Default for model=nogse_free_grad_offset: -20 20.",
    )

    ap.add_argument("--n_fit", type=int, default=None, help="Use only the first n_fit points after sorting by x.")
    ap.add_argument("--peak_grid_n", type=int, default=1000, help="Number of points used to search for the fitted peak.")
    ap.add_argument("--peak_D0_fix", type=float, default=3.2e-12, help="Fixed D0 used to convert the peak into tc_peak_ms.")
    ap.add_argument("--peak_gamma", type=float, default=267.5221900, help="Gamma in rad/(ms*mT) used to convert the peak into tc_peak_ms.")
    add_master_source_args(ap, default_row_kind="contrast", include_roi=False, include_direction=False, include_stat=False, include_N=False)
    add_parameter_mode_args(ap)
    add_fit_master_output_args(ap)
    args = ap.parse_args()
    validate_experiment_model("nogse_contrast_vs_g", args.model)
    backend_model = {
        "nogse_free": "free",
        "nogse_tort": "tort",
        "nogse_rest": "rest",
    }.get(args.model, args.model)
    has_param_plan = any(getattr(args, name) for name in ("param_mode", "param_init", "param_fixed", "param_bounds"))
    plan = None
    if has_param_plan:
        spec = get_contrast_model(canonical_contrast_model_name(args.model, family="nogse"), family="nogse")
        plan = build_common_parameter_plan(
            args,
            param_names=spec.param_names,
            default_modes=spec.default_modes,
            default_inits=spec.default_inits,
            default_bounds=spec.default_bounds,
            log_params=spec.log_params,
        )

    selected = selected_rows_or_legacy_table(
        args,
        legacy_path=args.contrast_parquet,
        default_row_kind=str(args.row_kind or "contrast"),
        temp_prefix="nogse_master_nogse_contrast_",
    )
    contrast_path = selected.paths[0]

    df = pd.read_parquet(contrast_path)
    analysis_id = _analysis_id_from_path(contrast_path)

    sheet_hint = canonical_sheet_name(analysis_id)
    if "sheet" not in df.columns:
        if "sheet_1" in df.columns:
            df["sheet"] = df["sheet_1"].map(canonical_sheet_name)
        elif "sheet_2" in df.columns:
            df["sheet"] = df["sheet_2"].map(canonical_sheet_name)
        else:
            df["sheet"] = sheet_hint
    else:
        df["sheet"] = df["sheet"].map(canonical_sheet_name)

    if "subj" not in df.columns:
        df["subj"] = [infer_subj_label(sheet, source_name=analysis_id) for sheet in df["sheet"]]
    df["subj"] = df["subj"].astype(str)

    n1_hint = unique_int(df, "N_1")
    n2_hint = unique_int(df, "N_2")
    outdir = Path(args.out_root) / analysis_id
    tables_dir = outdir
    plots_dir = outdir
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Correction
    use_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    f_by_direction = None
    td_ms_hint = infer_td_ms(df, analysis_id=analysis_id, override=args.corr_td_ms)

    if use_corr:
        if args.corr_xlsx is None:
            raise ValueError("--apply_grad_corr requires --corr_xlsx.")
        if td_ms_hint is None:
            raise ValueError("Could not infer td_ms for correction lookup. Pass --corr_td_ms or make sure td_ms_1 exists.")
        corr = read_correction_table(args.corr_xlsx)
        f_by_direction = build_direction_factors(
            corr,
            spec=CorrectionLookupSpec(
                roi_ref=str(args.corr_roi),
                td_ms=float(td_ms_hint),
                tol_ms=float(args.corr_tol_ms),
                sheet=(args.corr_sheet or sheet_hint),
                n1=n1_hint,
                n2=n2_hint,
            ),
            factor_mode="per_side",
        )

    # M0 flags
    if args.fix_M0 is not None:
        M0_vary = False
        M0_value = float(args.fix_M0)
    elif args.free_M0 is not None:
        M0_vary = True
        M0_value = float(args.free_M0)
    else:
        M0_vary = True
        M0_value = 1.0
    if plan is not None and "M0" in plan.configs:
        M0_vary = plan.mode("M0") != "fixed"
        M0_value = float(plan.fixed("M0", M0_value) if not M0_vary else plan.init("M0", M0_value))

    if args.fix_D0 is not None:
        D0_vary = False
        D0_value = float(args.fix_D0)
    elif args.free_D0 is not None:
        D0_vary = True
        D0_value = float(args.free_D0)
    else:
        D0_vary = True
        D0_value = 2.3e-12
    if plan is not None and "D0_m2_ms" in plan.configs:
        D0_vary = plan.mode("D0_m2_ms") != "fixed"
        D0_value = float(plan.fixed("D0_m2_ms", D0_value) if not D0_vary else plan.init("D0_m2_ms", D0_value))

    if args.fix_tc is not None:
        tc_vary = False
        tc_value = float(args.fix_tc)
    elif args.free_tc is not None:
        tc_vary = True
        tc_value = float(args.free_tc)
    else:
        tc_vary = True
        tc_value = 5.0
    if plan is not None and "tc_ms" in plan.configs:
        tc_vary = plan.mode("tc_ms") != "fixed"
        tc_value = float(plan.fixed("tc_ms", tc_value) if not tc_vary else plan.init("tc_ms", tc_value))

    if args.fix_g0 is not None:
        g0_vary = False
        g0_value = float(args.fix_g0)
    elif args.free_g0 is not None:
        g0_vary = True
        g0_value = float(args.free_g0)
    else:
        g0_vary = backend_model == "nogse_free_grad_offset"
        g0_value = 0.0
    if plan is not None and "g0_mTm" in plan.configs:
        g0_vary = plan.mode("g0_mTm") != "fixed"
        g0_value = float(plan.fixed("g0_mTm", g0_value) if not g0_vary else plan.init("g0_mTm", g0_value))

    m0_bounds = _validate_bounds("M0", tuple(args.M0_bounds)) if args.M0_bounds is not None else None
    d0_bounds = _validate_bounds("D0", tuple(args.D0_bounds)) if args.D0_bounds is not None else None
    tc_bounds = _validate_bounds("tc", tuple(args.tc_bounds)) if args.tc_bounds is not None else None
    g0_bounds = (
        _validate_bounds("g0", tuple(args.g0_bounds))
        if args.g0_bounds is not None
        else (-20.0, 20.0)
        if backend_model == "nogse_free_grad_offset"
        else None
    )
    _validate_fixed_value("M0", None if M0_vary else M0_value, m0_bounds)
    _validate_fixed_value("D0", None if D0_vary else D0_value, d0_bounds)
    if backend_model == "rest":
        _validate_fixed_value("tc", None if tc_vary else tc_value, tc_bounds)
    if backend_model == "nogse_free_grad_offset":
        _validate_fixed_value("g0", None if g0_vary else g0_value, g0_bounds)
    _validate_log_bounds("D0", d0_bounds)

    # Normalize filters
    directions = args.directions
    if directions is not None and len(directions) == 1 and str(directions[0]).upper() == "ALL":
        directions = None
    subjs = args.subjs
    if subjs is not None and len(subjs) == 1 and str(subjs[0]).upper() == "ALL":
        subjs = None
    rois = args.rois
    if rois is not None and len(rois) == 1 and str(rois[0]).upper() == "ALL":
        rois = None

    if subjs is not None:
        df = df[df["subj"].astype(str).isin([str(x) for x in subjs])].copy()
        if df.empty:
            print(f"Skipped: {analysis_id} (no match for subjs={subjs})")
            selected.cleanup()
            return

    stat_keep = args.stat
    if stat_keep is not None and str(stat_keep).upper() == "ALL":
        stat_keep = None

    fit_df = fit_nogse_contrast_long(
        df,
        model=backend_model,
        gbase=args.gbase,
        plot_xcol=args.plot_xcol,
        ycol=args.ycol,
        directions=directions,
        rois=rois,
        stat_keep=stat_keep,
        n_fit=args.n_fit,
        f_by_direction=f_by_direction,
        td_override_ms=args.corr_td_ms,
        M0_vary=M0_vary,
        D0_vary=D0_vary,
        M0_value=M0_value,
        D0_value=D0_value,
        g0_value=g0_value,
        g0_vary=g0_vary,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
        g0_bounds=g0_bounds,
        source_file=contrast_path.name,
        analysis_id=analysis_id,
        tc_value=tc_value,
        tc_vary=tc_vary,
        tc_bounds=tc_bounds,
        peak_grid_n=int(args.peak_grid_n),
        peak_D0_fix=float(args.peak_D0_fix),
        peak_gamma=float(args.peak_gamma),
        oneg=bool(args.oneg),
    )

    fit_params_name = fit_params_output_basename(
        model=str(backend_model),
        axis=str(args.gbase),
        ycol=str(args.ycol),
        directions=None if directions is None else [str(v) for v in directions],
    )
    out_parquet = tables_dir / f"{fit_params_name}.parquet"
    write_table_outputs(
        fit_df,
        out_parquet,
        xlsx_path=out_parquet.with_suffix(".xlsx"),
        csv_path=tables_dir / f"{fit_params_name}.csv",
    )
    append_fit_params_outputs(
        fit_df,
        args,
        fit_kind="nogse_contrast",
        model=str(backend_model),
        source=selected.source,
    )

    print("Saved fit table:", out_parquet)

    if args.no_plots:
        selected.cleanup()
        return

    # Plots: one per roi/direction/stat row
    for _, r in fit_df.iterrows():
        if not bool(r.get("ok", True)):
            continue
        roi = r["roi"]
        direction = r["direction"]
        stat = r.get("stat", None)

        g = df[(df["roi"].astype(str) == str(roi)) & (df["direction"].astype(str) == str(direction))].copy()
        if stat is not None and "stat" in g.columns:
            g = g[g["stat"].astype(str) == str(stat)]

        if g.empty:
            continue

        out_png = plots_dir / f"{roi}.{args.model}.{args.gbase}.{args.ycol}.direction_{direction}.png"
        plot_nogse_contrast_fit_one_group(g, r.to_dict(), out_png=out_png, gbase=args.gbase, ycol=args.ycol)

    print("Saved plots in:", plots_dir)
    selected.cleanup()


if __name__ == "__main__":
    main()
