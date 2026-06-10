from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from data_processing.io import fit_params_output_basename, write_table_outputs
from fitting.b_from_g import VALID_AXIS_BASES
from fitting.experiments import experiment_models, validate_experiment_model
from fitting.gradient_correction import (
    CorrectionLookupSpec,
    build_direction_factors,
    infer_td_ms,
    read_correction_table,
    unique_int,
)
from ogse_fitting.fit_ogse_contrast_vs_g import (
    _load_ogse_contrast_tables,
    _read_table,
    fit_ogse_contrast_global_long,
    fit_ogse_contrast_long,
    fit_ogse_contrast_mixed_global_long,
    plot_fit_one_group,
)
from tools.brain_labels import canonical_sheet_name, infer_subj_label


DEFAULT_TC_BOUNDS = (0.1, 1000.0)
DEFAULT_M0_BOUNDS = (0.0, 5.0)
DEFAULT_D0_BOUNDS = (1e-16, 1e-10)
DEFAULT_C_BOUNDS = (0.0, 1.0)


def _analysis_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _unique_text(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns:
        return None
    values = df[col].dropna().astype(str).unique().tolist()
    if len(values) == 1:
        return values[0]
    return None


def _attach_mixed_global_correction_factors(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    corr_roi: str,
    corr_td_ms: float | None,
    corr_tol_ms: float,
    corr_sheet: str | None,
) -> pd.DataFrame:
    out = df.copy()
    out["grad_correction_factor_1"] = 1.0
    out["grad_correction_factor_2"] = 1.0

    group_cols = ["analysis_id", "source_file", "direction"]
    missing = [col for col in group_cols if col not in out.columns]
    if missing:
        raise ValueError(f"mixed_global correction requires columns {missing}.")

    cache: dict[tuple[str | None, int | None, int | None, float, str], dict[str, float | tuple[float, float]]] = {}
    for _, group in out.groupby(group_cols, sort=False, dropna=False):
        analysis_id = _unique_text(group, "analysis_id") or ""
        direction = str(_unique_text(group, "direction") or "")
        sheet = corr_sheet or _unique_text(group, "sheet") or canonical_sheet_name(analysis_id)
        td_ms = infer_td_ms(group, analysis_id=analysis_id, override=corr_td_ms)
        if td_ms is None:
            raise ValueError(
                f"Could not infer td_ms for mixed_global correction lookup "
                f"(analysis_id={analysis_id}, direction={direction})."
            )
        n1 = unique_int(group, "N_1")
        n2 = unique_int(group, "N_2")
        cache_key = (sheet, n1, n2, round(float(td_ms), 6), str(corr_roi))
        factors = cache.get(cache_key)
        if factors is None:
            factors = build_direction_factors(
                corr,
                spec=CorrectionLookupSpec(
                    roi_ref=str(corr_roi),
                    td_ms=float(td_ms),
                    tol_ms=float(corr_tol_ms),
                    sheet=sheet,
                    n1=n1,
                    n2=n2,
                ),
                factor_mode="per_side",
            )
            cache[cache_key] = factors
        if direction not in factors:
            raise ValueError(
                f"No correction factor for direction={direction!r} "
                f"(analysis_id={analysis_id}, sheet={sheet}, td_ms={float(td_ms):.3f}). "
                f"Available directions: {sorted(factors)}"
            )
        f1, f2 = factors[direction] if isinstance(factors[direction], tuple) else (factors[direction], factors[direction])
        out.loc[group.index, "grad_correction_factor_1"] = float(f1)
        out.loc[group.index, "grad_correction_factor_2"] = float(f2)
    return out


def main() -> None:
    plot_axis_choices = sorted(
        {
            *VALID_AXIS_BASES,
            *[f"{axis}_1" for axis in VALID_AXIS_BASES],
            *[f"{axis}_2" for axis in VALID_AXIS_BASES],
        }
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet", type=Path, nargs="+", help="Input long-form contrast parquet(s) produced by make_contrast.py")

    ap.add_argument("--model", required=True, choices=sorted(experiment_models("ogse_contrast_vs_g")))
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
    grp_tc.add_argument("--fix_tc", type=float, default=None, help="Fix tc in ms.")
    grp_tc.add_argument(
        "--free_tc",
        nargs="?",
        const=5.0,
        type=float,
        default=None,
        help="Keep tc free. Optional value is the initial seed in ms. Default seed: 5.0.",
    )
    ap.add_argument("--tc_init", type=float, default=5.0, help="Initial tc seed in ms when --fix_tc/--free_tc are not used.")
    grp_c = ap.add_mutually_exclusive_group()
    grp_c.add_argument(
        "--fix_C",
        type=float,
        default=None,
        help="Fix rest-offset signal parameter C. Used only by model=rest_offset/rest_offset_globC.",
    )
    grp_c.add_argument(
        "--free_C",
        nargs="?",
        const=0.0,
        type=float,
        default=None,
        help="Keep rest-offset signal parameter C free. Optional value is the initial seed. Default seed: 0.0.",
    )
    ap.add_argument("--tc_bounds", "--tc-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--M0_bounds", "--M0-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--D0_bounds", "--D0-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--C_bounds", "--C-bounds", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--alpha_table", type=Path, default=None, help="Optional fixed-alpha table for model=mixed_global.")
    ap.add_argument("--alpha_col", default=None, help="Alpha column in --alpha_table. Defaults to alpha, then alpha_macro.")
    ap.add_argument("--alpha_td_col", default="td_ms", help="td column in --alpha_table used for per-td matching.")
    ap.add_argument("--alpha_td_tol_ms", type=float, default=1e-3, help="Tolerance in ms for matching alpha by td.")
    ap.add_argument(
        "--global_params",
        "--global-params",
        nargs="*",
        default=None,
        help=(
            "Fit all td curves jointly per subj/ROI/direction and share these model parameters. "
            "Examples: --global_params tc_ms D0_m2_ms, --global_params M0,D0,tc. "
            "Unlisted variable parameters remain local to each curve."
        ),
    )

    ap.add_argument("--n_fit", type=int, default=None, help="Use only the first n_fit points after sorting by x.")
    ap.add_argument("--peak_grid_n", type=int, default=1000, help="Number of points used to search for the fitted peak.")
    ap.add_argument("--peak_D0_fix", type=float, default=3.2e-12, help="Fixed D0 used to convert the peak into tc_peak_ms.")
    ap.add_argument("--peak_gamma", type=float, default=267.5221900, help="Gamma in rad/(ms*mT) used to convert the peak into tc_peak_ms.")
    ap.add_argument(
        "--peak_g_max_mTm",
        type=float,
        default=None,
        help="Optional raw x-axis gradient maximum, in mT/m, used to search the fitted contrast peak.",
    )
    ap.add_argument(
        "--peak_resample_gradient",
        action="store_true",
        help="Also compute tc_peak_resampled_ms after rebuilding the fitted OGSE contrast on a common gradient grid.",
    )
    ap.add_argument(
        "--peak_resample_g_max_corr_mTm",
        type=float,
        default=None,
        help="Optional corrected common-gradient maximum, in mT/m, used for tc_peak_resampled_ms.",
    )
    args = ap.parse_args()
    validate_experiment_model("ogse_contrast_vs_g", args.model)

    contrast_paths = [Path(p) for p in args.contrast_parquet]
    use_global_params = bool(args.global_params)
    if args.model != "mixed_global" and not use_global_params and len(contrast_paths) != 1:
        raise ValueError("Multiple contrast parquet inputs require --model mixed_global or --global_params.")
    if args.model == "mixed_global" and use_global_params:
        raise ValueError("--global_params is for regular models. Use --model mixed_global without --global_params.")

    df = pd.read_parquet(contrast_paths[0])
    analysis_id = _analysis_id_from_path(contrast_paths[0])

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

    if use_corr and args.model != "mixed_global":
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

    if args.fix_D0 is not None:
        D0_vary = False
        D0_value = float(args.fix_D0)
    elif args.free_D0 is not None:
        D0_vary = True
        D0_value = float(args.free_D0)
    else:
        D0_vary = True
        D0_value = 2.3e-12

    if args.fix_tc is not None:
        tc_vary = False
        tc_value = float(args.fix_tc)
    elif args.free_tc is not None:
        tc_vary = True
        tc_value = float(args.free_tc)
    else:
        tc_vary = True
        tc_value = float(args.tc_init)

    if args.fix_C is not None:
        C_vary = False
        C_value = float(args.fix_C)
    elif args.free_C is not None:
        C_vary = True
        C_value = float(args.free_C)
    else:
        C_vary = args.model in {"rest_offset", "rest_offset_globC"}
        C_value = 0.0

    tc_bounds = None if args.tc_bounds is None else tuple(float(v) for v in args.tc_bounds)
    m0_bounds = None if args.M0_bounds is None else tuple(float(v) for v in args.M0_bounds)
    d0_bounds = None if args.D0_bounds is None else tuple(float(v) for v in args.D0_bounds)
    c_bounds = None if args.C_bounds is None else tuple(float(v) for v in args.C_bounds)
    for name, bounds in [("tc_bounds", tc_bounds), ("M0_bounds", m0_bounds), ("D0_bounds", d0_bounds), ("C_bounds", c_bounds)]:
        if bounds is not None and (len(bounds) != 2 or bounds[0] >= bounds[1]):
            raise ValueError(f"{name} must contain two increasing values. Received {bounds}.")
    if tc_bounds is not None and tc_bounds[0] <= 0:
        raise ValueError(f"tc_bounds lower value must be positive. Received {tc_bounds}.")
    if d0_bounds is not None and d0_bounds[0] <= 0:
        raise ValueError(f"D0_bounds lower value must be positive. Received {d0_bounds}.")

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
            return

    stat_keep = args.stat
    if stat_keep is not None and str(stat_keep).upper() == "ALL":
        stat_keep = None

    if args.model == "mixed_global":
        combined = _load_ogse_contrast_tables(contrast_paths)
        if subjs is not None:
            combined = combined[combined["subj"].astype(str).isin([str(x) for x in subjs])].copy()
            if combined.empty:
                print(f"Skipped: mixed_global (no match for subjs={subjs})")
                return
        if use_corr:
            if args.corr_xlsx is None:
                raise ValueError("--apply_grad_corr requires --corr_xlsx.")
            corr = read_correction_table(args.corr_xlsx)
            combined = _attach_mixed_global_correction_factors(
                combined,
                corr,
                corr_roi=str(args.corr_roi),
                corr_td_ms=args.corr_td_ms,
                corr_tol_ms=float(args.corr_tol_ms),
                corr_sheet=args.corr_sheet,
            )
        alpha_df = _read_table(args.alpha_table) if args.alpha_table is not None else None
        outdir = Path(args.out_root) / "mixed_global"
        tables_dir = outdir
        plots_dir = outdir
        tables_dir.mkdir(parents=True, exist_ok=True)
        fit_df = fit_ogse_contrast_mixed_global_long(
            combined,
            gbase=args.gbase,
            plot_xcol=args.plot_xcol,
            ycol=args.ycol,
            directions=directions,
            rois=rois,
            stat_keep=stat_keep,
            n_fit=args.n_fit,
            f_by_direction=None,
            td_override_ms=args.corr_td_ms,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            tc_value=tc_value,
            tc_bounds=tc_bounds or DEFAULT_TC_BOUNDS,
            m0_bounds=m0_bounds or DEFAULT_M0_BOUNDS,
            d0_bounds=d0_bounds or DEFAULT_D0_BOUNDS,
            alpha_df=alpha_df,
            alpha_col=args.alpha_col,
            alpha_td_col=args.alpha_td_col,
            alpha_td_tol_ms=float(args.alpha_td_tol_ms),
            source_file="|".join(p.name for p in contrast_paths),
        )
        fit_params_name = fit_params_output_basename(
            model=str(args.model),
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
        print("Saved fit table:", out_parquet)
        if not args.no_plots:
            print("Plots for model=mixed_global are not generated yet; saved table only.")
        return

    if use_global_params:
        combined = _load_ogse_contrast_tables(contrast_paths)
        if subjs is not None:
            combined = combined[combined["subj"].astype(str).isin([str(x) for x in subjs])].copy()
            if combined.empty:
                print(f"Skipped: global fit (no match for subjs={subjs})")
                return
        if use_corr:
            if args.corr_xlsx is None:
                raise ValueError("--apply_grad_corr requires --corr_xlsx.")
            corr = read_correction_table(args.corr_xlsx)
            combined = _attach_mixed_global_correction_factors(
                combined,
                corr,
                corr_roi=str(args.corr_roi),
                corr_td_ms=args.corr_td_ms,
                corr_tol_ms=float(args.corr_tol_ms),
                corr_sheet=args.corr_sheet,
            )
        global_token = "-".join(str(v).replace(",", "-") for v in args.global_params if str(v).strip()) or "global"
        outdir = Path(args.out_root) / f"{args.model}_global_{global_token}"
        tables_dir = outdir
        tables_dir.mkdir(parents=True, exist_ok=True)
        fit_df = fit_ogse_contrast_global_long(
            combined,
            model=args.model,
            global_params=args.global_params,
            gbase=args.gbase,
            plot_xcol=args.plot_xcol,
            ycol=args.ycol,
            directions=directions,
            rois=rois,
            stat_keep=stat_keep,
            n_fit=args.n_fit,
            f_by_direction=None,
            td_override_ms=args.corr_td_ms,
            M0_vary=M0_vary,
            D0_vary=D0_vary,
            C_vary=C_vary,
            M0_value=M0_value,
            D0_value=D0_value,
            C_value=C_value,
            tc_value=tc_value,
            tc_vary=tc_vary,
            tc_bounds=tc_bounds,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
            c_bounds=c_bounds or DEFAULT_C_BOUNDS,
            source_file="|".join(p.name for p in contrast_paths),
            peak_grid_n=int(args.peak_grid_n),
            peak_D0_fix=float(args.peak_D0_fix),
            peak_gamma=float(args.peak_gamma),
            peak_g_max_mTm=args.peak_g_max_mTm,
            peak_resample_gradient=bool(args.peak_resample_gradient),
            peak_resample_g_max_corr_mTm=args.peak_resample_g_max_corr_mTm,
            oneg=bool(args.oneg),
        )
        fit_params_name = fit_params_output_basename(
            model=f"{args.model}_global",
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
        print("Saved fit table:", out_parquet)
        if not args.no_plots:
            print("Plots for --global_params fits are not generated yet; saved table only.")
        return

    fit_df = fit_ogse_contrast_long(
        df,
        model=args.model,
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
        C_vary=C_vary,
        M0_value=M0_value,
        D0_value=D0_value,
        C_value=C_value,
        tc_value=tc_value,
        tc_vary=tc_vary,
        tc_bounds=tc_bounds,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
        c_bounds=c_bounds or DEFAULT_C_BOUNDS,
        source_file=contrast_paths[0].name,
        analysis_id=analysis_id,
        peak_grid_n=int(args.peak_grid_n),
        peak_D0_fix=float(args.peak_D0_fix),
        peak_gamma=float(args.peak_gamma),
        peak_g_max_mTm=args.peak_g_max_mTm,
        peak_resample_gradient=bool(args.peak_resample_gradient),
        peak_resample_g_max_corr_mTm=args.peak_resample_g_max_corr_mTm,
        oneg=bool(args.oneg),
    )

    fit_params_name = fit_params_output_basename(
        model=str(args.model),
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

    print("Saved fit table:", out_parquet)

    if args.no_plots:
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
        plot_fit_one_group(g, r.to_dict(), out_png=out_png, gbase=args.gbase, ycol=args.ycol)

    print("Saved plots in:", plots_dir)


if __name__ == "__main__":
    main()
