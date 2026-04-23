from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from fitting.experiments import experiment_models, split_all_or_values, validate_experiment_model
from fitting.gradient_correction import (
    SignalCorrectionLookupSpec,
    build_signal_direction_factors,
    infer_td_ms,
    read_correction_table,
)
from ogse_fitting.fit_ogse_signal_vs_g import run_fit_ogse_signal_vs_g_from_parquet
from tools.strict_columns import raise_on_unrecognized_column_names


EXPERIMENT = "ogse_signal_vs_g"
VALID_YCOLS = {"value", "value_norm"}
VALID_G_TYPES = [
    "bvalue",
    "g",
    "bvalue_g",
    "g_max",
    "g_lin_max",
    "bvalue_g_lin_max",
    "g_thorsten",
    "bvalue_thorsten",
]
DEFAULT_FIT_POINTS = 6


def _unique_float_any(df: pd.DataFrame, cols: list[str], *, required: bool, name: str) -> float | None:
    for c in cols:
        if c in df.columns:
            u = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(u) == 1:
                return float(u[0])
    if required:
        raise ValueError(f"Could not infer {name}. Checked columns: {cols}")
    return None


def _load_parquet_context(parquet_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    raise_on_unrecognized_column_names(df.columns, context=f"_load_parquet_context({parquet_path})")
    required = ["direction", "roi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"_load_parquet_context({parquet_path}): missing required columns {missing}. Columns={list(df.columns)}")
    return df


def _infer_overrides_from_df(df: pd.DataFrame) -> dict[str, float | None]:
    N = _unique_float_any(df, ["N"], required=False, name="N")
    delta_ms = _unique_float_any(df, ["delta_ms"], required=False, name="delta_ms")
    Delta_app_ms = _unique_float_any(df, ["Delta_app_ms"], required=False, name="Delta_app_ms")
    td_ms = infer_td_ms(df, analysis_id="", override=None)
    return {"td_ms": td_ms, "N": N, "delta_ms": delta_ms, "Delta_app_ms": Delta_app_ms}


def _resolve_requested_rois(requested: list[str], available: list[str]) -> list[str] | None:
    if requested == ["ALL"]:
        return None
    available_set = {str(x) for x in available}
    missing = [str(x) for x in requested if str(x) not in available_set]
    if missing:
        raise ValueError(f"ROIs not found: {missing}. Available ROIs={available}")
    return [str(x) for x in requested]


def _resolve_requested_directions(requested: list[str] | None, available: list[str]) -> list[str] | None:
    if requested is None:
        return None
    available_set = {str(x) for x in available}
    missing = [str(x) for x in requested if str(x) not in available_set]
    if missing:
        raise ValueError(f"Directions not found: {missing}. Available directions={available}")
    return [str(x) for x in requested]


def _resolve_direction_factors(
    *,
    parquet: Path,
    df: pd.DataFrame,
    apply_grad_corr: bool,
    corr_xlsx: Path | None,
    corr_roi: str,
    corr_td_ms: float | None,
    corr_tol_ms: float,
    corr_sheet: str | None,
) -> dict[str, float] | None:
    if not apply_grad_corr:
        return None
    if corr_xlsx is None:
        raise ValueError("--apply_grad_corr requires --corr_xlsx.")

    analysis_id = parquet.stem.replace(".long", "")
    td_ms = infer_td_ms(df, analysis_id=analysis_id, override=corr_td_ms)
    if td_ms is None:
        raise ValueError("Could not infer td_ms for correction lookup. Pass --corr_td_ms or include td_ms columns in the input.")

    n_hint = pd.to_numeric(df.get("N"), errors="coerce").dropna().unique() if "N" in df.columns else []
    signal_n = int(round(float(n_hint[0]))) if len(n_hint) == 1 else None
    corr = read_correction_table(corr_xlsx)
    return build_signal_direction_factors(
        corr,
        spec=SignalCorrectionLookupSpec(
            roi_ref=str(corr_roi),
            td_ms=float(td_ms),
            signal_n=signal_n,
            tol_ms=float(corr_tol_ms),
            sheet=(corr_sheet or analysis_id),
            signal_source_file=parquet.name,
            preferred_side=None,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("parquet", type=Path, help="Input .long.parquet file (clean signal table)")
    ap.add_argument("--model", default="monoexp", choices=sorted(experiment_models(EXPERIMENT)))
    ap.add_argument("--directions", nargs="+", default=None, help="Direction values from the direction column to fit")
    ap.add_argument("--rois", nargs="+", default=["ALL"], help="ROIs to fit. The command fails if any requested ROI is missing.")
    ap.add_argument("--ycol", default="value_norm", choices=sorted(VALID_YCOLS))

    fit_group = ap.add_mutually_exclusive_group()
    fit_group.add_argument("--fit_points", type=int, default=None, help="Fixed number of points to use in the OGSE free-signal fit.")
    fit_group.add_argument(
        "--auto_fit_points",
        action="store_true",
        help="Automatically select how many initial points best match the OGSE free-signal model.",
    )
    ap.add_argument("--auto_fit_tol", type=float, default=0.05, help="Relative tolerance used by the automatic mode when adding a new point.")
    ap.add_argument("--auto_fit_err_floor", type=float, default=0.005, help="Absolute floor for rmse_log before comparing consecutive k values.")
    ap.add_argument("--auto_fit_min_points", type=int, default=3, help="First k value tested by the automatic mode.")
    ap.add_argument("--auto_fit_max_points", type=int, default=9, help="Last k value tested by the automatic mode.")

    ap.add_argument("--g_type", default="bvalue", choices=VALID_G_TYPES)
    ap.add_argument("--plot_xcol", default=None, choices=VALID_G_TYPES)
    ap.add_argument("--gamma", type=float, default=267.5221900)
    ap.add_argument("--td_ms", type=float, default=None)
    ap.add_argument("--N", type=float, default=None)
    ap.add_argument("--delta_ms", type=float, default=None)
    ap.add_argument("--Delta_app_ms", type=float, default=None)
    ap.add_argument("--D0_init", type=float, default=0.0023)
    ap.add_argument("--fix_M0", type=float, default=1.0)
    ap.add_argument("--free_M0", action="store_true")
    ap.add_argument("--out_root", default="ogse_experiments/fits/ogse_signal_vs_g_monoexp")
    ap.add_argument(
        "--out_dproj_root",
        default=None,
        help="Optional root where a synthetic *.Dproj.long.parquet table is written from the fitted OGSE reference D0 values.",
    )
    ap.add_argument("--stat", default="avg")

    corr_group = ap.add_mutually_exclusive_group()
    corr_group.add_argument("--apply_grad_corr", action="store_true")
    corr_group.add_argument("--no_grad_corr", action="store_true")
    ap.add_argument("--corr_xlsx", type=Path, default=None)
    ap.add_argument("--corr_roi", default="water")
    ap.add_argument("--corr_td_ms", type=float, default=None)
    ap.add_argument("--corr_tol_ms", type=float, default=1e-3)
    ap.add_argument("--corr_sheet", default=None)
    args = ap.parse_args()

    validate_experiment_model(EXPERIMENT, args.model)

    if args.fit_points is not None and args.fit_points <= 0:
        raise ValueError("--fit_points must be > 0.")
    if args.auto_fit_tol < 0:
        raise ValueError("--auto_fit_tol must be >= 0.")
    if args.auto_fit_err_floor < 0:
        raise ValueError("--auto_fit_err_floor must be >= 0.")
    if args.auto_fit_min_points < 1:
        raise ValueError("--auto_fit_min_points must be >= 1.")
    if args.auto_fit_max_points is not None and args.auto_fit_max_points < args.auto_fit_min_points:
        raise ValueError("--auto_fit_max_points must be >= --auto_fit_min_points.")

    df = _load_parquet_context(args.parquet)
    inferred = _infer_overrides_from_df(df)

    td_ms = args.td_ms if args.td_ms is not None else inferred["td_ms"]
    N = args.N if args.N is not None else inferred["N"]
    delta_ms = args.delta_ms if args.delta_ms is not None else inferred["delta_ms"]
    Delta_app_ms = args.Delta_app_ms if args.Delta_app_ms is not None else inferred["Delta_app_ms"]

    directions = _resolve_requested_directions(
        None if args.directions is None else [str(x) for x in args.directions],
        sorted(df["direction"].astype(str).dropna().unique().tolist()),
    )
    rois = _resolve_requested_rois(
        [str(x) for x in args.rois],
        sorted(df["roi"].astype(str).dropna().unique().tolist()),
    )

    fit_points = args.fit_points
    auto_fit_points = bool(args.auto_fit_points)
    if fit_points is None and not auto_fit_points:
        fit_points = DEFAULT_FIT_POINTS

    apply_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    f_by_direction = _resolve_direction_factors(
        parquet=args.parquet,
        df=df,
        apply_grad_corr=apply_corr,
        corr_xlsx=args.corr_xlsx,
        corr_roi=args.corr_roi,
        corr_td_ms=args.corr_td_ms,
        corr_tol_ms=args.corr_tol_ms,
        corr_sheet=args.corr_sheet,
    )

    run_fit_ogse_signal_vs_g_from_parquet(
        args.parquet,
        dirs=directions,
        rois=split_all_or_values(rois),
        ycol=args.ycol,
        g_type=args.g_type,
        fit_points=fit_points,
        auto_fit_points=auto_fit_points,
        auto_fit_min_points=args.auto_fit_min_points,
        auto_fit_max_points=args.auto_fit_max_points,
        auto_fit_rel_tol=args.auto_fit_tol,
        auto_fit_err_floor=args.auto_fit_err_floor,
        free_M0=args.free_M0,
        fix_M0=args.fix_M0,
        D0_init=args.D0_init,
        gamma=args.gamma,
        td_ms=td_ms,
        N=N,
        delta_ms=delta_ms,
        Delta_app_ms=Delta_app_ms,
        stat_keep=args.stat,
        out_root=args.out_root,
        out_dproj_root=args.out_dproj_root,
        f_by_direction=f_by_direction,
        plot_xcol=args.plot_xcol,
    )


if __name__ == "__main__":
    main()
