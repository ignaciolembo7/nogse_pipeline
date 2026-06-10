from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path
import re
import pandas as pd

from data_processing.io import fit_params_output_basename, write_table_outputs
from fitting.contrast import make_contrast
from fitting.experiments import experiment_models, validate_experiment_model
from ogse_fitting.contrast import build_fitted_resampled_ogse_contrast
from ogse_plotting.plot_ogse_contrast_vs_g import plot_ogse_contrast_summary
from plottings.core import render_xy_plot, sanitize_token
from ogse_fitting.fit_ogse_signal_vs_g import VALID_G_TYPES
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.fit_params_schema import standardize_fit_params
from tools.strict_columns import find_unrecognized_column_names
from tools.value_formatting import compact_unique_values, truthy_series

KEY_COLS = ("stat", "roi", "direction", "b_step")


def _normalize_direction_token(value: object) -> str:
    token = str(value).strip()
    if token == "":
        return ""
    try:
        num = float(token)
        if pd.notna(num) and abs(num - round(num)) < 1e-6:
            return str(int(round(num)))
    except Exception:
        pass
    return token


def _normalize_direction_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            norm = _normalize_direction_token(token)
            if norm:
                out.append(norm)
    return list(dict.fromkeys(out))


def _one(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        return default
    u = pd.Series(df[col]).dropna().unique()
    return u[0] if len(u) else default


def _fmt_num(x) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not pd.notna(x):
        return "NA"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _fmt_seq(x) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return _sanitize(str(x))
    if not pd.notna(x):
        return "NA"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return _sanitize(str(x))


def _has_oneg_marker(df: pd.DataFrame) -> bool:
    return "one_g_per_sequence" in df.columns and truthy_series(df["one_g_per_sequence"])


def _sequence_number(df: pd.DataFrame):
    seq = _one(df, "sequence", None)
    if seq is not None and str(seq).strip():
        return seq

    source = _one(df, "source_file", None)
    if source is None:
        return None

    m = re.search(r"_(\d+)_results(?:\.[A-Za-z0-9._-]+)?$", str(source))
    if m:
        return int(m.group(1))
    return None


def _sequence_label(df: pd.DataFrame, *, compact: bool = False) -> str:
    if compact and "sequence" in df.columns:
        values = pd.Series(df["sequence"]).dropna().unique().tolist()
        if values:
            return compact_unique_values(values)
    return _fmt_seq(_sequence_number(df))


def _build_analysis_core(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    directions: list[str],
    sheet_override: str | None,
) -> tuple[str, str]:
    sheet = str(_one(df_ref, "sheet", _one(df_cmp, "sheet", "EXP")))
    if sheet_override:
        sheet = str(sheet_override)

    N1 = _one(df_ref, "N", None)
    N2 = _one(df_cmp, "N", None)
    try:
        N1i = int(round(float(N1))) if N1 is not None else -1
    except Exception:
        N1i = -1
    try:
        N2i = int(round(float(N2))) if N2 is not None else -1
    except Exception:
        N2i = -1

    td1 = _one(df_ref, "td_ms", None)
    hz1 = _one(df_ref, "Hz", None)
    hz2 = _one(df_cmp, "Hz", None)

    dir_tag = "-".join([str(d) for d in directions]) if directions else "ALL"
    td_tag = f"td{_fmt_num(td1)}" if (td1 is not None and pd.notna(td1)) else "tdNA"

    hz_tag = ""
    if hz1 is not None and pd.notna(hz1):
        hz_tag = f"_Hz{_fmt_num(hz1)}"
        if hz2 is not None and pd.notna(hz2) and abs(float(hz2) - float(hz1)) > 1e-6:
            hz_tag = f"_Hz{_fmt_num(hz1)}-{_fmt_num(hz2)}"

    analysis_core = f"{sheet}_N{N1i}-N{N2i}_{td_tag}{hz_tag}_dir{dir_tag}"
    analysis_short = f"{sheet}"
    return _sanitize(analysis_core)[:160], _sanitize(analysis_short)[:160]


def _validate_input(df: pd.DataFrame, label: str) -> None:
    unrecognized = find_unrecognized_column_names(df.columns)
    if unrecognized:
        raise ValueError(
            f"{label}: unrecognized column names: {unrecognized}. "
            "Use canonical names such as 'direction', 'value_norm', and 'g_thorsten'."
        )
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required key columns {missing}. Expected {KEY_COLS}.")


def _normalize_key_dtypes(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    for c in ["stat", "roi", "direction"]:
        out[c] = out[c].astype(str)
    out["direction"] = out["direction"].map(_normalize_direction_token)

    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["stat", "roi", "direction", "b_step"]].head(10)
        raise ValueError(f"{label}: b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)
    return out


def _merge_side_columns(out: pd.DataFrame, side_df: pd.DataFrame, *, side: int) -> pd.DataFrame:
    """
    Carry all columns from side_df except KEY_COLS, using the _1 or _2 suffix.
    Skip columns that already exist as {col}_{side}.
    """
    extra_cols = [c for c in side_df.columns if c not in KEY_COLS]
    sub = side_df[list(KEY_COLS) + extra_cols].drop_duplicates(list(KEY_COLS), keep="first")

    rename = {}
    keep_extras = []
    for c in extra_cols:
        newc = f"{c}_{side}"
        if newc in out.columns:
            continue
        rename[c] = newc
        keep_extras.append(c)

    if not keep_extras:
        return out

    sub = sub[list(KEY_COLS) + keep_extras].rename(columns=rename)
    return out.merge(sub, on=list(KEY_COLS), how="left")


def _drop_aux_prefixed_cols(out: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in out.columns if c.startswith("param_") or c.startswith("meta_")]
    return out.drop(columns=drop_cols) if drop_cols else out


def _order_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    Final column order:
      roi, direction, b_step, stat,
      value, value_norm,
      [seq1: value_1, value_norm_1, S0_1, bvalues..., gradients..., params..., remaining],
      [seq2: ...],
      remaining unsuffixed columns
    """
    cols = list(out.columns)

    def present(xs):  # keep only existing columns while preserving order
        return [x for x in xs if x in cols]

    id_cols = present(["analysis_id", "subj", "sheet", "roi", "direction", "b_step", "stat"])
    head = id_cols + present(["value", "value_norm"])

    def side_block(suf: str) -> list[str]:
        block: list[str] = []
        # Core
        block += present([f"value{suf}", f"value_norm{suf}", f"S0{suf}"])

        # Put bvalue columns first
        b_pref = [
            f"bvalue{suf}",
            f"bvalue_g{suf}",
            f"bvalue_g_lin_max{suf}",
            f"bvalue_thorsten{suf}",
            f"bvalue_orig{suf}",
        ]
        block += present(b_pref)

        # Any remaining side-specific bvalue_* columns
        other_b = sorted([c for c in cols if c.endswith(suf) and c.startswith("bvalue") and c not in block])
        block += other_b

        # Gradients
        g_pref = [
            f"g{suf}",
            f"g_max{suf}",
            f"g_lin_max{suf}",
            f"g_thorsten{suf}",
        ]
        block += present(g_pref)

        other_g = sorted([c for c in cols if c.endswith(suf) and (c.startswith("g_") or c == f"g{suf}") and c not in block])
        block += other_g

        # Typical canonical parameters
        p_pref = [
            f"max_dur_ms{suf}", f"tm_ms{suf}", f"td_ms{suf}",
            f"Hz{suf}", f"N{suf}", f"TE{suf}", f"TR{suf}", f"bmax{suf}",
            f"protocol{suf}", f"sequence{suf}", f"sheet{suf}",
            f"Delta_app_ms{suf}", f"delta_ms{suf}",
            f"source_file{suf}",
        ]
        block += present(p_pref)

        # Remaining side-specific columns not yet included
        rest = sorted([c for c in cols if c.endswith(suf) and c not in block and c not in head])
        block += rest
        return block

    block1 = side_block("_1")
    block2 = side_block("_2")

    used = set(head + block1 + block2)
    tail = sorted([c for c in cols if c not in used])

    return out[head + block1 + block2 + tail]


def build_analysis_id(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    directions: list[str],
    sheet_override: str | None,
    oneg: bool = False,
) -> tuple[str, str]:
    analysis_core, analysis_short = _build_analysis_core(df_ref, df_cmp, directions, sheet_override)
    compact_sequences = bool(oneg or _has_oneg_marker(df_ref) or _has_oneg_marker(df_cmp))
    seq1 = _sequence_label(df_ref, compact=compact_sequences)
    seq2 = _sequence_label(df_cmp, compact=compact_sequences)
    seq_tag = f"_seq{seq1}-{seq2}"
    analysis = f"{analysis_core}{seq_tag}"
    return _sanitize(analysis)[:160], analysis_short


def build_analysis_id_without_sequence(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    directions: list[str],
    sheet_override: str | None,
) -> tuple[str, str]:
    return _build_analysis_core(df_ref, df_cmp, directions, sheet_override)


def _plot_fitted_resampled_signal_fits(
    *,
    points: pd.DataFrame,
    curves: pd.DataFrame,
    out_dir: Path,
    xcol: str,
    ycol: str,
) -> list[Path]:
    if points.empty or curves.empty:
        return []
    if xcol not in points.columns or xcol not in curves.columns:
        return []

    out_paths: list[Path] = []
    group_cols = [c for c in ["stat", "roi", "direction", "side"] if c in points.columns and c in curves.columns]
    if not group_cols:
        return out_paths

    for key, point_group in points.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))
        curve_mask = pd.Series(True, index=curves.index)
        for col, value in key_dict.items():
            curve_mask &= curves[col].astype(str) == str(value)
        curve_group = curves.loc[curve_mask].copy()
        if curve_group.empty:
            continue

        d_points = point_group.dropna(subset=[xcol, "signal_observed"]).sort_values(xcol, kind="stable")
        d_curve = curve_group.dropna(subset=[xcol, "signal_fit"]).sort_values(xcol, kind="stable")
        if d_points.empty or d_curve.empty:
            continue

        used = d_points
        if "fit_used" in d_points.columns:
            used = d_points[d_points["fit_used"].astype(bool)]

        roi = key_dict.get("roi", "roi")
        direction = key_dict.get("direction", "direction")
        side = key_dict.get("side", "side")
        stat = key_dict.get("stat", "stat")
        out_png = out_dir / (
            f"roi-{sanitize_token(roi)}.dir-{sanitize_token(direction)}."
            f"side-{sanitize_token(side)}.stat-{sanitize_token(stat)}.signal_fit.png"
        )
        render_xy_plot(
            x=pd.to_numeric(d_points[xcol], errors="coerce").to_numpy(dtype=float),
            y=pd.to_numeric(d_points["signal_observed"], errors="coerce").to_numpy(dtype=float),
            out_png=out_png,
            title=f"ROI={roi} | direction={direction} | side={side} | stat={stat}",
            xlabel=xcol,
            ylabel=ycol,
            data_label="signal",
            connect_data=False,
            fit_x=pd.to_numeric(d_curve[xcol], errors="coerce").to_numpy(dtype=float),
            fit_y=pd.to_numeric(d_curve["signal_fit"], errors="coerce").to_numpy(dtype=float),
            fit_label="fit",
            highlight_x=pd.to_numeric(used[xcol], errors="coerce").to_numpy(dtype=float),
            highlight_y=pd.to_numeric(used["signal_observed"], errors="coerce").to_numpy(dtype=float),
            highlight_label="fit points",
        )
        out_paths.append(out_png)

    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_parquet", help="signal parquet (ref)")
    ap.add_argument("cmp_parquet", help="signal parquet (cmp)")
    ap.add_argument("--direction", nargs="+", default=None, help="Filter by direction values, for example: 1 2 3 or long tra.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms to include, for example: BRAIN-3 LUDG-2 PHANTOM3.")
    ap.add_argument("--out_root", default="analysis/ogse_experiments/contrast", help="directory root")
    ap.add_argument("--exp", default=None, help="Override the sheet name used for naming only.")
    ap.add_argument("--oneg", action="store_true", help="Allow one-g-per-sequence inputs and compact sequence labels.")
    ap.add_argument(
        "--contrast-source",
        choices=["direct", "fitted_resampled"],
        default="direct",
        help="Build contrasts by direct point subtraction, or by fitting each signal curve and subtracting both fits on a common gradient grid.",
    )
    ap.add_argument(
        "--signal-model",
        default="monoexp",
        choices=sorted(experiment_models("ogse_signal_vs_g")),
        help="OGSE signal model used when --contrast-source=fitted_resampled.",
    )
    ap.add_argument(
        "--ycol",
        default="value_norm",
        choices=["value", "value_norm"],
        help="Signal column fitted when --contrast-source=fitted_resampled.",
    )
    ap.add_argument(
        "--g_type",
        default="g",
        choices=sorted(VALID_G_TYPES),
        help="Gradient axis used for signal fitting and the common resampling grid when --contrast-source=fitted_resampled.",
    )
    ap.add_argument("--resample_grid_n", type=int, default=None, help="Number of points in the fitted/resampled common gradient grid. Defaults to the smaller input curve length.")
    ap.add_argument("--resample_grid_min_mTm", type=float, default=None, help="Minimum common gradient in mT/m for fitted/resampled signal contrasts.")
    ap.add_argument("--resample_grid_max_mTm", type=float, default=None, help="Maximum common gradient in mT/m for fitted/resampled signal contrasts.")
    fit_group = ap.add_mutually_exclusive_group()
    fit_group.add_argument("--fit_points", type=int, default=6, help="Fixed number of leading points used for each fitted/resampled signal fit.")
    fit_group.add_argument("--auto_fit_points", action="store_true", help="Automatically choose the number of leading points for each fitted/resampled signal fit.")
    ap.add_argument("--auto_fit_tol", type=float, default=0.05, help="Relative tolerance for --auto_fit_points.")
    ap.add_argument("--auto_fit_err_floor", type=float, default=0.005, help="Absolute rmse_log floor for --auto_fit_points.")
    ap.add_argument("--auto_fit_min_points", type=int, default=3, help="First k value tested by --auto_fit_points.")
    ap.add_argument("--auto_fit_max_points", type=int, default=9, help="Last k value tested by --auto_fit_points.")
    ap.add_argument("--gamma", type=float, default=267.5221900, help="Gyromagnetic ratio used when deriving b-values from gradients.")
    ap.add_argument("--td_ms", type=float, default=None, help="Optional td_ms override for fitted/resampled signal fits.")
    ap.add_argument("--delta_ms", type=float, default=None, help="Optional delta_ms override for fitted/resampled signal fits.")
    ap.add_argument("--Delta_app_ms", type=float, default=None, help="Optional Delta_app_ms override for fitted/resampled signal fits.")
    ap.add_argument("--D0_init", type=float, default=0.0023, help="Initial D0 seed in mm^2/s for fitted/resampled signal fits.")
    ap.add_argument("--peak_D0_fix", type=float, default=3.2e-12, help="Fixed D0 in m^2/ms used to convert the resampled contrast peak into tc_peak_ms.")
    ap.add_argument("--fix_M0", type=float, default=1.0, help="Fixed M0 value for fitted/resampled signal fits unless --free_M0 is used.")
    ap.add_argument("--free_M0", action="store_true", help="Fit M0 for fitted/resampled signal fits instead of fixing it.")
    args = ap.parse_args()
    validate_experiment_model("ogse_signal_vs_g", args.signal_model)

    if args.resample_grid_n is not None and args.resample_grid_n <= 0:
        raise ValueError("--resample_grid_n must be > 0.")
    if (args.resample_grid_min_mTm is None) != (args.resample_grid_max_mTm is None):
        raise ValueError("Pass both --resample_grid_min_mTm and --resample_grid_max_mTm, or neither.")
    if args.resample_grid_min_mTm is not None and args.resample_grid_max_mTm is not None:
        if float(args.resample_grid_max_mTm) <= float(args.resample_grid_min_mTm):
            raise ValueError("--resample_grid_max_mTm must be greater than --resample_grid_min_mTm.")
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

    directions = _normalize_direction_list(args.direction)
    subjs = args.subjs
    if subjs is not None and len(subjs) == 1 and str(subjs[0]).upper() == "ALL":
        subjs = None

    df_ref = pd.read_parquet(Path(args.ref_parquet))
    df_cmp = pd.read_parquet(Path(args.cmp_parquet))

    _validate_input(df_ref, "ref")
    _validate_input(df_cmp, "cmp")

    df_ref = _normalize_key_dtypes(df_ref, "ref")
    df_cmp = _normalize_key_dtypes(df_cmp, "cmp")

    if directions:
        ref_dirs_before = sorted(df_ref["direction"].astype(str).dropna().unique().tolist())
        cmp_dirs_before = sorted(df_cmp["direction"].astype(str).dropna().unique().tolist())
        df_ref = df_ref[df_ref["direction"].isin(directions)]
        df_cmp = df_cmp[df_cmp["direction"].isin(directions)]
        if df_ref.empty or df_cmp.empty:
            raise ValueError(
                "Direction filter left empty inputs. "
                f"Requested directions={directions}, ref_available={ref_dirs_before}, cmp_available={cmp_dirs_before}."
            )

    oneg_mode = bool(args.oneg or _has_oneg_marker(df_ref) or _has_oneg_marker(df_cmp))

    analysis_id, analysis_short = build_analysis_id(df_ref, df_cmp, directions, args.exp, oneg=oneg_mode)
    old_analysis_id, _ = build_analysis_id_without_sequence(df_ref, df_cmp, directions, args.exp)
    if args.contrast_source == "fitted_resampled":
        source_tag = _sanitize(f"fitresamp-{args.signal_model}-{args.g_type}")
        analysis_id = _sanitize(f"{analysis_id}_{source_tag}")[:160]
    sheet = canonical_sheet_name(args.exp or _one(df_ref, "sheet", _one(df_cmp, "sheet", None)))
    subj = _one(df_ref, "subj", _one(df_cmp, "subj", infer_subj_label(sheet, source_name=analysis_id)))

    if subjs is not None and str(subj) not in {str(x) for x in subjs}:
        print(f"Skipped: {analysis_id} (subj={subj})")
        return

    signal_fit_params = pd.DataFrame()
    signal_fit_points = pd.DataFrame()
    signal_fit_curves = pd.DataFrame()
    contrast_peak_params = pd.DataFrame()
    if args.contrast_source == "direct":
        # Core contrast table: value/value_norm plus side-specific value_1/value_2 columns.
        res = make_contrast(
            df_ref,
            df_cmp,
            axes=tuple(directions) if directions else None,
            y_col="value",
            y_norm_col="value_norm",
            key_cols=KEY_COLS,
        )
        out = res.df.copy()

        _validate_input(out, "contrast_out")
        out = _normalize_key_dtypes(out, "contrast_out")

        # Carry all extra columns from ref and cmp.
        out = _merge_side_columns(out, df_ref, side=1)
        out = _merge_side_columns(out, df_cmp, side=2)
    else:
        res_fit = build_fitted_resampled_ogse_contrast(
            df_ref,
            df_cmp,
            axes=tuple(directions) if directions else None,
            ycol=args.ycol,
            signal_model=args.signal_model,
            g_type=args.g_type,
            grid_n=args.resample_grid_n,
            grid_min=args.resample_grid_min_mTm,
            grid_max=args.resample_grid_max_mTm,
            fit_points=None if args.auto_fit_points else args.fit_points,
            auto_fit_points=bool(args.auto_fit_points),
            auto_fit_min_points=int(args.auto_fit_min_points),
            auto_fit_max_points=args.auto_fit_max_points,
            auto_fit_rel_tol=float(args.auto_fit_tol),
            auto_fit_err_floor=float(args.auto_fit_err_floor),
            free_M0=bool(args.free_M0),
            fix_M0=float(args.fix_M0),
            D0_init=float(args.D0_init),
            gamma=float(args.gamma),
            peak_D0_fix=float(args.peak_D0_fix),
            delta_ms=args.delta_ms,
            Delta_app_ms=args.Delta_app_ms,
            td_ms=args.td_ms,
            key_cols=KEY_COLS,
        )
        out = res_fit.df.copy()
        signal_fit_params = res_fit.signal_fit_params.copy()
        signal_fit_points = res_fit.signal_fit_points.copy()
        signal_fit_curves = res_fit.signal_fit_curves.copy()
        contrast_peak_params = res_fit.contrast_peak_params.copy()
        if out.empty:
            raise ValueError("fitted_resampled contrast build produced no rows.")

    _validate_input(out, "contrast_out")
    out = _normalize_key_dtypes(out, "contrast_out")

    # Strict cleanup
    out = _drop_aux_prefixed_cols(out)
    _validate_input(out, "contrast_clean")

    out["analysis_id"] = str(analysis_id)
    out["sheet"] = sheet
    out["subj"] = str(subj)

    # Final column order
    out = _order_columns(out)

    tables_dir = Path(args.out_root) / "tables" / analysis_short
    plots_dir = Path(args.out_root) / "plots" / analysis_short
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = tables_dir / f"{analysis_id}.long.parquet"
    write_table_outputs(out, out_parquet, xlsx_path=out_parquet.with_suffix(".xlsx"))

    if args.contrast_source == "fitted_resampled" and not signal_fit_params.empty:
        signal_fit_params["analysis_id"] = str(analysis_id)
        signal_fit_params["sheet"] = sheet
        signal_fit_params["subj"] = str(subj)
        fit_params_parquet = tables_dir / f"{analysis_id}.signal_fit_params.parquet"
        write_table_outputs(signal_fit_params, fit_params_parquet, xlsx_path=fit_params_parquet.with_suffix(".xlsx"))

    if args.contrast_source == "fitted_resampled" and not signal_fit_points.empty:
        signal_fit_points["analysis_id"] = str(analysis_id)
        signal_fit_points["sheet"] = sheet
        signal_fit_points["subj"] = str(subj)
        points_parquet = tables_dir / f"{analysis_id}.signal_fit_points.parquet"
        write_table_outputs(signal_fit_points, points_parquet, xlsx_path=points_parquet.with_suffix(".xlsx"))

    if args.contrast_source == "fitted_resampled" and not signal_fit_curves.empty:
        signal_fit_curves["analysis_id"] = str(analysis_id)
        signal_fit_curves["sheet"] = sheet
        signal_fit_curves["subj"] = str(subj)
        curves_parquet = tables_dir / f"{analysis_id}.signal_fit_curves.parquet"
        write_table_outputs(signal_fit_curves, curves_parquet, xlsx_path=curves_parquet.with_suffix(".xlsx"))

    if args.contrast_source == "fitted_resampled":
        signal_plot_paths = _plot_fitted_resampled_signal_fits(
            points=signal_fit_points,
            curves=signal_fit_curves,
            out_dir=plots_dir / analysis_id / "signal_fits",
            xcol=str(args.g_type),
            ycol=str(args.ycol),
        )
        if signal_plot_paths:
            print("Saved signal-fit plots:", len(signal_plot_paths))

        contrast_xcol = str(args.g_type)
        if f"{contrast_xcol}_1" in out.columns:
            contrast_xcol = f"{contrast_xcol}_1"
        plot_out = out.copy()
        if "stat" in plot_out.columns:
            plot_out = plot_out[plot_out["stat"].astype(str) == "avg"].copy()
        if not plot_out.empty:
            contrast_plot_paths = plot_ogse_contrast_summary(
                plot_out,
                out_root=plots_dir / "contrasts",
                exp_id=analysis_id,
                xcol=contrast_xcol,
                ycol=str(args.ycol),
                directions=directions or sorted(plot_out["direction"].dropna().astype(str).unique().tolist()),
                rois_requested=None,
                stat="avg",
            )
            if contrast_plot_paths:
                print("Saved fitted/resampled contrast plots:", len(contrast_plot_paths))

    if args.contrast_source == "fitted_resampled" and not contrast_peak_params.empty:
        contrast_peak_params["analysis_id"] = str(analysis_id)
        contrast_peak_params["sheet"] = sheet
        contrast_peak_params["subj"] = str(subj)
        contrast_peak_params = standardize_fit_params(
            contrast_peak_params,
            fit_kind="ogse_contrast",
            source_file=Path(args.ref_parquet).name,
        )
        fit_params_name = fit_params_output_basename(
            model=str(args.signal_model),
            axis=str(args.g_type),
            ycol=str(args.ycol),
            directions=directions or None,
        )
        peak_params_parquet = tables_dir / f"{fit_params_name}.parquet"
        write_table_outputs(contrast_peak_params, peak_params_parquet, xlsx_path=peak_params_parquet.with_suffix(".xlsx"))

    # Remove older duplicate outputs that used the pre-sequence naming scheme.
    if args.contrast_source == "direct" and old_analysis_id != analysis_id:
        old_parquet = tables_dir / f"{old_analysis_id}.long.parquet"
        old_xlsx = old_parquet.with_suffix(".xlsx")
        for old_path in (old_parquet, old_xlsx):
            if old_path.exists():
                old_path.unlink()
                print("Removed duplicate output:", old_path)

    print("Saved:", out_parquet)


if __name__ == "__main__":
    main()
