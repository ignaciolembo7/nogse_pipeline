from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ogse_fitting.fit_ogse_contrast import fit_ogse_contrast_long, plot_fit_one_group
from data_processing.io import write_table_outputs
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.strict_columns import raise_on_unrecognized_column_names


def _analysis_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _unique_float(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    u = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(u) == 1:
        return float(u[0])
    return None


def _unique_int(df: pd.DataFrame, *cols: str) -> int | None:
    for col in cols:
        v = _unique_float(df, col)
        if v is not None:
            return int(round(float(v)))
    return None


def _infer_td_ms(df: pd.DataFrame, analysis_id: str, override: float | None) -> float | None:
    if override is not None:
        return float(override)

    # Prefer td_ms_1 when present in contrast tables.
    td1 = _unique_float(df, "td_ms_1")
    if td1 is not None:
        td2 = _unique_float(df, "td_ms_2")
        if td2 is None:
            return float(td1)
        if abs(float(td1) - float(td2)) < 1e-3:
            return float(0.5 * (td1 + td2))
        return None  # Let the core fitter raise a clearer error if needed.

    td = _unique_float(df, "td_ms")
    if td is not None:
        return float(td)

    # Fallback: parse tdXX from the analysis name.
    m = re.search(r"_td(\d+(?:p\d+)?)", analysis_id)
    if m:
        return float(m.group(1).replace("p", "."))

    return None


def _read_corr_table(path: Path) -> pd.DataFrame:
    """
    Strict reader for the correction table.
    Required columns: direction, roi, td_ms, correction_factor, N_1, N_2.
    """
    # Prefer the explicit "grad_correction" sheet when present.
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = "grad_correction" if "grad_correction" in xls.sheet_names else xls.sheet_names[0]
    corr = xls.parse(sheet_name=sheet).copy()

    raise_on_unrecognized_column_names(corr.columns, context=f"_read_corr_table({path})")

    rename = {}
    if "correction factor" in corr.columns and "correction_factor" not in corr.columns:
        rename["correction factor"] = "correction_factor"
    if "td_ms" in corr.columns and "td_ms" not in corr.columns:
        rename["td_ms"] = "td_ms"
    if rename:
        corr = corr.rename(columns=rename)

    required = {"roi", "direction", "td_ms", "correction_factor", "N_1", "N_2"}
    missing = required - set(corr.columns)
    if missing:
        raise ValueError(f"Invalid correction table. Missing required columns: {sorted(missing)}")

    corr["direction"] = corr["direction"].astype(str)
    corr["roi"] = corr["roi"].astype(str).str.strip()
    if "sheet" in corr.columns:
        corr["sheet"] = corr["sheet"].map(canonical_sheet_name)

    corr["td_ms"] = pd.to_numeric(corr["td_ms"], errors="coerce")
    corr["correction_factor"] = pd.to_numeric(corr["correction_factor"], errors="coerce")
    corr["N_1"] = pd.to_numeric(corr["N_1"], errors="coerce")
    corr["N_2"] = pd.to_numeric(corr["N_2"], errors="coerce")
    corr = corr[
        np.isfinite(corr["td_ms"])
        & np.isfinite(corr["correction_factor"])
        & np.isfinite(corr["N_1"])
        & np.isfinite(corr["N_2"])
    ].copy()
    return corr


def _build_f_by_direction(
    corr: pd.DataFrame,
    *,
    roi_ref: str,
    td_ms: float,
    tol_ms: float,
    sheet: str | None = None,
    n1: int | None = None,
    n2: int | None = None,
) -> dict[str, float]:
    c = corr[corr["roi"].astype(str) == str(roi_ref).strip()].copy()

    # If the correction table has a sheet column, filter it to avoid mixing datasets.
    if sheet is not None and "sheet" in c.columns:
        sheet_key = canonical_sheet_name(sheet)
        c = c[c["sheet"] == sheet_key].copy()

    if n1 is not None:
        c = c[pd.to_numeric(c["N_1"], errors="coerce") == int(n1)].copy()
    if n2 is not None:
        c = c[pd.to_numeric(c["N_2"], errors="coerce") == int(n2)].copy()

    c = c[np.isclose(c["td_ms"].astype(float), float(td_ms), rtol=0.0, atol=float(tol_ms))].copy()
    if c.empty:
        extra_parts: list[str] = []
        if sheet is not None and "sheet" in corr.columns:
            extra_parts.append(f"sheet={canonical_sheet_name(sheet)}")
        if n1 is not None:
            extra_parts.append(f"N_1={int(n1)}")
        if n2 is not None:
            extra_parts.append(f"N_2={int(n2)}")
        extra = f" and {'; '.join(extra_parts)}" if extra_parts else ""
        raise ValueError(f"No correction factors were found for roi={roi_ref}{extra} and td_ms={td_ms:.3f} (tol={tol_ms}).")

    # If duplicates exist, average them.
    c = c.groupby("direction", as_index=False)["correction_factor"].mean()

    out = {str(d): float(f) for d, f in zip(c["direction"], c["correction_factor"])}
    if not out:
        raise ValueError("The filtered correction table did not produce any valid factors.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet", type=Path, help="Input long-form contrast parquet produced by make_contrast.py")

    ap.add_argument("--model", choices=["free", "tort", "rest"])
    ap.add_argument("--gbase", default="g_lin_max", help="Gradient base: g, g_lin_max, or g_thorsten")
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

    ap.add_argument("--n_fit", type=int, default=None, help="Use only the first n_fit points after sorting by x.")
    ap.add_argument("--peak_grid_n", type=int, default=1000, help="Number of points used to search for the fitted peak.")
    ap.add_argument("--peak_D0_fix", type=float, default=3.2e-12, help="Fixed D0 used to convert the peak into tc_peak_ms.")
    ap.add_argument("--peak_gamma", type=float, default=267.5221900, help="Gamma in rad/(ms*mT) used to convert the peak into tc_peak_ms.")
    args = ap.parse_args()

    df = pd.read_parquet(args.contrast_parquet)
    analysis_id = _analysis_id_from_path(args.contrast_parquet)

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

    n1_hint = _unique_int(df, "N_1")
    n2_hint = _unique_int(df, "N_2")
    outdir = Path(args.out_root) / analysis_id
    tables_dir = outdir
    plots_dir = outdir
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Correction
    use_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    f_by_direction = None
    td_ms_hint = _infer_td_ms(df, analysis_id, args.corr_td_ms)

    if use_corr:
        if args.corr_xlsx is None:
            raise ValueError("--apply_grad_corr requires --corr_xlsx.")
        if td_ms_hint is None:
            raise ValueError("Could not infer td_ms for correction lookup. Pass --corr_td_ms or make sure td_ms_1 exists.")
        corr = _read_corr_table(args.corr_xlsx)
        f_by_direction = _build_f_by_direction(
            corr,
            roi_ref=args.corr_roi,
            td_ms=float(td_ms_hint),
            tol_ms=float(args.corr_tol_ms),
            sheet=(args.corr_sheet or sheet_hint),
            n1=n1_hint,
            n2=n2_hint,
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

    fit_df = fit_ogse_contrast_long(
        df,
        model=args.model,
        gbase=args.gbase,
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
        source_file=args.contrast_parquet.name,
        analysis_id=analysis_id,
        peak_grid_n=int(args.peak_grid_n),
        peak_D0_fix=float(args.peak_D0_fix),
        peak_gamma=float(args.peak_gamma),
        oneg=bool(args.oneg),
    )

    out_parquet = tables_dir / "fit_params.parquet"
    write_table_outputs(
        fit_df,
        out_parquet,
        xlsx_path=out_parquet.with_suffix(".xlsx"),
        csv_path=tables_dir / "fit_params.csv",
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
