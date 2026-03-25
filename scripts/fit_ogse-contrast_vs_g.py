from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ogse_fitting.fit_ogse_contrast import fit_ogse_contrast_long, plot_fit_one_group


def _analysis_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _canonical_sheet_name(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None

    m = re.match(r"^(\d{8}_[^_]+)", s)
    if m:
        return m.group(1)

    m = re.match(r"^(.+?)_(?:N\d|td)", s)
    if m:
        return m.group(1)

    return s


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

    # prefer td_ms_1 if exists (your contrast tables)
    td1 = _unique_float(df, "td_ms_1")
    if td1 is not None:
        td2 = _unique_float(df, "td_ms_2")
        if td2 is None:
            return float(td1)
        if abs(float(td1) - float(td2)) < 1e-3:
            return float(0.5 * (td1 + td2))
        return None  # let core raise with clear message if needed

    td = _unique_float(df, "td_ms")
    if td is not None:
        return float(td)

    # fallback: parse from name tdXX
    m = re.search(r"_td(\d+(?:p\d+)?)", analysis_id)
    if m:
        return float(m.group(1).replace("p", "."))

    return None


def _read_corr_table(path: Path) -> pd.DataFrame:
    """
    Strict: requires direction, roi, td_ms, correction_factor, N_1, N_2.
    (No axis fallback; keep pipeline consistent.)
    """
    # si el writer guardó "grad_correction", lo usamos; si no, usamos la primera
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = "grad_correction" if "grad_correction" in xls.sheet_names else xls.sheet_names[0]
    corr = xls.parse(sheet_name=sheet).copy()

    if "axis" in corr.columns:
        raise ValueError("Encontré columna 'axis' en tabla de corrección. Debe llamarse 'direction'.")

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
        raise ValueError(f"Tabla de corrección inválida. Faltan columnas: {sorted(missing)}")

    corr["direction"] = corr["direction"].astype(str)
    corr["roi"] = corr["roi"].astype(str).str.strip()
    if "sheet" in corr.columns:
        corr["sheet"] = corr["sheet"].map(_canonical_sheet_name)

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

    # ✅ si hay columna sheet, filtramos para evitar mezclar datasets
    if sheet is not None and "sheet" in c.columns:
        sheet_key = _canonical_sheet_name(sheet)
        c = c[c["sheet"] == sheet_key].copy()

    if n1 is not None:
        c = c[pd.to_numeric(c["N_1"], errors="coerce") == int(n1)].copy()
    if n2 is not None:
        c = c[pd.to_numeric(c["N_2"], errors="coerce") == int(n2)].copy()

    c = c[np.isclose(c["td_ms"].astype(float), float(td_ms), rtol=0.0, atol=float(tol_ms))].copy()
    if c.empty:
        extra_parts: list[str] = []
        if sheet is not None and "sheet" in corr.columns:
            extra_parts.append(f"sheet={_canonical_sheet_name(sheet)}")
        if n1 is not None:
            extra_parts.append(f"N_1={int(n1)}")
        if n2 is not None:
            extra_parts.append(f"N_2={int(n2)}")
        extra = f" y {'; '.join(extra_parts)}" if extra_parts else ""
        raise ValueError(f"No encontré factores para roi={roi_ref}{extra} y td_ms={td_ms:.3f} (tol={tol_ms}).")

    # if duplicates: average
    c = c.groupby("direction", as_index=False)["correction_factor"].mean()

    out = {str(d): float(f) for d, f in zip(c["direction"], c["correction_factor"])}
    if not out:
        raise ValueError("Tabla de corrección filtrada no produjo factores válidos.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast_parquet", type=Path, help="Parquet long de contraste (salida de make_contrast.py)")

    ap.add_argument("--model", choices=["free", "tort"])
    ap.add_argument("--gbase", default="g_lin_max", help="g, g_lin_max, g_thorsten")
    ap.add_argument("--ycol", default="value_norm", help="value o value_norm")

    ap.add_argument("--directions", nargs="*", default=None, help="Filtra por direction (ej: 1 2 3 o long tra).")
    ap.add_argument("--direction", nargs="*", dest="directions", help="Alias de --directions (por compat).")

    ap.add_argument("--rois", nargs="*", default=None, help="Filtra ROIs. Usa ALL para todas.")
    ap.add_argument("--stat", default="avg", help="Filtra stat (default avg). Usa ALL para todos.")

    ap.add_argument("--out_root", required=True)
    ap.add_argument("--no_plots", action="store_true")

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--apply_grad_corr", action="store_true")
    grp.add_argument("--no_grad_corr", action="store_true")

    ap.add_argument("--corr_xlsx", type=Path, default=None)
    ap.add_argument("--corr_roi", default="Agua")
    ap.add_argument("--corr_td_ms", type=float, default=None)
    ap.add_argument("--corr_tol_ms", type=float, default=1e-3)
    ap.add_argument("--corr_sheet", default=None, help="Opcional: sheet a usar dentro de la tabla de corrección. Si no, uso el prefijo del analysis_id.")

    ap.add_argument("--fix_M0", type=float, default=None)
    ap.add_argument("--free_M0", action="store_true")

    ap.add_argument("--n_fit", type=int, default=None, help="Usar solo los primeros n_fit puntos (después de ordenar por x).")
    args = ap.parse_args()

    df = pd.read_parquet(args.contrast_parquet)
    analysis_id = _analysis_id_from_path(args.contrast_parquet)

    sheet_hint = _canonical_sheet_name(analysis_id)
    n1_hint = _unique_int(df, "N_1")
    n2_hint = _unique_int(df, "N_2")
    outdir = Path(args.out_root) / analysis_id
    tables_dir = outdir
    plots_dir = outdir
    tables_dir.mkdir(parents=True, exist_ok=True)

    # correction
    use_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    f_by_direction = None
    td_ms_hint = _infer_td_ms(df, analysis_id, args.corr_td_ms)

    if use_corr:
        if args.corr_xlsx is None:
            raise ValueError("Pasaste --apply_grad_corr pero no pasaste --corr_xlsx.")
        if td_ms_hint is None:
            raise ValueError("No pude inferir td_ms para buscar corrección. Pasá --corr_td_ms o asegurate de tener td_ms_1.")
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
    if args.free_M0:
        M0_vary = True
        M0_value = 1.0
    elif args.fix_M0 is not None:
        M0_vary = False
        M0_value = float(args.fix_M0)
    else:
        M0_vary = True
        M0_value = 1.0

    D0_vary = True
    D0_value = 2.3e-12

    # normalize filters
    directions = args.directions
    if directions is not None and len(directions) == 1 and str(directions[0]).upper() == "ALL":
        directions = None
    rois = args.rois
    if rois is not None and len(rois) == 1 and str(rois[0]).upper() == "ALL":
        rois = None

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
    )

    out_parquet = tables_dir / "fit_params.parquet"
    fit_df.to_parquet(out_parquet, index=False)
    fit_df.to_excel(out_parquet.with_suffix(".xlsx"), index=False)
    fit_df.to_csv(tables_dir / "fit_params.csv", index=False)

    print("Saved fit table:", out_parquet)

    if args.no_plots:
        return

    # plots (one per roi/direction/stat row)
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
