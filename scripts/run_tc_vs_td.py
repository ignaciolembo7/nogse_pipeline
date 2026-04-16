from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tc_fittings.contrast_fit_table import canonicalize_contrast_fit_params, load_contrast_fit_params
from tc_fittings.tc_td_registry import METHODS
from tc_fittings.tc_td_pseudohuber import load_alpha_macro_summary


YCOL_LABELS = {
    "tc_ms": r"$t_c$ [ms]",
    "tc_fit_ms": r"$t_{c,fit}$ [ms]",
    "tc_peak_ms": r"$t_{c,peak}$ [ms]",
    "signal_peak": "Peak amplitude",
}


def _tc_vs_td_dirname(y_col: str) -> str:
    if y_col == "tc_peak_ms":
        return "tcpeak_vs_td"
    if y_col in {"tc_ms", "tc_fit_ms"}:
        return "tcfit_vs_td"
    return f"{y_col}_vs_td"


def _parse_exclude_match(spec: str) -> dict[str, object]:
    parts = [p.strip() for p in str(spec).split("|")]
    if len(parts) == 2:
        roi, direction = parts
        return {
            "roi": roi,
            "direction": direction,
            "td_ms": None,
        }
    if len(parts) == 3:
        if _looks_like_float(parts[2]):
            roi, direction, td_ms = parts
            return {
                "roi": roi,
                "direction": direction,
                "td_ms": float(td_ms),
            }
        subj, roi, direction = parts
        return {
            "subj": subj,
            "roi": roi,
            "direction": direction,
            "td_ms": None,
        }
    if len(parts) == 4:
        subj, roi, direction, td_ms = parts
        return {
            "subj": subj,
            "roi": roi,
            "direction": direction,
            "td_ms": float(td_ms),
        }
    raise ValueError(
        "Formato inválido en --exclude-match. Usa 'roi|direction', "
        "'subj|roi|direction', 'roi|direction|td_ms' "
        "o 'subj|roi|direction|td_ms'."
    )


def _looks_like_float(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _normalize_name_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                out.append(token)
    if not out:
        return None
    return list(dict.fromkeys(out))


def _load_df_params(args: argparse.Namespace) -> pd.DataFrame:
    groupfits_path = args.groupfits if args.groupfits is not None else args.globalfit
    if groupfits_path is not None:
        path = Path(groupfits_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        df = canonicalize_contrast_fit_params(df)
    else:
        if not args.fits:
            raise ValueError("Pasá --groupfits/--globalfit o al menos una raíz/archivo en --fits.")
        df = load_contrast_fit_params(
            args.fits,
            pattern=args.pattern,
            models=args.models,
            subjs=args.subjs,
            directions=args.directions,
            rois=args.rois,
            ok_only=True,
        )

    if args.subjs is not None and "subj" in df.columns:
        df = df[df["subj"].astype(str).isin([str(x) for x in args.subjs])].copy()
    if args.directions is not None and "direction" in df.columns:
        df = df[df["direction"].astype(str).isin([str(x) for x in args.directions])].copy()
    if args.rois is not None and "roi" in df.columns:
        target_rois = {str(x).replace("_norm", "").strip() for x in args.rois}
        roi_norm = df["roi"].astype(str).str.replace("_norm", "", regex=False).str.strip()
        df = df[roi_norm.isin(target_rois)].copy()

    if args.exclude_td_ms:
        td_vals = pd.to_numeric(df.get("td_ms"), errors="coerce")
        keep = np.ones(len(df), dtype=bool)
        for td_excl in args.exclude_td_ms:
            keep &= ~np.isclose(td_vals.to_numpy(dtype=float), float(td_excl), atol=1e-3, equal_nan=False)
        df = df.loc[keep].copy()

    if args.exclude_match:
        keep = np.ones(len(df), dtype=bool)
        td_vals = pd.to_numeric(df.get("td_ms"), errors="coerce")
        subj_vals = df["subj"].astype(str).str.strip() if "subj" in df.columns else pd.Series([""] * len(df), index=df.index)
        roi_vals = df["roi"].astype(str).str.strip() if "roi" in df.columns else pd.Series([""] * len(df), index=df.index)
        dir_vals = df["direction"].astype(str).str.strip() if "direction" in df.columns else pd.Series([""] * len(df), index=df.index)

        for spec in args.exclude_match:
            rule = _parse_exclude_match(spec)
            mask = np.ones(len(df), dtype=bool)
            mask &= roi_vals.to_numpy(dtype=str) == str(rule["roi"])
            mask &= dir_vals.to_numpy(dtype=str) == str(rule["direction"])
            if rule.get("td_ms") is not None:
                mask &= np.isclose(td_vals.to_numpy(dtype=float), float(rule["td_ms"]), atol=1e-3, equal_nan=False)
            if "subj" in rule:
                mask &= subj_vals.to_numpy(dtype=str) == str(rule["subj"])
            keep &= ~mask
        df = df.loc[keep].copy()

    if args.y_col not in df.columns:
        raise KeyError(f"No existe y_col={args.y_col!r} en la tabla combinada.")

    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    df = df[df[args.y_col].notna()].copy()
    if df.empty:
        raise ValueError(f"No quedó data válida para y_col={args.y_col!r}.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS.keys()))
    ap.add_argument("--k-last", type=int, default=None, help="Usar últimos K puntos (default: método).")
    ap.add_argument("--groupfits", default=None, help="Tabla groupfits ya combinada (xlsx/csv/parquet).")
    ap.add_argument("--globalfit", default=None, help="Alias legacy de --groupfits.")
    ap.add_argument("--fits", nargs="*", default=None, help="Raíces o archivos fit_params para cargar directamente.")
    ap.add_argument("--pattern", default="**/fit_params.*", help="Glob relativo si se usa --fits con directorios.")
    ap.add_argument("--models", nargs="+", default=None, help="Filtra modelos de contraste.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Filtra subjects/phantoms.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filtra directions.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filtra ROIs.")
    ap.add_argument("--y-col", default="tc_peak_ms", help="Columna a ajustar vs td_ms. Ej: tc_peak_ms o tc_fit_ms.")
    ap.add_argument("--exclude-td-ms", nargs="*", type=float, default=None, help="Lista de td_ms a excluir del ajuste. Ej: --exclude-td-ms 209.1")
    ap.add_argument(
        "--exclude-match",
        nargs="*",
        default=None,
        help="Excluye filas específicas. Formato: roi|direction, subj|roi|direction, roi|direction|td_ms o subj|roi|direction|td_ms.",
    )
    ap.add_argument("--no-errorbars", action="store_true", help="Si se pasa, los plots de pseudohuber se generan sin barras/bandas de error.")
    ap.add_argument("--td-min-ms", type=float, default=0.0, help="Límite inferior del eje Td para los plots.")
    ap.add_argument("--td-max-ms", type=float, default=2000.0, help="Límite superior del eje Td para los plots.")
    ap.add_argument("--c-fixed", type=float, default=None, help="Fija c en vez de ajustarlo.")
    ap.add_argument("--c-min", type=float, default=0.0, help="Límite inferior para c si se ajusta.")
    ap.add_argument("--c-max", type=float, default=float("inf"), help="Límite superior para c si se ajusta.")
    ap.add_argument("--delta-fixed", type=float, default=None, help="Fija delta [ms] en vez de ajustarlo.")
    ap.add_argument("--delta-min", type=float, default=1e-6, help="Límite inferior para delta [ms] si se ajusta.")
    ap.add_argument("--delta-max", type=float, default=float("inf"), help="Límite superior para delta [ms] si se ajusta.")
    ap.add_argument("--alpha-macro-fixed", type=float, default=None, help="Fija alpha_macro en pseudohuber_free.")
    ap.add_argument("--alpha-macro-min", type=float, default=0.1, help="Límite inferior para alpha_macro en pseudohuber_free.")
    ap.add_argument("--alpha-macro-max", type=float, default=0.3, help="Límite superior para alpha_macro en pseudohuber_free.")
    ap.add_argument("--summary-alpha", default=None, help="Ruta a summary_alpha_values.xlsx. Si no, no se usa salvo que el método lo requiera.")
    ap.add_argument("--out-dir", default=None, help="Directorio de salida. Si no, se arma a partir del input.")
    args = ap.parse_args()
    args.subjs = _normalize_name_list(args.subjs)
    args.directions = _normalize_name_list(args.directions)
    args.rois = _normalize_name_list(args.rois)
    if float(args.td_max_ms) <= float(args.td_min_ms):
        raise ValueError("--td-max-ms debe ser mayor que --td-min-ms.")

    df_params = _load_df_params(args)
    spec = METHODS[args.method]
    k_last = args.k_last if args.k_last is not None else spec.default_k_last

    alpha_macro_df = None
    if args.summary_alpha is not None:
        summary_path = Path(args.summary_alpha)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        alpha_macro_df = load_alpha_macro_summary(summary_path)
        if args.directions is not None and "direction" in alpha_macro_df.columns:
            alpha_macro_df = alpha_macro_df[alpha_macro_df["direction"].astype(str).isin([str(x) for x in args.directions])].copy()
        if args.rois is not None and "roi" in alpha_macro_df.columns:
            target_rois = {str(x).replace("_norm", "").strip() for x in args.rois}
            roi_norm = alpha_macro_df["roi"].astype(str).str.replace("_norm", "", regex=False).str.strip()
            alpha_macro_df = alpha_macro_df[roi_norm.isin(target_rois)].copy()
    elif spec.needs_alpha_macro:
        raise FileNotFoundError(f"{args.method} requiere --summary-alpha con summary_alpha_values.xlsx")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        tc_dirname = _tc_vs_td_dirname(args.y_col)
        groupfits_path = args.groupfits if args.groupfits is not None else args.globalfit
        if groupfits_path is not None:
            out_dir = Path(groupfits_path).resolve().parent / tc_dirname / args.method # / args.y_col
        else:
            first = Path(args.fits[0]).resolve()
            out_dir = (first if first.is_dir() else first.parent) / tc_dirname / args.method # / args.y_col

    out_dir.mkdir(parents=True, exist_ok=True)

    y_label = YCOL_LABELS.get(args.y_col, args.y_col)
    cfg = None
    if args.rois is not None:
        cfg = SimpleNamespace(regions=[str(r).replace("_norm", "") for r in args.rois])
    spec.func(
        cfg=cfg,
        df_params=df_params,
        out_dir=out_dir,
        k_last=k_last,
        alpha_macro_df=alpha_macro_df,
        y_col=args.y_col,
        y_label=y_label,
        show_errorbars=not args.no_errorbars,
        td_min_ms=float(args.td_min_ms),
        td_max_ms=float(args.td_max_ms),
        c_fixed=args.c_fixed,
        c_min=float(args.c_min),
        c_max=float(args.c_max),
        delta_fixed=args.delta_fixed,
        delta_min=float(args.delta_min),
        delta_max=float(args.delta_max),
        alpha_macro_fixed=args.alpha_macro_fixed,
        alpha_macro_min=float(args.alpha_macro_min),
        alpha_macro_max=float(args.alpha_macro_max),
    )


if __name__ == "__main__":
    main()
