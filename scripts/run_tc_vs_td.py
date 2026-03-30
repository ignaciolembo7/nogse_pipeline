from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tc_fittings.contrast_fit_table import canonicalize_contrast_fit_params, load_contrast_fit_params
from tc_fittings.tc_td_registry import METHODS
from tc_fittings.tc_td_pseudohuber import load_alpha_macro_summary


YCOL_LABELS = {
    "tc_ms": r"$t_c$ [ms]",
    "tc_peak_ms": r"$t_{c,peak}$ [ms]",
    "signal_peak": "Peak amplitude",
}


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
    ap.add_argument("--y-col", default="tc_peak_ms", help="Columna a ajustar vs td_ms. Ej: tc_peak_ms o tc_ms.")
    ap.add_argument("--summary-alpha", default=None, help="Ruta a summary_alpha_values.xlsx. Si no, no se usa salvo que el método lo requiera.")
    ap.add_argument("--out-dir", default=None, help="Directorio de salida. Si no, se arma a partir del input.")
    args = ap.parse_args()

    df_params = _load_df_params(args)
    spec = METHODS[args.method]
    k_last = args.k_last if args.k_last is not None else spec.default_k_last

    alpha_macro_df = None
    if args.summary_alpha is not None:
        summary_path = Path(args.summary_alpha)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        alpha_macro_df = load_alpha_macro_summary(summary_path)
    elif spec.needs_alpha_macro:
        raise FileNotFoundError(f"{args.method} requiere --summary-alpha con summary_alpha_values.xlsx")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        groupfits_path = args.groupfits if args.groupfits is not None else args.globalfit
        if groupfits_path is not None:
            out_dir = Path(groupfits_path).resolve().parent / "tc_vs_td" / args.method / args.y_col
        else:
            first = Path(args.fits[0]).resolve()
            out_dir = (first if first.is_dir() else first.parent) / "tc_vs_td" / args.method / args.y_col

    out_dir.mkdir(parents=True, exist_ok=True)

    y_label = YCOL_LABELS.get(args.y_col, args.y_col)
    spec.func(
        cfg=None,
        df_params=df_params,
        out_dir=out_dir,
        k_last=k_last,
        alpha_macro_df=alpha_macro_df,
        y_col=args.y_col,
        y_label=y_label,
    )


if __name__ == "__main__":
    main()
