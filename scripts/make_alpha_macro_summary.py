from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tc_fittings.alpha_macro_summary import (
    compute_alpha_macro_summary,
    load_dproj_measurements,
    parse_direction_aliases,
    plot_alpha_macro_vs_roi,
    write_alpha_macro_outputs,
)


def _safe_plot_alpha_macro_vs_roi(*args, **kwargs) -> None:
    try:
        plot_alpha_macro_vs_roi(*args, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "No hay datos para plotear" in msg:
            print(f"[INFO] Skipping plot: {msg}")
            return
        raise


def _ordered_unique(series: pd.Series) -> list[str]:
    return list(dict.fromkeys(series.dropna().astype(str).tolist()))


def _load_measurements_from_args(args: argparse.Namespace) -> pd.DataFrame:
    if args.combined_table is not None:
        path = Path(args.combined_table)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Formato no soportado para combined_table={path}")

        if args.subjs is not None and "subj" in df.columns:
            df = df[df["subj"].astype(str).isin([str(x) for x in args.subjs])]
        if args.rois is not None and "roi" in df.columns:
            df = df[df["roi"].astype(str).isin([str(x) for x in args.rois])]
        if args.dirs is not None and "direction" in df.columns:
            df = df[df["direction"].astype(str).isin([str(x) for x in args.dirs])]
        if args.N is not None and "N" in df.columns:
            df = df[df["N"].astype(float).sub(float(args.N)).abs() <= 1e-6]
        if args.Hz is not None and "Hz" in df.columns:
            df = df[df["Hz"].astype(float).sub(float(args.Hz)).abs() <= 1e-6]
        return df

    if not args.dproj_root:
        raise ValueError("Pasá --combined-table o --dproj-root.")

    return load_dproj_measurements(
        args.dproj_root,
        pattern=args.pattern,
        subjs=args.subjs,
        rois=args.rois,
        directions=args.dirs,
        N=args.N,
        Hz=args.Hz,
        bvalue_decimals=int(args.bvalue_decimals),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calcula alpha_macro = <D0>/0.0032 para un N/Hz dado y grafica alpha_macro vs ROI."
    )
    ap.add_argument("--combined-table", type=Path, default=None, help="Tabla combinada generada por plot_D0_vs_Delta.py.")
    ap.add_argument("--dproj-root", default=None, help="Raíz con *.Dproj.long.parquet si no se pasa --combined-table.")
    ap.add_argument("--pattern", default="**/*.Dproj.long.parquet", help="Glob relativo dentro de dproj-root.")
    subj_group = ap.add_mutually_exclusive_group()
    subj_group.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms a incluir (ej: BRAIN-3 LUDG-2 PHANTOM3).")
    subj_group.add_argument("--subj", nargs="+", dest="subjs", help="Alias singular para --subjs.")
    subj_group.add_argument("--brains", nargs="+", dest="subjs", help="Legacy alias for --subjs.")
    ap.add_argument("--rois", nargs="+", default=None, help="ROIs a incluir.")
    ap.add_argument("--dirs", nargs="+", default=None, help="Direcciones raw a incluir. Si se omite, no filtra.")

    selector = ap.add_mutually_exclusive_group()
    selector.add_argument("--N", type=float, default=None, help="Filtra por N.")
    selector.add_argument("--Hz", type=float, default=None, help="Filtra por Hz.")

    ap.add_argument("--bvalue-decimals", type=int, default=1, help="Decimales para redondear bvalue antes de agrupar.")
    ap.add_argument("--reference-D0", type=float, default=0.0032, help="Valor de referencia usado para alpha_macro.")
    ap.add_argument("--reference-D0-error", type=float, default=0.0000283512, help="Error del valor de referencia.")
    ap.add_argument("--direction-alias", action="append", default=None, help="Alias raw=agrupado. Repetible. Default: x=long, y=tra, z=tra.")
    ap.add_argument("--out-summary", type=Path, default=Path("plots/summary_alpha_values.xlsx"), help="Salida summary_alpha_values.xlsx")
    ap.add_argument("--out-avg", type=Path, default=Path("plots/alpha_macro_D0_avg.xlsx"), help="Salida tabla agregada D vs Delta_app.")
    ap.add_argument("--plot-rois", nargs="+", default=None, help="ROIs a mostrar en el plot alpha_macro vs ROI. Default: usa --rois o todas.")
    ap.add_argument("--plot-directions", nargs="+", default=None, help="Direcciones a mostrar en el plot. Default: usa --dirs o todas.")
    ap.add_argument("--out-plot", type=Path, default=None, help="PNG de salida para alpha_macro vs ROI. Default: <out-summary-dir>/alpha_macro_vs_roi.png")
    args = ap.parse_args()

    if args.N is None and args.Hz is None and args.combined_table is None:
        args.N = 1.0

    df_avg = _load_measurements_from_args(args)
    aliases = parse_direction_aliases(args.direction_alias)
    df_avg, df_summary = compute_alpha_macro_summary(
        df_avg,
        reference_D0=float(args.reference_D0),
        reference_D0_error=float(args.reference_D0_error),
        direction_aliases=aliases,
    )

    write_alpha_macro_outputs(
        df_avg,
        df_summary,
        out_summary_xlsx=args.out_summary,
        out_avg_xlsx=args.out_avg,
    )

    out_dir = args.out_summary.parent
    roi_order = args.plot_rois or args.rois or _ordered_unique(df_summary["roi"])
    directions = args.plot_directions or args.dirs or _ordered_unique(df_summary["direction"])
    out_plot = args.out_plot or (out_dir / "alpha_macro_vs_roi.png")
    _safe_plot_alpha_macro_vs_roi(
        df_summary,
        out_png=out_plot,
        roi_order=roi_order,
        directions=directions,
        subjs=args.subjs,
        title_prefix=rf"$\alpha_{{macro}}$ | N={args.N:g}" if args.N is not None else r"$\alpha_{macro}$",
    )

    print(f"[OK] summary: {args.out_summary}")
    print(f"[OK] avg:     {args.out_avg}")
    print(f"[OK] plot:    {out_plot}")


if __name__ == "__main__":
    main()
