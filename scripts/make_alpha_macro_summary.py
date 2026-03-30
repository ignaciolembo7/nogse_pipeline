from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tc_fittings.alpha_macro_summary import (
    DEFAULT_CC_REGIONS,
    DEFAULT_LATERAL_VENTRICLES,
    compute_alpha_macro_summary,
    load_dproj_measurements,
    parse_direction_aliases,
    plot_alpha_macro_vs_roi,
    write_alpha_macro_outputs,
)


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
        description="Calcula alpha_macro = <D0>/0.0032 usando curvas D vs Delta_app_ms a b fijo."
    )
    ap.add_argument("--combined-table", type=Path, default=None, help="Tabla combinada generada por plot_D0_vs_Delta.py.")
    ap.add_argument("--dproj-root", default=None, help="Raíz con *.rot_tensor.Dproj.long.parquet si no se pasa --combined-table.")
    ap.add_argument("--pattern", default="**/*.rot_tensor.Dproj.long.parquet", help="Glob relativo dentro de dproj-root.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms a incluir (ej: BRAIN-3 LUDG-2 PHANTOM3).")
    ap.add_argument("--rois", nargs="+", default=None, help="ROIs a incluir.")
    ap.add_argument("--dirs", nargs="+", default=["x", "y", "z"], help="Direcciones raw a incluir.")

    selector = ap.add_mutually_exclusive_group()
    selector.add_argument("--N", type=float, default=None, help="Filtra por N.")
    selector.add_argument("--Hz", type=float, default=None, help="Filtra por Hz.")

    ap.add_argument("--bvalue-decimals", type=int, default=1, help="Decimales para redondear bvalue antes de agrupar.")
    ap.add_argument("--reference-D0", type=float, default=0.0032, help="Valor de referencia usado para alpha_macro.")
    ap.add_argument("--reference-D0-error", type=float, default=0.0000283512, help="Error del valor de referencia.")
    ap.add_argument("--direction-alias", action="append", default=None, help="Alias raw=agrupado. Repetible. Default: x=long, y=tra, z=tra.")
    ap.add_argument("--out-summary", type=Path, default=Path("plots/summary_alpha_values.xlsx"), help="Salida summary_alpha_values.xlsx")
    ap.add_argument("--out-avg", type=Path, default=Path("plots/alpha_macro_D0_avg.xlsx"), help="Salida tabla agregada D vs Delta_app.")
    ap.add_argument("--cc-directions", nargs="+", default=["long", "tra"], help="Direcciones a mostrar en el mapa de CC.")
    ap.add_argument("--ventricle-directions", nargs="+", default=["long", "tra"], help="Direcciones a mostrar en el mapa de ventrículos.")
    ap.add_argument("--cc-rois", nargs="+", default=DEFAULT_CC_REGIONS, help="Orden de ROIs para el mapa del cuerpo calloso.")
    ap.add_argument(
        "--ventricle-rois",
        nargs="+",
        default=DEFAULT_LATERAL_VENTRICLES,
        help="Orden de ROIs para el mapa de ventrículos laterales.",
    )
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
    plot_alpha_macro_vs_roi(
        df_summary,
        out_png=out_dir / "alpha_macro_vs_region_cc.png",
        roi_order=args.cc_rois,
        directions=args.cc_directions,
        subjs=args.subjs,
        title_prefix=r"$\alpha_{macro}$ vs CC",
    )
    plot_alpha_macro_vs_roi(
        df_summary,
        out_png=out_dir / "alpha_macro_vs_lateral_ventricles.png",
        roi_order=args.ventricle_rois,
        directions=args.ventricle_directions,
        subjs=args.subjs,
        title_prefix=r"$\alpha_{macro}$ vs ventrículos",
    )

    print(f"[OK] summary: {args.out_summary}")
    print(f"[OK] avg:     {args.out_avg}")
    print(f"[OK] plots:   {out_dir}")


if __name__ == "__main__":
    main()
