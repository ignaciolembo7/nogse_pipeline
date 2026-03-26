from __future__ import annotations

import argparse
from pathlib import Path

from tc_fittings.contrast_fit_table import load_contrast_fit_params


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combina fit_params de fit_ogse-contrast_vs_g.py en una tabla groupfits coherente para el tramo tc-vs-td."
    )
    ap.add_argument(
        "fits",
        nargs="+",
        help="Uno o más directorios raíz de ajustes de contraste, o archivos fit_params.(parquet|xlsx|csv).",
    )
    ap.add_argument("--pattern", default="**/fit_params.*", help="Glob relativo para descubrir fit_params dentro de cada raíz.")
    ap.add_argument("--models", nargs="+", default=None, help="Filtra modelos de contraste (ej: rest tort free).")
    ap.add_argument("--brains", nargs="+", default=None, help="Filtra brains.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filtra directions.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filtra ROIs.")
    ap.add_argument("--include-failed", action="store_true", help="Incluye filas con ok=False.")
    ap.add_argument("--out-xlsx", type=Path, required=True, help="Salida combinada en xlsx.")
    ap.add_argument("--out-parquet", type=Path, default=None, help="Salida adicional en parquet.")
    args = ap.parse_args()

    df = load_contrast_fit_params(
        args.fits,
        pattern=args.pattern,
        models=args.models,
        brains=args.brains,
        directions=args.directions,
        rois=args.rois,
        ok_only=not bool(args.include_failed),
    )

    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.out_xlsx, index=False)
    df.to_csv(args.out_xlsx.with_suffix(".csv"), index=False)
    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)

    print(f"[OK] Tabla groupfits: {args.out_xlsx}")
    if args.out_parquet is not None:
        print(f"[OK] Parquet:      {args.out_parquet}")


if __name__ == "__main__":
    main()
