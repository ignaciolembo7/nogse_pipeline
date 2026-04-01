from __future__ import annotations

import argparse
from pathlib import Path

from monoexp_fitting.plot_D0_vs_Delta import (
    load_all_measurements,
    plot_all_groups,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Construye curvas D vs Delta_app_ms desde tablas *.Dproj.long.parquet."
    )
    ap.add_argument("--dproj-root", required=True, help="Carpeta raíz con tablas *.Dproj.long.parquet.")
    ap.add_argument("--pattern", default="**/*.Dproj.long.parquet", help="Glob relativo dentro de dproj-root.")
    ap.add_argument("--out-dir", required=True, help="Carpeta de salida para plots y tabla combinada.")
    subj_group = ap.add_mutually_exclusive_group()
    subj_group.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms a incluir (ej: BRAIN-3 LUDG-2 PHANTOM3).")
    subj_group.add_argument("--brains", nargs="+", dest="subjs", help="Legacy alias for --subjs.")
    ap.add_argument("--rois", nargs="+", default=None, help="ROIs a incluir.")
    ap.add_argument("--dirs", nargs="+", default=["x", "y", "z"], help="Direcciones a incluir.")

    selector = ap.add_mutually_exclusive_group()
    selector.add_argument("--N", type=float, default=None, help="Filtra por N.")
    selector.add_argument("--Hz", type=float, default=None, help="Filtra por Hz.")

    ap.add_argument("--bvalue-decimals", type=int, default=1, help="Decimales para redondear bvalue antes de agrupar.")
    ap.add_argument("--reference-D0", type=float, default=0.0032, help="Valor de referencia para anotar alpha en el plot.")
    ap.add_argument("--reference-D0-error", type=float, default=0.0000283512, help="Error del valor de referencia.")
    args = ap.parse_args()

    N = 1.0 if args.N is None and args.Hz is None else args.N

    df = load_all_measurements(
        args.dproj_root,
        pattern=args.pattern,
        dirs=args.dirs,
        rois=args.rois,
        subjs=args.subjs,
        N=N,
        Hz=args.Hz,
        bvalue_decimals=int(args.bvalue_decimals),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_all_groups(
        df,
        out_dir=out_dir,
        reference_D0=float(args.reference_D0),
        reference_D0_error=float(args.reference_D0_error),
    )

    df.to_csv(out_dir / "D_vs_delta_app.combined.csv", index=False)
    df.to_excel(out_dir / "D_vs_delta_app.combined.xlsx", index=False)
    print(f"[OK] Plots + tablas en: {out_dir}")


if __name__ == "__main__":
    main()
