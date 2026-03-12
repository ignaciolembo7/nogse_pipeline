from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from tc_fittings.alpha_macro_summary import (
    read_fit_params,
    compute_alpha_macro_powerlaw,
    write_alpha_macro_outputs,
)


def main():
    ap = argparse.ArgumentParser(
        description="Genera plots/summary_alpha_values.xlsx (alpha_macro) desde fit_params de monoexp."
    )
    ap.add_argument("--fits-root", type=str, required=True,
                    help="Carpeta raíz donde están los fit_params (xlsx/parquet/csv).")
    ap.add_argument("--pattern", type=str, default="**/fit_params*.xlsx",
                    help="Glob relativo dentro de fits-root (default **/fit_params*.xlsx).")
    ap.add_argument("--out-summary", type=Path, default=Path("plots/summary_alpha_values.xlsx"),
                    help="Salida summary (default plots/summary_alpha_values.xlsx).")
    ap.add_argument("--out-avg", type=Path, default=Path("plots/alpha_macro_D0_avg.xlsx"),
                    help="Salida tabla intermedia D0_mean vs td (default plots/alpha_macro_D0_avg.xlsx).")
    ap.add_argument("--min-points", type=int, default=3,
                    help="Mínimo de puntos td distintos para fit (default 3).")

    args = ap.parse_args()

    root = Path(args.fits_root)
    if not root.exists():
        raise FileNotFoundError(root)

    files = [Path(p) for p in glob(str(root / args.pattern), recursive=True)]
    if not files:
        raise FileNotFoundError(f"No encontré archivos con pattern={args.pattern} en {root}")

    df_all = read_fit_params(files)
    df_avg, df_summary = compute_alpha_macro_powerlaw(df_all, min_points=int(args.min_points))

    if df_summary.empty:
        raise RuntimeError(
            "df_summary quedó vacío: probablemente no hay >= min_points tiempos por (brain, roi, direction)."
        )

    write_alpha_macro_outputs(df_avg, df_summary, out_summary_xlsx=args.out_summary, out_avg_xlsx=args.out_avg)

    print(f"[OK] summary: {args.out_summary}")
    print(f"[OK] avg:     {args.out_avg}")


if __name__ == "__main__":
    main()