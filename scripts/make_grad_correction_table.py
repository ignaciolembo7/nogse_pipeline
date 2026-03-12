from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from ogse_fitting.make_grad_correction_table import ColumnMap, make_grad_correction_table


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--roi", required=True, help='ROI libre (ej: "Agua")')

    ap.add_argument(
        "--exp-fits-root",
        required=True,
        help="Root donde están los monoexp fits (fit_ogse-signal_vs_bval).",
    )

    ap.add_argument(
        "--nogse-root",
        required=True,
        help="Root o archivo con fits del contraste (fit_OGSE-contrast_vs_g). "
             "Se buscan fit_params*.{parquet,csv,xlsx} y también *.fit_*.parquet.",
    )

    ap.add_argument("--out-xlsx", required=True, help="Salida .xlsx (ej: grad_factor_corr.xlsx)")
    ap.add_argument("--out-csv", default=None, help="Opcional: salida .csv también")

    ap.add_argument("--fit_points", type=int, default=None, help="Si querés fijar k (fit_points) para monoexp")
    ap.add_argument("--stat", default="avg", help="Stat a conservar (avg por defecto). Usa ALL para no filtrar.")

    ap.add_argument("--tol-ms", type=float, default=1e-3, help="Tolerancia en ms para matchear td_ms")

    # Monoexp D0
    ap.add_argument("--exp-d0-col", default="D0_mm2_s")
    ap.add_argument("--exp-scale", type=float, default=1e-9, help="mm2/s -> m2/ms (default 1e-9)")

    # Contrast D0
    ap.add_argument("--nogse-d0-col", default="D0_m2_ms", help="Default coherente con fit_ogse_contrast.py")
    ap.add_argument("--nogse-scale", type=float, default=1.0)

    # NEW: normalización/QA
    ap.add_argument("--no-canonical-sheet", action="store_true",
                    help="No normalizar sheet (por default se reduce a un identificador base, ej OGSEvsMax_duration2 -> OGSEvsMax).")
    ap.add_argument("--no-average-over-N", action="store_true",
                    help="No promediar sobre N1/N2 (por default se promedia y queda 1 fila por direction y td_ms).")

    args = ap.parse_args()

    cmap = ColumnMap(
        roi_col="roi",
        dir_col="direction",
        td_col="td_ms",
        d0_col=args.nogse_d0_col,
        stat_col="stat",
        sheet_col="sheet",
        n1_col="N1",
        n2_col="N2",
    )

    out = make_grad_correction_table(
        roi=args.roi,
        exp_fits_root=args.exp_fits_root,
        nogse_root_or_file=args.nogse_root,
        fit_points=args.fit_points,
        stat_keep=args.stat,
        exp_d0_col=args.exp_d0_col,
        exp_scale=args.exp_scale,
        nogse_scale=args.nogse_scale,
        cmap=cmap,
        tol_ms=args.tol_ms,
        canonicalize_sheet=not args.no_canonical_sheet,
        average_over_N=not args.no_average_over_N,
    )

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx) as w:
        out.to_excel(w, sheet_name="grad_correction", index=False)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    print("OK:", out_xlsx)


if __name__ == "__main__":
    main()
