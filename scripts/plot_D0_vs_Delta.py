from __future__ import annotations

import argparse
from pathlib import Path

from monoexp_fitting.plot_D0_vs_Delta import (
    load_all_fits,
    plot_all_rois_same_scale,
    plot_per_roi,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-root", required=True, help="Carpeta raíz con monoexp_fits/<exp_id>/*.fit_params.csv")
    ap.add_argument("--out-dir", required=True, help="Carpeta salida para plots")
    ap.add_argument("--dirs", nargs="+", default=None, help="Direcciones a incluir (ej: x y z)")
    ap.add_argument("--rois", nargs="+", default=None, help="ROIs a incluir (si no, usa todas)")
    ap.add_argument("--agg", default="mean", choices=["mean", "median", "none"], help="Agregado si hay replicados")
    ap.add_argument("--ncols", type=int, default=3, help="Columnas en la figura comparativa de ROIs")
    args = ap.parse_args()

    df = load_all_fits(
        args.fits_root,
        dirs=args.dirs,
        rois=args.rois,
        agg=args.agg,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 plot por ROI (con todas las direcciones)
    rois_list = sorted(df["roi"].unique().tolist())
    for roi in rois_list:
        plot_per_roi(df, out_dir=out_dir, roi=roi, dirs=args.dirs)

    # figura comparativa (todas las ROIs, mismo eje Y)
    plot_all_rois_same_scale(df, out_dir=out_dir, dirs=args.dirs, rois=args.rois, ncols=args.ncols)

    # guardar tabla combinada
    df.to_csv(out_dir / "D0_vs_Delta.combined.csv", index=False)
    print(f"OK. Plots + tabla en: {out_dir}")

if __name__ == "__main__":
    main()
