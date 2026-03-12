from __future__ import annotations
import argparse
from pathlib import Path

from monoexp_fitting.plot_all_fits_grid import plot_grid_all_deltas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Carpeta con .long.parquet (ej OGSE_signal/data/<grupo>)")
    ap.add_argument("--fits-root", required=True, help="Carpeta con fits por experimento (ej monoexp_fits/<grupo>)")
    ap.add_argument("--out-png", required=True, help="Salida png")
    ap.add_argument("--dirs", nargs="+", default=["x","y","z"])
    ap.add_argument("--rois", nargs="+", default=None)
    ap.add_argument("--dir-col", default="direction")
    ap.add_argument("--roi-col", default="roi")
    ap.add_argument("--bcol", default="bvalue")
    ap.add_argument("--ycol", default="signal_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    plot_grid_all_deltas(
        data_root=args.data_root,
        fits_root=args.fits_root,
        out_png=args.out_png,
        dirs=args.dirs,
        rois=args.rois,
        dir_col=args.dir_col,
        roi_col=args.roi_col,
        bcol=args.bcol,
        ycol=args.ycol,
        stat_keep=args.stat,
        title=args.title,
    )

if __name__ == "__main__":
    main()
