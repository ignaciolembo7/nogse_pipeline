from __future__ import annotations

import argparse
from pathlib import Path

from ogse_fitting.contrast_fit_panels import plot_contrast_fit_panels


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Genera figuras de contraste vs g con ajustes, agrupadas por brain. "
            "Cada figura contiene una grilla ROI x direction y superpone todas las curvas td_ms disponibles."
        )
    )
    ap.add_argument("fits_root", help="Carpeta raíz con fit_params de contraste.")
    ap.add_argument(
        "--contrast-root",
        default="analysis/ogse_experiments/contrast-data-rotated",
        help="Raíz de tablas de contraste (root que contiene tables/).",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Directorio de salida. Si no, usa <fits_root>/contrast_fit_panels.")
    ap.add_argument("--pattern", default="**/fit_params.*", help="Glob relativo para descubrir fit_params.")
    ap.add_argument("--models", nargs="+", default=None, help="Filtra modelos.")
    ap.add_argument("--brains", nargs="+", default=None, help="Filtra brains.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filtra ROIs.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filtra directions.")
    ap.add_argument("--include-failed", action="store_true", help="Incluye filas con ok=False.")
    args = ap.parse_args()

    fits_root = Path(args.fits_root)
    out_dir = args.out_dir or (fits_root / "contrast_fit_panels")

    outputs = plot_contrast_fit_panels(
        fits_root=fits_root,
        contrast_root=args.contrast_root,
        out_dir=out_dir,
        pattern=args.pattern,
        models=args.models,
        brains=args.brains,
        rois=args.rois,
        directions=args.directions,
        ok_only=not bool(args.include_failed),
    )

    print(f"[OK] Figuras generadas: {len(outputs)}")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
