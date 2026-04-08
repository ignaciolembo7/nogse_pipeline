from __future__ import annotations

import argparse
from pathlib import Path

from ogse_fitting.contrast_tc_peak_panels import plot_contrast_tc_peak_panels


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Genera figuras de contraste con la misma grilla ROI x direction que los paneles de fit, "
            "marcando el tc_peak usado para cada curva y para varias transformaciones del eje x."
        )
    )
    ap.add_argument("fits_root", help="Carpeta raíz con fit_params de contraste.")
    ap.add_argument(
        "--contrast-root",
        default="analysis/ogse_experiments/contrast-data-rotated",
        help="Raíz de tablas de contraste (root que contiene tables/).",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Directorio de salida. Si no, usa <fits_root>/tc_peak_panels.")
    ap.add_argument("--pattern", default="**/fit_params.*", help="Glob relativo para descubrir fit_params.")
    ap.add_argument("--models", nargs="+", default=None, help="Filtra modelos.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Filtra subjects/phantoms.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filtra ROIs.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filtra directions.")
    ap.add_argument("--exclude-td-ms", nargs="*", type=float, default=None, help="Lista de td_ms a excluir de las figuras.")
    ap.add_argument("--x-vars", nargs="+", default=["g", "Ld", "lcf", "Lcf"], help="Variables del eje x a generar.")
    ap.add_argument("--peak-D0-fix", type=float, default=3.2e-12, help="D0 fijo usado para las transformaciones derivadas del pico.")
    ap.add_argument("--peak-gamma", type=float, default=267.5221900, help="Gamma en rad/(ms*mT) usada para las transformaciones derivadas del pico.")
    ap.add_argument("--include-failed", action="store_true", help="Incluye filas con ok=False.")
    args = ap.parse_args()

    fits_root = Path(args.fits_root)
    out_dir = args.out_dir or (fits_root / "tc_peak_panels")

    outputs = plot_contrast_tc_peak_panels(
        fits_root=fits_root,
        contrast_root=args.contrast_root,
        out_dir=out_dir,
        pattern=args.pattern,
        models=args.models,
        subjs=args.subjs,
        rois=args.rois,
        directions=args.directions,
        exclude_td_ms=args.exclude_td_ms,
        x_vars=args.x_vars,
        peak_D0_fix=float(args.peak_D0_fix),
        peak_gamma=float(args.peak_gamma),
        ok_only=not bool(args.include_failed),
    )

    print(f"[OK] Figuras generadas: {len(outputs)}")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
