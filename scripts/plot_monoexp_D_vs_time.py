from __future__ import annotations

import argparse
from pathlib import Path

from monoexp_fitting.plot_monoexp_D_vs_time import (
    aggregate_monoexp_by_x,
    load_monoexp_fit_measurements,
    plot_compare_N_within_sheet,
    plot_by_direction,
    plot_by_roi,
)


def _clear_compare_n_pngs(out_dir: Path, xcol: str) -> None:
    compare_dir = out_dir / xcol / "compare_N"
    if not compare_dir.exists():
        return
    for png in compare_dir.glob("*.png"):
        png.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Construye plots de D monoexp vs td_ms y Delta_app_ms desde fit_params.parquet."
    )
    ap.add_argument("--fits-root", required=True, help="Carpeta raíz con fit_params.parquet de monoexp.")
    ap.add_argument("--out-dir", required=True, help="Carpeta de salida para tablas combinadas y PNGs.")
    ap.add_argument("--pattern", default="**/fit_params.parquet", help="Glob relativo dentro de fits-root.")
    subj_group = ap.add_mutually_exclusive_group()
    subj_group.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms a incluir.")
    subj_group.add_argument("--brains", nargs="+", dest="subjs", help="Legacy alias for --subjs.")
    ap.add_argument("--rois", nargs="+", default=None, help="ROIs a incluir.")
    ap.add_argument("--dirs", nargs="+", default=None, help="Direcciones a incluir.")
    ap.add_argument("--stat", default="avg", help="Stat a conservar del fit monoexp.")
    ap.add_argument("--ycol", default="value_norm", help="ycol a conservar del fit monoexp.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_monoexp_fit_measurements(
        args.fits_root,
        pattern=args.pattern,
        subjs=args.subjs,
        rois=args.rois,
        directions=args.dirs,
        stat=args.stat,
        ycol=args.ycol,
    )

    raw.to_csv(out_dir / "monoexp_D.raw.csv", index=False)
    raw.to_excel(out_dir / "monoexp_D.raw.xlsx", index=False)

    for xcol in ("td_ms", "Delta_app_ms"):
        avg = aggregate_monoexp_by_x(raw, xcol=xcol)
        avg.to_csv(out_dir / f"monoexp_D_vs_{xcol}.combined.csv", index=False)
        avg.to_excel(out_dir / f"monoexp_D_vs_{xcol}.combined.xlsx", index=False)
        plot_by_roi(avg, xcol=xcol, out_dir=out_dir)
        plot_by_direction(avg, xcol=xcol, out_dir=out_dir)
        _clear_compare_n_pngs(out_dir, xcol)
        plot_compare_N_within_sheet(avg, xcol=xcol, out_dir=out_dir)

    print(f"[OK] Monoexp D summary plots + tables in: {out_dir}")


if __name__ == "__main__":
    main()
