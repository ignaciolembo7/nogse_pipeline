from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from ogse_fitting.contrast_tc_peak_panels import plot_contrast_tc_peak_panels


def _parse_xlims(rows: list[list[str]] | None) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not rows:
        return out
    for xvar, xmin, xmax in rows:
        out[str(xvar)] = (float(xmin), float(xmax))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate contrast figures with the same ROI x direction grid as the fit panels, "
            "with an option to mark tc_peak on selected x-axis variables."
        )
    )
    ap.add_argument("fits_root", help="Root folder with contrast fit_params.")
    ap.add_argument(
        "--contrast-root",
        default="analysis/ogse_experiments/contrast-data-rotated",
        help="Contrast table root (the root that contains tables/).",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to <fits_root>/tc_peak_panels.")
    ap.add_argument("--pattern", default="**/fit_params.*", help="Relative glob used to discover fit_params.")
    ap.add_argument("--models", nargs="+", default=None, help="Filter models.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Filter subjects/phantoms.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filter ROIs.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filter directions.")
    ap.add_argument("--exclude-td-ms", nargs="*", type=float, default=None, help="td_ms values to exclude from the figures.")
    ap.add_argument("--x-vars", nargs="+", default=["g", "Ld", "lcf", "lcf_a", "tc"], help="X-axis variables to generate.")
    ap.add_argument(
        "--peak-marker-x-vars",
        nargs="+",
        default=["tc"],
        help="X-axis variables where tc_peak is marked/annotated. Use NONE to disable or ALL to mark all.",
    )
    ap.add_argument(
        "--peak-source",
        choices=["standard", "resampled", "both"],
        default="standard",
        help="Which peak to mark/annotate.",
    )
    ap.add_argument(
        "--show-resampled-fit",
        action="store_true",
        help="Overlay the fit rebuilt on g_resampled as S1_fit(g_resampled)-S2_fit(g_resampled).",
    )
    ap.add_argument(
        "--hide-data-points",
        action="store_true",
        help="Do not draw experimental contrast points. With --show-resampled-fit, all x-axis transforms use g_resampled.",
    )
    ap.add_argument("--resampled-curve-n", type=int, default=300, help="Number of points in the resampled fitted curve.")
    ap.add_argument(
        "--xlim",
        nargs=3,
        action="append",
        metavar=("XVAR", "XMIN", "XMAX"),
        default=None,
        help="X-axis limit for one variable. Repeatable: --xlim lcf 0 20.",
    )
    ap.add_argument("--peak-D0-fix", type=float, default=3.2e-12, help="Fixed D0 used for peak-derived transforms.")
    ap.add_argument("--peak-gamma", type=float, default=267.5221900, help="Gamma in rad/(ms*mT) used for peak-derived transforms.")
    ap.add_argument("--include-failed", action="store_true", help="Include rows with ok=False.")
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
        peak_marker_x_vars=args.peak_marker_x_vars,
        peak_source=args.peak_source,
        show_resampled_fit=bool(args.show_resampled_fit),
        show_data_points=not bool(args.hide_data_points),
        resampled_curve_n=int(args.resampled_curve_n),
        peak_D0_fix=float(args.peak_D0_fix),
        peak_gamma=float(args.peak_gamma),
        x_lims=_parse_xlims(args.xlim),
        ok_only=not bool(args.include_failed),
    )

    print(f"[OK] Generated figures: {len(outputs)}")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
