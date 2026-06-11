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
        default=None,
        help=(
            "Contrast table root. Defaults to <fits_root>/contrast for fitted_resampled, "
            "or analysis/ogse_experiments/contrast-data-rotated for direct."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to <fits_root>/tc_peak_panels.")
    ap.add_argument("--pattern", default="**/fit_params.*", help="Relative glob used to discover fit_params.")
    ap.add_argument("--models", nargs="+", default=None, help="Filter models.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Filter subjects/phantoms.")
    ap.add_argument("--rois", nargs="+", default=None, help="Filter ROIs.")
    ap.add_argument("--directions", nargs="+", default=None, help="Filter directions.")
    ap.add_argument("--exclude-td-ms", nargs="*", type=float, default=None, help="td_ms values to exclude from the figures.")
    ap.add_argument("--n1", type=int, default=None, help="Keep only fits with this first OGSE N value.")
    ap.add_argument("--n2", type=int, default=None, help="Keep only fits with this second OGSE N value.")
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
        "--contrast-source",
        choices=["direct", "fitted_resampled", "auto"],
        default="auto",
        help="Interpret contrast tables as direct experimental contrasts or fitted/resampled contrasts.",
    )
    ap.add_argument(
        "--exclude-fitresamp",
        action="store_true",
        help="Ignore fit_params whose analysis_id comes from an already fitted/resampled contrast table.",
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
    if args.contrast_source == "auto":
        contrast_source = "fitted_resampled" if (fits_root / "contrast").is_dir() else "direct"
    else:
        contrast_source = args.contrast_source

    if args.contrast_root is None:
        if contrast_source == "fitted_resampled":
            contrast_root = fits_root / "contrast"
        else:
            contrast_root = Path("analysis/ogse_experiments/contrast-data-rotated")
    else:
        contrast_root = Path(args.contrast_root)

    if contrast_source == "fitted_resampled":
        expected = fits_root / "contrast"
        if contrast_root.resolve() != expected.resolve():
            raise ValueError(
                "For --contrast-source fitted_resampled, --contrast-root must point to "
                f"<fits_root>/contrast. Got {contrast_root}; expected {expected}."
            )
        if not contrast_root.is_dir():
            raise FileNotFoundError(
                f"Fitted/resampled contrast root does not exist: {contrast_root}"
            )

    outputs = plot_contrast_tc_peak_panels(
        fits_root=fits_root,
        contrast_root=contrast_root,
        out_dir=out_dir,
        pattern=args.pattern,
        models=args.models,
        subjs=args.subjs,
        rois=args.rois,
        directions=args.directions,
        exclude_td_ms=args.exclude_td_ms,
        x_vars=args.x_vars,
        n1=args.n1,
        n2=args.n2,
        peak_marker_x_vars=args.peak_marker_x_vars,
        peak_source=args.peak_source,
        show_resampled_fit=bool(args.show_resampled_fit),
        show_data_points=not bool(args.hide_data_points),
        resampled_curve_n=int(args.resampled_curve_n),
        peak_D0_fix=float(args.peak_D0_fix),
        peak_gamma=float(args.peak_gamma),
        x_lims=_parse_xlims(args.xlim),
        contrast_source=contrast_source,
        exclude_fitresamp=bool(args.exclude_fitresamp),
        ok_only=not bool(args.include_failed),
    )

    print(f"[OK] Generated figures: {len(outputs)}")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
