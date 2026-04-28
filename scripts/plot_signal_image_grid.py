from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from plottings.signal_image_grid import (
    build_grid_cells,
    collect_signal_image_entries,
    default_output_stem,
    render_signal_image_grid,
    write_grid_manifest,
)


def _float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        out.append(float(str(value).replace("p", ".")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a table of signal images by measurement type and gradient value.")
    ap.add_argument("--signal-root", type=Path, required=True, help="Base Data-signals folder.")
    ap.add_argument("--experiment", help="Experiment folder under --signal-root.")
    ap.add_argument("--name", help="Acquisition/name folder under --signal-root/experiment.")
    ap.add_argument("--out-root", type=Path, required=True, help="Output folder for PNG and manifest CSV.")
    ap.add_argument("--gradient-type", default="g", help="Gradient token to use for columns: g or b.")
    ap.add_argument("--gradient-values", nargs="+", required=True, help="Gradient values to plot, in display order.")
    ap.add_argument(
        "--rows",
        nargs="+",
        required=True,
        help="Row labels. Use quoted labels with ' - ' for difference rows, for example 'CPMG - HAHN'.",
    )
    ap.add_argument("--image-name", default="mean.nii.gz", help="Image filename inside each sequence folder.")
    ap.add_argument("--include-token", action="append", default=[], help="Require this token in matched sequence names.")
    ap.add_argument("--exclude-token", action="append", default=[], help="Reject matched sequence names containing this token.")
    ap.add_argument("--title", default=None, help="Figure title. Defaults to experiment / name.")
    ap.add_argument("--out-stem", default=None, help="Output stem. Defaults to a sanitized experiment/name/selection stem.")
    ap.add_argument("--allow-missing", action="store_true", help="Render missing cells instead of failing.")
    ap.add_argument("--gradient-tol", type=float, default=1e-6, help="Absolute tolerance for gradient matching.")
    ap.add_argument("--slice-axis", type=int, default=2, help="Slice axis for 3D images.")
    ap.add_argument("--slice-index", type=int, default=None, help="Slice index for 3D images. Defaults to the middle slice.")
    ap.add_argument("--volume-index", type=int, default=None, help="Volume index for 4D images. Defaults to the mean over volumes.")
    ap.add_argument("--no-crop", action="store_true", help="Do not crop around nonzero signal.")
    ap.add_argument("--intensity-percentile", type=float, default=99.0, help="Shared upper percentile for signal rows.")
    ap.add_argument("--diff-percentile", type=float, default=99.0, help="Symmetric percentile for difference rows.")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    experiment = args.experiment
    name = args.name

    if not experiment:
        raise SystemExit("--experiment is required.")
    if not name:
        raise SystemExit("--name is required.")

    gradient_values = _float_list(args.gradient_values)
    case_root = args.signal_root / experiment / name
    if not case_root.is_dir():
        raise SystemExit(f"Case root not found: {case_root}")

    entries = collect_signal_image_entries(
        case_root,
        gradient_type=args.gradient_type,
        image_name=args.image_name,
        include_tokens=args.include_token,
        exclude_tokens=args.exclude_token,
    )
    if not entries:
        raise SystemExit(
            "No matching signal image entries were found. "
            f"case_root={case_root}, image_name={args.image_name!r}, gradient_type={args.gradient_type!r}"
        )

    rows = build_grid_cells(
        entries,
        row_labels=args.rows,
        gradient_values=gradient_values,
        gradient_tol=args.gradient_tol,
        allow_missing=args.allow_missing,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        volume_index=args.volume_index,
    )

    title = args.title or f"exp={experiment} / name={name}"
    out_stem = args.out_stem or default_output_stem(experiment, name, args.gradient_type, args.rows, gradient_values)
    out_png = args.out_root / f"{out_stem}.png"
    out_csv = args.out_root / f"{out_stem}.manifest.csv"

    render_signal_image_grid(
        rows,
        gradient_type=args.gradient_type,
        title=title,
        out_png=out_png,
        crop_nonzero=not args.no_crop,
        intensity_percentile=args.intensity_percentile,
        diff_percentile=args.diff_percentile,
        dpi=args.dpi,
    )
    write_grid_manifest(rows, out_csv)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
