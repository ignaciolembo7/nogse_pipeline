from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from plotting.publication.contrast_lcf_tcpeak import (
    ContrastDatasetSpec,
    ContrastTdDatasetSpec,
    plot_contrast_td_tcpeak_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent


def _parse_aliases(items: list[str] | None, defaults: dict[str, str]) -> dict[str, str]:
    out = dict(defaults)
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid direction alias {raw!r}. Use source=target.")
        source, target = raw.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid direction alias {raw!r}. Use source=target.")
        out[source] = target
    return out


def _parse_floats(items: list[str] | None) -> tuple[float, ...]:
    if not items:
        return ()
    out: list[float] = []
    for raw in items:
        for part in str(raw).replace(",", " ").split():
            if part:
                out.append(float(part))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build publication-ready NOGSE resampled contrast panels with "
            "one selected fit/resampled contrast and multi-td contrast panels."
        )
    )
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--output-prefix", default="nogse_contrast_td_tcpeak_summary")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--point-size", type=float, default=82.0)
    ap.add_argument("--line-width", type=float, default=2.4)
    ap.add_argument("--lcf-min", type=float, default=None)
    ap.add_argument("--lcf-max", type=float, default=None)
    ap.add_argument("--inset-errorbars", action="store_true", help="Draw tc_peak error bars in the inset when available.")
    ap.add_argument("--peak-gamma", type=float, default=267.5221900)

    ap.add_argument("--brains-fits", type=Path, default=None)
    ap.add_argument("--brains-contrast-root", type=Path, default=None)
    ap.add_argument("--brains-tc-table", type=Path, default=None)
    ap.add_argument("--brains-model", default="rest_offset_globC")
    ap.add_argument("--brains-subj", default="BRAIN")
    ap.add_argument("--brains-roi", default="CentralCC")
    ap.add_argument("--brains-roi-label", default="CentralCC")
    ap.add_argument("--brains-direction", default="long")
    ap.add_argument("--brains-td-ms", type=float, default=143.4)
    ap.add_argument("--brains-direction-alias", action="append", default=None)
    ap.add_argument("--brains-peak-D0-fix", type=float, default=3.2e-12)
    ap.add_argument("--brains-exclude-td-ms", nargs="*", default=[])

    ap.add_argument("--phantoms-fits", type=Path, default=None)
    ap.add_argument("--phantoms-contrast-root", type=Path, default=None)
    ap.add_argument("--phantoms-tc-table", type=Path, default=None)
    ap.add_argument("--phantoms-model", default="rest_offset_globC")
    ap.add_argument("--phantoms-subj", default="PHANTOM3")
    ap.add_argument("--phantoms-roi", default="fiber1")
    ap.add_argument("--phantoms-roi-label", default="less packed")
    ap.add_argument("--phantoms-direction", default="1")
    ap.add_argument("--phantoms-td-ms", type=float, default=142.5)
    ap.add_argument("--phantoms-direction-alias", action="append", default=None)
    ap.add_argument("--phantoms-peak-D0-fix", type=float, default=2.3e-12)
    ap.add_argument("--phantoms-exclude-td-ms", nargs="*", default=["75.1"])
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (project_root / "analysis/publication_figures/nogse_contrast_td_tcpeak")

    brains_fit_root = project_root / "analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr"
    phantoms_fit_root = project_root / "analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr"

    brains_aliases = _parse_aliases(
        args.brains_direction_alias,
        {"long": "longitudinal", "tra": "transversal", "x": "longitudinal", "y": "transversal", "z": "transversal"},
    )
    phantoms_aliases = _parse_aliases(args.phantoms_direction_alias, {"1": "longitudinal", "3": "transversal"})

    specs = [
        ContrastTdDatasetSpec(
            base=ContrastDatasetSpec(
                name="brains",
                label="BRAIN",
                fits=args.brains_fits or (brains_fit_root / "groupfits_rest.parquet"),
                contrast_root=args.brains_contrast_root or (brains_fit_root / "contrast"),
                tc_table=args.brains_tc_table
                or (
                    brains_fit_root
                    / "tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx"
                ),
                roi=args.brains_roi,
                roi_label=args.brains_roi_label,
                subj=args.brains_subj,
                directions=(str(args.brains_direction),),
                direction_aliases=brains_aliases,
                peak_D0_fix=float(args.brains_peak_D0_fix),
                exclude_td_ms=_parse_floats(args.brains_exclude_td_ms),
                model=str(args.brains_model),
            ),
            selected_direction=str(args.brains_direction),
            selected_td_ms=float(args.brains_td_ms),
        ),
        ContrastTdDatasetSpec(
            base=ContrastDatasetSpec(
                name="phantoms",
                label="PHANTOMS",
                fits=args.phantoms_fits or (phantoms_fit_root / "groupfits_rest.parquet"),
                contrast_root=args.phantoms_contrast_root or (phantoms_fit_root / "contrast"),
                tc_table=args.phantoms_tc_table
                or (
                    phantoms_fit_root
                    / "tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx"
                ),
                roi=args.phantoms_roi,
                roi_label=args.phantoms_roi_label,
                subj=args.phantoms_subj,
                directions=(str(args.phantoms_direction),),
                direction_aliases=phantoms_aliases,
                peak_D0_fix=float(args.phantoms_peak_D0_fix),
                exclude_td_ms=_parse_floats(args.phantoms_exclude_td_ms),
                model=str(args.phantoms_model),
            ),
            selected_direction=str(args.phantoms_direction),
            selected_td_ms=float(args.phantoms_td_ms),
        ),
    ]

    outputs = plot_contrast_td_tcpeak_summary(
        specs,
        out_stem=out_dir / args.output_prefix,
        formats=args.formats,
        dpi=int(args.dpi),
        peak_gamma=float(args.peak_gamma),
        point_size=float(args.point_size),
        line_width=float(args.line_width),
        lcf_min=args.lcf_min,
        lcf_max=args.lcf_max,
        inset_errorbars=bool(args.inset_errorbars),
    )
    for path in outputs:
        print(f"[OK] figure: {path}")
    print(f"[OK] table: {out_dir / (args.output_prefix + '_plotted_rows.csv')}")


if __name__ == "__main__":
    main()
