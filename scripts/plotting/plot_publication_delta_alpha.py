from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from plotting.publication.delta_alpha_roi import (
    DatasetFigureSpec,
    build_plot_tables,
    plot_delta_alpha_figure,
    write_plot_tables,
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


def _default_brains_alpha(project_root: Path) -> Path:
    return project_root / "analysis/brains/ogse_experiments/alpha_macro/N1/summary_alpha_values.csv"


def _default_brains_delta(project_root: Path) -> Path:
    return (
        project_root
        / "analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr"
        / "tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx"
    )


def _default_phantoms_alpha(project_root: Path) -> Path:
    return project_root / "analysis/phantoms/ogse_experiments/alpha_macro/N1/summary_alpha_values.csv"


def _default_phantoms_delta(project_root: Path) -> Path:
    return (
        project_root
        / "analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr_D0=2.3"
        / "tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build publication-ready delta-vs-ROI and alpha-vs-ROI panels for OGSE brain and phantom results."
    )
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <project-root>/analysis/publication_figures/delta_alpha_ogse",
    )
    ap.add_argument("--output-prefix", default="delta_alpha_ogse_summary")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats, usually png pdf.")
    ap.add_argument("--plot-modes", nargs="+", choices=["points", "bars"], default=["points", "bars"])
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--no-separate", action="store_true", help="Only write the combined brain+phantom figure.")
    ap.add_argument("--delta-ymin", type=float, default=None, help="Optional shared lower y-limit for delta panels.")
    ap.add_argument("--phantoms-delta-ymin", type=float, default=1400.0, help="Lower y-limit for phantom delta panels.")
    ap.add_argument("--alpha-ymin", type=float, default=0.0, help="Shared lower y-limit for alpha panels.")
    ap.add_argument("--brains-alpha-ymax", type=float, default=0.4, help="Upper y-limit for brain alpha panels.")

    ap.add_argument("--brains-alpha-table", type=Path, default=None)
    ap.add_argument("--brains-delta-table", type=Path, default=None)
    ap.add_argument("--phantoms-alpha-table", type=Path, default=None)
    ap.add_argument("--phantoms-delta-table", type=Path, default=None)

    ap.add_argument("--brains-rois", nargs="+", default=["AntCC", "MidAntCC", "CentralCC", "MidPostCC", "PostCC"])
    ap.add_argument("--phantoms-rois", nargs="+", default=["fiber1", "fiber2"])
    ap.add_argument("--directions", nargs="+", default=["longitudinal", "transversal"], help="Final direction labels and plotting order.")
    ap.add_argument(
        "--brains-direction-alias",
        action="append",
        default=None,
        help="Repeatable source=target alias. Defaults include long=longitudinal, tra=transversal.",
    )
    ap.add_argument(
        "--phantoms-direction-alias",
        action="append",
        default=None,
        help="Repeatable source=target alias. Defaults include 1=transversal, 2=transversal, 3=longitudinal.",
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (project_root / "analysis/publication_figures/delta_alpha_ogse")

    brain_aliases = _parse_aliases(
        args.brains_direction_alias,
        {
            "long": "longitudinal",
            "tra": "transversal",
            "x": "longitudinal",
            "y": "transversal",
            "z": "transversal",
        },
    )
    phantom_aliases = _parse_aliases(
        args.phantoms_direction_alias,
        {"1": "transversal", "2": "transversal", "3": "longitudinal"},
    )

    specs = [
        DatasetFigureSpec(
            name="brains_ogse",
            label="BRAINS",
            alpha_table=args.brains_alpha_table or _default_brains_alpha(project_root),
            delta_table=args.brains_delta_table or _default_brains_delta(project_root),
            roi_order=tuple(args.brains_rois),
            direction_order=tuple(args.directions),
            direction_aliases=brain_aliases,
        ),
        DatasetFigureSpec(
            name="phantoms_ogse",
            label="PHANTOMS",
            alpha_table=args.phantoms_alpha_table or _default_phantoms_alpha(project_root),
            delta_table=args.phantoms_delta_table or _default_phantoms_delta(project_root),
            roi_order=tuple(args.phantoms_rois),
            direction_order=tuple(args.directions),
            direction_aliases=phantom_aliases,
        ),
    ]

    raw, agg = build_plot_tables(specs)
    table_paths = write_plot_tables(raw, agg, out_dir, prefix=args.output_prefix)

    outputs = []
    for mode in args.plot_modes:
        combined_stem = out_dir / f"{args.output_prefix}_{mode}"
        outputs.extend(
            plot_delta_alpha_figure(
                raw,
                agg,
                specs,
                out_stem=combined_stem,
                formats=args.formats,
                dpi=int(args.dpi),
                delta_ymin=None if args.delta_ymin is None else float(args.delta_ymin),
                phantoms_delta_ymin=None if args.phantoms_delta_ymin is None else float(args.phantoms_delta_ymin),
                alpha_ymin=float(args.alpha_ymin),
                brains_alpha_ymax=None if args.brains_alpha_ymax is None else float(args.brains_alpha_ymax),
                plot_mode=mode,
            )
        )
        if mode == "points":
            outputs.extend(
                plot_delta_alpha_figure(
                    raw,
                    agg,
                    specs,
                    out_stem=out_dir / args.output_prefix,
                    formats=args.formats,
                    dpi=int(args.dpi),
                    delta_ymin=None if args.delta_ymin is None else float(args.delta_ymin),
                    phantoms_delta_ymin=None if args.phantoms_delta_ymin is None else float(args.phantoms_delta_ymin),
                    alpha_ymin=float(args.alpha_ymin),
                    brains_alpha_ymax=None if args.brains_alpha_ymax is None else float(args.brains_alpha_ymax),
                    plot_mode=mode,
                )
            )

        if args.no_separate:
            continue
        for spec in specs:
            stem = out_dir / f"delta_alpha_{spec.name}_{mode}"
            outputs.extend(
                plot_delta_alpha_figure(
                    raw,
                    agg,
                    [spec],
                    out_stem=stem,
                    formats=args.formats,
                    dpi=int(args.dpi),
                    figsize=(5.5, 6.8),
                    delta_ymin=None if args.delta_ymin is None else float(args.delta_ymin),
                    phantoms_delta_ymin=None if args.phantoms_delta_ymin is None else float(args.phantoms_delta_ymin),
                    alpha_ymin=float(args.alpha_ymin),
                    brains_alpha_ymax=None if args.brains_alpha_ymax is None else float(args.brains_alpha_ymax),
                    plot_mode=mode,
                )
            )
            if mode == "points":
                outputs.extend(
                    plot_delta_alpha_figure(
                        raw,
                        agg,
                        [spec],
                        out_stem=out_dir / f"delta_alpha_{spec.name}",
                        formats=args.formats,
                        dpi=int(args.dpi),
                        figsize=(5.5, 6.8),
                        delta_ymin=None if args.delta_ymin is None else float(args.delta_ymin),
                        phantoms_delta_ymin=None if args.phantoms_delta_ymin is None else float(args.phantoms_delta_ymin),
                        alpha_ymin=float(args.alpha_ymin),
                        brains_alpha_ymax=None if args.brains_alpha_ymax is None else float(args.brains_alpha_ymax),
                        plot_mode=mode,
                    )
                )

    for path in table_paths:
        print(f"[OK] table: {path}")
    for path in outputs:
        print(f"[OK] figure: {path}")


if __name__ == "__main__":
    main()
