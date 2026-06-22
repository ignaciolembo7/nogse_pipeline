from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from plotting.publication.tc_param_vars import (
    TcParamVarsSpec,
    default_delta_vs_alpha_output_stem,
    default_output_stem,
    plot_delta_vs_alpha_publication,
    plot_tc_param_vars_publication,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent


def _parse_aliases(items: list[str] | None, defaults: dict[str, str]) -> dict[str, str]:
    out = dict(defaults)
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid alias {raw!r}. Use source=target.")
        source, target = raw.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid alias {raw!r}. Use source=target.")
        out[source] = target
    return out


def _default_brains_table(project_root: Path) -> Path:
    return (
        project_root
        / "analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr"
        / "tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx"
    )


def _default_phantoms_table(project_root: Path) -> Path:
    return (
        project_root
        / "analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr"
        / "tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build publication-ready pseudo-Huber parameter panels from tc_peak_resampled_data_ms tables."
        )
    )
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument(
        "--plot-types",
        nargs="+",
        choices=["vars", "delta-alpha"],
        default=["vars", "delta-alpha"],
        help="Publication figures to build.",
    )
    ap.add_argument(
        "--variables",
        nargs="+",
        default=["q_quad", "alpha_macro", "delta", "A", "c", "sqrt_q"],
        help="Rows to plot. Defaults reproduce the vars_sameY panel.",
    )
    ap.add_argument("--no-brains", action="store_true", help="Skip the brain figure.")
    ap.add_argument("--no-phantoms", action="store_true", help="Skip the phantom figure.")
    ap.add_argument("--no-errorbars", action="store_true", help="Draw lines and markers without error bars.")
    ap.add_argument("--no-point-labels", action="store_true", help="Hide ROI text labels in delta-alpha panels.")

    ap.add_argument("--brains-table", type=Path, default=None)
    ap.add_argument("--brains-rois", nargs="+", default=["AntCC", "MidAntCC", "CentralCC", "MidPostCC", "PostCC"])
    ap.add_argument("--brains-directions", nargs="+", default=["longitudinal", "transversal"])
    ap.add_argument("--brains-direction-alias", action="append", default=None)
    ap.add_argument("--brains-subject-alias", action="append", default=["BRAIN=subj1", "LUDG=subj2", "MBBL=subj3"])

    ap.add_argument("--phantoms-table", type=Path, default=None)
    ap.add_argument("--phantoms-rois", nargs="+", default=["fiber1", "fiber2"])
    ap.add_argument("--phantoms-directions", nargs="+", default=["longitudinal", "transversal"])
    ap.add_argument("--phantoms-direction-alias", action="append", default=None)
    ap.add_argument("--phantoms-subject-alias", action="append", default=["PHANTOM3=subj1"])
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (project_root / "analysis/publication_figures/tc_param_vars")

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

    specs: list[TcParamVarsSpec] = []
    if not args.no_brains:
        specs.append(
            TcParamVarsSpec(
                name="brains",
                label="Brains",
                table=args.brains_table or _default_brains_table(project_root),
                roi_order=tuple(args.brains_rois),
                direction_order=tuple(args.brains_directions),
                direction_aliases=brain_aliases,
                subject_aliases=_parse_aliases(args.brains_subject_alias, {}),
            )
        )
    if not args.no_phantoms:
        specs.append(
            TcParamVarsSpec(
                name="phantoms",
                label="Phantoms",
                table=args.phantoms_table or _default_phantoms_table(project_root),
                roi_order=tuple(args.phantoms_rois),
                direction_order=tuple(args.phantoms_directions),
                direction_aliases=phantom_aliases,
                subject_aliases=_parse_aliases(args.phantoms_subject_alias, {}),
            )
        )

    outputs: list[Path] = []
    for spec in specs:
        if "vars" in args.plot_types:
            outputs.extend(
                plot_tc_param_vars_publication(
                    spec,
                    out_stem=default_output_stem(out_dir, spec, args.variables),
                    variables=args.variables,
                    formats=args.formats,
                    dpi=int(args.dpi),
                    show_errorbars=not bool(args.no_errorbars),
                )
            )
        if "delta-alpha" in args.plot_types:
            outputs.extend(
                plot_delta_vs_alpha_publication(
                    spec,
                    out_stem=default_delta_vs_alpha_output_stem(out_dir, spec),
                    formats=args.formats,
                    dpi=int(args.dpi),
                    show_errorbars=not bool(args.no_errorbars),
                    label_points=not bool(args.no_point_labels),
                )
            )

    for path in outputs:
        print(f"[OK] figure: {path}")


if __name__ == "__main__":
    main()
