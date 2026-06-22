from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from plotting.publication.pseudohuber_simulation import (
    default_output_stem,
    plot_pseudohuber_transition_publication,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent


def _parse_figsize(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use WIDTH,HEIGHT or WIDTHxHEIGHT.")
    try:
        width, height = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Figure size values must be numeric.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Figure size values must be positive.")
    return width, height


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a square publication-ready tc vs Td pseudo-Huber simulation figure "
            "showing the large-Td linear asymptote for different transition deltas."
        )
    )
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-stem", type=Path, default=None)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--figsize", type=_parse_figsize, default=(5.6, 5.6), help="Figure size in inches, e.g. 5.6,5.6.")
    ap.add_argument("--c", type=float, default=20.0, help="Native pseudo-Huber c = tc(Td=0), in ms.")
    ap.add_argument("--alpha-macro", type=float, default=0.55, help="Shared large-Td slope.")
    ap.add_argument("--deltas-ms", nargs="+", type=float, default=[35.0, 60.0, 85.0], help="Transition deltas in ms.")
    ap.add_argument("--td-min-ms", type=float, default=0.0)
    ap.add_argument("--td-max-ms", type=float, default=180.0)
    ap.add_argument("--n-points", type=int, default=600)
    ap.add_argument(
        "--linear-start-factor",
        type=float,
        default=1.45,
        help="Draw the dotted large-Td approximation only for Td >= factor * delta.",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--xlabel", default=r"Diffusion time $T_d / \delta$ [dimensionless]")
    ap.add_argument("--ylabel", default=r"Correlation time $\tau_c / \delta$ [dimensionless]")
    ap.add_argument(
        "--common-slope-guide",
        action="store_true",
        help="Add a shared c + alpha*Td guide line. The pseudo-Huber curves still use the same native c at Td=0.",
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (project_root / "analysis/publication_figures/pseudohuber_simulation")
    out_stem = args.out_stem or default_output_stem(out_dir)

    outputs = plot_pseudohuber_transition_publication(
        out_stem=out_stem,
        c=float(args.c),
        alpha_macro=float(args.alpha_macro),
        deltas_ms=args.deltas_ms,
        td_min_ms=float(args.td_min_ms),
        td_max_ms=float(args.td_max_ms),
        n_points=int(args.n_points),
        formats=args.formats,
        dpi=int(args.dpi),
        figsize=args.figsize,
        xlabel=str(args.xlabel),
        ylabel=str(args.ylabel),
        title=args.title,
        show_common_slope_guide=bool(args.common_slope_guide),
        linear_start_factor=float(args.linear_start_factor),
    )

    for path in outputs:
        print(f"[OK] figure: {path}")


if __name__ == "__main__":
    main()
