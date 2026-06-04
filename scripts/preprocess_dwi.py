#!/usr/bin/env python
"""CLI entry point for DWI preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from preprocessing.workflow import (
    PreprocessingConfig,
    default_steps_for_dataset,
    run_preprocessing,
    supported_steps,
)


def _parse_subjects(values: list[str]) -> tuple[str, ...]:
    subjects: list[str] = []
    for value in values:
        subjects.extend(item.strip() for item in value.split(",") if item.strip())
    if not subjects:
        raise argparse.ArgumentTypeError("At least one subject is required")
    return tuple(subjects)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run DWI preprocessing for NOGSE/OGSE inputs.")
    parser.add_argument("--dataset", choices=("brain", "brains", "phantom", "phantoms"), required=True)
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs, separated by spaces or commas.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing converted NIfTI inputs.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root for preprocessed NIfTI outputs.")
    parser.add_argument("--steps", nargs="+", choices=supported_steps(), help="Selected preprocessing steps.")
    parser.add_argument("--session", default="ses-T0", help="BIDS-like session folder used when present.")
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing external tools.")
    parser.add_argument("--dwi-name", help="Override the default subject DWI file name.")
    parser.add_argument("--bval-name", help="Override the default subject bval file name.")
    parser.add_argument("--bvec-name", help="Override the default subject bvec file name.")
    parser.add_argument("--ap-json-name", help="Override the default AP/DWI JSON file name.")
    parser.add_argument("--pa-b0-name", help="Reverse phase-encoded b0 image file name.")
    parser.add_argument("--pa-json-name", help="Reverse phase-encoded b0 JSON file name.")
    parser.add_argument("--ap-b0-vols", default="0,1", help="AP b0 volume indices passed to fslselectvols.")
    parser.add_argument("--pa-b0-vols", default="0,1", help="PA b0 volume indices passed to fslselectvols.")
    parser.add_argument("--slspec-name", help="Optional existing FSL slspec file name for eddy.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    subjects = _parse_subjects(args.subjects)
    steps = tuple(args.steps) if args.steps else default_steps_for_dataset(args.dataset)

    if args.nthreads <= 0:
        parser.error("--nthreads must be positive")

    config = PreprocessingConfig(
        dataset=args.dataset,
        subjects=subjects,
        steps=steps,
        input_root=args.input_root,
        output_root=args.output_root,
        nthreads=args.nthreads,
        session=args.session,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        dwi_name=args.dwi_name,
        bval_name=args.bval_name,
        bvec_name=args.bvec_name,
        ap_json_name=args.ap_json_name,
        pa_b0_name=args.pa_b0_name,
        pa_json_name=args.pa_json_name,
        ap_b0_vols=args.ap_b0_vols,
        pa_b0_vols=args.pa_b0_vols,
        slspec_name=args.slspec_name,
    )

    try:
        run_preprocessing(config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
