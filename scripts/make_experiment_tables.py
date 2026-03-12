
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from data_processing_20220607_10_GONZANY.experiment_tables import (
    MetaVectors,
    format_name,
    make_tables_for_experiment,
    parse_meta_vectors,
    read_bvals_txt,
    read_results_csv,
    split_experiments,
    validate_meta_lengths,
    write_experiment_xlsx,
)


def _parse_stats(s: str) -> List[str]:
    # Accept comma-separated list: "Mean,Min,Max,Area"
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split Results.csv into one Excel file per experiment (chunk), using per-experiment b-values."
    )
    p.add_argument("results_csv", type=str, help="Path to Results.csv (stacked experiments).")
    p.add_argument("bvals_txt", type=str, help="Path to b-values txt (one line per experiment).")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    p.add_argument("--chunk-size", type=int, default=32, help="Rows per experiment (default: 32).")
    p.add_argument(
        "--stats",
        type=str,
        default="Mean,Min,Max,Area",
        help="Comma-separated stats to export as sheets, e.g. 'Mean,Min,Max,Area' (default).",
    )

    p.add_argument(
        "--base-name",
        type=str,
        default="EXP",
        help="Base name used in output files (you can paste your long Siemens-like prefix here).",
    )
    p.add_argument(
        "--name-template",
        type=str,
        default="{base}_exp{exp:02d}_d{d}_Delta{Delta}_Hz{Hz}_maxdur{max_dur}_b{bmax}_results.xlsx",
        help="Python format template for the output filename. Fields: base, exp, d, Delta, Hz, bmax",
    )

    # Meta vectors: user supplies one value per experiment, comma/space-separated.
    p.add_argument("--d", type=str, default=None, help="Comma/space-separated vector with d per experiment.")
    p.add_argument("--Delta", type=str, default=None, help="Comma/space-separated vector with Delta per experiment.")
    p.add_argument("--Hz", type=str, default=None, help="Comma/space-separated vector with Hz per experiment.")
    p.add_argument("--max_dur", type=str, default=None, help="Comma/space-separated vector with max_dur per experiment.")

    args = p.parse_args()

    df = read_results_csv(args.results_csv)
    chunks = split_experiments(df, chunk_size=args.chunk_size)

    bvals_lines = read_bvals_txt(args.bvals_txt)
    if len(bvals_lines) != len(chunks):
        raise SystemExit(
            f"bvals file has {len(bvals_lines)} lines but Results.csv implies {len(chunks)} experiments "
            f"({len(df)} rows / {args.chunk_size})."
        )

    meta = parse_meta_vectors(d=args.d, Delta=args.Delta, Hz=args.Hz, max_dur=args.max_dur)
    validate_meta_lengths(len(chunks), meta)

    out_dir = Path(args.out_dir)
    group_dir = out_dir / Path(args.bvals_txt).resolve().parent.name
    group_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = _parse_stats(args.stats)

    for i, (chunk, bvals) in enumerate(zip(chunks, bvals_lines), start=1):
        # bmax computed from this experiment's bvals
        bmax = max(bvals) if bvals else None
        d = meta.d[i-1] if meta.d else None
        Delta = meta.Delta[i-1] if meta.Delta else None
        Hz = meta.Hz[i-1] if meta.Hz else None
        max_dur = meta.max_dur[i-1] if getattr(meta, "max_dur", None) else None


        tables = make_tables_for_experiment(chunk, bvals=bvals, stats=stats)

        fname = format_name(
            args.name_template,
            base=args.base_name,
            exp=i,
            d=d,
            Delta=Delta,
            Hz=Hz,
            max_dur=max_dur,
            bmax=bmax,
        )
        out_path = group_dir / fname
        write_experiment_xlsx(out_path, tables)
        
    print(f"Done. Wrote {len(chunks)} files into: {out_dir}")


if __name__ == "__main__":
    main()
