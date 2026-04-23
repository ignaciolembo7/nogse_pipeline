from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data_processing.io import write_xlsx_sheets


BVALUES_32 = [
    0, 0,
    15, 15, 15,
    60, 60, 60,
    135, 135, 135,
    240, 240, 240,
    375, 375, 375,
    535, 535, 540,
    730, 730, 735,
    955, 955, 955,
    1205, 1205, 1210,
    1490, 1490, 1495,
]

DEFAULT_DELTAS = [36, 42, 48, 54, 60, 66, 72, 78, 84, 89.6]


def ffloat(x: float) -> str:
    """Convert a float to a short filename token (89.6 -> 89p6)."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Results.csv")
    ap.add_argument("--out_dir", default="Results_converted", help="Carpeta destino")
    ap.add_argument("--exp", default="CSVSET", help="Prefijo/experimento (ej 20230619_BRAIN-3)")
    ap.add_argument("--Hz", type=float, default=0.0)
    ap.add_argument("--delta_ms", type=float, default=10.0)
    ap.add_argument("--deltas", default=",".join(str(x) for x in DEFAULT_DELTAS),
                    help="Lista Delta_app (ms) separada por coma, ej: 36,42,48,...,89.6")
    ap.add_argument("--ndirs", type=int, default=3)
    ap.add_argument("--nbvals", type=int, default=10)
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = [float(x) for x in args.deltas.split(",")]

    df = pd.read_csv(csv_path)

    # Detect ROI columns.
    rois = ["Fibra1", "Agua", "Fibra2"]
    cols_avg = {r: f"Mean({r})" for r in rois}
    cols_min = {r: f"Min({r})" for r in rois}
    cols_max = {r: f"Max({r})" for r in rois}

    for k in list(cols_avg.values()) + list(cols_min.values()) + list(cols_max.values()):
        if k not in df.columns:
            raise ValueError(f"Missing CSV column: {k}. Available columns: {list(df.columns)}")

    rows_per_block = len(BVALUES_32)
    if len(df) != rows_per_block * len(deltas):
        raise ValueError(
            f"Rowcount no coincide: tengo {len(df)} filas, esperaba {rows_per_block*len(deltas)} "
            f"({len(deltas)} bloques × {rows_per_block} filas)."
        )

    # bmax used for naming (rounded pattern maximum).
    bmax = int(round(max(BVALUES_32)))

    for i, Delta_app in enumerate(deltas, start=1):
        block = df.iloc[(i-1)*rows_per_block : i*rows_per_block].copy()
        block.insert(0, "bvalues", BVALUES_32)

        avg_df = pd.DataFrame({"bvalues": block["bvalues"]})
        min_df = pd.DataFrame({"bvalues": block["bvalues"]})
        max_df = pd.DataFrame({"bvalues": block["bvalues"]})

        for r in rois:
            avg_df[r] = pd.to_numeric(block[cols_avg[r]], errors="coerce")
            min_df[r] = pd.to_numeric(block[cols_min[r]], errors="coerce")
            max_df[r] = pd.to_numeric(block[cols_max[r]], errors="coerce")

        # Short but unique filename.
        # IMPORTANT: the pipeline parses:
        #   *_10bval_*_03dir_*_dXXpY_*_Hz000_*_b1495_*_{seq}_results.xlsx
        fname = (
            f"{args.exp}_PGSE_"
            f"{args.nbvals:02d}bval_{args.ndirs:02d}dir_"
            f"d{ffloat(Delta_app)}_delta{ffloat(args.delta_ms)}_"
            f"Hz{int(round(args.Hz)):03d}_b{bmax:04d}_"
            f"{i}_results.xlsx"
        )

        out_path = out_dir / fname

        info = pd.DataFrame([{
            "exp": args.exp,
            "seq": i,
            "Hz": args.Hz,
            "bmax": bmax,
            "d_ms": Delta_app,
            "delta_ms": args.delta_ms,
            "ndirs": args.ndirs,
            "nbvals": args.nbvals,
            "notes": "Generated from Results.csv with a fixed bvalue pattern (32 rows per Delta_app).",
        }])
        write_xlsx_sheets(
            {
                "avg": avg_df,
                "min": min_df,
                "max": max_df,
                "info": info,
            },
            out_path,
        )

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
