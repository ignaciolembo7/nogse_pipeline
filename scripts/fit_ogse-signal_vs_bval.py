from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from monoexp_fitting.fit_signal_vs_bval import run_fit_from_parquet


def parse_dir_map(s: str | None):
    if not s:
        return None
    out = {}
    for chunk in s.split(","):
        k, v = chunk.split(":")
        out[k.strip()] = v.strip()
    return out


def _unique_float_any(df: pd.DataFrame, cols: list[str], *, required: bool, name: str) -> float | None:
    for c in cols:
        if c in df.columns:
            u = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(u) == 1:
                return float(u[0])
    if required:
        raise ValueError(f"No pude inferir {name}. Probé columnas: {cols}")
    return None


def _infer_overrides_from_parquet(parquet_path: str | Path) -> dict[str, float | None]:
    df = pd.read_parquet(parquet_path)

    if "axis" in df.columns:
        raise ValueError("Encontré columna 'axis'. Este pipeline usa SOLO 'direction'.")

    N = _unique_float_any(df, ["N"], required=False, name="N")
    delta_ms = _unique_float_any(df, ["delta_ms"], required=False, name="delta_ms")
    Delta_app_ms = _unique_float_any(df, ["Delta_app_ms", "delta_app_ms"], required=False, name="Delta_app_ms")

    td_ms = _unique_float_any(df, ["td_ms"], required=False, name="td_ms")
    if td_ms is None:
        max_dur_ms = _unique_float_any(df, ["max_dur_ms"], required=False, name="max_dur_ms")
        tm_ms = _unique_float_any(df, ["tm_ms"], required=False, name="tm_ms")
        if max_dur_ms is not None and tm_ms is not None:
            td_ms = 2.0 * float(max_dur_ms) + float(tm_ms)

    return {"td_ms": td_ms, "N": N, "delta_ms": delta_ms, "Delta_app_ms": Delta_app_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="Archivo .long.parquet (tabla de señal limpia)")

    # keep compat: --dirs / --direction / --directions
    ap.add_argument("--directions", nargs="+", default=None, help="Direcciones a fitear (ej: 1 2 3 o x y z)")
    ap.add_argument("--direction", nargs="+", dest="directions", help="Alias de --directions")
    ap.add_argument("--dirs", nargs="+", dest="directions", help="Alias legacy de --directions")

    ap.add_argument("--dir-map", default=None, help='Mapeo de valores direction. Ej: "1:x,2:y,3:z"')

    ap.add_argument("--rois", nargs="+", default=["ALL"])
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--fit_points", type=int, default=6)

    ap.add_argument(
        "--g_type",
        default="bvalue",
        choices=["bvalue", "g", "g_max", "g_lin_max", "g_thorsten", "gthorsten"],
    )
    ap.add_argument("--gamma", type=float, default=267.5221900)

    ap.add_argument("--td_ms", type=float, default=None)
    ap.add_argument("--N", type=float, default=None)
    ap.add_argument("--delta_ms", type=float, default=None)
    ap.add_argument("--Delta_app_ms", type=float, default=None)

    ap.add_argument("--D0_init", type=float, default=0.0023)
    ap.add_argument("--fix_M0", type=float, default=1.0)
    ap.add_argument("--free_M0", action="store_true")

    # ✅ nuevo default coherente con tu layout final
    ap.add_argument("--out_root", default="ogse_experiments/fits/fit-monoexp_ogse-signal")
    ap.add_argument("--stat", default="avg")

    args = ap.parse_args()

    inferred = _infer_overrides_from_parquet(args.parquet)

    td_ms = args.td_ms if args.td_ms is not None else inferred["td_ms"]
    N = args.N if args.N is not None else inferred["N"]
    delta_ms = args.delta_ms if args.delta_ms is not None else inferred["delta_ms"]
    Delta_app_ms = args.Delta_app_ms if args.Delta_app_ms is not None else inferred["Delta_app_ms"]

    g_type = args.g_type
    if g_type == "gthorsten":
        g_type = "g_thorsten"

    rois = args.rois if args.rois != ["ALL"] else ["ALL"]

    for roi in rois:
        run_fit_from_parquet(
            args.parquet,
            dir_map=parse_dir_map(args.dir_map),
            dirs=args.directions,
            roi=roi,
            ycol=args.ycol,
            g_type=g_type,
            fit_points=args.fit_points,
            free_M0=args.free_M0,
            fix_M0=args.fix_M0,
            D0_init=args.D0_init,
            gamma=args.gamma,
            td_ms=td_ms,
            N=N,
            delta_ms=delta_ms,
            Delta_app_ms=Delta_app_ms,
            stat_keep=args.stat,
            out_root=args.out_root,
        )


if __name__ == "__main__":
    main()