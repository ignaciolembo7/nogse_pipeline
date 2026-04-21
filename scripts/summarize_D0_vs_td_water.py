from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.io import write_xlsx_csv_outputs


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _read_all_fit_params(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("fit_params.csv"))
    if not files:
        raise SystemExit(f"Could not find fit_params.csv under: {root}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)

        # Normalize D0 column.
        if "D0_mm2_s" not in df.columns:
            d0c = _pick_col(df, ["D0_mm2_s", "D0", "D0_mm2_per_s"])
            if d0c is None:
                continue
            df = df.rename(columns={d0c: "D0_mm2_s"})

        # Normalize max_dur column.
        if "max_dur_ms" not in df.columns:
            mdc = _pick_col(df, ["max_dur_ms", "max_dur", "max_duration", "meta_max_dur", "param_max_dur"])
            if mdc is not None:
                df = df.rename(columns={mdc: "max_dur_ms"})

        # Traceability.
        df["fit_params_file"] = str(fp)
        df["fit_folder"] = fp.parent.name

        frames.append(df)

    if not frames:
        raise SystemExit("Found fit_params.csv files, but none had a usable D0 column.")

    out = pd.concat(frames, ignore_index=True)
    return out


def _relevant_cols(df: pd.DataFrame) -> list[str]:
    """
    Relevant columns follow the usual fit_signal_vs_bval.py output:
      direction, roi, g_type, fit_points, M0, M0_err, D0_mm2_s, D0_err, delta_ms, delta_app_ms, N, max_dur_ms
    plus any existing param_* / meta_* columns.
    """
    preferred = [
        "max_dur_ms",
        "roi",
        "direction",
        "g_type",
        "fit_points",
        "N",
        "delta_ms",
        "delta_app_ms",
        "M0",
        "M0_err",
        "D0_mm2_s",
        "D0_err_mm2_s",
        "D0_err",
        "stat",
        "fit_folder",
        "fit_params_file",
    ]
    cols = [c for c in preferred if c in df.columns]

    # Add param_*/meta_* columns when present.
    extra = [c for c in df.columns if (c.startswith("param_") or c.startswith("meta_")) and c not in cols]
    cols.extend(sorted(extra))

    # Finally append the remaining columns to preserve all context.
    rest = [c for c in df.columns if c not in cols]
    cols.extend(sorted(rest))

    return cols


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect D0 from all fit_params.csv files, then average by max_dur and plot D0 vs td."
    )
    ap.add_argument("--root", type=Path, default=None, help="root directory")
    ap.add_argument("--roi", type=str, default="Agua", help="ROI to filter. Use ALL to skip filtering.")
    ap.add_argument("--out_root", type=Path, default=None, help="out_root directory")
    ap.add_argument("--tm-ms", type=float, default=None, help="tm (ms). If omitted, it is requested interactively.")
    ap.add_argument("--use-std", action="store_true", help="Error bars = STD. Defaults to SEM.")
    args = ap.parse_args()

    root = args.root
    out = args.out_root or (root / "D0_summary")
    out.mkdir(parents=True, exist_ok=True)

    df = _read_all_fit_params(root)

    # Clean numeric types.
    df["D0_mm2_s"] = pd.to_numeric(df["D0_mm2_s"], errors="coerce")
    if "max_dur_ms" in df.columns:
        df["max_dur_ms"] = pd.to_numeric(df["max_dur_ms"], errors="coerce")

    df = df[np.isfinite(df["D0_mm2_s"])].copy()
    if "max_dur_ms" not in df.columns:
        raise SystemExit("Could not find max_dur_ms in fit_params.csv files, so ordering/grouping is not possible.")
    df = df[np.isfinite(df["max_dur_ms"])].copy()

    # ROI filter.
    if args.roi.upper() != "ALL" and "roi" in df.columns:
        df = df[df["roi"].astype(str) == str(args.roi)].copy()

    if df.empty:
        raise SystemExit("No rows remained after filtering. Check the ROI.")

    # Table A: all measurements sorted by max_dur.
    df_all = df[_relevant_cols(df)].sort_values(["max_dur_ms"], kind="stable").reset_index(drop=True)
    write_xlsx_csv_outputs(df_all, out / "D0_all_measurements.xlsx", csv_path=out / "D0_all_measurements.csv")

    # Table B: mean by max_dur.
    g = df.groupby("max_dur_ms", as_index=False)
    summary = g["D0_mm2_s"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "D0_mean_mm2_s", "std": "D0_std_mm2_s", "count": "n"})
    summary["D0_sem_mm2_s"] = summary["D0_std_mm2_s"] / np.sqrt(summary["n"].clip(lower=1))
    summary = summary.sort_values("max_dur_ms", kind="stable").reset_index(drop=True)

    # Read tm interactively when it was not provided as an argument.
    tm_ms = args.tm_ms
    if tm_ms is None:
        tm_ms = float(input("Enter tm (ms): ").strip())

    # td calculation and plot.
    summary["tm_ms"] = float(tm_ms)
    summary["td_ms"] = 2.0 * summary["max_dur_ms"] + float(tm_ms)

    write_xlsx_csv_outputs(summary, out / "D0_mean_by_maxdur.xlsx", csv_path=out / "D0_mean_by_maxdur.csv")

    x = summary["td_ms"].to_numpy(float)
    y = summary["D0_mean_mm2_s"].to_numpy(float)
    yerr = summary["D0_std_mm2_s"].to_numpy(float) if args.use_std else summary["D0_sem_mm2_s"].to_numpy(float)

    plt.figure(figsize=(7.5, 5.5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel("td_ms = 2*max_dur + tm")
    plt.ylabel("D0 (mm²/s)")
    plt.title(f"D0 vs td (ROI={args.roi}, tm={tm_ms} ms)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "D0_vs_td.png", dpi=200)
    plt.close()

    print(f"[OK] Table A (all):     {out / 'D0_all_measurements.xlsx'}")
    print(f"[OK] Table B (mean):    {out / 'D0_mean_by_maxdur.xlsx'}")
    print(f"[OK] Plot D0 vs td:     {out / 'D0_vs_td.png'}")


if __name__ == "__main__":
    main()
