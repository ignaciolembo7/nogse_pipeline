from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _analysis_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _split_all_or_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    if len(values) == 1 and str(values[0]).upper() == "ALL":
        return None
    return [str(v) for v in values]


def _unique_scalar(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    values = pd.Series(df[col]).dropna().unique().tolist()
    if len(values) != 1:
        return None
    return values[0]


def _plot_one_group(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    xcol: str,
    ycol: str,
    out_png: Path,
    title: str,
) -> None:
    data = avg_df.copy()
    data[xcol] = pd.to_numeric(data[xcol], errors="coerce")
    data[ycol] = pd.to_numeric(data[ycol], errors="coerce")
    data = data.dropna(subset=[xcol, ycol]).sort_values(xcol)
    if data.empty:
        return

    sigma = None
    if std_df is not None and not std_df.empty:
        err = std_df.copy()
        err[xcol] = pd.to_numeric(err[xcol], errors="coerce")
        err["value"] = pd.to_numeric(err["value"], errors="coerce")
        err = err.dropna(subset=[xcol, "value"]).sort_values(xcol)
        if not err.empty:
            if ycol == "value_norm":
                s0 = pd.to_numeric(data["S0"], errors="coerce")
                sigma = err["value"].to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma = sigma / s0.to_numpy(dtype=float)
            else:
                sigma = err["value"].to_numpy(dtype=float)
            if sigma.shape[0] != data.shape[0]:
                sigma = None

    plt.figure(figsize=(8, 6))
    if sigma is not None:
        plt.errorbar(
            data[xcol],
            data[ycol],
            yerr=sigma,
            fmt="o-",
            linewidth=2,
            markersize=6,
            capsize=3,
            label="signal",
        )
    else:
        plt.plot(data[xcol], data[ycol], "o-", linewidth=2, markersize=6, label="signal")

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("signal_parquet", type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g")
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    args = ap.parse_args()

    rois = _split_all_or_values(args.rois)
    directions = _split_all_or_values(args.directions)

    df = pd.read_parquet(args.signal_parquet)
    analysis_id = _analysis_id_from_path(args.signal_parquet)

    avg_df = df[df["stat"].astype(str) == str(args.stat)].copy()
    if avg_df.empty:
        raise SystemExit(f"No rows found for stat={args.stat!r} in {args.signal_parquet}.")

    std_df = df[df["stat"].astype(str) == "std"].copy()

    if rois is not None:
        avg_df = avg_df[avg_df["roi"].astype(str).isin(rois)].copy()
        std_df = std_df[std_df["roi"].astype(str).isin(rois)].copy()
    if directions is not None:
        avg_df = avg_df[avg_df["direction"].astype(str).isin(directions)].copy()
        std_df = std_df[std_df["direction"].astype(str).isin(directions)].copy()

    if avg_df.empty:
        raise SystemExit("No data remains after ROI/direction filtering.")

    out_dir = args.out_root / analysis_id

    for (roi, direction), group in avg_df.groupby(["roi", "direction"], sort=False):
        std_group = None
        if not std_df.empty:
            std_group = std_df[
                (std_df["roi"].astype(str) == str(roi))
                & (std_df["direction"].astype(str) == str(direction))
            ].copy()

        tn_value = _unique_scalar(group, "TN")
        n_value = _unique_scalar(group, "N")
        signal_type = _unique_scalar(group, "type")
        title = (
            f"{analysis_id} | roi={roi} | direction={direction} | "
            f"type={signal_type} | TN={tn_value} | N={n_value}"
        )
        out_png = out_dir / f"{analysis_id}.roi-{roi}.dir-{direction}.{args.ycol}_vs_{args.xcol}.png"
        _plot_one_group(
            group,
            std_group,
            xcol=args.xcol,
            ycol=args.ycol,
            out_png=out_png,
            title=title,
        )
        print("Saved:", out_png)


if __name__ == "__main__":
    main()
