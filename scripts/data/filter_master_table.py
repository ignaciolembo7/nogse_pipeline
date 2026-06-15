from __future__ import annotations

import argparse
from pathlib import Path

import repo_bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from data_processing.io import write_table_outputs
from data_processing.master_table import load_master_table


def parse_first_points_by_td(spec: str) -> dict[float, int | None]:
    out: dict[float, int | None] = {}
    for token in str(spec).replace(",", " ").split():
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected TD=POINTS, received {token!r}.")
        raw_td, raw_points = token.split("=", 1)
        td_ms = float(raw_td)
        points_text = raw_points.strip()
        if points_text.upper() == "ALL":
            out[td_ms] = None
            continue
        points = int(points_text)
        if points <= 0:
            raise ValueError(f"First-points limit for td_ms={td_ms:g} must be > 0.")
        out[td_ms] = points
    return out


def filter_first_points_by_td(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    limits = parse_first_points_by_td(spec)
    if not limits:
        return df.copy().reset_index(drop=True)
    if "td_ms" not in df.columns:
        raise KeyError("Cannot filter first points because column 'td_ms' is missing.")
    if "b_step" not in df.columns:
        raise KeyError("Cannot filter first points because column 'b_step' is missing.")

    out = df.copy()
    keep = pd.Series(True, index=out.index)
    td_values = pd.to_numeric(out["td_ms"], errors="coerce")
    point_values = pd.to_numeric(out["b_step"], errors="coerce")

    for td_ms, point_limit in limits.items():
        if point_limit is None:
            continue
        td_mask = np.isclose(td_values, float(td_ms), atol=1e-6, equal_nan=False)
        if not td_mask.any():
            continue
        point_order = np.sort(point_values[td_mask].dropna().unique())
        allowed = set(point_order[: int(point_limit)])
        keep.loc[td_mask] = point_values[td_mask].isin(allowed) | point_values[td_mask].isna()

    return out.loc[keep].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Write a filtered master table for downstream analysis steps.")
    ap.add_argument("master_parquet", type=Path, help="Input master.long.parquet.")
    ap.add_argument("--out-parquet", type=Path, required=True, help="Output filtered master parquet.")
    ap.add_argument(
        "--first-points-by-td",
        required=True,
        help="Comma- or space-separated TD=POINTS rules, for example 120=8,210=6. Unlisted td_ms values keep all points.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    df = load_master_table(args.master_parquet)
    out = filter_first_points_by_td(df, args.first_points_by_td)
    write_table_outputs(out, args.out_parquet)
    print(f"Saved filtered master: {args.out_parquet}")
    print(f"Rows: {len(df)} -> {len(out)}")


if __name__ == "__main__":
    main()
