from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tools.strict_columns import raise_on_unrecognized_column_names


@dataclass
class ContrastResult:
    df: pd.DataFrame


def make_contrast(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    *,
    axes: tuple[str, ...] | None = ("long", "tra"),
    y_col: str = "value",
    y_norm_col: str = "value_norm",
    key_cols: tuple[str, ...] = ("stat", "roi", "direction", "b_step"),
) -> ContrastResult:
    """
    Build a long-form contrast table with _1/_2 suffixes and two generic columns:

      value      = y_ref - y_cmp
      value_norm = y_norm_ref - y_norm_cmp   (when y_norm exists on both sides)
                = (y_ref / S0_ref) - (y_cmp / S0_cmp)  (fallback)

    Strict rules:
      - the canonical direction column is always named 'direction'
      - param_ / meta_ columns are not carried here
    """
    a = df_ref.copy()
    b = df_cmp.copy()
    raise_on_unrecognized_column_names(a.columns, context="make_contrast(df_ref)")
    raise_on_unrecognized_column_names(b.columns, context="make_contrast(df_cmp)")

    if "direction" not in a.columns or "direction" not in b.columns:
        raise KeyError("make_contrast: missing required column ['direction'] in one of the input tables.")

    for name, df in [("df_ref", a), ("df_cmp", b)]:
        missing = [c for c in key_cols if c not in df.columns]
        if missing:
            raise KeyError(f"make_contrast({name}): missing required key columns {missing}. Expected {key_cols}.")

    if y_col not in a.columns or y_col not in b.columns:
        raise KeyError(f"make_contrast: missing required signal column {y_col!r} in df_ref or df_cmp.")

    if axes is not None:
        axes = tuple(str(x) for x in axes)
        a["direction"] = a["direction"].astype(str)
        b["direction"] = b["direction"].astype(str)
        a = a[a["direction"].isin(axes)]
        b = b[b["direction"].isin(axes)]

    keep_cols = set(key_cols)
    keep_cols.add(y_col)

    have_norm = (y_norm_col in a.columns) and (y_norm_col in b.columns)
    if have_norm:
        keep_cols.add(y_norm_col)

    have_s0 = ("S0" in a.columns) and ("S0" in b.columns)
    if have_s0:
        keep_cols.add("S0")

    a = a[[c for c in keep_cols if c in a.columns]].copy()
    b = b[[c for c in keep_cols if c in b.columns]].copy()

    merged = a.merge(b, on=list(key_cols), suffixes=("_1", "_2"), how="inner")
    merged["value"] = merged[f"{y_col}_1"] - merged[f"{y_col}_2"]

    if have_norm and f"{y_norm_col}_1" in merged.columns and f"{y_norm_col}_2" in merged.columns:
        merged["value_norm"] = merged[f"{y_norm_col}_1"] - merged[f"{y_norm_col}_2"]
    else:
        if "S0_1" not in merged.columns or "S0_2" not in merged.columns:
            raise KeyError("make_contrast: missing normalized signal columns and missing S0_1/S0_2 fallback.")
        merged["value_norm"] = (merged[f"{y_col}_1"] / merged["S0_1"]) - (merged[f"{y_col}_2"] / merged["S0_2"])

    sort_cols = [c for c in ["stat", "roi", "direction", "b_step"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return ContrastResult(df=merged)
