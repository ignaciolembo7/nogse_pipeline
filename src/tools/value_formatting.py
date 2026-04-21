from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def compact_unique_values(values: Sequence[object]) -> str:
    text_values = [str(value) for value in values]
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        nums = sorted({float(value) for value in numeric.to_numpy(dtype=float)})
        if all(float(value).is_integer() for value in nums):
            ints = [int(value) for value in nums]
            if ints and ints == list(range(ints[0], ints[-1] + 1)):
                return f"{ints[0]}-{ints[-1]}"
            return ",".join(str(value) for value in ints)
        return ",".join(f"{value:g}" for value in nums)
    return ",".join(text_values)


def scalar_or_compact_series(series: pd.Series, *, name: str, required: bool = False) -> Any:
    values = pd.Series(series).dropna().unique().tolist()
    if len(values) == 0:
        if required:
            raise ValueError(f"Could not infer '{name}': column is empty or all-NaN.")
        return None
    if len(values) == 1:
        return values[0]
    return compact_unique_values(values)


def scalar_or_compact_column(df: pd.DataFrame, col: str, *, required: bool = False) -> Any:
    if col not in df.columns:
        if required:
            raise ValueError(f"Missing required column {col!r}.")
        return None
    return scalar_or_compact_series(df[col], name=col, required=required)


def truthy_series(series: pd.Series) -> bool:
    values = pd.Series(series).dropna().astype(str).str.strip().str.lower()
    return values.isin(["1", "true", "yes", "y", "on"]).any()
