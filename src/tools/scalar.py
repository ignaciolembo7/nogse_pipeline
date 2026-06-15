"""Shared helpers for extracting unique scalar values from a DataFrame group.

These functions are used by fitting modules that operate on grouped
master_table rows where each column should hold a single unique value
within a curve/group.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def unique_text(df: pd.DataFrame, col: str, default: str = "") -> str:
    """Return the single unique string value in *col*, or *default* if absent/empty.

    If the column holds multiple distinct values, returns them joined with '|'.
    """
    if col not in df.columns:
        return default
    values = pd.Series(df[col]).dropna().astype(str).unique().tolist()
    if len(values) == 1:
        return str(values[0])
    if len(values) == 0:
        return default
    return "|".join(str(v) for v in values)


def unique_float(df: pd.DataFrame, col: str) -> float:
    """Return the single unique numeric value in *col*, or nan if absent/empty.

    Raises ValueError if the column holds more than one distinct value.
    """
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(values) == 1:
        return float(values[0])
    if len(values) == 0:
        return np.nan
    raise ValueError(f"Column {col!r} is not unique inside a curve: {values[:10].tolist()}")


def unique_float_any(df: pd.DataFrame, cols: Sequence[str]) -> float:
    """Return the first finite unique value found among *cols*, or nan."""
    for col in cols:
        value = unique_float(df, col)
        if np.isfinite(value):
            return float(value)
    return np.nan


def unique_float_strict(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    required: bool,
    name: str,
) -> Optional[float]:
    """Return the first unique numeric value found among *cols*.

    Unlike *unique_float_any*, raises ValueError if a column exists but
    holds more than one distinct value.  If *required* is True and no
    value is found, also raises.
    """
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if len(v) == 0:
            continue
        if len(v) != 1:
            raise ValueError(
                f"Expected one unique value in {c!r} for {name}, found: {v[:10]}"
            )
        return float(v[0])
    if required:
        raise ValueError(f"Could not infer {name}. Checked columns: {list(cols)}")
    return None


def unique_str(df: pd.DataFrame, col: str) -> Optional[str]:
    """Return the single unique string value in *col*, or None if absent/non-unique."""
    if col not in df.columns:
        return None
    u = pd.Series(df[col]).dropna().astype(str).unique()
    if len(u) == 1:
        return str(u[0])
    return None
