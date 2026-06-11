from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from tools.strict_columns import raise_on_unrecognized_column_names


CONTRAST_KEY_COLUMNS = ("roi", "direction", "b_step")


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    """Check that a physical table has the columns needed by the next step."""
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns {missing}. Columns={list(df.columns)}")


def normalize_contrast_keys(
    rows: pd.DataFrame,
    *,
    label: str,
    key_cols: Sequence[str] = CONTRAST_KEY_COLUMNS,
) -> pd.DataFrame:
    """
    Receive contrast rows and return the same rows with stable grouping keys.

    The physical operation is small but important: each curve is selected by ROI,
    direction and b_step, so those keys must have predictable types before fitting.
    """
    out = rows.copy()
    raise_on_unrecognized_column_names(out.columns, context=label)
    require_columns(out, key_cols, label=label)

    if "direction" in key_cols:
        out["direction"] = out["direction"].astype(str)

    if "b_step" in key_cols:
        b_step = pd.to_numeric(out["b_step"], errors="coerce")
        if b_step.isna().any():
            bad = out.loc[b_step.isna(), ["roi", "direction", "b_step"]].head(10)
            raise ValueError(f"{label}: b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
        out["b_step"] = b_step.astype(int)

    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)

    return out


def unique_scalar(series: pd.Series, *, name: str, required: bool = False) -> Any:
    """Return one physical value from a group, failing when the group is mixed."""
    unique_values = pd.Series(series).dropna().unique()
    if len(unique_values) == 0:
        if required:
            raise ValueError(f"Could not infer '{name}': column is empty or all-NaN.")
        return None
    if len(unique_values) > 1:
        raise ValueError(f"'{name}' is not unique within the group. Values={unique_values.tolist()[:10]}")
    return unique_values[0]


def analysis_id_from_source_file(source_file: str | None) -> str:
    """Name the analysis from the signal table that produced these contrast rows."""
    if not source_file:
        return ""
    stem = Path(str(source_file)).stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def maybe_scale_gradient(axis_name: str, values: np.ndarray) -> np.ndarray:
    """Return gradient values in the units expected by the contrast models."""
    del axis_name
    return np.asarray(values, dtype=float)


def coerce_correction_pair(value: Any) -> tuple[float, float]:
    """Return physical correction factors for the two OGSE/NOGSE gradient sides."""
    if value is None:
        return 1.0, 1.0

    if isinstance(value, (tuple, list, np.ndarray, pd.Series)) and len(value) >= 2:
        f1 = float(value[0])
        f2 = float(value[1])
    else:
        f1 = float(value)
        f2 = f1

    if not np.isfinite(f1) or f1 <= 0:
        f1 = 1.0
    if not np.isfinite(f2) or f2 <= 0:
        f2 = 1.0
    return f1, f2


def fit_row_correction_pair(fit_row: dict[str, Any] | pd.Series) -> tuple[float, float]:
    """Read the two stored correction factors from a fitted-curve output row."""
    f1 = fit_row.get("f_corr_1", np.nan)
    f2 = fit_row.get("f_corr_2", np.nan)
    if pd.notna(f1) and pd.notna(f2):
        return coerce_correction_pair((f1, f2))
    return 1.0, 1.0
