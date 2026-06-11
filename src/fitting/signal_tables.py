from __future__ import annotations

from pathlib import Path

import pandas as pd

from fitting.contrast_tables import require_columns
from tools.strict_columns import raise_on_unrecognized_column_names


SIGNAL_CURVE_KEY_COLUMNS = ("roi", "direction", "b_step")


def signal_table_analysis_id(path: str | Path) -> str:
    """Return the physical analysis name encoded in a signal table filename."""
    name = Path(path).name
    for suffix in (".rot_tensor.long.parquet", ".long.parquet", ".parquet", ".xlsx", ".xls"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def normalize_signal_curve_keys(
    rows: pd.DataFrame,
    *,
    label: str,
    strict_columns: bool = True,
) -> pd.DataFrame:
    """
    Receive signal rows and return rows with stable curve keys.

    A signal curve is selected by ROI, direction and b_step. This function keeps
    that physical grouping explicit before any model fitting starts.
    """
    out = rows.copy()
    if strict_columns:
        raise_on_unrecognized_column_names(out.columns, context=label)
    require_columns(out, SIGNAL_CURVE_KEY_COLUMNS, label=label)

    out["roi"] = out["roi"].astype(str)
    out["direction"] = out["direction"].astype(str)
    b_step = pd.to_numeric(out["b_step"], errors="coerce")
    if b_step.isna().any():
        bad = out.loc[b_step.isna(), ["roi", "direction", "b_step"]].head(10)
        raise ValueError(f"{label}: b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
    out["b_step"] = b_step.astype(int)

    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)

    return out
