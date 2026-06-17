from __future__ import annotations

import re

import numpy as np
import pandas as pd

from tools.scalar import unique_float


def infer_td_ms(
    df: pd.DataFrame,
    *,
    analysis_id: str = "",
    override: float | None = None,
) -> float | None:
    if override is not None:
        return float(override)

    td1 = unique_float(df, "td_ms_1")
    if np.isfinite(td1):
        td2 = unique_float(df, "td_ms_2")
        if not np.isfinite(td2):
            return float(td1)
        if abs(float(td1) - float(td2)) < 1e-3:
            return float(0.5 * (td1 + td2))
        return None

    td = unique_float(df, "td_ms")
    if np.isfinite(td):
        return float(td)

    max_dur_ms = unique_float(df, "max_dur_ms")
    tm_ms = unique_float(df, "tm_ms")
    if np.isfinite(max_dur_ms) and np.isfinite(tm_ms):
        return float(2.0 * max_dur_ms + tm_ms)

    match = re.search(r"_td(\d+(?:p\d+)?)", str(analysis_id))
    if match:
        return float(match.group(1).replace("p", "."))
    return None
