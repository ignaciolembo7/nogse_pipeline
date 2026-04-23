from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tools.brain_labels import canonical_sheet_name
from tools.strict_columns import raise_on_unrecognized_column_names


@dataclass(frozen=True)
class CorrectionLookupSpec:
    roi_ref: str
    td_ms: float
    tol_ms: float = 1e-3
    sheet: str | None = None
    n1: int | None = None
    n2: int | None = None


def unique_float(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    u = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(u) == 1:
        return float(u[0])
    return None


def unique_int(df: pd.DataFrame, *cols: str) -> int | None:
    for col in cols:
        v = unique_float(df, col)
        if v is not None:
            return int(round(float(v)))
    return None


def infer_td_ms(
    df: pd.DataFrame,
    *,
    analysis_id: str = "",
    override: float | None = None,
) -> float | None:
    if override is not None:
        return float(override)

    # Prefer contrast-style columns when available.
    td1 = unique_float(df, "td_ms_1")
    if td1 is not None:
        td2 = unique_float(df, "td_ms_2")
        if td2 is None:
            return float(td1)
        if abs(float(td1) - float(td2)) < 1e-3:
            return float(0.5 * (td1 + td2))
        return None

    # Fall back to generic signal columns.
    td = unique_float(df, "td_ms")
    if td is not None:
        return float(td)

    max_dur_ms = unique_float(df, "max_dur_ms")
    tm_ms = unique_float(df, "tm_ms")
    if max_dur_ms is not None and tm_ms is not None:
        return float(2.0 * max_dur_ms + tm_ms)

    # Fallback: parse tdXX from the analysis id.
    m = re.search(r"_td(\d+(?:p\d+)?)", str(analysis_id))
    if m:
        return float(m.group(1).replace("p", "."))
    return None


def read_correction_table(path: str | Path) -> pd.DataFrame:
    """
    Strict reader for gradient correction tables.
    Required columns: direction, roi, td_ms, N_1, N_2, and either
    correction_factor or correction_factor_1/correction_factor_2.
    """
    path = Path(path)
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = "grad_correction" if "grad_correction" in xls.sheet_names else xls.sheet_names[0]
    corr = xls.parse(sheet_name=sheet).copy()

    raise_on_unrecognized_column_names(corr.columns, context=f"read_correction_table({path})")

    rename = {}
    if "correction factor" in corr.columns and "correction_factor" not in corr.columns:
        rename["correction factor"] = "correction_factor"
    if rename:
        corr = corr.rename(columns=rename)

    if "correction_factor" in corr.columns:
        if "correction_factor_1" not in corr.columns:
            corr["correction_factor_1"] = corr["correction_factor"]
        if "correction_factor_2" not in corr.columns:
            corr["correction_factor_2"] = corr["correction_factor"]

    required = {"roi", "direction", "td_ms", "N_1", "N_2"}
    missing = required - set(corr.columns)
    if missing:
        raise ValueError(f"Invalid correction table. Missing required columns: {sorted(missing)}")
    factor_missing = {"correction_factor_1", "correction_factor_2"} - set(corr.columns)
    if factor_missing:
        raise ValueError(f"Invalid correction table. Missing correction columns: {sorted(factor_missing)}")

    corr["direction"] = corr["direction"].astype(str)
    corr["roi"] = corr["roi"].astype(str).str.strip()
    if "sheet" in corr.columns:
        corr["sheet"] = corr["sheet"].map(canonical_sheet_name)

    corr["td_ms"] = pd.to_numeric(corr["td_ms"], errors="coerce")
    corr["correction_factor_1"] = pd.to_numeric(corr["correction_factor_1"], errors="coerce")
    corr["correction_factor_2"] = pd.to_numeric(corr["correction_factor_2"], errors="coerce")
    if "correction_factor" in corr.columns:
        corr["correction_factor"] = pd.to_numeric(corr["correction_factor"], errors="coerce")
    else:
        corr["correction_factor"] = np.sqrt(corr["correction_factor_1"] * corr["correction_factor_2"])
    corr["N_1"] = pd.to_numeric(corr["N_1"], errors="coerce")
    corr["N_2"] = pd.to_numeric(corr["N_2"], errors="coerce")
    corr = corr[
        np.isfinite(corr["td_ms"])
        & np.isfinite(corr["correction_factor_1"])
        & np.isfinite(corr["correction_factor_2"])
        & np.isfinite(corr["N_1"])
        & np.isfinite(corr["N_2"])
    ].copy()
    return corr


def build_direction_factors(
    corr: pd.DataFrame,
    *,
    spec: CorrectionLookupSpec,
) -> dict[str, Union[float, tuple[float, float]]]:
    c = corr[corr["roi"].astype(str) == str(spec.roi_ref).strip()].copy()

    # If the correction table has a sheet column, filter it to avoid mixing datasets.
    if spec.sheet is not None and "sheet" in c.columns:
        sheet_key = canonical_sheet_name(spec.sheet)
        c = c[c["sheet"] == sheet_key].copy()

    if spec.n1 is not None:
        c = c[pd.to_numeric(c["N_1"], errors="coerce") == int(spec.n1)].copy()
    if spec.n2 is not None:
        c = c[pd.to_numeric(c["N_2"], errors="coerce") == int(spec.n2)].copy()

    c = c[
        np.isclose(
            pd.to_numeric(c["td_ms"], errors="coerce").to_numpy(dtype=float),
            float(spec.td_ms),
            rtol=0.0,
            atol=float(spec.tol_ms),
        )
    ].copy()
    if c.empty:
        extra_parts: list[str] = []
        if spec.sheet is not None and "sheet" in corr.columns:
            extra_parts.append(f"sheet={canonical_sheet_name(spec.sheet)}")
        if spec.n1 is not None:
            extra_parts.append(f"N_1={int(spec.n1)}")
        if spec.n2 is not None:
            extra_parts.append(f"N_2={int(spec.n2)}")
        extra = f" and {'; '.join(extra_parts)}" if extra_parts else ""
        raise ValueError(
            f"No correction factors were found for roi={spec.roi_ref}{extra} "
            f"and td_ms={float(spec.td_ms):.3f} (tol={spec.tol_ms})."
        )

    # If duplicates exist, average them.
    if {"correction_factor_1", "correction_factor_2"}.issubset(c.columns):
        c = c.groupby("direction", as_index=False)[["correction_factor_1", "correction_factor_2"]].mean()
        out = {
            str(row["direction"]): (
                float(row["correction_factor_1"]),
                float(row["correction_factor_2"]),
            )
            for _, row in c.iterrows()
        }
    else:
        c = c.groupby("direction", as_index=False)["correction_factor"].mean()
        out = {str(d): float(f) for d, f in zip(c["direction"], c["correction_factor"])}
    if not out:
        raise ValueError("The filtered correction table did not produce any valid factors.")
    return out
