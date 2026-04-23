from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

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


@dataclass(frozen=True)
class SignalCorrectionLookupSpec:
    roi_ref: str
    td_ms: float
    signal_n: int | None = None
    tol_ms: float = 1e-3
    sheet: str | None = None
    signal_source_file: str | None = None
    preferred_side: Literal[1, 2] | None = None


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
    Required columns: direction, roi, td_ms, N_1, N_2,
    correction_factor_1, and correction_factor_2.
    """
    path = Path(path)
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = "grad_correction" if "grad_correction" in xls.sheet_names else xls.sheet_names[0]
    corr = xls.parse(sheet_name=sheet).copy()

    raise_on_unrecognized_column_names(corr.columns, context=f"read_correction_table({path})")

    if (
        ("correction factor" in corr.columns or "correction_factor" in corr.columns)
        and not {"correction_factor_1", "correction_factor_2"}.issubset(corr.columns)
    ):
        raise ValueError(
            "Legacy correction tables with a single correction_factor are no longer supported. "
            "Regenerate the table with the new side-specific pipeline so it includes "
            "correction_factor_1 and correction_factor_2."
        )

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
    factor_mode: Literal["side_1", "side_2", "per_side"] = "per_side",
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
    if factor_mode == "per_side":
        c = c.groupby("direction", as_index=False)[["correction_factor_1", "correction_factor_2"]].mean()
        out = {
            str(row["direction"]): (
                float(row["correction_factor_1"]),
                float(row["correction_factor_2"]),
            )
            for _, row in c.iterrows()
        }
    elif factor_mode == "side_1":
        c = c.groupby("direction", as_index=False)["correction_factor_1"].mean()
        out = {str(d): float(f) for d, f in zip(c["direction"], c["correction_factor_1"])}
    else:
        c = c.groupby("direction", as_index=False)["correction_factor_2"].mean()
        out = {str(d): float(f) for d, f in zip(c["direction"], c["correction_factor_2"])}
    if not out:
        raise ValueError("The filtered correction table did not produce any valid factors.")
    return out


def _source_key(text: str | None) -> str:
    if text is None:
        return ""
    name = Path(str(text)).name.strip()
    lower = name.lower()
    if lower.endswith(".long.parquet"):
        return name[: -len(".long.parquet")]
    if lower.endswith(".parquet"):
        return name[: -len(".parquet")]
    return Path(name).stem


def build_signal_direction_factors(
    corr: pd.DataFrame,
    *,
    spec: SignalCorrectionLookupSpec,
) -> dict[str, float]:
    c = corr[corr["roi"].astype(str) == str(spec.roi_ref).strip()].copy()

    if spec.sheet is not None and "sheet" in c.columns:
        sheet_key = canonical_sheet_name(spec.sheet)
        c = c[c["sheet"] == sheet_key].copy()

    c = c[
        np.isclose(
            pd.to_numeric(c["td_ms"], errors="coerce").to_numpy(dtype=float),
            float(spec.td_ms),
            rtol=0.0,
            atol=float(spec.tol_ms),
        )
    ].copy()

    if c.empty:
        raise ValueError(
            f"No signal correction factors were found for roi={spec.roi_ref} "
            f"and td_ms={float(spec.td_ms):.3f} (tol={spec.tol_ms})."
        )

    source_key = _source_key(spec.signal_source_file)
    matched_rows: list[dict[str, object]] = []
    for _, row in c.iterrows():
        side_candidates: list[int] = []
        if spec.preferred_side is not None:
            side_candidates = [int(spec.preferred_side)]
        else:
            for side in (1, 2):
                file_col = f"signal_source_file_{side}"
                n_col = f"N_{side}"
                row_source_key = _source_key(row.get(file_col))
                n_val = pd.to_numeric(pd.Series([row.get(n_col)]), errors="coerce").iloc[0]
                source_match = bool(source_key) and row_source_key == source_key
                n_match = spec.signal_n is not None and np.isfinite(n_val) and int(round(float(n_val))) == int(spec.signal_n)
                if source_match or n_match:
                    side_candidates.append(side)

        for side in side_candidates:
            factor_col = f"correction_factor_{side}"
            factor = pd.to_numeric(pd.Series([row.get(factor_col)]), errors="coerce").iloc[0]
            if not np.isfinite(factor):
                continue
            matched_rows.append(
                {
                    "direction": str(row["direction"]),
                    "side": int(side),
                    "factor": float(factor),
                }
            )

    if not matched_rows:
        extra = []
        if source_key:
            extra.append(f"signal_source_file={source_key}")
        if spec.signal_n is not None:
            extra.append(f"N={int(spec.signal_n)}")
        if spec.preferred_side is not None:
            extra.append(f"preferred_side={int(spec.preferred_side)}")
        detail = f" ({', '.join(extra)})" if extra else ""
        raise ValueError(
            "Could not match any side-specific correction factor for the requested signal"
            f"{detail}. The new pipeline only supports per-signal side-specific correction."
        )

    matches = pd.DataFrame(matched_rows)
    side_counts = matches.groupby("direction")["side"].nunique()
    ambiguous = side_counts[side_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "Ambiguous side-specific correction lookup for some directions. "
            "The signal could not be matched to a unique side in the correction table: "
            f"{sorted(ambiguous.index.astype(str).tolist())}"
        )

    out = matches.groupby("direction", as_index=False)["factor"].mean()
    return {str(d): float(f) for d, f in zip(out["direction"], out["factor"])}
