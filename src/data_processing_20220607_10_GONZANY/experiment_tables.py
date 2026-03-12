
"""
Utilities to convert a long Results.csv (stacked experiments) into one Excel file per experiment,
matching the 'bvalues + ROI columns' layout used by Siemens-style *_results.xls tables.

Assumptions (default):
- The input Results.csv contains N experiments stacked vertically.
- Each experiment corresponds to `chunk_size` rows (default 32).
- b-values for each experiment are provided as one line per experiment in a text file
  (e.g., bval-Hahn.txt), with exactly `chunk_size` values per line.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _parse_num_list(text: str) -> List[float]:
    # Accept comma and/or whitespace separated lists.
    parts = [p for p in text.replace(",", " ").split() if p.strip() != ""]
    return [float(p) for p in parts]


def read_bvals_txt(path: str | Path) -> List[List[float]]:
    """
    Read b-values file where each non-empty line contains a list of numbers.
    Returns list-of-lines, each line is a list of floats.
    """
    path = Path(path)
    lines: List[List[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        lines.append(_parse_num_list(s))
    return lines


def read_results_csv(path: str | Path) -> pd.DataFrame:
    """
    Read Results.csv and normalize column names (strip whitespace).
    Drops the first column if it looks like an index column.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Common cases: first column is an index with name '' or 'Unnamed: 0' or ' '.
    first = df.columns[0]
    if first in ("", " ", "Unnamed: 0", "index") or df[first].is_monotonic_increasing:
        # Only drop if it's an integer-like counting column 1..chunk_size repeating.
        # We'll be conservative and drop if it's numeric and has many duplicates.
        try:
            s = pd.to_numeric(df[first], errors="coerce")
            if s.notna().mean() > 0.95:
                df = df.drop(columns=[first])
        except Exception:
            pass
    return df


def split_experiments(df: pd.DataFrame, chunk_size: int = 32) -> List[pd.DataFrame]:
    """
    Split a long dataframe into chunks of `chunk_size` rows.
    """
    if len(df) % chunk_size != 0:
        raise ValueError(
            f"Input has {len(df)} rows, not divisible by chunk_size={chunk_size}."
        )
    n = len(df) // chunk_size
    return [df.iloc[i*chunk_size:(i+1)*chunk_size].reset_index(drop=True) for i in range(n)]


def _roi_name_from_col(col: str) -> Optional[str]:
    # Expected: "Mean(Fibra1)" -> "Fibra1"
    m = None
    if "(" in col and col.endswith(")"):
        m = col.split("(", 1)[1][:-1]
    return m.strip() if m else None


def make_wide_table(chunk: pd.DataFrame, bvals: Sequence[float], stat: str = "Mean") -> pd.DataFrame:
    """
    Build a wide table with columns:
      bvalues, <ROI1>, <ROI2>, ...
    taking values from columns like f"{stat}(ROI)".
    """
    if len(bvals) != len(chunk):
        raise ValueError(f"bvals length {len(bvals)} does not match chunk rows {len(chunk)}")

    # Gather stat columns
    prefix = f"{stat}("
    cols = [c for c in chunk.columns if str(c).startswith(prefix) and str(c).endswith(")")]
    if not cols:
        raise ValueError(
            f"No columns found matching '{stat}(...)'. Available columns: {list(chunk.columns)}"
        )

    out = pd.DataFrame({"bvalues": list(bvals)})
    for c in cols:
        roi = _roi_name_from_col(str(c)) or str(c)
        out[roi] = chunk[c].values
    return out


def make_tables_for_experiment(
    chunk: pd.DataFrame,
    bvals: Sequence[float],
    stats: Sequence[str] = ("Mean",),
) -> Dict[str, pd.DataFrame]:
    """
    Return {sheet_name: wide_table_df}.
    """
    tables: Dict[str, pd.DataFrame] = {}
    for st in stats:
        tables[st] = make_wide_table(chunk, bvals=bvals, stat=st)
    return tables


def format_name(template: str, *, base: str, exp: int, d=None, Delta=None, Hz=None, max_dur=None, bmax=None) -> str:
    """
    Format an output filename from a template. Available fields:
      {base}, {exp}, {d}, {Delta}, {Hz}, {max_dur}, {bmax}
    Example:
      "{base}_d{d}_Delta{Delta}_Hz{Hz:03d}_maxdur{max_dur}_b{bmax}exp{exp:02d}_results.xlsx"
    """
    return template.format(base=base, exp=exp, d=d, Delta=Delta, Hz=Hz, max_dur=max_dur, bmax=bmax)


def write_experiment_xlsx(
    out_path: str | Path,
    tables: Dict[str, pd.DataFrame],
) -> None:
    """
    Write tables to an Excel file, one sheet per stat.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        for sheet, df in tables.items():
            # Excel sheet names max 31 chars.
            sheet_name = sheet[:31]
            df.to_excel(w, sheet_name=sheet_name, index=False)


@dataclass(frozen=True)
class MetaVectors:
    d: Optional[List[float]] = None
    Delta: Optional[List[float]] = None
    Hz: Optional[List[float]] = None
    max_dur: Optional[List[float]] = None



def parse_meta_vectors(*, d: Optional[str], Delta: Optional[str], Hz: Optional[str], max_dur: Optional[str]) -> MetaVectors:
    def _maybe(v: Optional[str]) -> Optional[List[float]]:
        if v is None:
            return None
        return _parse_num_list(v)
    return MetaVectors(d=_maybe(d), Delta=_maybe(Delta), Hz=_maybe(Hz), max_dur=_maybe(max_dur))


def validate_meta_lengths(n_exp: int, meta: MetaVectors) -> None:
    for name, vec in (("d", meta.d), ("Delta", meta.Delta), ("Hz", meta.Hz), ("max_dur", meta.max_dur)):
        if vec is None:
            continue
        if len(vec) != n_exp:
            raise ValueError(f"Vector '{name}' has length {len(vec)} but there are {n_exp} experiments.")
