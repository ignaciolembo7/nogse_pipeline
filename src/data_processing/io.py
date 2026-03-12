from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
DEFAULT_STAT_SHEETS = {0: "avg", 1: "std", 2: "med", 3: "mad", 4: "mode"}
@dataclass(frozen=True)
class Layout:
    nbvals: int | None
    ndirs: int | None

def infer_layout_from_filename(path: str | Path) -> Layout:
    p = Path(path)
    m_bval = re.search(r"(\d+)bval", p.name)
    m_dir  = re.search(r"(\d+)dir", p.name)
    nbvals = int(m_bval.group(1)) if m_bval else None
    ndirs  = int(m_dir.group(1)) if m_dir else None
    return Layout(nbvals=nbvals, ndirs=ndirs)

NEW_STAT_SHEETS = {"Mean": "avg", "Min": "min", "Max": "max", "Area": "area"}

def read_result_xls(path: str | Path, stat_sheets=DEFAULT_STAT_SHEETS) -> dict[str, pd.DataFrame]:
    """
    Lee resultados legacy (.xls) y nuevos (.xlsx) y devuelve dict[stat -> DataFrame].
    - Legacy .xls: sheets por índice 0..4 -> avg/std/med/mad/mode
    - Nuevo .xlsx: sheets por nombre Mean/Min/Max/Area -> avg/min/max/area
    """
    p = Path(path)
    suf = p.suffix.lower()

    out: dict[str, pd.DataFrame] = {}

    if suf == ".xls":
        # legacy: requiere xlrd
        xls = pd.ExcelFile(p, engine="xlrd")
        for sheet_idx, stat_name in stat_sheets.items():
            out[stat_name] = pd.read_excel(xls, sheet_name=sheet_idx, engine="xlrd")
        return out

    # nuevo: .xlsx
    xls = pd.ExcelFile(p, engine="openpyxl")
    sheets = set(xls.sheet_names)

    # si es tu formato nuevo:
    hits = [s for s in NEW_STAT_SHEETS.keys() if s in sheets]
    if hits:
        for s in hits:
            out[NEW_STAT_SHEETS[s]] = pd.read_excel(xls, sheet_name=s, engine="openpyxl")
        return out

    # fallback: si no matchea nada, leer primera hoja como avg
    out["avg"] = pd.read_excel(xls, sheet_name=0, engine="openpyxl")
    return out
