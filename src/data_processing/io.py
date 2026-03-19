from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
DEFAULT_STAT_SHEETS = {0: "avg", 1: "std", 2: "med", 3: "mad", 4: "mode"}
LEGACY_XLSX_STAT_SHEETS = {
    "Sheet1": "avg",
    "Sheet2": "std",
    "Sheet3": "med",
    "Sheet4": "mad",
    "Sheet5": "mode",
}
NAMED_STAT_SHEETS = {
    "mean": "avg",
    "avg": "avg",
    "std": "std",
    "median": "med",
    "med": "med",
    "mad": "mad",
    "mode": "mode",
}
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


def _read_xlsx_stats(xls: pd.ExcelFile) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    sheet_names = list(xls.sheet_names)
    exact_map = {str(name): str(name) for name in sheet_names}
    norm_map = {str(name).strip().lower(): str(name) for name in sheet_names}

    # formato nuevo de contraste / área
    for sheet_name, stat_name in NEW_STAT_SHEETS.items():
        if sheet_name in exact_map:
            out[stat_name] = pd.read_excel(xls, sheet_name=exact_map[sheet_name], engine="openpyxl")
    if out:
        return out

    # salida estilo MATLAB escrita como Sheet1..Sheet5
    for sheet_name, stat_name in LEGACY_XLSX_STAT_SHEETS.items():
        if sheet_name in exact_map:
            out[stat_name] = pd.read_excel(xls, sheet_name=exact_map[sheet_name], engine="openpyxl")
    if out:
        return out

    # compatibilidad adicional si las hojas ya vienen nombradas por la stat
    for sheet_name, stat_name in NAMED_STAT_SHEETS.items():
        if sheet_name in norm_map:
            out[stat_name] = pd.read_excel(xls, sheet_name=norm_map[sheet_name], engine="openpyxl")
    if out:
        return out

    # fallback: si no matchea nada, leer primera hoja como avg
    return {"avg": pd.read_excel(xls, sheet_name=0, engine="openpyxl")}

def read_result_xls(path: str | Path, stat_sheets=DEFAULT_STAT_SHEETS) -> dict[str, pd.DataFrame]:
    """
    Lee resultados legacy (.xls) y nuevos (.xlsx) y devuelve dict[stat -> DataFrame].
    - Legacy .xls: sheets por índice 0..4 -> avg/std/med/mad/mode
    - Legacy .xlsx: sheets tipo Sheet1..Sheet5 o mean/std/median/mad/mode
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
    return _read_xlsx_stats(xls)
