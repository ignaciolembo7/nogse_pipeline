from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _normalize_direction(value: object, aliases: Mapping[str, str]) -> str:
    text = str(value).strip()
    return str(aliases.get(text, text)).strip()
