from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_processing.schema import finalize_clean_signal_long

MASTER_SIGNAL_KEY_COLUMNS = ["subj", "source_file", "sheet", "stat", "roi", "direction", "b_step", "N", "td_ms", "g", "bvalue"]


def load_master_signal_table(path: str | Path) -> pd.DataFrame:
    master_path = Path(path)
    if not master_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(master_path)


def upsert_master_signal_table(records: pd.DataFrame, path: str | Path, key_columns: list[str] | None =