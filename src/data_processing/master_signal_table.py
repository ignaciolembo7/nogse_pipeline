from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_processing.schema import finalize_clean_signal_long

KEY_COLUMNS = ["subj", "sheet", "source_file", "stat", "roi", "direction", "b_step", "N", "td_ms", "tm_ms", "delta_ms", "g", "bvalue", "gradient_axis_kind"]


def _present_key_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in KEY_COLUMNS if col in df.columns]


def validate_master_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = finalize_clean_signal_long(df)
    key_cols