from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_processing.schema import CLEAN_SIGNAL_LONG_COLUMNS, finalize_clean_signal_long

DEFAULT_MASTER_SIGNAL_KEY_COLUMNS = [
    "subj",
    "source_file",
    "sheet",
    "stat",
    "roi",
    "direction",
    "b_step",
    "N",
    "td_ms",
    "tm_ms",
    "delta_ms",
    "g",
    "bvalue",
   