from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_processing.schema import finalize_clean_signal_long

MASTER_SIGNAL_KEY_COLUMNS = [
    "subj",
    "sheet",
    "source_file",
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
    "gradient