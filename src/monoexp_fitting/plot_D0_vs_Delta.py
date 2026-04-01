from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from tc_fittings.alpha_macro_summary import (
    load_dproj_measurements,
    plot_d_vs_delta_curves,
)


def load_all_measurements(
    dproj_root: str | Path,
    *,
    pattern: str = "**/*.Dproj.long.parquet",
    dirs: Sequence[str] | None = None,
    rois: Sequence[str] | None = None,
    subjs: Sequence[str] | None = None,
    N: float | None = None,
    Hz: float | None = None,
    bvalue_decimals: int = 1,
) -> pd.DataFrame:
    return load_dproj_measurements(
        dproj_root,
        pattern=pattern,
        subjs=subjs,
        rois=rois,
        directions=dirs,
        N=N,
        Hz=Hz,
        bvalue_decimals=bvalue_decimals,
    )


def plot_all_groups(
    df_avg: pd.DataFrame,
    *,
    out_dir: str | Path,
    reference_D0: float | None = None,
    reference_D0_error: float | None = None,
) -> list[Path]:
    return plot_d_vs_delta_curves(
        df_avg,
        out_dir=out_dir,
        reference_D0=reference_D0,
        reference_D0_error=reference_D0_error,
    )
