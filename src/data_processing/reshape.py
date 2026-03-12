from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def to_long(
    stats: dict[str, pd.DataFrame],
    *,
    ndirs: int,
    nbvals: int,
    bcol: str = "bvalues",
    b0_reps: int = 2,
    source_file: str | None = None,
) -> pd.DataFrame:
    """
    Convierte las tablas intercaladas (stride=ndirs) a formato long/tidy.
    Replica una fila b0 (promedio de b0_reps) para cada direction, igual que tu notebook.
    """
    if not stats:
        raise ValueError("stats está vacío")

    any_df = next(iter(stats.values()))
    if bcol not in any_df.columns:
        for alt in ["bvalue", "bval", "b"]:
            if alt in any_df.columns:
                bcol = alt
                break
    if bcol not in any_df.columns:
        raise ValueError(f"No encuentro la columna '{bcol}'. Columnas: {list(any_df.columns)}")

    rois = [c for c in any_df.columns if c != bcol]

    rows_expected = ndirs * nbvals
    long_parts: list[pd.DataFrame] = []

    for stat, df in stats.items():
        # b0 promedio (por ROI)
        b0 = df.loc[: b0_reps - 1, rois].mean(axis=0)

        # datos sin los b0 repetidos
        data = df.loc[b0_reps:, [bcol] + rois].reset_index(drop=True)
        if len(data) != rows_expected:
            raise ValueError(
                f"[{stat}] Esperaba {rows_expected} filas (ndirs*nbvals), pero tengo {len(data)}. "
                "Esto suele pasar si ndirs/nbvals no coinciden con el archivo."
            )

        idx = np.arange(len(data))
        direction = (idx % ndirs) + 1          # 1..ndirs
        b_step = (idx // ndirs) + 1            # 1..nbvals

        data = data.assign(direction=direction, b_step=b_step)

        # long sin b0
        long = data.melt(
            id_vars=[bcol, "direction", "b_step"],
            value_vars=rois,
            var_name="roi",
            value_name="value",
        )
        long = long.rename(columns={bcol: "bvalue"})
        long["stat"] = stat

        # b0 duplicado por direction (b_step=0)
        b0_long = pd.DataFrame({
            "bvalue": 0.0,
            "direction": np.repeat(np.arange(1, ndirs + 1), len(rois)),
            "b_step": 0,
            "roi": np.tile(np.array(rois), ndirs),
            "value": np.tile(b0.to_numpy(), ndirs),
            "stat": stat,
        })

        long_parts.append(pd.concat([b0_long, long], ignore_index=True))

    out = pd.concat(long_parts, ignore_index=True)
    out["source_file"] = source_file if source_file is not None else ""

    # Orden final: por ROI y direction; y dentro de cada curva, b_step (0 primero)
    sort_cols = [c for c in ["stat", "roi", "direction", "b_step"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return out