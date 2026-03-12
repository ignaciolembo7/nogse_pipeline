from __future__ import annotations
import numpy as np
import pandas as pd

def add_ogse_features(
    df: pd.DataFrame,
    *,
    gamma: float,          # 1/(ms*mT)
    N: int,
    delta_ms: float,
    delta_app_ms: float,
    gthorsten_val: float | None = None,
    norm_stat: str = "avg",
) -> pd.DataFrame:
    out = df.copy()

    # --- compute g if missing ---
    if "g" not in out.columns:
        b = out["bvalue"].to_numpy(dtype=float)  # b en s/mm^2
        g = np.sqrt((b * 1e9) / (N * (gamma**2) * (delta_ms**2) * delta_app_ms))
        g[b == 0] = 0.0
        out["g"] = g

    if "b_step" not in out.columns:
        raise KeyError("b_step no existe; asegurate de haber pasado por reshape.to_long()")

    # Elegir stat para normalización (prioriza avg/Mean si existen)
    stats_present = set(out["stat"].unique())
    if norm_stat not in stats_present:
        if "avg" in stats_present:
            norm_stat = "avg"
        elif "Mean" in stats_present:
            norm_stat = "Mean"
        else:
            norm_stat = next(iter(stats_present))

    # --- g_max / g_lin_max / gthorsten CONSISTENTES ---
    b_step_max = int(pd.to_numeric(out["b_step"], errors="coerce").max())
    if b_step_max <= 0:
        b_step_max = 1

    gmax_by_dir = (
        out.loc[out["b_step"] == b_step_max]
        .groupby("direction")["g"]
        .max()
    )
    out["g_max"] = out["direction"].map(gmax_by_dir).fillna(pd.to_numeric(out["g"], errors="coerce").max())

    out["g_lin_max"] = out["g_max"] * (pd.to_numeric(out["b_step"], errors="coerce") / float(b_step_max))

    if gthorsten_val is not None:
        out["gthorsten"] = float(gthorsten_val) * (pd.to_numeric(out["b_step"], errors="coerce") / float(b_step_max))

    # --- Normalización SOLO para stat==norm_stat ---
    out["signal_norm"] = np.nan
    mask = out["stat"] == norm_stat

    b0 = out.loc[mask & (out["b_step"] == 0), ["direction", "roi", "value"]].rename(columns={"value": "b0_value"})
    out = out.merge(b0, on=["direction", "roi"], how="left")

    out.loc[mask, "signal_norm"] = out.loc[mask, "value"] / out.loc[mask, "b0_value"]
    out = out.drop(columns=["b0_value"])

    return out