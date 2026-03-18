from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.io import infer_layout_from_filename, read_result_xls
from data_processing.match_params import parse_results_filename, select_params_row
from data_processing.params import read_sequence_params_xlsx
from data_processing.reshape import to_long
from data_processing.schema import finalize_clean_signal_long
from ogse_fitting.b_from_g import b_from_g


# -------------------------
# Helpers
# -------------------------
def _norm_key(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _row_get(row: pd.Series, keys: list[str], default=None):
    """Busca en row por keys (case/space-insensitive) y devuelve el primer valor no-NaN."""
    idx = list(row.index)
    norm_map = {_norm_key(c): c for c in idx}

    for k in keys:
        kk = _norm_key(k)
        if kk in norm_map:
            col = norm_map[kk]
            v = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
    return default


def _to_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _add_S0_and_signal_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_numeric(out, ["b_step", "value"])

    # S0 por (stat, roi, direction): mean de value cuando b_step==0
    s0 = (
        out.loc[out["b_step"] == 0]
        .groupby(["stat", "roi", "direction"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "S0"})
    )
    out = out.merge(s0, on=["stat", "roi", "direction"], how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["value_norm"] = out["value"] / out["S0"]

    # forzamos 1.0 en b_step==0 si hay S0
    m0 = (out["b_step"] == 0) & out["S0"].notna()
    out.loc[m0, "value_norm"] = 1.0
    return out


def _add_g_and_b_derivatives(
    df: pd.DataFrame,
    *,
    gamma: float,
    N: int,
    delta_ms: float,
    Delta_app_ms: float,
    gthorsten_mTm: float | None,
) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_numeric(out, ["bvalue", "b_step"])

    # g desde bvalue (mT/m)  [misma fórmula que venías usando]
    b = out["bvalue"].to_numpy(dtype=float)  # s/mm^2
    denom = (N * (gamma**2) * (delta_ms**2) * (Delta_app_ms))
    g = np.sqrt((b * 1e9) / denom)
    g[np.isclose(b, 0.0)] = 0.0
    out["g"] = g

    # g_max por direction usando el último b_step
    b_step_max = int(np.nanmax(out["b_step"].to_numpy(dtype=float)))
    if not np.isfinite(b_step_max) or b_step_max <= 0:
        b_step_max = 1

    gmax_by_dir = (
        out.loc[out["b_step"] == b_step_max]
        .groupby("direction")["g"]
        .max()
    )
    out["g_max"] = out["direction"].map(gmax_by_dir)
    # fallback si alguna direction quedó sin match
    out["g_max"] = out["g_max"].fillna(np.nanmax(out["g"].to_numpy(dtype=float)))

    # g_lin_max
    out["g_lin_max"] = out["g_max"] * (out["b_step"] / float(b_step_max))

    # g_thorsten (si existe el escalar)
    if gthorsten_mTm is None or not np.isfinite(gthorsten_mTm):
        out["g_thorsten"] = np.nan
    else:
        out["g_thorsten"] = float(gthorsten_mTm) * (out["b_step"] / float(b_step_max))

    # bvalue derivados desde g*
    out["bvalue_g"] = b_from_g(
        pd.to_numeric(out["g"], errors="coerce").to_numpy(float),
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=Delta_app_ms,
        g_type="g",
    )
    out["bvalue_g_lin_max"] = b_from_g(
        pd.to_numeric(out["g_lin_max"], errors="coerce").to_numpy(float),
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=Delta_app_ms,
        g_type="g_lin_max",
    )
    # Thorsten puede ser NaN -> b_from_g devuelve NaN
    out["bvalue_thorsten"] = b_from_g(
        pd.to_numeric(out["g_thorsten"], errors="coerce").to_numpy(float),
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=Delta_app_ms,
        g_type="gthorsten",
    )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file", type=Path)
    ap.add_argument("params_xlsx", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/ogse_experiments/data"))
    ap.add_argument("--gamma", type=float, default=267.5221900, help="1/(ms*mT)")
    args = ap.parse_args()

    meta = parse_results_filename(args.results_file)
    layout = infer_layout_from_filename(args.results_file)

    ndirs = meta.ndirs or layout.ndirs
    nbvals = meta.nbvals or layout.nbvals
    if ndirs is None or nbvals is None:
        raise SystemExit(f"No pude inferir ndirs/nbvals del filename: {args.results_file.name}")

    stats = read_result_xls(args.results_file)
    df_long = to_long(stats, ndirs=ndirs, nbvals=nbvals, source_file=args.results_file.name)

    # --- params (una sola fila) ---
    params = read_sequence_params_xlsx(args.params_xlsx)
    row = select_params_row(params, meta)

    # extraer parámetros (robusto a nombres con mayúsculas/espacios)
    sheet = str(_row_get(row, ["sheet"], meta.sheet))
    protocol = _row_get(row, ["protocol", "Protocol*"], None)
    sequence = _row_get(row, ["seq", "sequence"], None)

    Hz = _to_float(_row_get(row, ["Hz", "Frecuency [Hz]"], meta.Hz))
    bmax = _to_float(_row_get(row, ["bmax", "bval_max [s/mm2]"], meta.bmax))
    N = int(_to_float(_row_get(row, ["N"], 1)))

    delta_ms = _to_float(_row_get(row, ["delta_ms", "delta [ms]"], meta.delta_ms))
    Delta_app_ms = _to_float(_row_get(row, ["Delta_app_ms", "delta_app_ms", "Delta_app [ms]"], meta.Delta_ms))

    # max_dur y TM (del Excel; si tu params.py aún usa d_ms/TM_ms, igual lo levantamos)
    max_dur_ms = _to_float(_row_get(row, ["max_dur_ms", "d_ms", "Max duration d  [ms]"], None))
    tm_ms = _to_float(_row_get(row, ["tm_ms", "TM_ms", "mixing time TM  [ms]"], None))

    td_ms = np.nan
    if np.isfinite(max_dur_ms) and np.isfinite(tm_ms):
        td_ms = 2.0 * max_dur_ms + tm_ms

    TE = _to_float(_row_get(row, ["TE", "TE_ms", "Echo time TE  [ms]"], np.nan))
    TR = _to_float(_row_get(row, ["TR", "TR_ms", "Repetition time TR  [ms]"], np.nan))

    gthorsten_mTm = _to_float(_row_get(row, ["gthorsten_mTm", "G thorsten [mT/m]"], np.nan))
    if not np.isfinite(gthorsten_mTm):
        gthorsten_mTm = None

    # --- agregar parámetros LIMPIOS (sin param_/meta_) ---
    df_long["sheet"] = sheet
    df_long["protocol"] = protocol
    df_long["sequence"] = sequence

    df_long["Hz"] = Hz
    df_long["bmax"] = bmax
    df_long["N"] = N
    df_long["delta_ms"] = delta_ms
    df_long["Delta_app_ms"] = Delta_app_ms

    df_long["max_dur_ms"] = max_dur_ms
    df_long["tm_ms"] = tm_ms
    df_long["td_ms"] = td_ms

    df_long["TE"] = TE
    df_long["TR"] = TR

    # --- features g + bvalue derivados + normalización ---
    df_long = _add_g_and_b_derivatives(
        df_long,
        gamma=float(args.gamma),
        N=int(N),
        delta_ms=float(delta_ms),
        Delta_app_ms=float(Delta_app_ms),
        gthorsten_mTm=gthorsten_mTm,
    )
    df_long = _add_S0_and_signal_norm(df_long)

    df_long = finalize_clean_signal_long(df_long)
    df_long = df_long.sort_values(["stat", "roi", "direction", "b_step"], kind="stable").reset_index(drop=True)

    # guardar
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = args.out_dir / sheet
    exp_dir.mkdir(parents=True, exist_ok=True)

    out_path = exp_dir / (args.results_file.stem + ".long.parquet")
    df_long.to_parquet(out_path, index=False)
    df_long.to_excel(out_path.with_suffix(".xlsx"), index=False)

    print("Selected params (clean):")
    print(
        pd.Series(
            {
                "sheet": sheet,
                "protocol": protocol,
                "sequence": sequence,
                "Hz": Hz,
                "bmax": bmax,
                "N": N,
                "delta_ms": delta_ms,
                "Delta_app_ms": Delta_app_ms,
                "max_dur_ms": max_dur_ms,
                "tm_ms": tm_ms,
                "td_ms": td_ms,
                "TE": TE,
                "TR": TR,
            }
        ).to_string()
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
