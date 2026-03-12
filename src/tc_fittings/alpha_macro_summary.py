from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None


def _infer_id_from_filename(p: Path) -> str:
    """
    ID estable desde filename para agrupar replicados exp01/exp02, etc.
    Quita '_expNN' y también cosas de timing muy específicas.
    """
    s = p.stem
    s = re.sub(r"_exp\d+", "", s)
    return s


@dataclass(frozen=True)
class Columns:
    brain: str
    sheet: str | None
    region: str
    direction: str
    td_ms: str
    D0: str


def detect_columns(df: pd.DataFrame) -> Columns:
    cols = list(df.columns)

    brain = _pick_col(cols, ["brain"])
    sheet = _pick_col(cols, ["sheet"])
    region = _pick_col(cols, ["roi", "region", "roi"])
    direction = _pick_col(cols, ["direction", "direction", "dir"])
    td = _pick_col(cols, ["delta_app_ms", "Delta_app_ms", "Td (ms)", "td_ms", "Td_ms", "td_ms_1"])
    D0 = _pick_col(cols, ["D0_mm2_s", "D0_m2_ms", "D0", "D_fit", "D0_fit"])

    if region is None:
        raise KeyError("No pude detectar columna de roi (roi/region/roi).")
    if direction is None:
        raise KeyError("No pude detectar columna de direction (direction/direction).")
    if td is None:
        raise KeyError("No pude detectar columna de tiempo (delta_app_ms/Delta_app_ms/td_ms/Td (ms)).")
    if D0 is None:
        raise KeyError("No pude detectar columna de difusión (D0_mm2_s / D0_m2_ms / D0).")

    # brain puede faltar -> se llena luego
    return Columns(
        brain=brain or "__MISSING__",
        sheet=sheet,
        region=region,
        direction=direction,
        td_ms=td,
        D0=D0,
    )


def read_fit_params(files: list[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        if p.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        elif p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            continue

        df["__source_file"] = str(p)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No encontré archivos de fit_params (.xlsx/.parquet/.csv).")

    return pd.concat(frames, ignore_index=True)


def compute_alpha_macro_powerlaw(
    df_fit_params: pd.DataFrame,
    *,
    min_points: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Agrega D0 promedio vs td (delta_app_ms) por (brain, roi, direction, td)
    2) Ajusta ley de potencia: D0(td) = a * td^{-alpha_macro}
       => log D0 = log a - alpha_macro * log td

    Devuelve:
      - df_avg: tabla D0_mean vs td
      - df_summary: summary_alpha_values.xlsx compatible
    """
    df = df_fit_params.copy()
    cols = detect_columns(df)

    # normalizar columnas básicas
    df["roi"] = _as_str(df[cols.region]).str.replace("_norm", "", regex=False)
    df["direction"] = _as_str(df[cols.direction])

    df["td_ms"] = pd.to_numeric(df[cols.td_ms], errors="coerce")
    df["D0_raw"] = pd.to_numeric(df[cols.D0], errors="coerce")

    # brain (si no existe, inferir desde filename o sheet)
    if cols.brain != "__MISSING__":
        df["brain"] = _as_str(df[cols.brain])
    elif cols.sheet is not None:
        df["brain"] = _as_str(df[cols.sheet])
    else:
        df["brain"] = df["__source_file"].astype(str).apply(lambda s: _infer_id_from_filename(Path(s)))

    # sheet opcional (para matching alternativo)
    if cols.sheet is not None:
        df["sheet"] = _as_str(df[cols.sheet])
    else:
        df["sheet"] = ""

    # limpiar
    df = df[np.isfinite(df["td_ms"]) & (df["td_ms"] > 0) & np.isfinite(df["D0_raw"]) & (df["D0_raw"] > 0)].copy()
    if df.empty:
        raise ValueError("Luego de limpiar NaNs/<=0, no quedó data para alpha_macro.")

    # promedio por td
    gcols = ["brain", "sheet", "roi", "direction", "td_ms"]
    df_avg = (
        df.groupby(gcols, as_index=False)
          .agg(D0_mean=("D0_raw", "mean"), D0_std=("D0_raw", "std"), n=("D0_raw", "count"))
    )
    df_avg["D0_se"] = df_avg["D0_std"] / np.sqrt(df_avg["n"].clip(lower=1))

    # fit por grupo (brain, roi, direction)
    rows = []
    for (brain, sheet, region, direc), sub in df_avg.groupby(["brain", "sheet", "roi", "direction"], sort=False):
        sub = sub.sort_values("td_ms")
        # requiere puntos distintos
        if sub["td_ms"].nunique() < min_points:
            continue

        x = np.log(sub["td_ms"].to_numpy(float))
        y = np.log(sub["D0_mean"].to_numpy(float))

        # pesos ~ 1/var(log D) ; var(log D) ≈ (se/D)^2
        se = sub["D0_se"].to_numpy(float)
        D = sub["D0_mean"].to_numpy(float)
        se_log = np.where((se > 0) & (D > 0), se / D, np.nan)
        w = np.where(np.isfinite(se_log) & (se_log > 0), 1.0 / (se_log**2), 1.0)

        # regresión lineal ponderada
        X = np.column_stack([np.ones_like(x), x])
        W = np.diag(w)
        XtWX = X.T @ W @ X
        try:
            beta = np.linalg.solve(XtWX, X.T @ W @ y)
        except np.linalg.LinAlgError:
            continue

        yhat = X @ beta
        resid = y - yhat
        dof = max(1, len(y) - 2)
        s2 = float((resid.T @ W @ resid) / dof)

        try:
            cov = np.linalg.inv(XtWX) * s2
            se_beta = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se_beta = np.array([np.nan, np.nan])

        intercept, slope = float(beta[0]), float(beta[1])
        intercept_se, slope_se = float(se_beta[0]), float(se_beta[1])

        alpha_macro = float(-slope)
        alpha_macro_err = float(slope_se) if np.isfinite(slope_se) else np.nan

        rows.append({
            "brain": str(brain),
            "sheet": str(sheet),
            "roi": str(region),
            "direction": str(direc),
            "alpha_macro": alpha_macro,
            "alpha_macro_error": alpha_macro_err,
            "model": "powerlaw_loglog",
            "n_points": int(sub["td_ms"].nunique()),
            "td_min_ms": float(sub["td_ms"].min()),
            "td_max_ms": float(sub["td_ms"].max()),
            "D0_at_1ms": float(np.exp(intercept)),  # D0 cuando td=1ms (en las unidades de D0)
        })

    df_summary = pd.DataFrame(rows)
    return df_avg, df_summary


def write_alpha_macro_outputs(
    df_avg: pd.DataFrame,
    df_summary: pd.DataFrame,
    *,
    out_summary_xlsx: Path,
    out_avg_xlsx: Path | None = None,
) -> None:
    out_summary_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_excel(out_summary_xlsx, index=False)
    if out_avg_xlsx is not None:
        out_avg_xlsx.parent.mkdir(parents=True, exist_ok=True)
        df_avg.to_excel(out_avg_xlsx, index=False)