from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ResultMeta:
    sheet: str | None
    seq: int | None
    Hz: float | None
    bmax: float | None

    # legacy: _d55 (ANTES; lo tratamos como max_dur_ms)
    d_ms: float | None

    # nuevo: _d1.5 (delta) y _Delta11.6 (Delta_app)
    delta_ms: float | None
    Delta_ms: float | None

    ndirs: int | None
    nbvals: int | None
    encoding: str | None


def parse_results_filename(path: str | Path) -> ResultMeta:
    p = Path(path)
    name = p.name

    if "_ep2d" in name:
        sheet = name.split("_ep2d")[0]
    elif "_d" in name:
        sheet = name.split("_d")[0]
    elif "_Delta" in name:
        sheet = name.split("_Delta")[0]
    else:
        sheet = p.stem.split("_")[0]
    sheet = re.sub(r"_(\d+)bval_(\d+)dir.*$", "", sheet, flags=re.IGNORECASE)

    def _int(rx: str) -> int | None:
        m = re.search(rx, name, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _float(rx: str) -> float | None:
        m = re.search(rx, name, re.IGNORECASE)
        return float(m.group(1)) if m else None

    nbvals = _int(r"(\d+)bval")
    ndirs  = _int(r"(\d+)dir")

    # nuevo: _d1.5 y _Delta11.6  (solo interpretamos _d como delta_ms si también hay _Delta)
    delta_ms = _float(r"_d(\d+(?:\.\d+)?)") if re.search(r"_Delta", name, re.IGNORECASE) else None
    Delta_ms = _float(r"_Delta(\d+(?:\.\d+)?)")

    # legacy: _d55 (solo si NO hay _Delta...)
    d_ms = _int(r"_d(\d+)") if delta_ms is None else None

    Hz   = _float(r"Hz(\d+(?:\.\d+)?)")
    bmax = _float(r"_b(\d+(?:\.\d+)?)")

    seq  = _int(r"_(\d+)_results")
    encoding = "OGSE" if re.search(r"OGSE", name, re.IGNORECASE) else ("PGSE" if re.search(r"PGSE", name, re.IGNORECASE) else None)

    return ResultMeta(
        sheet=sheet, seq=seq, Hz=Hz, bmax=bmax, d_ms=(float(d_ms) if d_ms is not None else None),
        delta_ms=delta_ms, Delta_ms=Delta_ms,
        ndirs=ndirs, nbvals=nbvals, encoding=encoding
    )


def _norm_sheet(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("-", "_")
    s = re.sub(r"_(\d+)BVAL_(\d+)DIR.*$", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _filter_close(df: pd.DataFrame, col: str, target: float, *, atol: float = 1e-6) -> pd.DataFrame:
    if col not in df.columns:
        return df
    v = pd.to_numeric(df[col], errors="coerce")
    return df[np.isclose(v.to_numpy(float), float(target), rtol=0.0, atol=float(atol))]


def select_params_row(params: pd.DataFrame, meta: ResultMeta) -> pd.Series:
    """
    Matching consistente con nombres canónicos:
      - meta.delta_ms  -> params.delta_ms
      - meta.Delta_ms  -> params.delta_app_ms
      - meta.d_ms (legacy) -> params.max_dur_ms (o params.d_ms si existiera)
    """
    df = params.copy()

    # 1) sheet
    if meta.sheet and "sheet" in df.columns:
        target = _norm_sheet(meta.sheet)
        s_norm = df["sheet"].map(_norm_sheet)
        cand = df[s_norm == target]
        if cand.empty:
            cand = df[s_norm.apply(lambda ss: target.startswith(ss) or ss.startswith(target))]
        if not cand.empty:
            df = cand

    # 2) seq
    if meta.seq is not None and "seq" in df.columns:
        cand = df[pd.to_numeric(df["seq"], errors="coerce").fillna(-1).astype(int) == int(meta.seq)]
        if len(cand) == 1:
            return cand.iloc[0]
        if len(cand) > 1:
            df = cand

    # 3) Hz / bmax
    Hz = 0 if (meta.Hz is None and meta.encoding == "PGSE") else meta.Hz
    if Hz is not None and "Hz" in df.columns:
        hzcol = pd.to_numeric(df["Hz"], errors="coerce")
        if float(Hz) == 0.0:
            df = df[(hzcol.isna()) | (np.isclose(hzcol.to_numpy(float), 0.0, rtol=0.0, atol=1e-9))]
        else:
            df = _filter_close(df, "Hz", float(Hz), atol=1e-6)

    if meta.bmax is not None and "bmax" in df.columns:
        df = _filter_close(df, "bmax", float(meta.bmax), atol=1e-6)

    # 4) tiempos
    if meta.delta_ms is not None and "delta_ms" in df.columns:
        df = _filter_close(df, "delta_ms", float(meta.delta_ms), atol=1e-3)

    if meta.Delta_ms is not None and "delta_app_ms" in df.columns:
        df = _filter_close(df, "delta_app_ms", float(meta.Delta_ms), atol=1e-3)

    if meta.d_ms is not None:
        if "max_dur_ms" in df.columns:
            df = _filter_close(df, "max_dur_ms", float(meta.d_ms), atol=1e-3)
        elif "d_ms" in df.columns:
            df = _filter_close(df, "d_ms", float(meta.d_ms), atol=1e-3)

    if df.empty:
        raise ValueError("No encontré ninguna fila de parámetros que matchee este archivo.")
    if len(df) > 1:
        if "protocol" in df.columns:
            df = df.sort_values("protocol", kind="stable")
        return df.iloc[0]
    return df.iloc[0]