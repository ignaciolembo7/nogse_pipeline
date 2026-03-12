from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd


# ---------------------------
# IO + discovery
# ---------------------------
def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    # xlsx/xls
    try:
        return pd.read_excel(path, sheet_name="fit_params", engine="openpyxl")
    except Exception:
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")


def _discover_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root no existe: {root.resolve()}")
    if root.is_file():
        return [root]

    files: set[Path] = set()
    for pat in patterns:
        files.update(p for p in root.rglob(pat) if p.is_file())
    return sorted(files)


def _infer_exp_id(p: Path) -> str:
    """
    - Para parquet fit: <analysis_id>.fit_free.g_lin_max.value_norm.parquet -> analysis_id
    - Para fit_params.{csv,xlsx,parquet}: usa parent (que en tu layout es el analysis_id)
    """
    name = p.name
    lower = name.lower()
    if ".fit_" in lower:
        return name.split(".fit_", 1)[0]
    if lower.startswith("fit_params"):
        return p.parent.name
    if name.endswith(".fit_params.csv"):
        return name[: -len(".fit_params.csv")]
    if name.endswith(".fit_params.xlsx"):
        return name[: -len(".fit_params.xlsx")]
    if name.endswith(".fit_params.parquet"):
        return name[: -len(".fit_params.parquet")]
    return p.stem


def _canonical_sheet(sheet: str) -> str:
    """
    Normaliza nombres de 'sheet' entre pipelines.
    Objetivo: que monoexp y contrast usen el MISMO identificador base.
    Ejemplos:
      OGSEvsMax_duration2 -> OGSEvsMax
      OGSEvsMax -> OGSEvsMax
    """
    s = str(sheet).strip()
    # cortar versiones típicas
    for token in ["_duration", "_dur", "_version", "_ver", "_v"]:
        idx = s.find(token)
        if idx > 0:
            return s[:idx]
    # fallback: primer chunk
    return s.split("_")[0]


def _infer_sheet_from_df_or_id(df: pd.DataFrame, exp_id: str, *, canonicalize: bool = True) -> str:
    """
    Preferimos sheet explícito en la tabla.
    - NOGSE fit: suele traer sheet_1/sheet_2
    - Monoexp fit: suele traer sheet
    Si no existe: parseo simple del exp_id antes de "_N" o "_td" (tu naming típico).
    """
    for c in ["sheet", "sheet_1"]:
        if c in df.columns:
            u = pd.Series(df[c]).dropna().astype(str).unique()
            if len(u) == 1:
                s = str(u[0])
                return _canonical_sheet(s) if canonicalize else s

    # fallback: <sheet>_N8-N4_td...  o  <sheet>_td...
    m = re.match(r"^([^_]+)_(?:N|td)", exp_id)
    if m:
        s = m.group(1)
        return _canonical_sheet(s) if canonicalize else s

    s = exp_id.split("_")[0]
    return _canonical_sheet(s) if canonicalize else s


def _infer_td_ms(df: pd.DataFrame) -> Optional[float]:
    """
    Orden de preferencia:
      1) td_ms
      2) td_ms_1 (si existe y es único)
      3) 2*max_dur_ms + tm_ms (si ambas únicas)
      4) 2*max_dur_ms_1 + tm_ms_1
    """
    def _uniq(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        if len(v) == 1:
            return float(v[0])
        return None

    td = _uniq("td_ms")
    if td is not None:
        return td

    td1 = _uniq("td_ms_1")
    if td1 is not None:
        return td1

    # compute from max_dur + tm
    d = _uniq("max_dur_ms")
    tm = _uniq("tm_ms")
    if d is not None and tm is not None:
        return float(2.0 * d + tm)

    d1 = _uniq("max_dur_ms_1")
    tm1 = _uniq("tm_ms_1")
    if d1 is not None and tm1 is not None:
        return float(2.0 * d1 + tm1)

    return None


def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------
# Column mapping (for contrast fits)
# ---------------------------
@dataclass
class ColumnMap:
    roi_col: str = "roi"
    dir_col: str = "direction"
    td_col: str = "td_ms"
    d0_col: str = "D0_m2_ms"     # default coherente con fit_ogse_contrast.py
    stat_col: str = "stat"       # opcional
    sheet_col: str = "sheet"     # opcional (si no está, se infiere)
    # opcional informativo
    n1_col: str = "N1"
    n2_col: str = "N2"


# ---------------------------
# Core computations
# ---------------------------
def compute_d0_exp_mean(
    exp_fits_root: str | Path,
    *,
    roi: str,
    fit_points: Optional[int] = None,
    stat_keep: str = "avg",
    exp_d0_col: str = "D0_mm2_s",
    exp_scale: float = 1e-9,  # mm²/s -> m²/ms
    canonicalize_sheet: bool = True,
) -> pd.DataFrame:
    """
    Lee monoexp fits y devuelve una tabla por (sheet, td_ms):
      D0_fit_monoexp = promedio sobre N y direcciones (y replicados), en m²/ms
    """
    root = Path(exp_fits_root)

    # IMPORTANT: no busques *.parquet genérico porque suele incluir tablas grandes (fit_points, datos, etc.)
    files = _discover_files(root, patterns=[
        "fit_params*.csv", "fit_params*.xlsx", "fit_params*.xls", "fit_params*.parquet",
        "*.fit_params.csv", "*.fit_params.xlsx", "*.fit_params.xls", "*.fit_params.parquet",
    ])
    if not files:
        raise FileNotFoundError(f"No encontré tablas bajo: {root.resolve()}")

    rows = []
    for f in files:
        df = _read_table(f)
        exp_id = _infer_exp_id(f)
        sheet = _infer_sheet_from_df_or_id(df, exp_id, canonicalize=canonicalize_sheet)
        td_ms = _infer_td_ms(df)
        if td_ms is None:
            continue

        if "roi" not in df.columns:
            continue

        # D0 col: permitimos alias por coherencia
        d0_col = exp_d0_col if exp_d0_col in df.columns else _pick_existing(df, ["D0_mm2_s", "D0_m2_ms", "D0"])
        if d0_col is None:
            continue

        sub = df.copy()
        sub_roi = sub["roi"].astype(str).str.strip()
        sub = sub[sub_roi == str(roi).strip()]

        if "stat" in sub.columns and stat_keep is not None and str(stat_keep).upper() != "ALL":
            st = sub["stat"].astype(str).str.lower()
            keep = [str(stat_keep).lower(), "mean", "avg"]
            sub = sub[st.isin(keep)]

        if fit_points is not None and "fit_points" in sub.columns:
            sub = sub[sub["fit_points"] == int(fit_points)]

        sub[d0_col] = pd.to_numeric(sub[d0_col], errors="coerce")
        sub = sub.dropna(subset=[d0_col])
        if sub.empty:
            continue

        # convertir a unidad común m²/ms
        d0_vals = sub[d0_col].to_numpy(dtype=float)
        if d0_col == "D0_mm2_s":
            d0_vals = d0_vals * float(exp_scale)

        rows.append(
            {
                "sheet": sheet,
                "td_ms": float(td_ms),
                "D0_fit_monoexp": float(np.mean(d0_vals)),
                "D0_fit_monoexp_std": float(np.std(d0_vals, ddof=1)) if len(d0_vals) > 1 else np.nan,
                "n_monoexp": int(len(d0_vals)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No pude construir D0_fit_monoexp. Revisá ROI, columnas y exp_fits_root.")

    # promediamos por (sheet, td) por si hay múltiples archivos por el mismo td
    out = out.groupby(["sheet", "td_ms"], as_index=False).agg(
        D0_fit_monoexp=("D0_fit_monoexp", "mean"),
        D0_fit_monoexp_std=("D0_fit_monoexp_std", "mean"),
        n_monoexp=("n_monoexp", "sum"),
    )
    return out


def load_nogse_free_fits(
    nogse_root_or_file: str | Path,
    *,
    roi: str,
    cmap: ColumnMap = ColumnMap(),
    nogse_scale: float = 1.0,  # si ya viene en m²/ms, dejar 1.0
    canonicalize_sheet: bool = True,
) -> pd.DataFrame:
    """
    Lee fits del contraste (free NOGSE/OGSE contrast) y devuelve:
      sheet, roi, direction, td_ms, D0_fit_nogse (+ opcional N1,N2)
    """
    p = Path(nogse_root_or_file)
    files = _discover_files(p, patterns=[
        "*.fit_*.parquet",  # si alguna vez guardás así
        "fit_params*.parquet", "fit_params*.csv", "fit_params*.xlsx", "fit_params*.xls",
        "*.fit_params.parquet", "*.fit_params.csv", "*.fit_params.xlsx", "*.fit_params.xls",
    ])
    if not files:
        raise FileNotFoundError(f"No encontré contrast fit tables bajo: {p.resolve()}")

    blocks = []
    for f in files:
        df = _read_table(f)
        exp_id = _infer_exp_id(f)

        roi_col = cmap.roi_col if cmap.roi_col in df.columns else _pick_existing(df, ["roi"])
        dir_col = cmap.dir_col if cmap.dir_col in df.columns else _pick_existing(df, ["direction"])
        d0_col = cmap.d0_col if cmap.d0_col in df.columns else _pick_existing(df, ["D0_m2_ms", "D0_mm2_s", "D0"])

        if roi_col is None or dir_col is None or d0_col is None:
            continue

        sub = df.copy()
        sub = sub[sub[roi_col].astype(str).str.strip() == str(roi).strip()]

        # stat
        if cmap.stat_col in sub.columns:
            st = sub[cmap.stat_col].astype(str).str.lower()
            sub = sub[st.isin(["avg", "mean"])]

        # td
        if cmap.td_col in sub.columns:
            sub[cmap.td_col] = pd.to_numeric(sub[cmap.td_col], errors="coerce")
        else:
            td_ms = _infer_td_ms(sub)
            sub[cmap.td_col] = td_ms

        sub[d0_col] = pd.to_numeric(sub[d0_col], errors="coerce") * float(nogse_scale)

        sub = sub.dropna(subset=[cmap.td_col, d0_col])
        if sub.empty:
            continue

        sheet = _infer_sheet_from_df_or_id(sub, exp_id, canonicalize=canonicalize_sheet)
        sub["sheet"] = sheet

        sub[dir_col] = sub[dir_col].astype(str)

        # optional N1/N2
        n1 = cmap.n1_col if cmap.n1_col in sub.columns else None
        n2 = cmap.n2_col if cmap.n2_col in sub.columns else None

        cols = ["sheet", roi_col, dir_col, cmap.td_col, d0_col]
        rename = {roi_col: "roi", dir_col: "direction", cmap.td_col: "td_ms", d0_col: "D0_fit_nogse"}

        if n1 is not None:
            cols.append(n1); rename[n1] = "N1"
        if n2 is not None:
            cols.append(n2); rename[n2] = "N2"

        blocks.append(sub[cols].rename(columns=rename))

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if out.empty:
        raise ValueError("No pude construir D0_fit_nogse. Revisá ruta, columnas y ROI.")
    return out


def make_grad_correction_table(
    *,
    roi: str,
    exp_fits_root: str | Path,
    nogse_root_or_file: str | Path,
    fit_points: Optional[int] = None,
    stat_keep: str = "avg",
    exp_d0_col: str = "D0_mm2_s",
    exp_scale: float = 1e-9,
    nogse_scale: float = 1.0,
    cmap: ColumnMap = ColumnMap(),
    tol_ms: float = 1e-3,
    canonicalize_sheet: bool = True,
    average_over_N: bool = True,
) -> pd.DataFrame:
    """
    Output canónico:
      sheet, roi, direction, td_ms, D0_fit_nogse, D0_fit_monoexp, correction_factor
    (+ columnas de QA: std y counts)

    correction_factor = sqrt(D0_fit_nogse / D0_fit_monoexp)
    y luego en el fit del contraste multiplicás G1,G2 por correction_factor.
    """
    exp_mean = compute_d0_exp_mean(
        exp_fits_root,
        roi=roi,
        fit_points=fit_points,
        stat_keep=stat_keep,
        exp_d0_col=exp_d0_col,
        exp_scale=exp_scale,
        canonicalize_sheet=canonicalize_sheet,
    )

    nogse = load_nogse_free_fits(
        nogse_root_or_file,
        roi=roi,
        cmap=cmap,
        nogse_scale=nogse_scale,
        canonicalize_sheet=canonicalize_sheet,
    )

    exp_mean = exp_mean.copy()
    nogse = nogse.copy()

    exp_mean["_td_key"] = np.round(exp_mean["td_ms"].astype(float) / float(tol_ms)).astype(int)
    nogse["_td_key"] = np.round(nogse["td_ms"].astype(float) / float(tol_ms)).astype(int)

    # merge by (sheet, td)
    out = nogse.merge(exp_mean, on=["sheet", "_td_key"], how="left", suffixes=("", "_exp"))
    out = out.drop(columns=["td_ms_exp"], errors="ignore")  # preservo td_ms de nogse

    if out["D0_fit_monoexp"].isna().any():
        miss = out[out["D0_fit_monoexp"].isna()][["sheet", "td_ms"]].drop_duplicates()
        sheets_nogse = sorted(out["sheet"].dropna().astype(str).unique().tolist())[:25]
        sheets_exp = sorted(exp_mean["sheet"].dropna().astype(str).unique().tolist())[:25]
        raise ValueError(
            "Faltan D0_fit_monoexp para algunos (sheet, td_ms).\n"
            "Ejemplos:\n" + miss.head(10).to_string(index=False) + "\n\n"
            f"Sheets en nogse: {sheets_nogse}\n"
            f"Sheets en monoexp: {sheets_exp}\n"
            "Tip: si los nombres difieren (p.ej. OGSEvsMax_duration2 vs OGSEvsMax) usá canonicalize_sheet=True."
        )

    out["ratio"] = out["D0_fit_nogse"] / out["D0_fit_monoexp"]

    if average_over_N:
        # promediamos sobre N1/N2 (y cualquier duplicado) pero mantenemos direction (porque el factor puede ser direccional)
        gcols = ["sheet", "roi", "direction", "td_ms"]
        out = out.groupby(gcols, as_index=False).agg(
            D0_fit_nogse=("D0_fit_nogse", "mean"),
            D0_fit_nogse_std=("D0_fit_nogse", "std"),
            n_nogse=("D0_fit_nogse", "size"),
            D0_fit_monoexp=("D0_fit_monoexp", "first"),
            D0_fit_monoexp_std=("D0_fit_monoexp_std", "first"),
            n_monoexp=("n_monoexp", "first"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
        )
        out["correction_factor"] = np.sqrt(out["ratio_mean"])
        # error aprox (si ratio_std existe)
        out["correction_factor_std"] = 0.5 * out["ratio_std"] / np.sqrt(out["ratio_mean"])
        out = out.drop(columns=["ratio_mean", "ratio_std"])
    else:
        out["correction_factor"] = np.sqrt(out["ratio"])
        out["correction_factor_std"] = np.nan

    cols = [
        "sheet", "roi", "direction", "td_ms",
        "D0_fit_nogse", "D0_fit_nogse_std", "n_nogse",
        "D0_fit_monoexp", "D0_fit_monoexp_std", "n_monoexp",
        "correction_factor", "correction_factor_std",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(["sheet", "td_ms", "direction"], kind="stable").reset_index(drop=True)
    return out
