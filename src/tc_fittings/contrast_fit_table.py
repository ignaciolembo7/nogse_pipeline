from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from tools.brain_labels import canonical_sheet_name, infer_subj_label

def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported format: {path}")


def _score_fit_params_path(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return 0
    if suffix in {".xlsx", ".xls"}:
        return 1
    if suffix == ".csv":
        return 2
    return 99


def discover_fit_param_files(root: str | Path, pattern: str = "**/fit_params.*") -> list[Path]:
    base = Path(root)
    if base.is_file():
        return [base]

    candidates = [
        p for p in base.glob(pattern)
        if p.name.startswith("fit_params.") and p.suffix.lower() in {".parquet", ".xlsx", ".xls", ".csv"}
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find fit_params in {base} with pattern={pattern!r}")

    best_by_dir: dict[Path, Path] = {}
    for path in sorted(candidates):
        parent = path.parent
        current = best_by_dir.get(parent)
        if current is None or _score_fit_params_path(path) < _score_fit_params_path(current):
            best_by_dir[parent] = path
    return sorted(best_by_dir.values())


def canonicalize_contrast_fit_params(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "analysis_id" not in out.columns and "source_file" in out.columns:
        out["analysis_id"] = out["source_file"].astype(str).map(lambda s: Path(s).stem.replace(".long", ""))
    if "sheet" not in out.columns:
        if "sheet_1" in out.columns:
            out["sheet"] = out["sheet_1"]
        elif "sheet_2" in out.columns:
            out["sheet"] = out["sheet_2"]
        else:
            out["sheet"] = np.nan
    out["sheet"] = out["sheet"].map(canonical_sheet_name)

    if "subj" not in out.columns:
        src = out["source_file"] if "source_file" in out.columns else pd.Series([""] * len(out))
        out["subj"] = [infer_subj_label(sheet, source_name=str(source)) for sheet, source in zip(out["sheet"], src)]
    out["subj"] = out["subj"].astype(str)

    if "td_ms" not in out.columns:
        if "td_ms_1" in out.columns:
            out["td_ms"] = pd.to_numeric(out["td_ms_1"], errors="coerce")
        elif "td_ms_2" in out.columns:
            out["td_ms"] = pd.to_numeric(out["td_ms_2"], errors="coerce")

    if "tc_peak_ms" not in out.columns and "tc_at_max" in out.columns:
        out["tc_peak_ms"] = pd.to_numeric(out["tc_at_max"], errors="coerce")
    if "tc_fit_ms" not in out.columns and "tc_ms" in out.columns:
        out["tc_fit_ms"] = pd.to_numeric(out["tc_ms"], errors="coerce")
    if "signal_peak" not in out.columns and "signal_max" in out.columns:
        out["signal_peak"] = pd.to_numeric(out["signal_max"], errors="coerce")
    if "lcf_peak_m" not in out.columns and "lcf_at_max" in out.columns:
        out["lcf_peak_m"] = pd.to_numeric(out["lcf_at_max"], errors="coerce")
    if "x_peak_raw_mTm" not in out.columns and "g_at_max" in out.columns:
        out["x_peak_raw_mTm"] = pd.to_numeric(out["g_at_max"], errors="coerce")

    numeric_cols = [
        "td_ms",
        "tc_ms",
        "tc_fit_ms",
        "tc_err_ms",
        "tc_peak_ms",
        "tc_peak_resampled_ms",
        "tc_peak_resampled_data_ms",
        "signal_peak",
        "signal_peak_resampled_data",
        "g_peak_resampled_data_corr_mTm",
        "f_corr",
        "f_corr_1",
        "f_corr_2",
        "peak_fraction",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "direction" in out.columns:
        out["direction"] = out["direction"].astype(str).str.strip()
    if "roi" in out.columns:
        out["roi"] = out["roi"].astype(str).str.replace("_norm", "", regex=False)
    if "model" in out.columns:
        out["model"] = out["model"].astype(str).str.strip()

    return out


def _fitresamp_mask(df: pd.DataFrame) -> pd.Series:
    markers = (
        "_fitresamp-",
        "fitresamp-",
        "fitted_resampled",
        "contrast-data-fitresampled",
        "ogse_contrast_vs_gresampled",
        "nogse_contrast_vs_gresampled",
    )
    mask = pd.Series(False, index=df.index)
    for col in ("analysis_id", "source_file", "__fit_params_path"):
        if col not in df.columns:
            continue
        values = df[col].astype(str).str.lower()
        col_mask = pd.Series(False, index=df.index)
        for marker in markers:
            col_mask = col_mask | values.str.contains(marker, regex=False, na=False)
        mask = mask | col_mask
    return mask


def load_contrast_fit_params(
    roots_or_files: Sequence[str | Path],
    *,
    pattern: str = "**/fit_params.*",
    models: Sequence[str] | None = None,
    subjs: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    rois: Sequence[str] | None = None,
    exclude_fitresamp: bool = False,
    only_fitresamp: bool = False,
    ok_only: bool = True,
) -> pd.DataFrame:
    files: list[Path] = []
    for item in roots_or_files:
        path = Path(item)
        if path.is_file():
            files.append(path)
        else:
            files.extend(discover_fit_param_files(path, pattern=pattern))

    if not files:
        raise FileNotFoundError("Could not find fit_params to load.")

    frames = []
    for path in files:
        frame = canonicalize_contrast_fit_params(_read_table(path))
        frame["__fit_params_path"] = str(path)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)

    if exclude_fitresamp and only_fitresamp:
        raise ValueError("exclude_fitresamp and only_fitresamp are mutually exclusive.")
    if exclude_fitresamp or only_fitresamp:
        is_fitresamp = _fitresamp_mask(out)
        if exclude_fitresamp:
            out = out[~is_fitresamp].copy()
        if only_fitresamp:
            out = out[is_fitresamp].copy()
            if out.empty:
                raise ValueError("No fitted/resampled fit_params remained after applying only_fitresamp.")

    if ok_only and "ok" in out.columns:
        out = out[out["ok"].fillna(False).astype(bool)].copy()
    if models is not None and "model" in out.columns:
        out = out[out["model"].isin([str(x) for x in models])].copy()
    if subjs is not None and "subj" in out.columns:
        out = out[out["subj"].isin([str(x) for x in subjs])].copy()
    if directions is not None and "direction" in out.columns:
        out = out[out["direction"].isin([str(x) for x in directions])].copy()
    if rois is not None and "roi" in out.columns:
        out = out[out["roi"].isin([str(x) for x in rois])].copy()

    dedup_cols = [
        c
        for c in [
            "analysis_id",
            "sheet",
            "subj",
            "roi",
            "direction",
            "stat",
            "td_ms",
            "N_1",
            "N_2",
            "model",
            "gbase",
            "ycol",
            "xplot",
        ]
        if c in out.columns
    ]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="last")

    out = out.drop(columns=["__fit_params_path"], errors="ignore")
    return out.reset_index(drop=True)


def ensure_required_target_column(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    out = canonicalize_contrast_fit_params(df)
    if y_col not in out.columns:
        raise KeyError(f"y_col={y_col!r} does not exist in fit_params. Columns: {list(out.columns)}")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    return out
