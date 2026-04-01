from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------
# IO + discovery
# ---------------------------
def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == '.parquet':
        return pd.read_parquet(path)
    if suf == '.csv':
        return pd.read_csv(path)
    try:
        return pd.read_excel(path, sheet_name='fit_params', engine='openpyxl')
    except Exception:
        return pd.read_excel(path, sheet_name=0, engine='openpyxl')


def _table_priority(path: Path) -> tuple[int, str]:
    suffix_priority = {
        '.parquet': 0,
        '.csv': 1,
        '.xlsx': 2,
        '.xls': 3,
    }
    return suffix_priority.get(path.suffix.lower(), 99), path.name


def _dedupe_table_files(files: Iterable[Path]) -> list[Path]:
    preferred: dict[tuple[Path, str], Path] = {}
    for path in files:
        key = (path.parent, path.stem)
        current = preferred.get(key)
        if current is None or _table_priority(path) < _table_priority(current):
            preferred[key] = path
    return sorted(preferred.values())


def _discover_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Root no existe: {root.resolve()}')
    if root.is_file():
        return [root]

    files: set[Path] = set()
    for pat in patterns:
        files.update(p for p in root.rglob(pat) if p.is_file())
    return _dedupe_table_files(sorted(files))


def _infer_exp_id(p: Path) -> str:
    name = p.name
    lower = name.lower()
    if '.fit_' in lower:
        return name.split('.fit_', 1)[0]
    if lower.startswith('fit_params'):
        return p.parent.name
    if name.endswith('.fit_params.csv'):
        return name[: -len('.fit_params.csv')]
    if name.endswith('.fit_params.xlsx'):
        return name[: -len('.fit_params.xlsx')]
    if name.endswith('.fit_params.parquet'):
        return name[: -len('.fit_params.parquet')]
    return p.stem


def _canonical_sheet(sheet: str) -> str:
    s = str(sheet).strip()
    if not s:
        return s

    for token in ['_duration', '_dur', '_version', '_ver']:
        idx = s.find(token)
        if idx > 0:
            s = s[:idx]
            break

    m = re.match(r'^(\d{8}[-_][^_]+)', s)
    if m:
        return m.group(1)

    m = re.match(r'^(.+?)_(?:N\d|td)', s)
    if m:
        return m.group(1)

    return s


def _infer_sheet_from_df_or_id(
    df: pd.DataFrame,
    exp_id: str,
    *,
    canonicalize: bool = True,
    source_path: Path | None = None,
) -> str:
    for c in ['sheet', 'sheet_1']:
        if c in df.columns:
            u = pd.Series(df[c]).dropna().astype(str).unique()
            if len(u) == 1:
                s = str(u[0])
                return _canonical_sheet(s) if canonicalize else s

    candidates: list[str] = []
    if exp_id:
        candidates.append(exp_id)
    if source_path is not None:
        candidates.append(source_path.parent.name)
        parent = source_path.parent
        if parent.parent != parent:
            candidates.append(parent.parent.name)

    for candidate in candidates:
        candidate = str(candidate).strip()
        if not candidate:
            continue
        normalized = _canonical_sheet(candidate) if canonicalize else candidate
        if re.match(r'^\d{8}_[^_]+$', normalized):
            return normalized

    if candidates:
        candidate = str(candidates[0]).strip()
        return _canonical_sheet(candidate) if canonicalize else candidate

    return _canonical_sheet(exp_id) if canonicalize else exp_id


def _infer_subj_from_df_or_id(
    df: pd.DataFrame,
    exp_id: str,
    *,
    source_path: Path | None = None,
) -> str:
    for c in ['subj', 'subject', 'brain']:
        if c in df.columns:
            u = pd.Series(df[c]).dropna().astype(str).str.strip().unique()
            if len(u) == 1 and u[0]:
                return str(u[0])

    candidates: list[str] = []
    if exp_id:
        candidates.append(exp_id)
    if source_path is not None:
        candidates.append(source_path.parent.name)
        parent = source_path.parent
        if parent.parent != parent:
            candidates.append(parent.parent.name)

    for candidate in candidates:
        s = str(candidate).strip()
        if not s:
            continue
        m = re.match(r'^\d{8}[-_](.+?)(?:_|$)', s)
        if m:
            return m.group(1)

    return ''


def _infer_td_ms(df: pd.DataFrame) -> Optional[float]:
    def _uniq(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors='coerce').dropna().unique()
        if len(v) == 1:
            return float(v[0])
        return None

    td = _uniq('td_ms')
    if td is not None:
        return td

    td1 = _uniq('td_ms_1')
    if td1 is not None:
        return td1

    d = _uniq('max_dur_ms')
    tm = _uniq('tm_ms')
    if d is not None and tm is not None:
        return float(2.0 * d + tm)

    d1 = _uniq('max_dur_ms_1')
    tm1 = _uniq('tm_ms_1')
    if d1 is not None and tm1 is not None:
        return float(2.0 * d1 + tm1)

    return None


def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _filter_stat(df: pd.DataFrame, *, stat_col: str, stat_keep: str | None) -> pd.DataFrame:
    if stat_col not in df.columns or stat_keep is None or str(stat_keep).upper() == 'ALL':
        return df

    st = df[stat_col].astype(str).str.lower()
    keep = {str(stat_keep).lower()}
    if str(stat_keep).lower() in {'avg', 'mean'}:
        keep |= {'avg', 'mean'}
    return df[st.isin(keep)].copy()


def _convert_d0_to_m2_ms(series: pd.Series, *, source_col: str, mm2s_scale: float) -> pd.Series:
    out = pd.to_numeric(series, errors='coerce')
    if source_col == 'D0_mm2_s':
        return out * float(mm2s_scale)
    return out


def _td_key_from_series(series: pd.Series, tol_ms: float) -> pd.Series:
    return np.round(pd.to_numeric(series, errors='coerce').astype(float) / float(tol_ms)).astype('Int64')


# ---------------------------
# Column mapping (contrast fits)
# ---------------------------
@dataclass
class ColumnMap:
    roi_col: str = 'roi'
    dir_col: str = 'direction'
    td_col: str = 'td_ms'
    d0_col: str = 'D0_m2_ms'
    stat_col: str = 'stat'
    sheet_col: str = 'sheet'
    n1_col: str = 'N_1'
    n2_col: str = 'N_2'


# ---------------------------
# Monoexp reference
# ---------------------------
def compute_d0_monoexp_reference(
    exp_fits_root: str | Path,
    *,
    roi: str,
    fit_points: Optional[int] = None,
    stat_keep: str = 'avg',
    exp_d0_col: str = 'D0_mm2_s',
    exp_scale: float = 1e-9,
    canonicalize_sheet: bool = True,
) -> pd.DataFrame:
    root = Path(exp_fits_root)
    files = _discover_files(
        root,
        patterns=[
            'fit_params*.csv', 'fit_params*.xlsx', 'fit_params*.xls', 'fit_params*.parquet',
            '*.fit_params.csv', '*.fit_params.xlsx', '*.fit_params.xls', '*.fit_params.parquet',
        ],
    )
    if not files:
        raise FileNotFoundError(f'No encontré tablas monoexp bajo: {root.resolve()}')

    blocks: list[pd.DataFrame] = []
    for f in files:
        df = _read_table(f)
        if df.empty:
            continue

        exp_id = _infer_exp_id(f)
        sheet = _infer_sheet_from_df_or_id(df, exp_id, canonicalize=canonicalize_sheet, source_path=f)
        subj = _infer_subj_from_df_or_id(df, exp_id, source_path=f)

        if 'fit_kind' in df.columns:
            df = df[df['fit_kind'].astype(str) == 'monoexp'].copy()
        if 'ok' in df.columns:
            df = df[df['ok'].fillna(False).astype(bool)].copy()
        if 'roi' not in df.columns:
            continue

        d0_col = exp_d0_col if exp_d0_col in df.columns else _pick_existing(df, ['D0_mm2_s', 'D0_m2_ms', 'D0'])
        if d0_col is None:
            continue

        sub = df.copy()
        sub['roi'] = sub['roi'].astype(str).str.strip()
        sub = sub[sub['roi'] == str(roi).strip()].copy()
        sub = _filter_stat(sub, stat_col='stat', stat_keep=stat_keep)

        if fit_points is not None and 'fit_points' in sub.columns:
            fit_points_num = pd.to_numeric(sub['fit_points'], errors='coerce')
            sub = sub[fit_points_num == int(fit_points)].copy()

        if sub.empty:
            continue

        if 'td_ms' in sub.columns:
            sub['td_ms'] = pd.to_numeric(sub['td_ms'], errors='coerce')
        else:
            sub['td_ms'] = _infer_td_ms(sub)

        sub['subj'] = str(subj).strip()
        sub['sheet'] = sheet
        sub['D0_fit_monoexp'] = _convert_d0_to_m2_ms(sub[d0_col], source_col=d0_col, mm2s_scale=exp_scale)

        keep = sub[['subj', 'sheet', 'roi', 'td_ms', 'D0_fit_monoexp']].copy()
        keep = keep.dropna(subset=['subj', 'sheet', 'roi', 'td_ms', 'D0_fit_monoexp'])
        if not keep.empty:
            blocks.append(keep)

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if out.empty:
        raise ValueError('No pude construir la referencia monoexp por experimento/td_ms. Revisá ROI, columnas y exp_fits_root.')

    out = out.groupby(['subj', 'sheet', 'roi', 'td_ms'], as_index=False).agg(
        D0_fit_monoexp=('D0_fit_monoexp', 'mean'),
        D0_fit_monoexp_std=('D0_fit_monoexp', 'std'),
        n_monoexp=('D0_fit_monoexp', 'size'),
    )
    return out


# ---------------------------
# NOGSE contrast D0
# ---------------------------
def load_nogse_fit_d0(
    nogse_root_or_file: str | Path,
    *,
    roi: str,
    stat_keep: str = 'avg',
    cmap: ColumnMap = ColumnMap(),
    nogse_scale: float = 1.0,
    canonicalize_sheet: bool = True,
) -> pd.DataFrame:
    p = Path(nogse_root_or_file)
    files = _discover_files(
        p,
        patterns=[
            '*.fit_*.parquet',
            'fit_params*.parquet', 'fit_params*.csv', 'fit_params*.xlsx', 'fit_params*.xls',
            '*.fit_params.parquet', '*.fit_params.csv', '*.fit_params.xlsx', '*.fit_params.xls',
        ],
    )
    if not files:
        raise FileNotFoundError(f'No encontré contrast fit tables bajo: {p.resolve()}')

    blocks: list[pd.DataFrame] = []
    for f in files:
        df = _read_table(f)
        if df.empty:
            continue

        exp_id = _infer_exp_id(f)
        if 'fit_kind' in df.columns:
            df = df[df['fit_kind'].astype(str) == 'nogse_contrast'].copy()
        if 'ok' in df.columns:
            df = df[df['ok'].fillna(False).astype(bool)].copy()
        if 'model' in df.columns:
            model = df['model'].astype(str).str.lower()
            if (model == 'free').any():
                df = df[model == 'free'].copy()

        roi_col = cmap.roi_col if cmap.roi_col in df.columns else _pick_existing(df, ['roi'])
        dir_col = cmap.dir_col if cmap.dir_col in df.columns else _pick_existing(df, ['direction'])
        td_col = cmap.td_col if cmap.td_col in df.columns else _pick_existing(df, ['td_ms', 'td_ms_1'])
        d0_col = cmap.d0_col if cmap.d0_col in df.columns else _pick_existing(df, ['D0_m2_ms', 'D0_mm2_s', 'D0'])
        n1_col = cmap.n1_col if cmap.n1_col in df.columns else None
        n2_col = cmap.n2_col if cmap.n2_col in df.columns else None

        if roi_col is None or dir_col is None or d0_col is None or n1_col is None or n2_col is None:
            continue

        sub = df.copy()
        sub = sub[sub[roi_col].astype(str).str.strip() == str(roi).strip()].copy()
        sub = _filter_stat(sub, stat_col=cmap.stat_col, stat_keep=stat_keep)
        if sub.empty:
            continue

        if td_col is not None:
            sub['td_ms'] = pd.to_numeric(sub[td_col], errors='coerce')
        else:
            sub['td_ms'] = _infer_td_ms(sub)

        sub['subj'] = _infer_subj_from_df_or_id(sub, exp_id, source_path=f)
        sub['sheet'] = _infer_sheet_from_df_or_id(sub, exp_id, canonicalize=canonicalize_sheet, source_path=f)
        sub['roi'] = sub[roi_col].astype(str).str.strip()
        sub['direction'] = sub[dir_col].astype(str)
        sub['N_1'] = pd.to_numeric(sub[n1_col], errors='coerce')
        sub['N_2'] = pd.to_numeric(sub[n2_col], errors='coerce')
        sub['D0_fit_nogse'] = _convert_d0_to_m2_ms(sub[d0_col], source_col=d0_col, mm2s_scale=1e-9) * float(nogse_scale)

        keep = sub[['subj', 'sheet', 'roi', 'direction', 'td_ms', 'N_1', 'N_2', 'D0_fit_nogse']].copy()
        keep = keep.dropna(subset=['subj', 'sheet', 'roi', 'direction', 'td_ms', 'N_1', 'N_2', 'D0_fit_nogse'])
        if not keep.empty:
            blocks.append(keep)

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if out.empty:
        raise ValueError('No pude construir D0_fit_nogse. Revisá ruta, columnas, ROI y N_1/N_2.')

    out = out.groupby(['subj', 'sheet', 'roi', 'direction', 'td_ms', 'N_1', 'N_2'], as_index=False).agg(
        D0_fit_nogse=('D0_fit_nogse', 'mean'),
        D0_fit_nogse_std=('D0_fit_nogse', 'std'),
        n_nogse=('D0_fit_nogse', 'size'),
    )
    return out


# ---------------------------
# Matching + correction factor
# ---------------------------
def _format_missing_matches(missing: pd.DataFrame, monoexp_ref: pd.DataFrame) -> str:
    sample_missing = missing[['subj', 'sheet', 'roi', 'td_ms', 'direction', 'N_1', 'N_2']].drop_duplicates().head(10)
    sample_exp = monoexp_ref[['subj', 'sheet', 'roi', 'td_ms']].drop_duplicates().head(20)
    return (
        'Faltan referencias monoexp para algunos fits de contraste.\n\n'
        'Ejemplos faltantes:\n'
        f"{sample_missing.to_string(index=False)}\n\n"
        'Referencias monoexp disponibles (ejemplos):\n'
        f"{sample_exp.to_string(index=False)}"
    )


def _propagate_correction_std(row: pd.Series) -> float:
    corr = row.get('correction_factor', np.nan)
    if not np.isfinite(corr) or corr <= 0:
        return np.nan

    rel_var = 0.0
    used = False
    for mean_col, std_col in [
        ('D0_fit_nogse', 'D0_fit_nogse_std'),
        ('D0_fit_monoexp', 'D0_fit_monoexp_std'),
    ]:
        mean_v = row.get(mean_col, np.nan)
        std_v = row.get(std_col, np.nan)
        if np.isfinite(mean_v) and mean_v > 0 and np.isfinite(std_v) and std_v >= 0:
            rel_var += (float(std_v) / float(mean_v)) ** 2
            used = True

    if not used:
        return np.nan
    return float(0.5 * corr * np.sqrt(rel_var))


def make_grad_correction_table(
    *,
    roi: str,
    exp_fits_root: str | Path,
    nogse_root_or_file: str | Path,
    fit_points: Optional[int] = None,
    stat_keep: str = 'avg',
    exp_d0_col: str = 'D0_mm2_s',
    exp_scale: float = 1e-9,
    nogse_scale: float = 1.0,
    cmap: ColumnMap = ColumnMap(),
    tol_ms: float = 1e-3,
    canonicalize_sheet: bool = True,
) -> pd.DataFrame:
    monoexp_ref = compute_d0_monoexp_reference(
        exp_fits_root,
        roi=roi,
        fit_points=fit_points,
        stat_keep=stat_keep,
        exp_d0_col=exp_d0_col,
        exp_scale=exp_scale,
        canonicalize_sheet=canonicalize_sheet,
    )
    nogse = load_nogse_fit_d0(
        nogse_root_or_file,
        roi=roi,
        stat_keep=stat_keep,
        cmap=cmap,
        nogse_scale=nogse_scale,
        canonicalize_sheet=canonicalize_sheet,
    )

    monoexp_ref = monoexp_ref.copy()
    nogse = nogse.copy()
    monoexp_ref['_td_key'] = _td_key_from_series(monoexp_ref['td_ms'], tol_ms)
    nogse['_td_key'] = _td_key_from_series(nogse['td_ms'], tol_ms)

    out = nogse.merge(
        monoexp_ref[['subj', 'sheet', 'roi', '_td_key', 'D0_fit_monoexp', 'D0_fit_monoexp_std', 'n_monoexp']],
        on=['subj', 'sheet', 'roi', '_td_key'],
        how='left',
    )

    missing = out[out['D0_fit_monoexp'].isna()].copy()
    if not missing.empty:
        raise ValueError(_format_missing_matches(missing, monoexp_ref))

    out['ratio'] = out['D0_fit_nogse'] / out['D0_fit_monoexp']
    out['correction_factor'] = np.sqrt(out['ratio'])
    out['correction_factor_std'] = out.apply(_propagate_correction_std, axis=1)

    cols = [
        'subj', 'sheet', 'roi', 'direction', 'td_ms', 'N_1', 'N_2',
        'D0_fit_nogse', 'D0_fit_nogse_std', 'n_nogse',
        'D0_fit_monoexp', 'D0_fit_monoexp_std', 'n_monoexp',
        'ratio', 'correction_factor', 'correction_factor_std',
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(['subj', 'sheet', 'td_ms', 'N_1', 'N_2', 'direction'], kind='stable').reset_index(drop=True)
    return out
