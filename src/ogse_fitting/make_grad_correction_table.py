from __future__ import annotations

import re
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

from nogse_models.nogse_model_fitting import M_nogse_free


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


def _int_key_from_series(series: pd.Series) -> pd.Series:
    return np.round(pd.to_numeric(series, errors='coerce')).astype('Int64')


# ---------------------------
# Column mapping (contrast fits)
# ---------------------------
@dataclass
class ColumnMap:
    roi_col: str = 'roi'
    dir_col: str = 'direction'
    td_col: str = 'td_ms'
    d0_col: str = 'D0_m2_ms'
    d0_err_col: str = 'D0_err_m2_ms'
    stat_col: str = 'stat'
    sheet_col: str = 'sheet'
    n1_col: str = 'N_1'
    n2_col: str = 'N_2'


@dataclass(frozen=True)
class GradCorrectionOutputs:
    grad_correction: pd.DataFrame
    ogse_signal_fits: pd.DataFrame


VALID_GBASES = {'g', 'g_max', 'g_lin_max', 'g_thorsten'}


def _normalize_gbase(gbase: str) -> str:
    b = str(gbase).strip()
    if b.endswith('_1') or b.endswith('_2'):
        b = b[:-2]
    if b not in VALID_GBASES:
        raise ValueError(f'Unrecognized gbase {gbase!r}. Allowed values: {sorted(VALID_GBASES)}.')
    return b


def _side_col(base: str, side: int) -> str:
    return f'{_normalize_gbase(base)}_{int(side)}'


def _maybe_scale_g_thorsten(gbase: str, arr: np.ndarray) -> np.ndarray:
    if _normalize_gbase(gbase) == 'g_thorsten':
        return np.sqrt(2.0) * np.abs(arr)
    return arr


def _analysis_id_from_contrast_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith('.long'):
        stem = stem[: -len('.long')]
    return stem


def _sanitize_token(value: object) -> str:
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value))
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'NA'


def _source_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ''
    name = Path(str(value)).name.strip()
    if not name:
        return ''
    for suffix in ['.parquet', '.xlsx', '.xls', '.csv']:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.lower().endswith('.long'):
        name = name[: -len('.long')]
    for suffix in ['.rot_tensor']:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _source_key_series(series: pd.Series) -> pd.Series:
    return series.map(_source_key).astype(str)


def _infer_contrast_data_root(nogse_root_or_file: str | Path) -> Path | None:
    p = Path(nogse_root_or_file)
    starts = [p] + list(p.parents)

    candidates: list[Path] = []
    for base in starts:
        if base.name == 'fits':
            analysis_root = base.parent
            if 'rotated' in str(p):
                candidates.extend([
                    analysis_root / 'contrast-data-rotated' / 'tables',
                    analysis_root / 'contrast-data' / 'tables',
                ])
            else:
                candidates.extend([
                    analysis_root / 'contrast-data' / 'tables',
                    analysis_root / 'contrast-data-rotated' / 'tables',
                ])
        candidates.extend([
            base / 'contrast-data' / 'tables',
            base / 'contrast-data-rotated' / 'tables',
            base / 'tables',
        ])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _discover_contrast_tables(root_or_file: str | Path) -> list[Path]:
    p = Path(root_or_file)
    if p.is_file():
        return [p]
    root = p / 'tables' if (p / 'tables').is_dir() else p
    return _discover_files(root, patterns=['*.long.parquet'])


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
        raise FileNotFoundError(f'Could not find monoexp tables under: {root.resolve()}')

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
        if 'direction' in sub.columns:
            sub['direction'] = sub['direction'].astype(str)
        else:
            sub['direction'] = ''
        if 'N' in sub.columns:
            sub['N'] = pd.to_numeric(sub['N'], errors='coerce')
        else:
            sub['N'] = np.nan
        if 'source_file' in sub.columns:
            sub['source_file'] = sub['source_file'].astype(str)
            sub['source_key'] = _source_key_series(sub['source_file'])
        else:
            sub['source_file'] = f.parent.name
            sub['source_key'] = _source_key(f.parent.name)
        sub['D0_fit_monoexp'] = _convert_d0_to_m2_ms(sub[d0_col], source_col=d0_col, mm2s_scale=exp_scale)

        keep = sub[['subj', 'sheet', 'roi', 'direction', 'source_file', 'source_key', 'td_ms', 'N', 'D0_fit_monoexp']].copy()
        keep = keep.dropna(subset=['subj', 'sheet', 'roi', 'direction', 'source_key', 'td_ms', 'N', 'D0_fit_monoexp'])
        if not keep.empty:
            blocks.append(keep)

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if out.empty:
        raise ValueError('Could not build the monoexp reference by experiment/td_ms. Check ROI, columns, and exp_fits_root.')

    out = out.groupby(['subj', 'sheet', 'roi', 'direction', 'source_key', 'td_ms', 'N'], as_index=False).agg(
        source_file=('source_file', 'first'),
        D0_fit_monoexp=('D0_fit_monoexp', 'mean'),
        D0_fit_monoexp_std=('D0_fit_monoexp', 'std'),
        n_monoexp=('D0_fit_monoexp', 'size'),
    )
    return out


def load_nogse_expected_keys(
    nogse_root_or_file: str | Path,
    *,
    stat_keep: str = 'avg',
    cmap: ColumnMap = ColumnMap(),
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
        raise FileNotFoundError(f'Could not find contrast fit tables under: {p.resolve()}')

    blocks: list[pd.DataFrame] = []
    for f in files:
        df = _read_table(f)
        if df.empty:
            continue

        exp_id = _infer_exp_id(f)
        if 'fit_kind' in df.columns:
            df = df[df['fit_kind'].astype(str).isin(['ogse_contrast', 'nogse_contrast'])].copy()
        if 'ok' in df.columns:
            df = df[df['ok'].fillna(False).astype(bool)].copy()
        if 'model' in df.columns:
            model = df['model'].astype(str).str.lower()
            if (model == 'free').any():
                df = df[model == 'free'].copy()

        dir_col = cmap.dir_col if cmap.dir_col in df.columns else _pick_existing(df, ['direction'])
        td_col = cmap.td_col if cmap.td_col in df.columns else _pick_existing(df, ['td_ms', 'td_ms_1'])
        n1_col = cmap.n1_col if cmap.n1_col in df.columns else None
        n2_col = cmap.n2_col if cmap.n2_col in df.columns else None
        if dir_col is None or n1_col is None or n2_col is None:
            continue

        sub = _filter_stat(df.copy(), stat_col=cmap.stat_col, stat_keep=stat_keep)
        if sub.empty:
            continue

        if td_col is not None:
            sub['td_ms'] = pd.to_numeric(sub[td_col], errors='coerce')
        else:
            td_val = _infer_td_ms(sub)
            sub['td_ms'] = np.nan if td_val is None else float(td_val)

        sub['subj'] = _infer_subj_from_df_or_id(sub, exp_id, source_path=f)
        sub['sheet'] = _infer_sheet_from_df_or_id(sub, exp_id, canonicalize=canonicalize_sheet, source_path=f)
        sub['direction'] = sub[dir_col].astype(str)
        sub['N_1'] = pd.to_numeric(sub[n1_col], errors='coerce')
        sub['N_2'] = pd.to_numeric(sub[n2_col], errors='coerce')
        keep = sub[['subj', 'sheet', 'direction', 'td_ms', 'N_1', 'N_2']].copy()
        keep = keep.dropna(subset=['subj', 'sheet', 'direction', 'td_ms', 'N_1', 'N_2'])
        if not keep.empty:
            blocks.append(keep)

    out = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if out.empty:
        raise ValueError('Could not build the expected NOGSE contrast keys.')

    out = out.groupby(['subj', 'sheet', 'direction', 'td_ms', 'N_1', 'N_2'], as_index=False).size()
    out = out.drop(columns=['size'])
    return out


# ---------------------------
# NOGSE side-specific signal D0
# ---------------------------
def _nogse_signal_model(td_ms: float, G: np.ndarray, N: int, M0: float, D0: float) -> np.ndarray:
    return M_nogse_free(float(td_ms), G, int(N), float(td_ms) / float(N), M0, D0)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _chi2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum((y - yhat) ** 2))


def _fit_nogse_signal_free(
    *,
    td_ms: float,
    G: np.ndarray,
    N: int,
    y: np.ndarray,
    M0_vary: bool,
    M0_value: float,
    D0_init: float,
) -> dict:
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(G) & np.isfinite(y) & (y > 0)
    G_fit = G[valid]
    y_fit = y[valid]
    n_fit = int(len(y_fit))

    if n_fit < 3:
        return {
            'ok': False,
            'n_fit': n_fit,
            'M0': np.nan,
            'M0_err': np.nan,
            'D0_m2_ms': np.nan,
            'D0_err_m2_ms': np.nan,
            'rmse': np.nan,
            'chi2': np.nan,
            'method': 'failed',
            'msg': 'Too few valid points for M_nogse_free fit.',
        }

    D0_seed = float(D0_init)
    if not np.isfinite(D0_seed) or D0_seed <= 0:
        D0_seed = 2.3e-12

    D_lo = max(D0_seed / 100.0, 1e-15)
    D_hi = min(D0_seed * 100.0, 1e-9)

    try:
        if M0_vary:
            def f(_dummy, M0, D0):
                return _nogse_signal_model(td_ms, G_fit, N, M0, D0)

            popt, pcov = curve_fit(
                f,
                np.zeros_like(y_fit),
                y_fit,
                p0=[float(M0_value), D0_seed],
                bounds=([0.0, D_lo], [5.0, D_hi]),
                maxfev=400000,
            )
            yhat = f(None, *popt)
            perr = (
                np.sqrt(np.diag(pcov))
                if pcov is not None and np.all(np.isfinite(pcov))
                else np.array([np.nan, np.nan])
            )
            return {
                'ok': True,
                'n_fit': n_fit,
                'M0': float(popt[0]),
                'M0_err': float(perr[0]) if np.isfinite(perr[0]) else np.nan,
                'D0_m2_ms': float(popt[1]),
                'D0_err_m2_ms': float(perr[1]) if np.isfinite(perr[1]) else np.nan,
                'rmse': _rmse(y_fit, yhat),
                'chi2': _chi2(y_fit, yhat),
                'method': 'scipy_curve_fit(M0,D0)',
                'msg': '',
            }

        def loss(log_D0: float) -> float:
            D0 = float(np.exp(log_D0))
            yhat = _nogse_signal_model(td_ms, G_fit, N, float(M0_value), D0)
            if yhat.shape != y_fit.shape or not np.all(np.isfinite(yhat)):
                return np.inf
            return float(np.sum((y_fit - yhat) ** 2))

        log_lo = float(np.log(D_lo))
        log_hi = float(np.log(D_hi))
        grid = np.linspace(log_lo, log_hi, 96)
        losses = np.array([loss(v) for v in grid], dtype=float)
        i_best = int(np.nanargmin(losses))
        best_log = float(grid[i_best])
        best_loss = float(losses[i_best])

        ref_lo = grid[max(0, i_best - 1)]
        ref_hi = grid[min(len(grid) - 1, i_best + 1)]
        if ref_hi > ref_lo:
            opt = minimize_scalar(
                loss,
                bounds=(float(ref_lo), float(ref_hi)),
                method='bounded',
                options={'xatol': 1e-8},
            )
            if bool(opt.success) and np.isfinite(float(opt.fun)) and float(opt.fun) <= best_loss:
                best_log = float(opt.x)

        D0 = float(np.exp(best_log))
        yhat = _nogse_signal_model(td_ms, G_fit, N, float(M0_value), D0)
        return {
            'ok': True,
            'n_fit': n_fit,
            'M0': float(M0_value),
            'M0_err': np.nan,
            'D0_m2_ms': D0,
            'D0_err_m2_ms': np.nan,
            'rmse': _rmse(y_fit, yhat),
            'chi2': _chi2(y_fit, yhat),
            'method': 'logD_scalar_search(M0_fixed)',
            'msg': '',
        }
    except Exception as exc:
        return {
            'ok': False,
            'n_fit': n_fit,
            'M0': np.nan,
            'M0_err': np.nan,
            'D0_m2_ms': np.nan,
            'D0_err_m2_ms': np.nan,
            'rmse': np.nan,
            'chi2': np.nan,
            'method': 'failed',
            'msg': str(exc),
        }


def _unique_for_group(df: pd.DataFrame, col: str) -> object | None:
    if col not in df.columns:
        return None
    u = pd.Series(df[col]).dropna().unique()
    if len(u) == 1:
        return u[0]
    return None


def _optional_float(value: object) -> float:
    val = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    return float(val) if np.isfinite(val) else np.nan


def _infer_td_ms_for_side(df: pd.DataFrame, side: int) -> float | None:
    td = _unique_for_group(df, f'td_ms_{side}')
    if td is not None:
        return float(td)

    max_dur = _unique_for_group(df, f'max_dur_ms_{side}')
    tm = _unique_for_group(df, f'tm_ms_{side}')
    if max_dur is not None and tm is not None:
        return float(2.0 * float(max_dur) + float(tm))
    return None


def _correction_lookup_key(row: pd.Series, tol_ms: float) -> tuple[str, str, str, str, int, int, int]:
    td_key = int(round(float(row['td_ms']) / float(tol_ms)))
    return (
        str(row['subj']),
        str(row['sheet']),
        str(row['roi']),
        str(row['direction']),
        td_key,
        int(round(float(row['N']))),
        int(row['side']),
    )


def _build_side_factor_lookup(corr_table: pd.DataFrame, tol_ms: float) -> dict[tuple[str, str, str, str, int, int, int], float]:
    lookup: dict[tuple[str, str, str, str, int, int, int], float] = {}
    if corr_table.empty:
        return lookup

    for _, row in corr_table.iterrows():
        for side in (1, 2):
            factor_col = f'correction_factor_{side}'
            if factor_col not in corr_table.columns:
                continue
            factor = pd.to_numeric(pd.Series([row.get(factor_col)]), errors='coerce').iloc[0]
            if not np.isfinite(factor):
                continue
            td_key = int(round(float(row['td_ms']) / float(tol_ms)))
            key = (
                str(row['subj']),
                str(row['sheet']),
                str(row['roi']),
                str(row['direction']),
                td_key,
                int(round(float(row[f'N_{side}']))),
                side,
            )
            lookup[key] = float(factor)
    return lookup


def load_nogse_signal_free_fits(
    contrast_root_or_file: str | Path,
    *,
    roi: str,
    stat_keep: str = 'avg',
    gbase: str = 'g_lin_max',
    ycol: str = 'value_norm',
    M0_vary: bool = False,
    M0_value: float = 1.0,
    D0_init: float = 2.3e-12,
    correction_table: pd.DataFrame | None = None,
    tol_ms: float = 1e-3,
    canonicalize_sheet: bool = True,
) -> pd.DataFrame:
    files = _discover_contrast_tables(contrast_root_or_file)
    if not files:
        raise FileNotFoundError(f'No encontré contrast tables bajo: {Path(contrast_root_or_file).resolve()}')

    factor_lookup = _build_side_factor_lookup(correction_table, tol_ms) if correction_table is not None else {}
    correction_state = 'corrected' if correction_table is not None else 'raw'

    rows: list[dict] = []
    for f in files:
        df = _read_table(f)
        if df.empty:
            continue

        analysis_id = _analysis_id_from_contrast_path(f)
        sheet = _infer_sheet_from_df_or_id(df, analysis_id, canonicalize=canonicalize_sheet, source_path=f)
        subj = _infer_subj_from_df_or_id(df, analysis_id, source_path=f)
        work = df.copy()

        if 'roi' not in work.columns or 'direction' not in work.columns:
            continue
        work['roi'] = work['roi'].astype(str).str.strip()
        work['direction'] = work['direction'].astype(str)
        work = work[work['roi'] == str(roi).strip()].copy()
        work = _filter_stat(work, stat_col='stat', stat_keep=stat_keep)
        if work.empty:
            continue

        group_cols = ['roi', 'direction'] + (['stat'] if 'stat' in work.columns else [])
        for key, gg in work.groupby(group_cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            key_dict = dict(zip(group_cols, key))
            stat_val = str(key_dict.get('stat', stat_keep))

            for side in (1, 2):
                g_col = _side_col(gbase, side)
                y_col = f'{ycol}_{side}'
                n_col = f'N_{side}'
                if y_col not in gg.columns or g_col not in gg.columns or n_col not in gg.columns:
                    continue

                td_ms = _infer_td_ms_for_side(gg, side)
                n_val = _unique_for_group(gg, n_col)
                if td_ms is None or n_val is None:
                    continue
                N = int(round(float(n_val)))
                signal_source_file = _unique_for_group(gg, f'source_file_{side}')
                signal_source_key = _source_key(signal_source_file)
                hz_val = _unique_for_group(gg, f'Hz_{side}')
                sequence_val = _unique_for_group(gg, f'sequence_{side}')
                delta_val = _unique_for_group(gg, f'delta_ms_{side}')
                delta_app_val = _unique_for_group(gg, f'Delta_app_ms_{side}')

                y = pd.to_numeric(gg[y_col], errors='coerce').to_numpy(dtype=float)
                G_raw = pd.to_numeric(gg[g_col], errors='coerce').to_numpy(dtype=float)
                G_raw = _maybe_scale_g_thorsten(gbase, G_raw)

                factor = 1.0
                if correction_table is not None:
                    probe = pd.Series(
                        {
                            'subj': str(subj).strip(),
                            'sheet': str(sheet),
                            'roi': str(key_dict['roi']),
                            'direction': str(key_dict['direction']),
                            'td_ms': float(td_ms),
                            'N': int(N),
                            'side': int(side),
                        }
                    )
                    factor = factor_lookup.get(_correction_lookup_key(probe, tol_ms), np.nan)
                    if not np.isfinite(factor):
                        continue

                G_fit = G_raw * float(factor)
                fit = _fit_nogse_signal_free(
                    td_ms=float(td_ms),
                    G=G_fit,
                    N=N,
                    y=y,
                    M0_vary=bool(M0_vary),
                    M0_value=float(M0_value),
                    D0_init=float(D0_init),
                )

                rows.append(
                    {
                        'source_file': f.name,
                        'source_path': str(f),
                        'analysis_id': analysis_id,
                        'subj': str(subj).strip(),
                        'sheet': str(sheet),
                        'roi': str(key_dict['roi']),
                        'direction': str(key_dict['direction']),
                        'stat': stat_val,
                        'side': int(side),
                        'correction_state': correction_state,
                        'correction_factor': float(factor),
                        'signal_source_file': '' if signal_source_file is None else str(signal_source_file),
                        'signal_source_key': signal_source_key,
                        'td_ms': float(td_ms),
                        'N': int(N),
                        'Hz': _optional_float(hz_val),
                        'sequence': sequence_val,
                        'delta_ms': _optional_float(delta_val),
                        'Delta_app_ms': _optional_float(delta_app_val),
                        'gbase': _normalize_gbase(gbase),
                        'ycol': str(ycol),
                        'n_points': int(np.sum(np.isfinite(y) & np.isfinite(G_fit))),
                        'g_max_raw_mTm': float(np.nanmax(G_raw)) if np.isfinite(G_raw).any() else np.nan,
                        'g_max_fit_mTm': float(np.nanmax(G_fit)) if np.isfinite(G_fit).any() else np.nan,
                        **fit,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        label = 'corrected NOGSE signal fits' if correction_table is not None else 'raw NOGSE signal fits'
        raise ValueError(f'No pude construir {label}. Revisá ruta, ROI, columnas side-specific y gbase/ycol.')

    out['D0_mm2_s'] = pd.to_numeric(out['D0_m2_ms'], errors='coerce') * 1e9
    out['D0_err_mm2_s'] = pd.to_numeric(out['D0_err_m2_ms'], errors='coerce') * 1e9
    return out.sort_values(
        ['subj', 'sheet', 'analysis_id', 'td_ms', 'roi', 'direction', 'side', 'correction_state'],
        kind='stable',
    ).reset_index(drop=True)


def _plot_side_signal_fit(
    *,
    df_group: pd.DataFrame,
    raw_row: pd.Series,
    corrected_row: pd.Series,
    gbase: str,
    ycol: str,
    out_png: Path,
) -> None:
    side = int(raw_row['side'])
    g_col = _side_col(gbase, side)
    y_col = f'{ycol}_{side}'
    if g_col not in df_group.columns or y_col not in df_group.columns:
        return

    y = pd.to_numeric(df_group[y_col], errors='coerce').to_numpy(dtype=float)
    G_raw = pd.to_numeric(df_group[g_col], errors='coerce').to_numpy(dtype=float)
    G_raw = _maybe_scale_g_thorsten(gbase, G_raw)
    valid = np.isfinite(y) & np.isfinite(G_raw)
    y = y[valid]
    G_raw = G_raw[valid]
    if y.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for ax, row, title in [
        (axes[0], raw_row, 'before correction'),
        (axes[1], corrected_row, 'after correction'),
    ]:
        factor = float(row.get('correction_factor', 1.0) or 1.0)
        G = G_raw * factor
        order = np.argsort(G)
        Gp = G[order]
        yp = y[order]
        ax.plot(Gp, yp, 'o', markersize=5.0, label='data')

        if bool(row.get('ok', True)) and Gp.size:
            Gs = np.linspace(0.0, float(np.nanmax(Gp)), 250)
            M0 = float(row['M0'])
            D0 = float(row['D0_m2_ms'])
            ys = _nogse_signal_model(float(row['td_ms']), Gs, int(row['N']), M0, D0)
            ax.plot(Gs, ys, '-', linewidth=2.0, label=f'M0={M0:.3g}, D0={D0:.3g} m2/ms')

        ax.set_title(title)
        ax.set_xlabel(f'{_normalize_gbase(gbase)}_{side} [mT/m]')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(str(ycol))
    fig.suptitle(
        (
            f"OGSE signal free fit | ROI={raw_row['roi']} | dir={raw_row['direction']} | "
            f"td={float(raw_row['td_ms']):g} ms | N={int(raw_row['N'])} | side={side}"
        )
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_ogse_signal_correction_fits(
    *,
    contrast_root_or_file: str | Path,
    fits: pd.DataFrame,
    plots_root: str | Path,
    gbase: str,
    ycol: str,
) -> list[Path]:
    if fits.empty:
        return []

    paths_by_name = {p.name: p for p in _discover_contrast_tables(contrast_root_or_file)}
    plots_root = Path(plots_root)
    outputs: list[Path] = []

    index_cols = ['source_file', 'roi', 'direction', 'stat', 'side']
    for key, sub in fits.groupby(index_cols, sort=False):
        source_file, roi, direction, stat, side = key
        raw = sub[sub['correction_state'].astype(str) == 'raw']
        corr = sub[sub['correction_state'].astype(str) == 'corrected']
        if raw.empty or corr.empty:
            continue

        source_path = paths_by_name.get(str(source_file))
        if source_path is None:
            raw_path = raw.iloc[0].get('source_path')
            source_path = Path(raw_path) if raw_path else None
        if source_path is None or not source_path.exists():
            continue

        df = _read_table(source_path)
        group = df[
            (df['roi'].astype(str).str.strip() == str(roi))
            & (df['direction'].astype(str) == str(direction))
        ].copy()
        if 'stat' in group.columns and str(stat) != 'nan':
            group = group[group['stat'].astype(str) == str(stat)].copy()
        if group.empty:
            continue

        raw_row = raw.iloc[0]
        corr_row = corr.iloc[0]
        out_png = plots_root / (
            f"{_sanitize_token(raw_row['analysis_id'])}."
            f"roi={_sanitize_token(roi)}."
            f"dir={_sanitize_token(direction)}."
            f"N={int(raw_row['N'])}."
            f"side={int(side)}."
            f"{_sanitize_token(_normalize_gbase(gbase))}.{_sanitize_token(ycol)}.raw_vs_corr.png"
        )
        _plot_side_signal_fit(
            df_group=group,
            raw_row=raw_row,
            corrected_row=corr_row,
            gbase=gbase,
            ycol=ycol,
            out_png=out_png,
        )
        outputs.append(out_png)

    return outputs


def _propagate_correction_std_side(row: pd.Series, side: int) -> float:
    corr = row.get(f'correction_factor_{side}', np.nan)
    if not np.isfinite(corr) or corr <= 0:
        return np.nan

    rel_var = 0.0
    used = False
    for mean_col, std_col in [
        (f'D0_fit_nogse_{side}', f'D0_fit_nogse_err_{side}'),
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


def _format_missing_signal_matches(missing: pd.DataFrame, monoexp_ref: pd.DataFrame) -> str:
    sample_missing_cols = [
        c for c in ['subj', 'sheet', 'roi', 'direction', 'side', 'signal_source_key', 'td_ms', 'N']
        if c in missing.columns
    ]
    sample_missing = missing[sample_missing_cols].drop_duplicates().head(10)
    sample_exp_cols = [c for c in ['subj', 'sheet', 'roi', 'direction', 'source_key', 'td_ms', 'N'] if c in monoexp_ref.columns]
    sample_exp = monoexp_ref[sample_exp_cols].drop_duplicates().head(20)
    return (
        'Faltan referencias monoexp para algunos fits OGSE individuales.\n\n'
        'Ejemplos faltantes:\n'
        f"{sample_missing.to_string(index=False)}\n\n"
        'Referencias monoexp disponibles (ejemplos):\n'
        f"{sample_exp.to_string(index=False)}"
    )


def _normalize_monoexp_ref_Ns(monoexp_ref_Ns: Iterable[int] | None) -> tuple[int, ...] | None:
    if monoexp_ref_Ns is None:
        return None
    values = sorted({int(round(float(n))) for n in monoexp_ref_Ns if pd.notna(n)})
    return tuple(values) if values else None


def _join_unique_ints(values: pd.Series) -> str:
    nums = pd.to_numeric(values, errors='coerce').dropna()
    unique = sorted({int(round(float(v))) for v in nums})
    return ','.join(str(v) for v in unique)


def _count_unique_ints(values: pd.Series) -> int:
    nums = pd.to_numeric(values, errors='coerce').dropna()
    return len({int(round(float(v))) for v in nums})


def _join_unique_strings(values: pd.Series) -> str:
    unique = sorted({str(v) for v in values.dropna().astype(str) if str(v)})
    return ';'.join(unique)


def _format_missing_selected_monoexp(
    missing: pd.DataFrame,
    monoexp_ref: pd.DataFrame,
    *,
    monoexp_ref_Ns: tuple[int, ...],
) -> str:
    sample_missing_cols = [
        c for c in ['subj', 'sheet', 'roi', 'direction', 'td_ms', 'N_1', 'N_2', 'monoexp_ref_Ns']
        if c in missing.columns
    ]
    sample_missing = missing[sample_missing_cols].drop_duplicates().head(10)
    sample_exp_cols = [c for c in ['subj', 'sheet', 'roi', 'direction', 'source_key', 'td_ms', 'N'] if c in monoexp_ref.columns]
    sample_exp = monoexp_ref[sample_exp_cols].drop_duplicates().head(20)
    return (
        f'Faltan referencias monoexp para promediar Ns={list(monoexp_ref_Ns)}.\n\n'
        'Ejemplos faltantes:\n'
        f"{sample_missing.to_string(index=False)}\n\n"
        'Referencias monoexp disponibles (ejemplos):\n'
        f"{sample_exp.to_string(index=False)}"
    )


def _attach_selected_monoexp_reference(
    out: pd.DataFrame,
    monoexp_ref: pd.DataFrame,
    *,
    monoexp_ref_Ns: tuple[int, ...],
    tol_ms: float,
) -> pd.DataFrame:
    ref = monoexp_ref.copy()
    ref['_td_key'] = _td_key_from_series(ref['td_ms'], tol_ms)
    ref['_N_key'] = _int_key_from_series(ref['N'])
    ref = ref[ref['_N_key'].isin(list(monoexp_ref_Ns))].copy()
    if ref.empty:
        raise ValueError(_format_missing_selected_monoexp(out, monoexp_ref, monoexp_ref_Ns=monoexp_ref_Ns))

    group_cols = ['subj', 'sheet', 'roi', 'direction', '_td_key']
    ref_agg = ref.groupby(group_cols, as_index=False).agg(
        D0_fit_monoexp=('D0_fit_monoexp', 'mean'),
        D0_fit_monoexp_std=('D0_fit_monoexp', 'std'),
        n_monoexp=('D0_fit_monoexp', 'size'),
        monoexp_ref_Ns=('_N_key', _join_unique_ints),
        monoexp_ref_N_count=('_N_key', _count_unique_ints),
        monoexp_ref_source_files=('source_file', _join_unique_strings),
    )

    work = out.copy()
    work['_td_key'] = _td_key_from_series(work['td_ms'], tol_ms)
    merged = work.merge(ref_agg, on=group_cols, how='left')
    missing = merged[
        merged['D0_fit_monoexp'].isna()
        | (pd.to_numeric(merged['monoexp_ref_N_count'], errors='coerce') < len(monoexp_ref_Ns))
    ].copy()
    if not missing.empty:
        raise ValueError(_format_missing_selected_monoexp(missing, monoexp_ref, monoexp_ref_Ns=monoexp_ref_Ns))
    return merged.drop(columns=['_td_key'])


def _build_side_specific_table(
    raw_signal_fits: pd.DataFrame,
    monoexp_ref: pd.DataFrame,
    *,
    tol_ms: float,
    monoexp_ref_Ns: Iterable[int] | None = None,
) -> pd.DataFrame:
    raw = raw_signal_fits.copy()
    raw = raw[raw['correction_state'].astype(str) == 'raw'].copy()
    if 'ok' in raw.columns:
        raw = raw[raw['ok'].fillna(False).astype(bool)].copy()
    if raw.empty:
        raise ValueError('No hay fits OGSE individuales crudos válidos para construir la corrección.')

    monoexp_ref = monoexp_ref.copy()
    raw['_td_key'] = _td_key_from_series(raw['td_ms'], tol_ms)
    raw['_N_key'] = _int_key_from_series(raw['N'])
    raw['signal_source_key'] = raw.get('signal_source_key', pd.Series('', index=raw.index)).astype(str)
    monoexp_ref['_td_key'] = _td_key_from_series(monoexp_ref['td_ms'], tol_ms)
    monoexp_ref['_N_key'] = _int_key_from_series(monoexp_ref['N'])
    monoexp_ref['source_key'] = monoexp_ref['source_key'].astype(str)

    mono_cols = [
        'subj', 'sheet', 'roi', 'direction', 'source_key', '_td_key', '_N_key',
        'source_file', 'D0_fit_monoexp', 'D0_fit_monoexp_std', 'n_monoexp',
    ]
    mono = monoexp_ref[mono_cols].rename(
        columns={
            'source_key': 'signal_source_key',
            'source_file': 'monoexp_source_file',
        }
    )

    merged = raw.merge(
        mono,
        on=['subj', 'sheet', 'roi', 'direction', 'signal_source_key', '_td_key', '_N_key'],
        how='left',
    )

    missing = merged[merged['D0_fit_monoexp'].isna()].copy()
    if not missing.empty:
        raise ValueError(_format_missing_signal_matches(missing, monoexp_ref))

    common = ['subj', 'sheet', 'analysis_id', 'source_file', 'roi', 'direction', 'stat']
    side_blocks: list[pd.DataFrame] = []
    for side in (1, 2):
        sub = merged[merged['side'].astype(int) == side].copy()
        if sub.empty:
            continue
        keep = common + [
            'signal_source_file', 'signal_source_key', 'monoexp_source_file',
            'td_ms', 'N', 'Hz', 'sequence', 'delta_ms', 'Delta_app_ms',
            'D0_m2_ms', 'D0_err_m2_ms', 'n_fit', 'n_points', 'g_max_raw_mTm',
            'D0_fit_monoexp', 'D0_fit_monoexp_std', 'n_monoexp',
        ]
        sub = sub[keep].rename(
            columns={
                'signal_source_file': f'signal_source_file_{side}',
                'signal_source_key': f'signal_source_key_{side}',
                'monoexp_source_file': f'monoexp_source_file_{side}',
                'td_ms': f'td_ms_{side}',
                'N': f'N_{side}',
                'Hz': f'Hz_{side}',
                'sequence': f'sequence_{side}',
                'delta_ms': f'delta_ms_{side}',
                'Delta_app_ms': f'Delta_app_ms_{side}',
                'D0_m2_ms': f'D0_fit_nogse_{side}',
                'D0_err_m2_ms': f'D0_fit_nogse_err_{side}',
                'n_fit': f'n_nogse_{side}',
                'n_points': f'n_points_nogse_{side}',
                'g_max_raw_mTm': f'g_max_raw_mTm_{side}',
                'D0_fit_monoexp': f'D0_fit_monoexp_{side}',
                'D0_fit_monoexp_std': f'D0_fit_monoexp_std_{side}',
                'n_monoexp': f'n_monoexp_{side}',
            }
        )
        side_blocks.append(sub)

    if len(side_blocks) != 2:
        raise ValueError('No pude construir correcciones para ambos lados OGSE (side=1 y side=2).')

    out = side_blocks[0].merge(side_blocks[1], on=common, how='inner')
    if out.empty:
        raise ValueError('No hubo matches entre side=1 y side=2 para construir la tabla de corrección.')

    out['td_ms'] = 0.5 * (
        pd.to_numeric(out['td_ms_1'], errors='coerce') + pd.to_numeric(out['td_ms_2'], errors='coerce')
    )
    td_delta = np.abs(pd.to_numeric(out['td_ms_1'], errors='coerce') - pd.to_numeric(out['td_ms_2'], errors='coerce'))
    if (td_delta > float(tol_ms)).any():
        bad = out.loc[td_delta > float(tol_ms), ['source_file', 'roi', 'direction', 'td_ms_1', 'td_ms_2']].head(10)
        raise ValueError(
            'td_ms_1 y td_ms_2 no coinciden para algunas curvas OGSE:\n'
            f'{bad.to_string(index=False)}'
        )

    selected_Ns = _normalize_monoexp_ref_Ns(monoexp_ref_Ns)
    if selected_Ns is None:
        mono_side_vals = out[['D0_fit_monoexp_1', 'D0_fit_monoexp_2']].apply(pd.to_numeric, errors='coerce')
        out['D0_fit_monoexp'] = mono_side_vals.mean(axis=1)
        out['D0_fit_monoexp_std'] = mono_side_vals.std(axis=1)
        mono_internal_std = out[['D0_fit_monoexp_std_1', 'D0_fit_monoexp_std_2']].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        out['D0_fit_monoexp_std'] = out['D0_fit_monoexp_std'].fillna(mono_internal_std)
        out['n_monoexp'] = (
            pd.to_numeric(out['n_monoexp_1'], errors='coerce').fillna(0)
            + pd.to_numeric(out['n_monoexp_2'], errors='coerce').fillna(0)
        ).astype(int)
        out['monoexp_ref_Ns'] = out.apply(lambda row: ','.join(str(int(round(float(row[f'N_{side}'])))) for side in (1, 2)), axis=1)
        out['monoexp_ref_N_count'] = 2
        out['monoexp_ref_source_files'] = out[['monoexp_source_file_1', 'monoexp_source_file_2']].apply(
            lambda row: ';'.join(str(v) for v in row.dropna().astype(str) if str(v)),
            axis=1,
        )
    else:
        out = _attach_selected_monoexp_reference(
            out,
            monoexp_ref,
            monoexp_ref_Ns=selected_Ns,
            tol_ms=tol_ms,
        )

    out['ratio_1'] = out['D0_fit_nogse_1'] / out['D0_fit_monoexp']
    out['ratio_2'] = out['D0_fit_nogse_2'] / out['D0_fit_monoexp']
    out['correction_factor_1'] = np.sqrt(out['ratio_1'])
    out['correction_factor_2'] = np.sqrt(out['ratio_2'])
    out['correction_factor_1_std'] = out.apply(lambda row: _propagate_correction_std_side(row, 1), axis=1)
    out['correction_factor_2_std'] = out.apply(lambda row: _propagate_correction_std_side(row, 2), axis=1)
    out['correction_source'] = 'ogse_signal_free_per_side'
    out['correction_pool_n'] = 1

    cols = [
        'subj', 'sheet', 'analysis_id', 'source_file', 'roi', 'direction', 'stat', 'td_ms', 'td_ms_1', 'td_ms_2',
        'N_1', 'N_2',
        'Hz_1', 'Hz_2', 'sequence_1', 'sequence_2',
        'signal_source_file_1', 'signal_source_file_2',
        'monoexp_source_file_1', 'monoexp_source_file_2',
        'D0_fit_nogse_1', 'D0_fit_nogse_err_1', 'n_nogse_1', 'n_points_nogse_1', 'g_max_raw_mTm_1',
        'D0_fit_nogse_2', 'D0_fit_nogse_err_2', 'n_nogse_2', 'n_points_nogse_2', 'g_max_raw_mTm_2',
        'D0_fit_monoexp_1', 'D0_fit_monoexp_2',
        'D0_fit_monoexp', 'D0_fit_monoexp_std', 'n_monoexp',
        'monoexp_ref_Ns', 'monoexp_ref_N_count', 'monoexp_ref_source_files',
        'ratio_1', 'ratio_2',
        'correction_factor_1', 'correction_factor_1_std',
        'correction_factor_2', 'correction_factor_2_std',
        'correction_source', 'correction_pool_n',
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(
        ['subj', 'sheet', 'td_ms', 'N_1', 'N_2', 'direction'],
        kind='stable',
    ).reset_index(drop=True)


def _add_side_specific_pooled_rows(
    out: pd.DataFrame,
    *,
    expected_keys: pd.DataFrame,
    roi: str,
    tol_ms: float,
) -> pd.DataFrame:
    if out.empty or expected_keys.empty:
        return out

    work = out.copy()
    work['_td_key'] = _td_key_from_series(work['td_ms'], tol_ms)

    expected = expected_keys.copy()
    expected['_td_key'] = _td_key_from_series(expected['td_ms'], tol_ms)

    key_cols = ['subj', 'sheet', 'direction', '_td_key', 'N_1', 'N_2']
    existing_keys = work[key_cols].drop_duplicates().copy()
    missing_keys = expected.merge(existing_keys, on=key_cols, how='left', indicator=True)
    missing_keys = missing_keys[missing_keys['_merge'] == 'left_only'].drop(columns=['_merge'])
    if missing_keys.empty:
        return out

    pooled_rows: list[dict[str, float | int | str]] = []
    donor_match_cols = ['_td_key', 'direction', 'N_1', 'N_2']
    def _num_col(frame: pd.DataFrame, col: str) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[col], errors='coerce')

    for _, miss in missing_keys.iterrows():
        donors = work.copy()
        for c in donor_match_cols:
            donors = donors[donors[c] == miss[c]]
        if donors.empty:
            continue
        ref_count = (
            pd.to_numeric(donors['monoexp_ref_N_count'], errors='coerce').max()
            if 'monoexp_ref_N_count' in donors.columns
            else np.nan
        )

        row: dict[str, float | int | str] = {
            'subj': str(miss['subj']),
            'sheet': str(miss['sheet']),
            'analysis_id': '',
            'source_file': '',
            'roi': str(roi),
            'direction': str(miss['direction']),
            'stat': 'pooled',
            'td_ms': float(miss['td_ms']),
            'td_ms_1': float(miss['td_ms']),
            'td_ms_2': float(miss['td_ms']),
            'N_1': float(miss['N_1']),
            'N_2': float(miss['N_2']),
            'D0_fit_monoexp': float(pd.to_numeric(donors['D0_fit_monoexp'], errors='coerce').mean()),
            'D0_fit_monoexp_std': float(pd.to_numeric(donors['D0_fit_monoexp'], errors='coerce').std()),
            'n_monoexp': int(len(donors)),
            'monoexp_ref_Ns': _join_unique_strings(donors['monoexp_ref_Ns']) if 'monoexp_ref_Ns' in donors.columns else '',
            'monoexp_ref_N_count': int(ref_count) if np.isfinite(ref_count) else np.nan,
            'monoexp_ref_source_files': _join_unique_strings(donors['monoexp_ref_source_files']) if 'monoexp_ref_source_files' in donors.columns else '',
            'correction_source': 'ogse_signal_free_per_side_pooled_average',
            'correction_pool_n': int(len(donors)),
        }
        if not np.isfinite(float(row['D0_fit_monoexp_std'])):
            row['D0_fit_monoexp_std'] = float(pd.to_numeric(donors['D0_fit_monoexp_std'], errors='coerce').mean())

        for side in (1, 2):
            d0_col = f'D0_fit_nogse_{side}'
            factor_col = f'correction_factor_{side}'
            factor_std_col = f'correction_factor_{side}_std'
            mono_col = f'D0_fit_monoexp_{side}'

            d0_vals = pd.to_numeric(donors[d0_col], errors='coerce').dropna()
            factor_vals = pd.to_numeric(donors[factor_col], errors='coerce').dropna()
            mono_vals = _num_col(donors, mono_col).dropna()
            row[f'signal_source_file_{side}'] = ''
            row[f'monoexp_source_file_{side}'] = ''
            row[d0_col] = float(d0_vals.mean()) if not d0_vals.empty else np.nan
            row[f'D0_fit_nogse_err_{side}'] = float(d0_vals.std()) if len(d0_vals) > 1 else np.nan
            if not np.isfinite(float(row[f'D0_fit_nogse_err_{side}'])):
                row[f'D0_fit_nogse_err_{side}'] = float(_num_col(donors, f'D0_fit_nogse_err_{side}').mean())
            row[f'n_nogse_{side}'] = int(len(donors))
            row[f'n_points_nogse_{side}'] = int(_num_col(donors, f'n_points_nogse_{side}').sum())
            row[f'g_max_raw_mTm_{side}'] = float(_num_col(donors, f'g_max_raw_mTm_{side}').mean())
            row[mono_col] = float(mono_vals.mean()) if not mono_vals.empty else np.nan
            row[f'D0_fit_monoexp_std_{side}'] = float(mono_vals.std()) if len(mono_vals) > 1 else np.nan
            if not np.isfinite(float(row[f'D0_fit_monoexp_std_{side}'])):
                row[f'D0_fit_monoexp_std_{side}'] = float(_num_col(donors, f'D0_fit_monoexp_std_{side}').mean())
            row[f'n_monoexp_{side}'] = int(_num_col(donors, f'n_monoexp_{side}').sum())
            row[factor_col] = float(factor_vals.mean()) if not factor_vals.empty else np.nan
            row[factor_std_col] = float(factor_vals.std()) if len(factor_vals) > 1 else np.nan
            if not np.isfinite(float(row[factor_std_col])):
                row[factor_std_col] = float(_num_col(donors, factor_std_col).mean())
            row[f'ratio_{side}'] = float(row[factor_col] ** 2) if np.isfinite(float(row[factor_col])) else np.nan

        pooled_rows.append(row)

    if not pooled_rows:
        return out

    out = pd.concat([out, pd.DataFrame(pooled_rows)], ignore_index=True, sort=False)
    return out.sort_values(['subj', 'sheet', 'td_ms', 'N_1', 'N_2', 'direction'], kind='stable').reset_index(drop=True)


def make_grad_correction_outputs(
    *,
    roi: str,
    exp_fits_root: str | Path,
    nogse_root_or_file: str | Path,
    contrast_data_root_or_file: str | Path | None = None,
    fit_points: Optional[int] = None,
    stat_keep: str = 'avg',
    exp_d0_col: str = 'D0_mm2_s',
    exp_scale: float = 1e-9,
    nogse_scale: float = 1.0,
    cmap: ColumnMap = ColumnMap(),
    tol_ms: float = 1e-3,
    canonicalize_sheet: bool = True,
    gbase: str = 'g_lin_max',
    ycol: str = 'value_norm',
    M0_vary: bool = False,
    M0_value: float = 1.0,
    D0_init: float = 2.3e-12,
    monoexp_ref_Ns: Iterable[int] | None = (1, 4, 8),
    plots_root: str | Path | None = None,
) -> GradCorrectionOutputs:
    monoexp_ref = compute_d0_monoexp_reference(
        exp_fits_root,
        roi=roi,
        fit_points=fit_points,
        stat_keep=stat_keep,
        exp_d0_col=exp_d0_col,
        exp_scale=exp_scale,
        canonicalize_sheet=canonicalize_sheet,
    )

    contrast_root = Path(contrast_data_root_or_file) if contrast_data_root_or_file is not None else _infer_contrast_data_root(nogse_root_or_file)
    if contrast_root is None or not contrast_root.exists():
        requested = Path(contrast_data_root_or_file) if contrast_data_root_or_file is not None else None
        missing_root = requested if requested is not None else Path(str(nogse_root_or_file))
        raise FileNotFoundError(
            "The new gradient-correction pipeline requires the side-specific contrast tables (*.long.parquet). "
            "Legacy correction-factor tables are no longer supported. "
            f"Could not resolve a valid contrast-data root from: {missing_root}"
        )

    raw_fits = load_nogse_signal_free_fits(
        contrast_root,
        roi=roi,
        stat_keep=stat_keep,
        gbase=gbase,
        ycol=ycol,
        M0_vary=M0_vary,
        M0_value=M0_value,
        D0_init=D0_init,
        correction_table=None,
        tol_ms=tol_ms,
        canonicalize_sheet=canonicalize_sheet,
    )
    grad_correction = _build_side_specific_table(
        raw_fits,
        monoexp_ref,
        tol_ms=tol_ms,
        monoexp_ref_Ns=monoexp_ref_Ns,
    )
    try:
        expected_keys = load_nogse_expected_keys(
            nogse_root_or_file,
            stat_keep=stat_keep,
            cmap=cmap,
            canonicalize_sheet=canonicalize_sheet,
        )
        grad_correction = _add_side_specific_pooled_rows(
            grad_correction,
            expected_keys=expected_keys,
            roi=roi,
            tol_ms=tol_ms,
        )
    except FileNotFoundError:
        pass

    corrected_fits = load_nogse_signal_free_fits(
        contrast_root,
        roi=roi,
        stat_keep=stat_keep,
        gbase=gbase,
        ycol=ycol,
        M0_vary=M0_vary,
        M0_value=M0_value,
        D0_init=D0_init,
        correction_table=grad_correction,
        tol_ms=tol_ms,
        canonicalize_sheet=canonicalize_sheet,
    )
    signal_fits = pd.concat([raw_fits, corrected_fits], ignore_index=True)

    if plots_root is not None:
        plot_ogse_signal_correction_fits(
            contrast_root_or_file=contrast_root,
            fits=signal_fits,
            plots_root=plots_root,
            gbase=gbase,
            ycol=ycol,
        )

    return GradCorrectionOutputs(grad_correction=grad_correction, ogse_signal_fits=signal_fits)


def make_grad_correction_table(
    *,
    roi: str,
    exp_fits_root: str | Path,
    nogse_root_or_file: str | Path,
    contrast_data_root_or_file: str | Path | None = None,
    fit_points: Optional[int] = None,
    stat_keep: str = 'avg',
    exp_d0_col: str = 'D0_mm2_s',
    exp_scale: float = 1e-9,
    nogse_scale: float = 1.0,
    cmap: ColumnMap = ColumnMap(),
    tol_ms: float = 1e-3,
    canonicalize_sheet: bool = True,
    gbase: str = 'g_lin_max',
    ycol: str = 'value_norm',
    M0_vary: bool = False,
    M0_value: float = 1.0,
    D0_init: float = 2.3e-12,
    monoexp_ref_Ns: Iterable[int] | None = (1, 4, 8),
    plots_root: str | Path | None = None,
) -> pd.DataFrame:
    return make_grad_correction_outputs(
        roi=roi,
        exp_fits_root=exp_fits_root,
        nogse_root_or_file=nogse_root_or_file,
        contrast_data_root_or_file=contrast_data_root_or_file,
        fit_points=fit_points,
        stat_keep=stat_keep,
        exp_d0_col=exp_d0_col,
        exp_scale=exp_scale,
        nogse_scale=nogse_scale,
        cmap=cmap,
        tol_ms=tol_ms,
        canonicalize_sheet=canonicalize_sheet,
        gbase=gbase,
        ycol=ycol,
        M0_vary=M0_vary,
        M0_value=M0_value,
        D0_init=D0_init,
        monoexp_ref_Ns=monoexp_ref_Ns,
        plots_root=plots_root,
    ).grad_correction
