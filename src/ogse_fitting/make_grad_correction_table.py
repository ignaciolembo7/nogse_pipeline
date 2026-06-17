from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

from models.model_fitting import M_nogse_free
from tools.brain_labels import canonical_sheet_name


# ---------------------------------------------------------------------------
# NOGSE free fit (signal vs G)
# ---------------------------------------------------------------------------

def _nogse_signal_model(td_ms: float, G: np.ndarray, N: int, M0: float, D0: float) -> np.ndarray:
    return M_nogse_free(float(td_ms), G, int(N), float(td_ms) / float(N), M0, D0)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


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
    """
    Fit a single OGSE signal curve with the M_nogse_free model.
    G in mT/m, D0 in m2/ms.
    Returns dict with keys: ok, D0_m2_ms, D0_mm2_s, M0, rmse, n_fit, msg.
    """
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(G) & np.isfinite(y) & (y > 0)
    G_fit = G[valid]
    y_fit = y[valid]
    n_fit = int(len(y_fit))

    if n_fit < 3:
        return {
            'ok': False, 'n_fit': n_fit,
            'M0': np.nan, 'D0_m2_ms': np.nan, 'D0_mm2_s': np.nan,
            'rmse': np.nan, 'msg': 'Too few valid points.',
        }

    D0_seed = float(D0_init) if np.isfinite(D0_init) and D0_init > 0 else 2.3e-12
    D_lo = max(D0_seed / 100.0, 1e-15)
    D_hi = min(D0_seed * 100.0, 1e-9)

    try:
        if M0_vary:
            def f(_dummy, M0, D0):
                return _nogse_signal_model(td_ms, G_fit, N, M0, D0)

            popt, pcov = curve_fit(
                f, np.zeros_like(y_fit), y_fit,
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
            D0 = float(popt[1])
            M0 = float(popt[0])
        else:
            def loss(log_D0: float) -> float:
                D0_ = float(np.exp(log_D0))
                yhat_ = _nogse_signal_model(td_ms, G_fit, N, float(M0_value), D0_)
                if yhat_.shape != y_fit.shape or not np.all(np.isfinite(yhat_)):
                    return np.inf
                return float(np.sum((y_fit - yhat_) ** 2))

            grid = np.linspace(float(np.log(D_lo)), float(np.log(D_hi)), 96)
            losses = np.array([loss(v) for v in grid], dtype=float)
            i_best = int(np.nanargmin(losses))
            ref_lo = grid[max(0, i_best - 1)]
            ref_hi = grid[min(len(grid) - 1, i_best + 1)]
            best_log = float(grid[i_best])
            if ref_hi > ref_lo:
                opt = minimize_scalar(loss, bounds=(float(ref_lo), float(ref_hi)),
                                      method='bounded', options={'xatol': 1e-8})
                if bool(opt.success) and np.isfinite(float(opt.fun)) and float(opt.fun) <= losses[i_best]:
                    best_log = float(opt.x)
            D0 = float(np.exp(best_log))
            M0 = float(M0_value)

        yhat = _nogse_signal_model(td_ms, G_fit, N, M0, D0)
        return {
            'ok': True, 'n_fit': n_fit,
            'M0': M0, 'D0_m2_ms': D0, 'D0_mm2_s': D0 * 1e9,
            'rmse': _rmse(y_fit, yhat), 'msg': '',
        }
    except Exception as exc:
        return {
            'ok': False, 'n_fit': n_fit,
            'M0': np.nan, 'D0_m2_ms': np.nan, 'D0_mm2_s': np.nan,
            'rmse': np.nan, 'msg': str(exc),
        }


# ---------------------------------------------------------------------------
# Monoexp fit (signal vs b-value)
# ---------------------------------------------------------------------------

def _fit_monoexp(
    *,
    b: np.ndarray,
    y: np.ndarray,
    D0_init_mm2_s: float,
    M0_value: float,
    M0_vary: bool,
) -> dict:
    """
    Fit S = M0 * exp(-b * D0) where b in s/mm2, D0 in mm2/s.
    Returns dict with keys: ok, D0_mm2_s, D0_m2_ms, M0, rmse, n_fit, msg.
    D0_m2_ms = D0_mm2_s * 1e-9.
    """
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(b) & np.isfinite(y) & (y > 0) & (b >= 0)
    b_fit = b[valid]
    y_fit = y[valid]
    n_fit = int(len(b_fit))

    if n_fit < 3:
        return {
            'ok': False, 'n_fit': n_fit,
            'M0': np.nan, 'D0_mm2_s': np.nan, 'D0_m2_ms': np.nan,
            'rmse': np.nan, 'msg': 'Too few valid points.',
        }

    order = np.argsort(b_fit)
    b_fit = b_fit[order]
    y_fit = y_fit[order]

    D0_seed = float(D0_init_mm2_s) if np.isfinite(D0_init_mm2_s) and D0_init_mm2_s > 0 else 2.3e-3
    D_lo = max(D0_seed / 100.0, 1e-12)
    D_hi = min(D0_seed * 100.0, 1.0)

    try:
        if M0_vary:
            def model(b_, M0, D0):
                return M0 * np.exp(-b_ * D0)
            popt, _ = curve_fit(
                model, b_fit, y_fit,
                p0=[float(M0_value), D0_seed],
                bounds=([0.0, D_lo], [5.0, D_hi]),
                maxfev=40000,
            )
            D0 = float(popt[1])
            M0 = float(popt[0])
        else:
            M0_fix = float(M0_value)
            def model_fixed(b_, D0):
                return M0_fix * np.exp(-b_ * D0)
            popt, _ = curve_fit(
                model_fixed, b_fit, y_fit,
                p0=[D0_seed],
                bounds=([D_lo], [D_hi]),
                maxfev=40000,
            )
            D0 = float(popt[0])
            M0 = M0_fix

        yhat = M0 * np.exp(-b_fit * D0)
        return {
            'ok': True, 'n_fit': n_fit,
            'M0': M0, 'D0_mm2_s': D0, 'D0_m2_ms': D0 * 1e-9,
            'rmse': _rmse(y_fit, yhat), 'msg': '',
        }
    except Exception as exc:
        return {
            'ok': False, 'n_fit': n_fit,
            'M0': np.nan, 'D0_mm2_s': np.nan, 'D0_m2_ms': np.nan,
            'rmse': np.nan, 'msg': str(exc),
        }


# ---------------------------------------------------------------------------
# Main: build correction table from manifest + master parquet
# ---------------------------------------------------------------------------

def make_grad_correction_from_manifest(
    master_parquet: str | Path,
    manifest: str | Path,
    *,
    stat_keep: str = 'avg',
    row_kind: str = 'signal_rotated',
    gbase: str = 'g_lin_max',
    bbase: str = 'bvalue_thorsten',
    M0_vary: bool = False,
    M0_value: float = 1.0,
    D0_init: float = 2.3e-12,
    tol_ms: float = 1e-3,
) -> pd.DataFrame:
    """
    Read the grad-correction manifest and the master parquet, then for each
    listed curve fit it with both M_nogse_free and monoexp.

    correction_factor = sqrt(D0_nogse / D0_monoexp)

    The manifest must have columns: subj, sheet, roi, direction, td_ms, N
    (Hz and model are optional and ignored).

    Returns one row per manifest entry with fitting results and correction_factor.
    """
    manifest = Path(manifest)
    master_parquet = Path(master_parquet)

    mf = pd.read_csv(manifest)
    if mf.empty:
        raise ValueError(f'Manifest {manifest} is empty.')

    master = pd.read_parquet(master_parquet)

    rois = set(mf['roi'].astype(str).str.strip().unique())
    row_kinds = {row_kind, 'signal_rotated', 'signal'}

    signal = master[
        master['row_kind'].astype(str).isin(row_kinds) &
        master['roi'].astype(str).str.strip().isin(rois)
    ].copy()

    if stat_keep and str(stat_keep).upper() != 'ALL' and 'stat' in signal.columns:
        signal = signal[signal['stat'].astype(str) == str(stat_keep)].copy()

    if signal.empty:
        raise ValueError(
            f'No rows with row_kind in {sorted(row_kinds)} found in master for ROIs {sorted(rois)}.\n'
            f'Master contains row_kinds: {master["row_kind"].astype(str).unique().tolist()}'
        )

    g_col = gbase if gbase in signal.columns else 'g'
    b_col = bbase if bbase in signal.columns else 'bvalue'
    D0_init_mm2_s = float(D0_init) * 1e9

    rows: list[dict] = []
    for _, mrow in mf.iterrows():
        subj = str(mrow['subj']).strip()
        sheet = str(mrow['sheet']).strip()
        roi = str(mrow['roi']).strip()
        direction = str(mrow['direction']).strip()
        td_ms = float(mrow['td_ms'])
        N = int(round(float(mrow['N'])))

        mask = (
            (signal['subj'].astype(str).str.strip() == subj) &
            (signal['roi'].astype(str).str.strip() == roi) &
            (signal['direction'].astype(str) == direction) &
            np.isclose(pd.to_numeric(signal['td_ms'], errors='coerce'), td_ms, atol=tol_ms) &
            np.isclose(pd.to_numeric(signal['N'], errors='coerce'), float(N), atol=0.5)
        )
        if 'sheet' in signal.columns:
            sheet_canon = canonical_sheet_name(sheet)
            mask &= signal['sheet'].astype(str).apply(canonical_sheet_name) == sheet_canon

        curve = signal[mask].copy()
        if curve.empty:
            print(f'WARNING: no signal data for subj={subj} sheet={sheet} roi={roi} '
                  f'dir={direction} td_ms={td_ms} N={N}')
            continue

        curve_sorted = curve.sort_values(g_col)
        G = pd.to_numeric(curve_sorted[g_col], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(curve_sorted[b_col], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(curve_sorted['value_norm'], errors='coerce').to_numpy(dtype=float)

        fit_nogse = _fit_nogse_signal_free(
            td_ms=td_ms, G=G, N=N, y=y,
            M0_vary=M0_vary, M0_value=M0_value, D0_init=D0_init,
        )
        fit_mono = _fit_monoexp(
            b=b, y=y,
            D0_init_mm2_s=D0_init_mm2_s,
            M0_value=M0_value, M0_vary=M0_vary,
        )

        D0_nogse = fit_nogse.get('D0_m2_ms', np.nan)
        D0_mono = fit_mono.get('D0_m2_ms', np.nan)

        both_ok = bool(fit_nogse.get('ok', False)) and bool(fit_mono.get('ok', False))
        if both_ok and np.isfinite(D0_nogse) and np.isfinite(D0_mono) and D0_mono > 0:
            ratio = D0_nogse / D0_mono
            correction_factor = float(np.sqrt(ratio)) if ratio > 0 else np.nan
        else:
            ratio = np.nan
            correction_factor = np.nan

        rows.append({
            'subj': subj,
            'sheet': sheet,
            'roi': roi,
            'direction': direction,
            'td_ms': td_ms,
            'N': N,
            'ok_nogse': bool(fit_nogse.get('ok', False)),
            'ok_monoexp': bool(fit_mono.get('ok', False)),
            'n_fit_nogse': int(fit_nogse.get('n_fit', 0)),
            'n_fit_monoexp': int(fit_mono.get('n_fit', 0)),
            'D0_fit_nogse_m2_ms': float(D0_nogse) if np.isfinite(D0_nogse) else np.nan,
            'D0_fit_nogse_mm2_s': float(D0_nogse * 1e9) if np.isfinite(D0_nogse) else np.nan,
            'D0_fit_monoexp_mm2_s': float(fit_mono.get('D0_mm2_s', np.nan)),
            'D0_fit_monoexp_m2_ms': float(D0_mono) if np.isfinite(D0_mono) else np.nan,
            'rmse_nogse': float(fit_nogse.get('rmse', np.nan)),
            'rmse_monoexp': float(fit_mono.get('rmse', np.nan)),
            'ratio': float(ratio) if np.isfinite(ratio) else np.nan,
            'correction_factor': correction_factor,
            'g_col': g_col,
            'b_col': b_col,
            'msg_nogse': str(fit_nogse.get('msg', '')),
            'msg_monoexp': str(fit_mono.get('msg', '')),
        })

    if not rows:
        raise ValueError(
            'No correction factors computed. Check that the manifest curves exist in the master parquet.'
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(['subj', 'sheet', 'direction', 'td_ms', 'N'], kind='stable').reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Master table update
# ---------------------------------------------------------------------------

def _update_master_correction_factors(master_path: Path, corr: pd.DataFrame, tol_ms: float = 1e-3) -> None:
    """
    Write correction_factor from the correction table into master.long.parquet.

    For signal/signal_rotated rows: matches on (subj, sheet, direction, td_ms, N).
    For contrast rows: matches N against N_1 → grad_correction_factor_1,
                       and N against N_2 → grad_correction_factor_2.
    The correction propagates to ALL rois in the master, not just the syringe roi.
    """
    from data_processing.master_table import load_master_table, write_master_table

    master = load_master_table(master_path)
    master['grad_correction_factor'] = pd.to_numeric(
        master.get('grad_correction_factor'), errors='coerce'
    )

    master_subj = master['subj'].astype(str).str.strip()
    master_sheet_canon = master['sheet'].astype(str).apply(canonical_sheet_name)
    master_dir = master['direction'].astype(str)
    master_td = pd.to_numeric(master['td_ms'], errors='coerce')

    signal_mask = master['row_kind'].astype(str).isin(['signal', 'signal_rotated'])
    contrast_mask = master['row_kind'].astype(str) == 'contrast'

    for _, row in corr.iterrows():
        factor = row.get('correction_factor')
        if not np.isfinite(factor):
            continue
        factor = float(factor)

        subj = str(row['subj']).strip()
        sheet_canon = canonical_sheet_name(str(row['sheet']))
        direction = str(row['direction'])
        td_ms = float(row['td_ms'])
        N = float(row['N'])

        base = (
            (master_subj == subj) &
            (master_sheet_canon == sheet_canon) &
            (master_dir == direction) &
            np.isclose(master_td, td_ms, atol=tol_ms)
        )

        # Signal rows (all rois): match on N
        master_N = pd.to_numeric(master['N'], errors='coerce')
        sig_mask = signal_mask & base & np.isclose(master_N, N, atol=0.5)
        master.loc[sig_mask, 'grad_correction_factor'] = factor

        # Contrast rows (all rois): match N against N_1 and N_2
        if 'N_1' in master.columns:
            master_N1 = pd.to_numeric(master['N_1'], errors='coerce')
            c1_mask = contrast_mask & base & np.isclose(master_N1, N, atol=0.5)
            if 'grad_correction_factor_1' not in master.columns:
                master['grad_correction_factor_1'] = np.nan
            master.loc[c1_mask, 'grad_correction_factor_1'] = factor

        if 'N_2' in master.columns:
            master_N2 = pd.to_numeric(master['N_2'], errors='coerce')
            c2_mask = contrast_mask & base & np.isclose(master_N2, N, atol=0.5)
            if 'grad_correction_factor_2' not in master.columns:
                master['grad_correction_factor_2'] = np.nan
            master.loc[c2_mask, 'grad_correction_factor_2'] = factor

    write_master_table(master, master_path)
    print('Updated master grad_correction_factor:', master_path)
