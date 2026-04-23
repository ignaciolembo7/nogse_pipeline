from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from tools.strict_columns import raise_on_unrecognized_column_names


_MM2_S_TO_M2_MS = 1e-9
_M2_MS_TO_MM2_S = 1e9


MONOEXP_COLS = [
    'source_file',
    'subj',
    'roi',
    'direction',
    'max_dur_ms',
    'tm_ms',
    'td_ms',
    'N',
    'delta_ms',
    'Delta_app_ms',
    'fit_kind',
    'model',
    'ycol',
    'stat',
    'n_points',
    'n_fit',
    'M0',
    'M0_err',
    'D0_m2_ms',
    'D0_err_m2_ms',
    'D0_mm2_s',
    'D0_err_mm2_s',
    'rmse',
    'rmse_log',
    'chi2',
    'g_type',
    'fit_points',
    'fit_strategy',
    'auto_fit_metric',
    'auto_fit_score',
    'method',
    'ok',
    'msg',
]

NOGSE_CONTRAST_COLS = [
    'source_file',
    'analysis_id',
    'subj',
    'sheet',
    'roi',
    'direction',
    'td_ms',
    'max_dur_ms_1',
    'tm_ms_1',
    'td_ms_1',
    'N_1',
    'delta_ms_1',
    'Delta_app_ms_1',
    'Hz_1',
    'TE_1',
    'TR_1',
    'bmax_1',
    'protocol_1',
    'sequence_1',
    'sheet_1',
    'max_dur_ms_2',
    'tm_ms_2',
    'td_ms_2',
    'N_2',
    'delta_ms_2',
    'Delta_app_ms_2',
    'Hz_2',
    'TE_2',
    'TR_2',
    'bmax_2',
    'protocol_2',
    'sequence_2',
    'sheet_2',
    'fit_kind',
    'model',
    'ycol',
    'stat',
    'n_points',
    'n_fit',
    'M0',
    'M0_err',
    'D0_m2_ms',
    'D0_err_m2_ms',
    'D0_mm2_s',
    'D0_err_mm2_s',
    'rmse',
    'chi2',
    'gbase',
    'xplot',
    'f_corr',
    'f_corr_1',
    'f_corr_2',
    'peak_method',
    'peak_grid_n',
    'g1_max_raw_mTm',
    'g2_max_raw_mTm',
    'g1_max_corr_mTm',
    'g2_max_corr_mTm',
    'peak_fraction',
    'g1_peak_raw_mTm',
    'g2_peak_raw_mTm',
    'g1_peak_corr_mTm',
    'g2_peak_corr_mTm',
    'x_peak_raw_mTm',
    'x_peak_corr_mTm',
    'signal_peak',
    'l_G_peak_m',
    'L_cf_peak',
    'lcf_peak_m',
    'tc_peak_ms',
    'tc_ms',
    'tc_err_ms',
    'alpha',
    'alpha_err',
    'method',
    'ok',
    'msg',
]


def _schema_cols_for_kind(fit_kind: str) -> list[str]:
    if fit_kind == 'monoexp':
        return MONOEXP_COLS
    if fit_kind == 'nogse_contrast':
        return NOGSE_CONTRAST_COLS
    raise ValueError(f'fit_kind desconocido: {fit_kind}')


def standardize_fit_params(
    df: pd.DataFrame,
    *,
    fit_kind: str,
    source_file: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    target_cols = _schema_cols_for_kind(fit_kind)

    if 'source_file' not in out.columns:
        out['source_file'] = str(source_file) if source_file is not None else ''
    elif source_file is not None:
        out['source_file'] = out['source_file'].fillna(str(source_file)).replace('', str(source_file))

    raise_on_unrecognized_column_names(out.columns, context="standardize_fit_params")

    if 'D0_m2_ms' not in out.columns and 'D0' in out.columns:
        out['D0_m2_ms'] = pd.to_numeric(out['D0'], errors='coerce')
    if 'D0_err_m2_ms' not in out.columns and 'D0_err' in out.columns:
        out['D0_err_m2_ms'] = pd.to_numeric(out['D0_err'], errors='coerce')

    if 'D0_m2_ms' not in out.columns and 'D0_mm2_s' in out.columns:
        out['D0_m2_ms'] = pd.to_numeric(out['D0_mm2_s'], errors='coerce') * _MM2_S_TO_M2_MS
    if 'D0_err_m2_ms' not in out.columns and 'D0_err_mm2_s' in out.columns:
        out['D0_err_m2_ms'] = pd.to_numeric(out['D0_err_mm2_s'], errors='coerce') * _MM2_S_TO_M2_MS

    if 'D0_mm2_s' not in out.columns and 'D0_m2_ms' in out.columns:
        out['D0_mm2_s'] = pd.to_numeric(out['D0_m2_ms'], errors='coerce') * _M2_MS_TO_MM2_S
    if 'D0_err_mm2_s' not in out.columns and 'D0_err_m2_ms' in out.columns:
        out['D0_err_mm2_s'] = pd.to_numeric(out['D0_err_m2_ms'], errors='coerce') * _M2_MS_TO_MM2_S

    out['fit_kind'] = fit_kind
    if 'ok' not in out.columns:
        out['ok'] = True
    if 'msg' not in out.columns:
        out['msg'] = ''

    for c in target_cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[target_cols].copy()
