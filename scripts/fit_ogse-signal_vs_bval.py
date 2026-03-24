from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from monoexp_fitting.fit_signal_vs_bval import run_fit_from_parquet


VALID_YCOLS = {'value', 'value_norm'}
LEGACY_DEFAULT_FIT_POINTS = 6


def _unique_float_any(df: pd.DataFrame, cols: list[str], *, required: bool, name: str) -> float | None:
    for c in cols:
        if c in df.columns:
            u = pd.to_numeric(df[c], errors='coerce').dropna().unique()
            if len(u) == 1:
                return float(u[0])
    if required:
        raise ValueError(f'No pude inferir {name}. Probe columnas: {cols}')
    return None


def _load_parquet_context(parquet_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if 'axis' in df.columns:
        raise ValueError("Encontre columna 'axis'. Este pipeline usa SOLO 'direction'.")
    required = ['direction', 'roi']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas requeridas {missing}. Columns={list(df.columns)}')
    return df


def _infer_overrides_from_df(df: pd.DataFrame) -> dict[str, float | None]:
    N = _unique_float_any(df, ['N'], required=False, name='N')
    delta_ms = _unique_float_any(df, ['delta_ms'], required=False, name='delta_ms')
    Delta_app_ms = _unique_float_any(df, ['Delta_app_ms'], required=False, name='Delta_app_ms')

    td_ms = _unique_float_any(df, ['td_ms'], required=False, name='td_ms')
    if td_ms is None:
        max_dur_ms = _unique_float_any(df, ['max_dur_ms'], required=False, name='max_dur_ms')
        tm_ms = _unique_float_any(df, ['tm_ms'], required=False, name='tm_ms')
        if max_dur_ms is not None and tm_ms is not None:
            td_ms = 2.0 * float(max_dur_ms) + float(tm_ms)

    return {'td_ms': td_ms, 'N': N, 'delta_ms': delta_ms, 'Delta_app_ms': Delta_app_ms}


def _resolve_requested_rois(requested: list[str], available: list[str]) -> list[str] | None:
    if requested == ['ALL']:
        return None

    available_set = {str(x) for x in available}
    missing = [str(x) for x in requested if str(x) not in available_set]
    if missing:
        raise ValueError(f'ROIs no encontradas: {missing}. ROIs disponibles={available}')
    return [str(x) for x in requested]


def _resolve_requested_directions(requested: list[str] | None, available: list[str]) -> list[str] | None:
    if requested is None:
        return None

    available_set = {str(x) for x in available}
    missing = [str(x) for x in requested if str(x) not in available_set]
    if missing:
        raise ValueError(
            f'Directions no encontradas: {missing}. Directions disponibles={available}'
        )
    return [str(x) for x in requested]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('parquet', help='Archivo .long.parquet (tabla de senal limpia)')
    ap.add_argument('--directions', nargs='+', default=None, help='Direcciones de la columna direction a fitear')
    ap.add_argument('--rois', nargs='+', default=['ALL'], help='ROIs a fitear. Si no existen en la tabla, falla.')
    ap.add_argument('--ycol', default='value_norm', choices=sorted(VALID_YCOLS))

    fit_group = ap.add_mutually_exclusive_group()
    fit_group.add_argument('--fit_points', type=int, default=None, help='Cantidad fija de puntos para el ajuste monoexp.')
    fit_group.add_argument(
        '--auto_fit_points',
        action='store_true',
        help='Busca automáticamente cuántos puntos iniciales ajustan mejor al modelo monoexp.',
    )
    ap.add_argument('--auto_fit_tol', type=float, default=0.05, help='Tolerancia relativa del modo automático al agregar un punto nuevo.')
    ap.add_argument('--auto_fit_err_floor', type=float, default=0.005, help='Piso absoluto para rmse_log antes de compararlo entre k consecutivos.')
    ap.add_argument('--auto_fit_min_points', type=int, default=3, help='Primer k a probar en el modo automático.')
    ap.add_argument('--auto_fit_max_points', type=int, default=9, help='Último k a probar en el modo automático.')

    ap.add_argument(
        '--g_type',
        default='bvalue',
        choices=['bvalue', 'g', 'g_max', 'g_lin_max', 'g_thorsten'],
    )
    ap.add_argument('--gamma', type=float, default=267.5221900)
    ap.add_argument('--td_ms', type=float, default=None)
    ap.add_argument('--N', type=float, default=None)
    ap.add_argument('--delta_ms', type=float, default=None)
    ap.add_argument('--Delta_app_ms', type=float, default=None)
    ap.add_argument('--D0_init', type=float, default=0.0023)
    ap.add_argument('--fix_M0', type=float, default=1.0)
    ap.add_argument('--free_M0', action='store_true')
    ap.add_argument('--out_root', default='ogse_experiments/fits/fit-monoexp_ogse-signal')
    ap.add_argument('--stat', default='avg')

    args = ap.parse_args()

    if args.fit_points is not None and args.fit_points <= 0:
        raise ValueError('--fit_points debe ser > 0.')
    if args.auto_fit_tol < 0:
        raise ValueError('--auto_fit_tol debe ser >= 0.')
    if args.auto_fit_err_floor < 0:
        raise ValueError('--auto_fit_err_floor debe ser >= 0.')
    if args.auto_fit_min_points < 1:
        raise ValueError('--auto_fit_min_points debe ser >= 1.')
    if args.auto_fit_max_points is not None and args.auto_fit_max_points < args.auto_fit_min_points:
        raise ValueError('--auto_fit_max_points debe ser >= --auto_fit_min_points.')

    df = _load_parquet_context(args.parquet)
    inferred = _infer_overrides_from_df(df)

    td_ms = args.td_ms if args.td_ms is not None else inferred['td_ms']
    N = args.N if args.N is not None else inferred['N']
    delta_ms = args.delta_ms if args.delta_ms is not None else inferred['delta_ms']
    Delta_app_ms = args.Delta_app_ms if args.Delta_app_ms is not None else inferred['Delta_app_ms']

    directions = _resolve_requested_directions(
        None if args.directions is None else [str(x) for x in args.directions],
        sorted(df['direction'].astype(str).dropna().unique().tolist()),
    )
    rois = _resolve_requested_rois(
        [str(x) for x in args.rois],
        sorted(df['roi'].astype(str).dropna().unique().tolist()),
    )

    fit_points = args.fit_points
    auto_fit_points = bool(args.auto_fit_points)
    if fit_points is None and not auto_fit_points:
        fit_points = LEGACY_DEFAULT_FIT_POINTS

    run_fit_from_parquet(
        args.parquet,
        dirs=directions,
        rois=rois,
        ycol=args.ycol,
        g_type=args.g_type,
        fit_points=fit_points,
        auto_fit_points=auto_fit_points,
        auto_fit_min_points=args.auto_fit_min_points,
        auto_fit_max_points=args.auto_fit_max_points,
        auto_fit_rel_tol=args.auto_fit_tol,
        auto_fit_err_floor=args.auto_fit_err_floor,
        free_M0=args.free_M0,
        fix_M0=args.fix_M0,
        D0_init=args.D0_init,
        gamma=args.gamma,
        td_ms=td_ms,
        N=N,
        delta_ms=delta_ms,
        Delta_app_ms=Delta_app_ms,
        stat_keep=args.stat,
        out_root=args.out_root,
    )


if __name__ == '__main__':
    main()
