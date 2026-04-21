from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.io import write_xlsx_csv_outputs
from ogse_fitting.make_grad_correction_table import ColumnMap, make_grad_correction_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi', required=True, help='ROI to use in both monoexp and contrast fits.')
    ap.add_argument('--exp-fits-root', required=True, help='Root containing monoexp fits (fit_ogse-signal_vs_bval).')
    ap.add_argument(
        '--nogse-root',
        required=True,
        help='Root or file with NOGSE contrast fits. Searches fit_params*.{parquet,csv,xlsx} and *.fit_*.parquet.',
    )
    ap.add_argument('--out-xlsx', required=True, help='Correction table .xlsx output.')
    ap.add_argument('--out-csv', default=None, help='Optional .csv output too.')

    ap.add_argument('--fit_points', type=int, default=None, help='Optional k filter for monoexp fits.')
    ap.add_argument('--stat', default='avg', help='Stat to keep. Defaults to avg. Use ALL to skip filtering.')
    ap.add_argument('--tol-ms', type=float, default=1e-3, help='Tolerance in ms for matching td_ms between monoexp and contrast.')

    ap.add_argument('--exp-d0-col', default='D0_mm2_s')
    ap.add_argument('--exp-scale', type=float, default=1e-9, help='Scale used to convert monoexp D0 from mm2/s to m2/ms.')
    ap.add_argument('--nogse-d0-col', default='D0_m2_ms', help='D0 column from the NOGSE contrast fit.')
    ap.add_argument('--nogse-d0-err-col', default='D0_err_m2_ms', help='D0 error column from the NOGSE contrast fit.')
    ap.add_argument('--nogse-scale', type=float, default=1.0, help='Additional scale for NOGSE D0 when needed.')

    ap.add_argument(
        '--no-canonical-sheet',
        action='store_true',
        help='Do not normalize sheet. By default, it is reduced to a base identifier shared by monoexp and contrast.',
    )

    args = ap.parse_args()

    cmap = ColumnMap(
        roi_col='roi',
        dir_col='direction',
        td_col='td_ms',
        d0_col=args.nogse_d0_col,
        d0_err_col=args.nogse_d0_err_col,
        stat_col='stat',
        sheet_col='sheet',
        n1_col='N_1',
        n2_col='N_2',
    )

    out = make_grad_correction_table(
        roi=args.roi,
        exp_fits_root=args.exp_fits_root,
        nogse_root_or_file=args.nogse_root,
        fit_points=args.fit_points,
        stat_keep=args.stat,
        exp_d0_col=args.exp_d0_col,
        exp_scale=args.exp_scale,
        nogse_scale=args.nogse_scale,
        cmap=cmap,
        tol_ms=args.tol_ms,
        canonicalize_sheet=not args.no_canonical_sheet,
    )

    out_xlsx = write_xlsx_csv_outputs(
        out,
        args.out_xlsx,
        csv_path=args.out_csv if args.out_csv else None,
        sheet_name='grad_correction',
    )

    print('OK:', out_xlsx)


if __name__ == '__main__':
    main()
