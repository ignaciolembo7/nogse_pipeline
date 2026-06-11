from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path

from data_processing.io import write_xlsx_sheets
from ogse_fitting.make_grad_correction_table import ColumnMap, make_grad_correction_outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi', required=True, help='ROI to use in both monoexp and contrast fits.')
    ap.add_argument('--exp-fits-root', required=True, help='Root containing OGSE signal monoexp fits (ogse_signal_vs_g_monoexp).')
    ap.add_argument(
        '--nogse-root',
        required=True,
        help='Root or file with NOGSE contrast fits. Searches fit_params*.{parquet,csv,xlsx} and *.fit_*.parquet.',
    )
    ap.add_argument(
        '--contrast-data-root',
        default=None,
        help='Optional root with *.long.parquet contrast tables. If omitted, it is inferred from --nogse-root.',
    )
    ap.add_argument('--out-xlsx', required=True, help='Correction table .xlsx output.')
    ap.add_argument('--out-csv', default=None, help='Optional .csv output too.')
    ap.add_argument('--plots-root', default=None, help='Optional root for per-fit OGSE plots before/after correction.')
    ap.add_argument('--no-plots', action='store_true', help='Do not generate per-fit OGSE plots.')

    ap.add_argument('--fit_points', type=int, default=None, help='Optional k filter for monoexp fits.')
    ap.add_argument(
        '--monoexp-ref-Ns',
        nargs='+',
        type=int,
        default=[1, 4, 8],
        help='Ns from the monoexp curves that are averaged into D0_fit_monoexp (with direction-averaging per subj/sheet/roi/td_ms).',
    )
    ap.add_argument('--stat', default='avg', help='Stat to keep. Defaults to avg. Use ALL to skip filtering.')
    ap.add_argument('--tol-ms', type=float, default=1e-3, help='Tolerance in ms for matching td_ms between monoexp and contrast.')

    ap.add_argument('--exp-d0-col', default='D0_mm2_s')
    ap.add_argument('--exp-scale', type=float, default=1e-9, help='Scale used to convert monoexp D0 from mm2/s to m2/ms.')
    ap.add_argument('--nogse-d0-col', default='D0_m2_ms', help='D0 column from the NOGSE contrast fit.')
    ap.add_argument('--nogse-d0-err-col', default='D0_err_m2_ms', help='D0 error column from the NOGSE contrast fit.')
    ap.add_argument('--nogse-scale', type=float, default=1.0, help='Additional scale for NOGSE D0 when needed.')
    ap.add_argument('--gbase', default='g_lin_max', help='Gradient base for the per-fit OGSE signal fits.')
    ap.add_argument('--ycol', default='value_norm', help='Signal column for the per-fit OGSE signal fits.')
    ap.add_argument('--D0-init', type=float, default=2.3e-12, help='D0 seed for M_nogse_free, in m2/ms.')

    m0_group = ap.add_mutually_exclusive_group()
    m0_group.add_argument('--fix_M0', type=float, default=1.0, help='Fix M0 in the per-fit OGSE signal fits.')
    m0_group.add_argument('--free_M0', nargs='?', const=1.0, type=float, default=None, help='Keep M0 free with an optional seed.')

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

    out_xlsx = Path(args.out_xlsx)
    plots_root = None
    if not args.no_plots:
        plots_root = Path(args.plots_root) if args.plots_root else out_xlsx.with_suffix('').parent / f'{out_xlsx.stem}.plots'

    M0_vary = args.free_M0 is not None
    M0_value = float(args.free_M0) if args.free_M0 is not None else float(args.fix_M0)

    outputs = make_grad_correction_outputs(
        roi=args.roi,
        exp_fits_root=args.exp_fits_root,
        nogse_root_or_file=args.nogse_root,
        contrast_data_root_or_file=args.contrast_data_root,
        fit_points=args.fit_points,
        stat_keep=args.stat,
        exp_d0_col=args.exp_d0_col,
        exp_scale=args.exp_scale,
        nogse_scale=args.nogse_scale,
        cmap=cmap,
        tol_ms=args.tol_ms,
        canonicalize_sheet=not args.no_canonical_sheet,
        gbase=args.gbase,
        ycol=args.ycol,
        M0_vary=M0_vary,
        M0_value=M0_value,
        D0_init=args.D0_init,
        monoexp_ref_Ns=args.monoexp_ref_Ns,
        plots_root=plots_root,
    )

    out = outputs.grad_correction
    tables = {'grad_correction': out}
    if not outputs.ogse_signal_fits.empty:
        tables['ogse_signal_fits'] = outputs.ogse_signal_fits
    out_xlsx = write_xlsx_sheets(tables, out_xlsx)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        if not outputs.ogse_signal_fits.empty:
            signal_csv = out_csv.with_name(f'{out_csv.stem}.ogse_signal_fits.csv')
            outputs.ogse_signal_fits.to_csv(signal_csv, index=False)
    print('OK:', out_xlsx)
    if 'correction_source' in out.columns:
        sources = ', '.join(sorted(out['correction_source'].dropna().astype(str).unique()))
        print('Correction source:', sources)
    if 'monoexp_ref_Ns' in out.columns:
        ref_Ns = ', '.join(sorted(out['monoexp_ref_Ns'].dropna().astype(str).unique()))
        print('Monoexp reference Ns:', ref_Ns)
    if not outputs.ogse_signal_fits.empty:
        print('OGSE signal fits:', len(outputs.ogse_signal_fits))
    if plots_root is not None and not outputs.ogse_signal_fits.empty:
        print('Plots:', plots_root)


if __name__ == '__main__':
    main()
