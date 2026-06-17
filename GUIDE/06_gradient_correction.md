# Gradient Correction

Gradient correction is master-driven. The `grad_correction` step computes the
factors and writes them back into `master.long.parquet`; corrected fit steps read
those embedded columns directly.

## Creation

Entry point:

- `scripts/data/make_grad_correction_table.py`

Reusable implementation:

- `src/ogse_fitting/make_grad_correction_table.py`

Inputs are:

- monoexponential OGSE signal fits;
- reference contrast fits from a selected ROI;
- selected reference `N` values;
- a matching tolerance for `td_ms`;
- `MASTER_PARQUET`, which is enriched in place.

The step writes audit `.xlsx`/`.csv` copies, but those files are not fit inputs.

## Embedded Columns

Signal rows use:

- `grad_correction_factor`

Contrast rows use:

- `grad_correction_factor_1`
- `grad_correction_factor_2`

## How Fitters Use It

The batch scripts pass correction mode explicitly:

- `--apply_grad_corr`
- `--no_grad_corr`

When correction is requested, fitters read the factor columns from the selected
master rows and fail clearly if those columns are missing or empty.
