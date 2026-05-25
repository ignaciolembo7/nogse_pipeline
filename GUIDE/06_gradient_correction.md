# Gradient Correction

Gradient correction is implemented as a table-driven lookup and is applied by
fitters through explicit CLI flags. It is not hidden in plotting code.

## Correction Table Creation

Entry point:

- `scripts/make_grad_correction_table.py`

Reusable implementation:

- `src/ogse_fitting/make_grad_correction_table.py`

Inputs are:

- monoexponential OGSE signal fits from `ogse_signal_vs_g_monoexp`;
- reference contrast fits from a selected ROI;
- selected reference `N` values;
- a matching tolerance for `td_ms`;
- output paths for `.xlsx` and `.csv`.

The output table contains side-specific factors:

- `correction_factor_1`
- `correction_factor_2`

Legacy single-factor correction tables are rejected by the strict reader.

## Correction Lookup

Reusable lookup implementation:

- `src/fitting/gradient_correction.py`

Important objects:

- `CorrectionLookupSpec`
- `SignalCorrectionLookupSpec`

Important functions:

- `read_correction_table`
- `build_direction_factors`
- `build_signal_direction_factors`
- `infer_td_ms`

The lookup can filter by:

- reference ROI,
- `td_ms`,
- sheet,
- `N_1`,
- `N_2`,
- signal-side `N`,
- source file,
- preferred side.

## How Fitters Use It

The batch scripts pass correction mode explicitly:

- `--apply_grad_corr`
- `--no_grad_corr`
- `--corr_xlsx`
- `--corr_roi`
- optional `--corr_td_ms`
- optional `--corr_sheet`

Signal fits use a direction-factor map. Contrast fits use per-side direction
factors. Corrected and uncorrected outputs are written to distinct directories
through the naming convention in `src/fitting/experiments.py`.

## Implementation Constraints

- Correction factors are side-specific for contrast tables.
- Correction tables must contain the required columns checked by
  `read_correction_table`.
- Fitters should fail clearly when the requested correction row is missing.
- New workflows should reuse `src/fitting/gradient_correction.py` instead of
  parsing correction spreadsheets directly.

