# Fitting Architecture

Fitters are split into thin CLI wrappers under `scripts/` and reusable modules
under `src/`.

## Shared Fitting Core

Reusable fitting helpers live in:

- `src/fitting/core.py`

Important objects and functions:

- `ParametricFit`
- `CurveFitParameter`
- `fit_curve_fit`
- `fit_curve_fit_parameters`
- `fit_least_squares`
- `rmse`, `rmse_log`, `chi2`, `r2_score`
- `parameter_error_column`

This module centralizes parameter values, parameter errors, fitted curves, and
fit-quality metrics. It also standardizes names such as:

- `D0_err_mm2_s`
- `D0_err_m2_ms`
- `tc_err_ms`
- `g0_err_mTm`

## Model Registry

Experiment families and valid model names are registered in:

- `src/fitting/experiments.py`

Current families:

- `ogse_signal_vs_g`
- `ogse_contrast_vs_g`
- `nogse_signal_vs_g`
- `nogse_contrast_vs_g`

Before adding a model, update or inspect this registry so output names and CLI
choices remain consistent.

## Mathematical Model Functions

Model functions live in:

- `src/models/model_fitting.py`

The fitters import model functions from this module and wrap them with the
appropriate axis, timing, fixed parameters, and bounds.

## Monoexponential Fits

Entry points:

- `scripts/fitting/fit_ogse_signal_vs_g.py` with `--model monoexp`

Reusable implementation:

- `src/monoexp_fitting/fit_monoexp_signal_vs_bval.py`

The monoexponential fitter handles:

- fixed or fitted `M0`,
- automatic prefix selection for fit points,
- `D0_mm2_s` and `D0_err_mm2_s`,
- generated fit plots,
- optional synthetic `Dproj` tables,
- standardized fit-parameter output names.

Summary plotting is implemented in:

- `src/monoexp_fitting/plot_monoexp_D_vs_time.py`
- `src/monoexp_fitting/plot_D0_vs_Delta.py`

## OGSE Fits

Entry points:

- `scripts/fitting/fit_ogse_signal_vs_g.py`
- `scripts/fitting/fit_ogse_contrast_vs_g.py`

Reusable implementation:

- `src/ogse_fitting/fit_ogse_signal_vs_g.py`
- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`

Supported OGSE signal models are registered as:

- `monoexp`
- `free_ogse`

Supported OGSE contrast models include:

- `free`
- `mixed`
- `mixed_global`
- `rest`
- `tort`

## NOGSE Fits

Entry points:

- `scripts/fitting/fit_nogse_signal_vs_g.py`
- `scripts/fitting/fit_nogse_contrast_vs_g.py`

Reusable implementation:

- `src/nogse_fitting/fit_nogse_signal_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`

NOGSE fitters use the shared model functions and the same schema conventions as
the OGSE fitters. The fit axis and plot axis are explicit CLI choices so the
model can use the physically required variable while the output plot uses a
separate display axis when needed.

## Group-Level `t_c` Fitting

Entry points:

- `scripts/fitting/run_tc_vs_td.py`
- `scripts/fitting/run_tc_pipeline.py`

Reusable implementation:

- `src/tc_fittings/tc_td_pseudohuber.py`
- `src/tc_fittings/tc_td_registry.py`
- `src/tc_fittings/contrast_fit_table.py`
- `src/tc_fittings/alpha_macro_summary.py`

This layer reads standardized contrast-fit tables and produces grouped
summaries and robust `t_c` vs `t_d` fits.

