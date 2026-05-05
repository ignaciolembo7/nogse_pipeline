#### How each fitter now works

##### `ogse_signal_vs_g`

- User choice:
  - fit variable: `g_type`
  - plotting variable: `plot_xcol`
  - corrected vs uncorrected mode: `--apply_grad_corr` / `--no_grad_corr`
- Internal fit variable:
  - depends on model:
  - `monoexp`: corrected `b` axis (equivalent to deriving `b` from corrected
    gradient for the same acquisition timing);
  - `free_ogse`: corrected gradient `g_corr` passed to `M_ogse_free`;
- Internal plot variable:
  - `plot_xcol`, built from the same corrected gradient family

Key code:

```python
if model == "monoexp":
    run_fit_monoexp_from_parquet(...)
elif model == "free_ogse":
    yhat = M_ogse_free(td_ms, fit_bundle.gradient_corr, N, td_ms / N, M0, D0)
```

Code reference:

- `scripts/fit_ogse_signal_vs_g.py`
- `src/ogse_fitting/fit_ogse_signal_vs_g.py`

##### `nogse_signal_vs_g`

- User choice:
  - fit variable: `xcol`
  - plotting variable: `plot_xcol`
  - corrected vs uncorrected mode: `--apply_grad_corr` / `--no_grad_corr`
- Internal fit variable:
  - the corrected gradient `g_corr`, because `M_nogse_free` is parameterized in
    gradient, not in `b`
- Internal plot variable:
  - `plot_xcol`, which may be either a corrected gradient axis or a corrected
    `bvalue` axis derived from the same `g_corr`

The implementation makes that explicit by building the corrected axis bundle,
then passing the corrected gradient to the model:

```python
fit_bundle = build_axis_bundle(...)
group_fit["__fit_x__"] = fit_bundle.gradient_corr
fit_row, x_data, y_data, fit_curve = fit_one_group(..., xcol="__fit_x__")
```

So even when the requested axis family is written as `bvalue`, the model is
still evaluated on the physically relevant corrected gradient, and the matching
corrected `b` representation is used only as an axis representation when needed.

Code reference:

- `scripts/fit_nogse_signal_vs_g.py`
- `src/nogse_fitting/fit_nogse_signal_vs_g.py`

##### `ogse_contrast_vs_g`

- User choice:
  - fit variable family: `gbase`
  - plotting variable: `plot_xcol` such as `g_1`, `g_thorsten_2`,
    `bvalue_1`, or `bvalue_thorsten_2`
  - corrected vs uncorrected mode: `--apply_grad_corr` / `--no_grad_corr`
- Internal fit variables:
  - corrected side-1 gradient `G1_corr`
  - corrected side-2 gradient `G2_corr`
- Internal plot variable:
  - the side selected by `plot_xcol`, expressed either as corrected gradient or
    as corrected `b` derived from that side’s corrected gradient

Key code:

```python
fit_bundle_1 = build_axis_bundle(..., side=1, correction_factor=f_corr_1)
fit_bundle_2 = build_axis_bundle(..., side=2, correction_factor=f_corr_2)
plot_bundle = build_axis_bundle(..., axis=plot_axis, side=plot_side, ...)
```

This is still a two-side fit physically, but both sides now use the same
gradient-first correction flow as the signal fits.

Code reference:

- `scripts/fit_ogse_contrast_vs_g.py`
- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`

##### `nogse_contrast_vs_g`

- User choice:
  - fit variable family: `gbase`
  - plotting variable: `plot_xcol` on side 1
  - corrected vs uncorrected mode: `--apply_grad_corr` / `--no_grad_corr`
- Internal fit variable:
  - corrected side-1 gradient `G_corr`, because the NOGSE contrast model is
    parameterized in a single gradient axis
- Internal plot variable:
  - corrected side-1 `plot_xcol`, which may be either gradient or derived
    `bvalue`

The fitter still keeps side-2 metadata and side-2 correction factors because
they are needed for provenance and for the correction-table logic, but the
model itself remains a side-1-gradient model.

Code reference:

- `scripts/fit_nogse_contrast_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`

#### How fitting and plotting variables are recorded

The fit tables now make the chosen axes explicit:

- `ogse_signal_vs_g`: `g_type` and `plot_xcol`
- `nogse_signal_vs_g`: `xcol` and `plot_xcol`
- `ogse_contrast_vs_g`: `fit_xcol`, `plot_xcol`, plus legacy `gbase` and `xplot`
- `nogse_contrast_vs_g`: `fit_xcol`, `plot_xcol`, plus legacy `gbase` and `xplot`

This keeps the user-visible choice simple while preserving compatibility with
older downstream code that still expects `gbase` and `xplot`.

#### How the repository knows which factor to use in each case

There are two lookup patterns in `src/fitting/gradient_correction.py`.

For contrast fits, the correction table is filtered by ROI, `td_ms`, and
optionally `N_1`, `N_2`, then returned as a pair per direction:

```python
if factor_mode == "per_side":
    out = {
        str(row["direction"]): (
            float(row["correction_factor_1"]),
            float(row["correction_factor_2"]),
        )
        for _, row in c.iterrows()
    }
```

For single-signal fits, the lookup is stricter: it tries to match the signal to
side 1 or side 2 using the original `signal_source_file` and/or `N`, so a
single curve receives the factor belonging to its own side:

```python
source_match = bool(source_key) and row_source_key == source_key
n_match = spec.signal_n is not None and np.isfinite(n_val) and int(round(float(n_val))) == int(spec.signal_n)
if source_match or n_match:
    side_candidates.append(side)
```

This is why the current pipeline can support side-specific correction robustly
without confusing the two acquisitions that formed a contrast.

**What comes out**

- `grad_correction` tables with `correction_factor_1` and `correction_factor_2`

**Why this step is needed**

- it enforces consistency between the effective diffusion scale seen by the signal-fit branch and the one inferred by the contrast-fit branch.

**Key physical or mathematical idea**

- the correction is not a generic intensity normalization;
- it is specifically a recalibration of the effective gradient axis using a physically interpretable reference ROI.
- the same correction law is used for brains and phantoms; what changes is mainly the reference ROI and whether the downstream model uses corrected `b(g_corr)` or corrected `g_corr` directly.

**Code**

- `scripts/make_grad_correction_table.py`
- `src/ogse_fitting/make_grad_correction_table.py`
- `src/fitting/b_from_g.py`
- `src/ogse_fitting/fit_ogse_signal_vs_g.py`
- `src/nogse_fitting/fit_nogse_signal_vs_g.py`
- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`

#### End-to-end correction code traces (signals and contrasts)

This section spells out the exact code path used to apply correction factors in
the four `*_vs_g` workflows.

1. Correction factors are created as side-specific values in the correction table:

```python
out['ratio_1'] = out['D0_fit_nogse_1'] / out['D0_fit_monoexp']
out['ratio_2'] = out['D0_fit_nogse_2'] / out['D0_fit_monoexp']
out['correction_factor_1'] = np.sqrt(out['ratio_1'])
out['correction_factor_2'] = np.sqrt(out['ratio_2'])
```

Code: `src/ogse_fitting/make_grad_correction_table.py`.

2. Those factors are read and matched before fitting:

```python
corr = read_correction_table(args.corr_xlsx)
f_by_direction = build_direction_factors(
    corr,
    spec=CorrectionLookupSpec(...),
    factor_mode="per_side",
)
```

Code: `scripts/fit_ogse_contrast_vs_g.py`, `scripts/fit_nogse_contrast_vs_g.py`.

For single-signal fits, matching is done per signal file and/or `N`, then
reduced to one factor per direction:

```python
return build_signal_direction_factors(
    corr,
    spec=SignalCorrectionLookupSpec(
        ...,
        signal_source_file=parquet.name,
        signal_n=signal_n,
    )
)
```

Code: `scripts/fit_ogse_signal_vs_g.py`, `scripts/fit_nogse_signal_vs_g.py`,
`src/fitting/gradient_correction.py`.

3. OGSE signal fit (`ogse_signal_vs_g`):

- `free_ogse` model consumes corrected gradient directly via `build_axis_bundle`:

```python
fit_bundle = build_axis_bundle(..., axis=g_type, correction_factor=float(f_corr), ...)
fit_res = _select_fit_result(fit_bundle.gradient_corr, y, ...)
```

Code: `src/ogse_fitting/fit_ogse_signal_vs_g.py`.

- `monoexp` model branch delegates to the monoexponential fitter, which applies
  the equivalent `b`-axis correction:

```python
run_fit_monoexp_from_parquet(..., f_by_direction=f_by_direction, b_corr_power=2.0)
...
b_corr_scale = float(f_corr) ** float(b_corr_power)
b = b * b_corr_scale
```

Code: `src/ogse_fitting/fit_ogse_signal_vs_g.py`,
`src/monoexp_fitting/fit_monoexp_signal_vs_bval.py`.

4. NOGSE signal fit (`nogse_signal_vs_g`) always fits on corrected gradient:

```python
fit_bundle = build_axis_bundle(..., axis=fit_axis, correction_factor=float(f_corr), ...)
group_fit["__fit_x__"] = fit_bundle.gradient_corr
fit_row, x_data, y_data, fit_curve = fit_one_group(..., xcol="__fit_x__")
```

Code: `src/nogse_fitting/fit_nogse_signal_vs_g.py`.

5. OGSE contrast fit (`ogse_contrast_vs_g`) uses side-specific corrected gradients:

```python
fit_bundle_1 = build_axis_bundle(..., side=1, correction_factor=float(f_corr_1), ...)
fit_bundle_2 = build_axis_bundle(..., side=2, correction_factor=float(f_corr_2), ...)
G1 = fit_bundle_1.gradient_corr
G2 = fit_bundle_2.gradient_corr
```

Code: `src/ogse_fitting/fit_ogse_contrast_vs_g.py`.

6. NOGSE contrast fit (`nogse_contrast_vs_g`) fits side-1 corrected gradient
   (model axis), while keeping side-2 correction metadata for provenance and
   peak/summary reporting:

```python
fit_bundle = build_axis_bundle(..., side=1, correction_factor=float(f_corr_1), ...)
G_corr = fit_bundle.gradient_corr
...
side2_bundle = build_axis_bundle(..., side=2, correction_factor=float(f_corr_2), ...)
```

Code: `src/nogse_fitting/fit_nogse_contrast_vs_g.py`.

7. Shared transformation law used by all four workflows:

```python
gradient_raw = extract_gradient_array(df, axis=axis_base, side=resolved_side)
gradient_corr = gradient_raw * float(f_corr)
bvalue_corr = bvalue_from_gradient(gradient_corr, axis=axis_base, ...)
```

Code: `src/fitting/b_from_g.py`.

So, operationally:

- factors are always looked up as `correction_factor_1/2`,
- correction is applied first on gradient amplitude,
- any corrected `b` axis is derived from that corrected gradient (or, for the
  monoexp branch, equivalently as `b *= f^2`).

### Stage 11: Grouped `t_c` summaries and `t_c` vs `t_d` fitting

**What goes in**

- corrected `rest`-model contrast fits
- `alpha_macro` summary from the monoexponential branch

**What happens conceptually**

- all compatible rest-model fits are gathered into one table;
- each fit contributes a fitted `t_c` and a peak-derived `t_c`;
- the repository then fits `t_c` as a function of diffusion time `t_d` using a pseudo-Huber transition model.

The implemented curve is:

```text
tc(Td) = c + alpha_macro * delta * (sqrt(1 + (Td/delta)^2) - 1)
```

This behaves:

- quadratically for small `Td`,
- linearly for large `Td`,
- with `delta` controlling the transition scale.

**What comes out**

- grouped fit tables
- fit panels
- `tc_peak` summary panels
- final `tc`-vs-`td` fit results

**Why this step is needed**

- it is the final stage that compresses many individual contrast fits into a small number of interpretable trend parameters.

**Key physical or mathematical idea**

- `alpha_macro` is treated as a known macro-scale slope,
- `c` and `delta` describe how the observed `t_c` departs from the asymptotic linear regime at shorter diffusion times.

**Code**

- `scripts/run_tc_pipeline.py`
- `scripts/run_tc_vs_td.py`
- `src/tc_fittings/contrast_fit_table.py`
- `src/tc_fittings/tc_td_pseudohuber.py`

## Phantom pipeline

### Inputs

The phantom workflow expects:

- phantom DWI NIfTI files under `Data-NIFTI/...`
- `.bval/.bvec` or `.gval/.gvec` sidecars
- manually drawn ROI masks already in DWI space
- the phantom sequence-parameter spreadsheet

Important entry points are:

- `bash_template/phantoms/0.*`
- `bash_template/phantoms/1.0-run_PHANTOM-denoised_signal_extraction.sh`
- `bash_template/phantoms/2.*`
- `bash_template/phantoms/3.*`, `5.*`, `6.*`

### Stage 1: Phantom-specific setup

**What goes in**

- raw phantom NIfTI or DICOM data

**What happens conceptually**

- DICOM can be converted to NIfTI;
- for direct-`g` phantom acquisitions, `.gval/.gvec` files can be synthesized from filenames;
- repeated acquisitions can be collapsed to a mean image after discarding dummy scans;
- selected masks can be copied into each per-sequence folder for consistent ROI handling.

**What comes out**

- NIfTI sequences with usable gradient sidecars
- per-sequence folders prepared for mask-based extraction

**Why this step is needed**

- phantom data are often organized as repeated single-condition sequences rather than full multi-`b` image series;
- the pipeline therefore needs an explicit direct-`g` preparation branch.

**Code**

- `bash_template/phantoms/0.0-run_dicom2nifti.sh`
- `bash_template/phantoms/0.1-run_make_gval_gvec.sh`
- `scripts/make_gval_gvec_from_filenames.py`
- `bash_template/phantoms/0.2-prep_phantom_b0.sh`
- `bash_template/phantoms/0.3-copy_selected_files.sh`

### Stage 2: Direct ROI handling in DWI space

**What goes in**

- phantom DWI sequence
- manual binary masks stored inside that sequence folder

**What happens conceptually**

- unlike the brain workflow, phantom ROIs are not warped from another space;
- the pipeline searches the sequence folder for binary mask files, ignores non-mask outputs, and builds one combined `ALL_ROIS` label image from them.

**What comes out**

- direct DWI-space ROI masks
- ROI signal tables in Excel form

**Why this step is needed**

- phantom geometry is simple and stable enough that manual DWI-space masking is the intended source of truth.

**Key physical or mathematical idea**

- no anatomical registration is performed;
- the crucial assumption is instead that all masks already share the exact DWI grid.

**Code**

- `src/signal_extraction/coreg_extract_phantom.py`
  - `discover_sequence_masks`
  - `build_multilabel_from_binary_masks`
  - `main`

### Stage 3: Canonical long tables and direct-`g` curve assembly

**What goes in**

- phantom ROI Excel tables
- phantom sequence parameters

**What happens conceptually**

- the tables are converted into long form as in the brain workflow;
- when each input file is a single `g` point, the pipeline can merge multiple files into one grouped `g` curve using `--oneg`;
- for these direct-`g` curves, `direction` is inferred from the gradient vector and `b_step` is reconstructed from the ordered `g` values.

**What comes out**

- long-form phantom signal tables under `analysis/phantoms/.../data`

**Why this step is needed**

- direct-`g` phantom protocols are spread across many files, so the pipeline must reconstruct a continuous signal-vs-`g` curve before fitting.

**Code**

- `scripts/process_one_results.py`
  - `_aggregate_g_results`
  - `_merge_into_group_curve`
  - `_add_direct_g_derivatives`

### Stage 4: Phantom NOGSE signal fitting

**What goes in**

- grouped phantom signal-vs-`g` tables

**What happens conceptually**

- the pipeline fits individual CPMG or Hahn signal curves with the analytical free NOGSE signal model;
- the only difference between the CPMG and Hahn versions is how the internal timing variable `x` is chosen:
  - CPMG: `x = TN / 2`
  - Hahn: `x = 0`

**What comes out**

- fitted phantom signal parameters, mainly `M0` and `D0`

**Why this step is needed**

- this provides a direct signal-based estimate of the diffusion scale before any CPMG-minus-Hahn subtraction is taken.

**Key physical or mathematical idea**

- the model formulas come from `src/models/model_fitting.py`, with `M_nogse_free` as the core signal model.

**Code**

- `bash_template/phantoms/2.2-run_fit_nogse_signal_vs_g.sh`
- `src/nogse_fitting/fit_nogse_signal_vs_g.py`
- `src/models/model_fitting.py`

### Stage 5: Phantom contrast construction

**What goes in**

- matched phantom signal tables

**What happens conceptually**

- the current phantom templates pair CPMG and Hahn acquisitions and define the contrast as:
  - `CPMG - HAHN`
- this is done explicitly in the NOGSE branch and, in the current repository state, also in the legacy-named phantom `3.1` contrast script.

**What comes out**

- long-form phantom contrast tables

**Why this step is needed**

- the contrast emphasizes the part of the signal that depends on the difference between the two sequence constructions rather than their shared baseline attenuation.

**Code**

- `bash_template/phantoms/5.0-run_make_nogse_contrast_selected.sh`
- `bash_template/helpers/run_make_nogse_contrast_auto.sh`
- `bash_template/phantoms/3.1-run_make_contrast_selected.sh`
- `scripts/make_contrast.py`

### Stage 6: Phantom contrast fitting

**What goes in**

- phantom contrast tables

**What happens conceptually**

- the same `free`, `tort`, and `rest` contrast model families are applied as in the brain workflow;
- the phantom branch typically uses the water ROI as the reference ROI for correction.
- when gradient correction is enabled, the same side-specific factors described in the brain workflow are used here too; the difference is that the reference ROI is usually `water` rather than `Syringe`.

**What comes out**

- fitted phantom contrast tables
- optional corrected versions
- peak-derived `t_c` summaries

**Why this step is needed**

- it converts the observed CPMG-minus-Hahn contrast into estimates of diffusivity scale and restriction time scale.

**Code**

- `bash_template/phantoms/5.1-run_fit_nogse_contrast_vs_g.sh`
- `bash_template/phantoms/5.2-run_make_grad_correction_table.sh`
- `bash_template/phantoms/5.3-run_fit_free_all_nogse_contrast_vs_g_corr.sh`
- `bash_template/phantoms/6.1-run_fit_rest_all_nogse_contrast_vs_g_corr.sh`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`

### Stage 7: Phantom macro and `t_c`-vs-`t_d` summaries

**What goes in**

- phantom `Dproj` or contrast-fit tables

**What happens conceptually**

- the same macro-scale summary logic can be applied:
  - `D` vs `Delta_app`
  - `alpha_macro`
  - grouped rest fits
  - pseudo-Huber `t_c` vs `t_d`

In practice, the phantom batch templates leave some of these stages optional and skip them when the needed fit tables are absent.

**What comes out**

- summary tables and cross-experiment plots, when the required earlier products exist

**Code**

- `bash_template/phantoms/4.*`
- `bash_template/phantoms/6.*`
- `scripts/make_alpha_macro_summary.py`
- `scripts/run_tc_pipeline.py`
- `scripts/run_tc_vs_td.py`

## Shared fitting logic and shared utilities

Across both workflows, the repository uses a common fitting infrastructure:

- `src/fitting/core.py`
  - generic wrappers for `curve_fit` and `least_squares`
  - RMSE, chi-square, and parameter-error handling
- `src/fitting/experiments.py`
  - canonical experiment/model names
- `src/fitting/gradient_correction.py`
  - lookup of correction factors from correction tables
- `src/tools/fit_params_schema.py`
  - standardization of output fit tables

This design matters scientifically because it means the pipelines share:

- the same column conventions,
- the same idea of what a “fit parameter table” is,
- the same correction-lookup mechanism,
- the same grouping variables (`roi`, `direction`, `sheet`, `subj`, `td_ms`, `N`, `Hz`).

## Key differences between brains and phantoms

### What is shared

- ROI signals are summarized into the same long-form table model.
- Contrasts are always built as differences between matched curves.
- The same physical model families (`free`, `tort`, `rest`) are fitted to contrasts.
- The same gradient-correction and `t_c`-vs-`t_d` machinery is reused.
- The same mathematical rule links gradient correction to diffusivity mismatch: `f = sqrt(D0_side / D0_ref)`.

### What is different

- Brains use structural registration; phantoms do not.
- Brains define ROIs from FreeSurfer labels plus optional masks; phantoms use manual DWI-space masks.
- Brains add tensor rotation to obtain `long` and `tra`; phantoms usually keep direct axes.
- Phantom direct-`g` experiments often need grouped one-point-per-file reconstruction; brain data are mostly full directional curves already.
- The reference ROI for gradient correction is typically:
  - `Syringe` in brains
  - `water` in phantoms
- In OGSE signal fits the correction is usually consumed through the derived `b` axis, while phantom/NOGSE direct-`g` fits often consume it directly on the `g` axis.

## Code references for the main steps

| Stage | Brain workflow | Phantom workflow | Shared implementation |
|---|---|---|---|
| Batch orchestration | `bash_template/brains/run_brains_pipeline.sh` | `bash_template/phantoms/run_phantoms_pipeline*.sh` | `bash_template/helpers/pipeline_runner_lib.sh` |
| ROI extraction | `src/signal_extraction/coreg_extract_brain.py` | `src/signal_extraction/coreg_extract_phantom.py` | `src/signal_extraction/extract_roi_tables.py` |
| Results to long tables | `scripts/process_one_results.py` | `scripts/process_one_results.py` | `src/data_processing/*` |
| Signal rotation | `scripts/rotate_ogse_tensor.py` | not used in the direct phantom branch | `src/signal_rotation/rotation_tensor.py` |
| Contrast construction | `bash_template/brains/3.1`, `5.0` | `bash_template/phantoms/3.1`, `5.0` | `scripts/make_contrast.py`, `src/fitting/contrast.py` |
| OGSE signal fit | `scripts/fit_ogse_signal_vs_g.py` | `scripts/fit_ogse_signal_vs_g.py` | `src/ogse_fitting/fit_ogse_signal_vs_g.py` |
| NOGSE signal fit | not the main brain branch | `scripts/fit_nogse_signal_vs_g.py` | `src/nogse_fitting/fit_nogse_signal_vs_g.py` |
| Contrast fits | `scripts/fit_ogse_contrast_vs_g.py`, `scripts/fit_nogse_contrast_vs_g.py` | same | `src/ogse_fitting/fit_ogse_contrast_vs_g.py`, `src/nogse_fitting/fit_nogse_contrast_vs_g.py` |
| Physical model formulas | same | same | `src/models/model_fitting.py` |
| Gradient correction | `scripts/make_grad_correction_table.py` | `scripts/make_grad_correction_table.py` | `src/ogse_fitting/make_grad_correction_table.py`, `src/fitting/gradient_correction.py` |
| Final `t_c` vs `t_d` stage | `scripts/run_tc_pipeline.py`, `scripts/run_tc_vs_td.py` | same | `src/tc_fittings/*` |

## Summary

The repository implements one coherent analysis philosophy with two acquisition-specific front ends.

For brains, the logic is:

1. move anatomical ROIs into diffusion space,
2. extract ROI curves,
3. standardize them into long experiment tables,
4. rotate directional OGSE data into tensor-informed axes,
5. build matched contrasts,
6. fit those contrasts with increasingly structured models,
7. summarize the fitted correlation time scale across diffusion times.

For phantoms, the logic is:

1. use manually defined DWI-space masks,
2. assemble direct-`g` signal curves when needed,
3. optionally fit individual NOGSE signal families,
4. build CPMG-minus-Hahn contrasts,
5. fit the same contrast-model family,
6. use water as the reference for correction and summary.

The design is therefore intentionally layered:

- image geometry first,
- signal tables second,
- physical models last.

That separation is what makes the pipeline reusable across brains and phantoms while still preserving the distinct physical meaning of each workflow.
