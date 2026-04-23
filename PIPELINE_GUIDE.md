# NOGSE / OGSE Pipeline Guide

## Purpose of the pipeline

This repository implements a diffusion-MRI analysis pipeline that starts from ROI-averaged sequence signals and ends with physically interpretable summaries such as:

- normalized signal curves,
- OGSE and NOGSE contrast curves,
- monoexponential diffusivities,
- fitted correlation times `t_c`,
- macro-scale diffusivity ratios such as `alpha_macro`.

The key point is that the pipeline is not mainly about image processing in the abstract. It is about turning a set of brain or phantom diffusion measurements into a small number of aligned, comparable signal representations and then fitting those representations with specific physical models.

## Main pipeline stages found in the repository

Reading the repository shows the pipeline is organized around these conceptual stages:

1. Convert DICOM series to NIfTI, and attach gradient sidecars.
2. Extract ROI-level signals from each acquisition.
3. Convert MATLAB-style ROI tables into canonical long-form experiment tables.
4. Derive gradient and `b`-value representations needed for later fitting.
5. For brains only: rotate directional OGSE signals into tensor-informed axes.
6. Build contrasts by subtracting matched acquisitions.
7. Fit signal or contrast models to estimate `D0`, `alpha`, or `t_c`.
8. Use a reference ROI to derive gradient-correction factors.
9. Aggregate fitted contrast curves across experiments and fit `t_c` as a function of diffusion time `t_d`.

The implementation is driven mainly by:

- brain orchestration: `bash_template/brains/run_brains_pipeline.sh`
- phantom orchestration: `bash_template/phantoms/run_phantoms_pipeline*.sh`
- signal extraction: `src/signal_extraction/coreg_extract_brain.py`, `src/signal_extraction/coreg_extract_phantom.py`
- experiment-table construction: `scripts/process_one_results.py`
- contrast construction: `scripts/make_contrast.py`, `src/fitting/contrast.py`
- signal rotation: `scripts/rotate_ogse_tensor.py`, `src/signal_rotation/rotation_tensor.py`
- signal and contrast fitting: `src/ogse_fitting/*`, `src/nogse_fitting/*`
- physical model formulas: `src/nogse_models/nogse_model_fitting.py`
- final `t_c` vs `t_d` fitting: `scripts/run_tc_vs_td.py`, `src/tc_fittings/tc_td_pseudohuber.py`

## High-level overview

At a high level, both workflows do the same scientific job:

1. define ROIs,
2. extract ROI-average signals for each acquisition,
3. normalize those signals relative to a zero-weighting reference,
4. align metadata such as `N`, `Hz`, `delta`, `Delta_app`, and `t_d`,
5. build matched differences between acquisitions when a contrast is the desired observable,
6. fit models that map the observed signal or contrast to a smaller set of physical parameters.

The brains and phantoms workflows differ mainly in how ROIs are defined and how directional information is handled:

- brains use structural registration and atlas transfer from T1/FreeSurfer space into DWI space;
- phantoms use manually drawn masks already in DWI space;
- brains add an extra tensor-rotation stage to reduce 6-direction OGSE data into physically meaningful axes such as `long` and `tra`;
- phantoms rely more heavily on direct gradient-amplitude (`g`) curves and on manually paired CPMG/Hahn comparisons.

## Shared concepts across workflows

### ROI signal tables

The extraction stage always produces ROI-wise statistics across the DWI volumes:

- mean (`avg`)
- standard deviation (`std`)
- median (`med`)
- mean absolute deviation (`mad`)
- mode (`mode`)

This logic is implemented in `src/signal_extraction/extract_roi_tables.py`, especially:

- `extract_tables`
- `_extract_collapsed_mean_tables`
- `write_excel_like_matlab`

The extracted signal is a region average, not a voxelwise fit. The pipeline is therefore built around ROI-wise curves.

### Signal normalization

Most downstream fitting uses normalized signal:

```text
value_norm = value / S0
```

where `S0` is the ROI-average signal at `b_step == 0` or the zero-gradient reference. This is added in `scripts/process_one_results.py` and enforced again in `src/data_processing/schema.py`.

Why this is needed:

- it removes arbitrary ROI intensity scaling,
- it makes different acquisitions comparable,
- it lets later models focus on attenuation shape rather than absolute image intensity.

### Converting gradient amplitude into `b`-value

Many later steps work on a `b`-value axis even when the acquisition was organized by gradient amplitude. The repository uses:

```python
b = N * gamma**2 * delta_ms**2 * delta_app_ms * g**2 / 1e9
```

This is implemented in `src/fitting/b_from_g.py`.

Conceptually:

- `g` sets gradient strength,
- `delta` and `Delta_app` set the timing scale of the encoding,
- `N` accounts for the number of oscillation periods or lobes,
- the resulting `b` gives the diffusion-weighting strength.

The repository keeps several related axes:

- `g`
- `g_max`
- `g_lin_max`
- `g_thorsten`
- `bvalue`
- `bvalue_g`
- `bvalue_g_lin_max`
- `bvalue_thorsten`

This lets the same data be re-expressed in the axis most appropriate for plotting or fitting.

### Contrast construction

The generic contrast definition is always a difference between matched signals:

```python
value = value_1 - value_2
value_norm = value_norm_1 - value_norm_2
```

implemented in `src/fitting/contrast.py` and used by `scripts/make_contrast.py`.

The important scientific point is that the pipeline treats contrast as a derived observable built from two already-aligned experiments. The two sides are kept explicitly in the table, so later fits still know:

- which side was acquisition 1 and which was acquisition 2,
- each side's `N`, `Hz`, `sequence`, `source_file`, and gradient axis.

## Brain pipeline

### Inputs

The brain workflow expects:

- denoised/topup-corrected DWI NIfTI files under `Data-NIFTI-BRAINS-denoised_topup/...`
- matching gradient sidecars (`.bval/.bvec` or `.gval/.gvec`)
- FreeSurfer subject folders under `Data-signals/DATA_PROCESSED/subjects/sub-<subject>`
- optional syringe masks in T1 or DWI space

The main batch entry points are:

- `bash_template/brains/1.0-run_BRAINS-denoised_topup_signal_extraction.sh`
- `bash_template/brains/2.0-run_process_all_results.sh`
- `bash_template/brains/3.*` through `6.*`

### Stage 1: Structural export and reference-image creation

**What goes in**

- FreeSurfer outputs: `T1.mgz`, `brain.mgz`, `wmparc.mgz`
- one DWI sequence

**What happens conceptually**

- the T1, brain-only T1, and FreeSurfer label volume are exported to NIfTI;
- a diffusion reference image is created from the DWI sequence, usually the mean of the `b=0` volumes;
- if the sequence is organized as repeated measurements of the same condition, the full mean image can be used instead.

**What comes out**

- `T1.nii.gz`
- `T1_brain.nii.gz`
- `wmparc.nii.gz`
- `NII_b0.nii.gz` and `b0_mean.nii.gz`, or `NII_mean.nii.gz` and `mean.nii.gz`

**Why this step is needed**

- the structural images define anatomically meaningful ROIs;
- the DWI reference provides the moving image for registration and the intensity reference for ROI transfer.

**Key physical or mathematical idea**

- the `b=0` mean is used because it is the least diffusion-weighted DWI contrast and is therefore the best structural proxy inside the diffusion series.

**Code**

- `src/signal_extraction/coreg_extract_brain.py`
  - `prep_struct_once`
  - `make_b0_mean_with_mrtrix_fsl`
  - `make_full_mean_with_fsl`
  - `make_zero_gradient_mean_from_values`

### Stage 2: Skull stripping and DWI-to-T1 registration

**What goes in**

- diffusion reference image
- brain-only T1

**What happens conceptually**

- the DWI reference is skull-stripped with BET;
- ANTs performs rigid plus affine registration between the stripped DWI reference and the stripped T1;
- the fitted transform is then inverted so atlas labels and masks can be brought from T1 space into DWI space.

**What comes out**

- skull-stripped diffusion reference
- affine transform between DWI and T1
- T1 and T1-brain resampled into DWI space

**Why this step is needed**

- the ROI definitions live in anatomical space but the signal measurements live in diffusion space;
- accurate ROI transfer requires a common geometry.

**Key physical or mathematical idea**

- this is geometric alignment, not a diffusion-model fit;
- nearest-neighbor interpolation is used for label images so atlas integers remain integers, while linear interpolation is used for scalar images such as T1.

**Code**

- `src/signal_extraction/coreg_extract_brain.py`
  - `bet_b0`
  - `ants_register_b0_to_t1`
  - `ants_apply_inverse_label`
  - `ants_apply_inverse_image`

### Stage 3: ROI definition in diffusion space

**What goes in**

- FreeSurfer `wmparc` labels
- optional syringe mask
- optional manual masks
- the DWI-space reference grid

**What happens conceptually**

- selected anatomical labels are kept from `wmparc`;
- by default the corpus callosum subdivisions `251..255` are included;
- ventricle labels are added explicitly in the brain batch script;
- the selected atlas labels are warped into DWI space;
- a syringe mask can also be warped from T1 or used directly if already drawn in DWI space;
- binary masks are created for each ROI and merged into one multi-label image for visual checking.

**What comes out**

- per-ROI binary masks in DWI space
- one combined `ALL_ROIS` label image and label map

**Why this step is needed**

- downstream fitting is ROI-based, so every acquisition must be summarized over exactly the same anatomical targets.

**Key physical or mathematical idea**

- the entire ROI workflow depends on preserving the DWI voxel grid;
- shape and affine consistency are checked before extraction so masks truly refer to the same physical voxels as the DWI.

**Code**

- `src/signal_extraction/coreg_extract_brain.py`
  - `write_selected_label_image`
  - `write_binary_mask_from_label_image`
  - `build_all_rois_multilabel`
- `src/signal_extraction/extract_roi_tables.py`
  - `build_roi_from_binary_mask`
  - `build_rois_from_labelmask`
  - `build_rois_from_cc_mask`

### Stage 4: ROI signal extraction

**What goes in**

- the 4D DWI sequence
- the DWI-space ROI masks
- gradient sidecars

**What happens conceptually**

- for each volume and each ROI, the pipeline computes summary statistics;
- for single-condition repeated acquisitions it can collapse the whole sequence into one mean image before summarizing.

**What comes out**

- MATLAB-like Excel tables per sequence under `Data-signals/Results/...`

**Why this step is needed**

- it reduces each sequence to a compact ROI-by-volume table that can later be matched to acquisition parameters and fitted.

**Key physical or mathematical idea**

- this is where the image data become curves;
- from this point onward the pipeline mostly manipulates signal tables rather than NIfTI volumes.

**Code**

- `src/signal_extraction/extract_roi_tables.py`
  - `extract_tables`
  - `_extract_collapsed_mean_tables`
  - `write_excel_like_matlab`

### Stage 5: Canonical experiment-table construction

**What goes in**

- extracted Excel tables
- sequence-parameter spreadsheet
- filename-derived metadata

**What happens conceptually**

- the interleaved ROI tables are reshaped into long form: one row per ROI, direction, and `b_step`;
- the matching sequence-parameter row is attached;
- if the original acquisition was `b`-organized, gradient-amplitude surrogates are derived from the sequence timing;
- if the original acquisition was direct-`g`, the direct `g` axis is preserved and corresponding `bvalue_*` columns are derived where possible;
- `S0` and `value_norm` are added.

**What comes out**

- cleaned long-form signal tables (`*.long.parquet`, `*.xlsx`)

**Why this step is needed**

- later stages require a uniform representation no matter how the original sequence was named or organized;
- the long-form table is the central data model of the repository.

**Key physical or mathematical idea**

- the pipeline explicitly separates raw signal organization from physical metadata;
- the same signal can later be viewed either as a function of `g` or as a function of a derived `b`.

**Code**

- `scripts/process_one_results.py`
- `src/data_processing/reshape.py`
- `src/data_processing/schema.py`

### Stage 6: Tensor-based rotation of brain OGSE signals

**What goes in**

- long-form brain signal tables with six diffusion directions

**What happens conceptually**

- for each ROI and `b`-step, the pipeline estimates a diffusion tensor `D` from the directional attenuations;
- the tensor is diagonalized to obtain its principal axes;
- the signal is re-expressed along fixed axes such as:
  - tensor eigenvectors `eig1`, `eig2`, `eig3`
  - laboratory axes `x`, `y`, `z`
  - combined axes `long` and `tra`
- it also writes `D_proj`, the projection of the tensor along each rotated axis.

**What comes out**

- rotated signal tables
- projected diffusivity tables (`*.Dproj.long.parquet`)

**Why this step is needed**

- the raw six-direction measurements are hard to compare directly across subjects;
- rotation reduces them to axes that better reflect underlying anisotropy, especially for structures such as the corpus callosum.

**Key physical or mathematical idea**

The tensor fit uses the standard relation:

```text
-log(S/S0) / b = n^T D n
```

and then projects the fitted tensor onto chosen directions:

```text
D_proj = n^T D n
```

**Code**

- `scripts/rotate_ogse_tensor.py`
- `src/signal_rotation/rotation_tensor.py`
  - `fit_tensor_from_signals`
  - `D_proj`
  - `rotate_signals_tensor`

### Stage 7: OGSE contrast construction

**What goes in**

- matched rotated OGSE signal tables

**What happens conceptually**

- manually selected pairs of acquisitions are subtracted to build an OGSE contrast curve;
- the paired files share the same subject and diffusion-time context but differ in oscillation setting, typically through `N`, `Hz`, or both;
- both sides of the subtraction remain attached to the output table.

**What comes out**

- long-form OGSE contrast tables under `contrast-data-rotated/tables/...`

**Why this step is needed**

- the contrast isolates how the signal changes when the oscillating encoding is changed while keeping the broader acquisition context fixed.

**Key physical or mathematical idea**

- the contrast is treated as the observable to be fitted, not as an intermediate plotting convenience.

**Code**

- `bash_template/brains/3.1-run_make_contrast_selected_rotated.sh`
- `scripts/make_contrast.py`
- `src/fitting/contrast.py`

### Stage 8: OGSE monoexponential signal fitting and macro-scale summaries

**What goes in**

- rotated signal tables

**What happens conceptually**

- each ROI/direction curve is fitted with a monoexponential attenuation model;
- the fit can use either a fixed number of lowest-`b` points or an automatically selected prefix of the curve;
- from the fitted attenuation, the pipeline produces:
  - `D0`
  - synthetic `D_proj` tables
  - `D` vs `Delta_app` plots
  - `alpha_macro = <D0> / D0_ref`

**What comes out**

- monoexponential fit tables
- `Dproj` tables
- `D`-vs-`Delta_app` plots
- `summary_alpha_values.xlsx`

**Why this step is needed**

- it provides a simple diffusion-scale reference before moving to more specialized contrast models;
- `alpha_macro` is later used as a fixed slope parameter in the final `t_c`-vs-`t_d` fit.

**Key physical or mathematical idea**

The fitted model is:

```text
S(b) = M0 * exp(-b * D0)
```

The automatic prefix selection stops when adding another point worsens `rmse_log` beyond a tolerance, so the fit stays in the part of the curve that remains approximately monoexponential.

**Code**

- `scripts/fit_ogse_signal_vs_g.py`
- `src/ogse_fitting/fit_ogse_signal_vs_g.py`
- `scripts/plot_D0_vs_Delta.py`
- `scripts/make_alpha_macro_summary.py`
- `src/tc_fittings/alpha_macro_summary.py`

### Stage 9: NOGSE-style contrast construction and fitting in brains

**What goes in**

- selected rotated signal tables
- later, optionally corrected gradient factors

**What happens conceptually**

- the repository builds additional contrast tables from selected rotated acquisitions;
- these contrasts are then fitted with three model families:
  - `free`
  - `tort`
  - `rest`

For brains, the practical sequence is:

1. fit the uncorrected `free` model,
2. build a gradient-correction table from the syringe ROI,
3. refit with correction,
4. fit the corrected `rest` model over all target ROIs.

**What comes out**

- per-analysis fit tables with parameters such as `D0`, `alpha`, `tc_ms`
- peak-derived quantities such as `tc_peak_ms`, `lcf_peak_m`, and peak gradient location

**Why this step is needed**

- the contrast models are the part of the pipeline that tries to map the measured contrast to microstructural timescales rather than just to an effective diffusivity.

**Key physical or mathematical idea**

The model formulas come from `src/nogse_models/nogse_model_fitting.py`. In the repository’s naming:

- `free` uses unrestricted diffusion with effective diffusivity `D0`,
- `tort` scales the free-diffusion contribution by `alpha`,
- `rest` introduces a finite correlation time `t_c`.

Peak metrics are then converted to characteristic length and time scales using the notebook-derived formulas implemented in:

- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`

**Code**

- `bash_template/brains/5.0-run_make_nogse_contrast_selected_rotated.sh`
- `bash_template/brains/5.1-run_fit_nogse_contrast_vs_g.sh`
- `bash_template/brains/6.1-run_fit_rest_all_ogse_contrast_vs_g_corr.sh`
- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`
- `src/nogse_models/nogse_model_fitting.py`

### Stage 10: Gradient correction

**What goes in**

- OGSE monoexponential `D0` fits from a reference ROI
- free-model contrast fits from the same reference ROI

**What happens conceptually**

- the repository compares a reference diffusivity inferred from OGSE signal fits with the diffusivity implied by each side of the contrast fit;
- from that comparison it computes side-specific correction factors that act on the gradient axis.

The implemented formula is:

```text
ratio_side = D0_fit_nogse_side / D0_fit_monoexp
correction_factor_side = sqrt(ratio_side)
```

Because `b` scales as `g^2`, correcting gradient amplitude by a factor `f` implies a `b`-axis scaling of `f^2`, which is exactly how the later fitting scripts use these factors.

**What comes out**

- `grad_correction` tables with `correction_factor_1` and `correction_factor_2`

**Why this step is needed**

- it enforces consistency between the effective diffusion scale seen by the signal-fit branch and the one inferred by the contrast-fit branch.

**Key physical or mathematical idea**

- the correction is not a generic intensity normalization;
- it is specifically a recalibration of the effective gradient axis using a physically interpretable reference ROI.

**Code**

- `scripts/make_grad_correction_table.py`
- `src/ogse_fitting/make_grad_correction_table.py`

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

- the model formulas come from `src/nogse_models/nogse_model_fitting.py`, with `M_nogse_free` as the core signal model.

**Code**

- `bash_template/phantoms/2.2-run_fit_nogse_signal_vs_g.sh`
- `src/nogse_fitting/fit_nogse_signal_vs_g.py`
- `src/nogse_models/nogse_model_fitting.py`

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

### What is different

- Brains use structural registration; phantoms do not.
- Brains define ROIs from FreeSurfer labels plus optional masks; phantoms use manual DWI-space masks.
- Brains add tensor rotation to obtain `long` and `tra`; phantoms usually keep direct axes.
- Phantom direct-`g` experiments often need grouped one-point-per-file reconstruction; brain data are mostly full directional curves already.
- The reference ROI for gradient correction is typically:
  - `Syringe` in brains
  - `water` in phantoms

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
| Physical model formulas | same | same | `src/nogse_models/nogse_model_fitting.py` |
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
