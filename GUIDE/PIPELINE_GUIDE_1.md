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
- physical model formulas: `src/models/model_fitting.py`
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

The repository implements this explicitly by estimating `S0` from the `b_step == 0`
rows inside each `(stat, roi, direction)` group:

```python
s0 = (
    out.loc[out["b_step"] == 0]
    .groupby(["stat", "roi", "direction"], as_index=False)["value"]
    .mean()
    .rename(columns={"value": "S0"})
)
out["value_norm"] = out["value"] / out["S0"]
out.loc[(out["b_step"] == 0) & out["S0"].notna(), "value_norm"] = 1.0
```

Code reference: `scripts/process_one_results.py`, `_add_S0_and_value_norm`.