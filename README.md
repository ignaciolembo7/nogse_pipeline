# NOGSE / OGSE Pipeline

This repository contains the end-to-end analysis pipeline used here for brain and phantom diffusion experiments. In practice, the pipeline turns ROI-level diffusion signals into:

- clean long-form signal tables,
- OGSE or NOGSE contrast tables,
- monoexponential diffusivity fits,
- contrast-model fits (`free`, `tort`, `rest`),
- grouped `t_c` summaries and final `t_c`-vs-`t_d` fits.

The detailed scientific walkthrough now lives in [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).

## What the pipeline does

At a high level, the repository implements these stages:

1. prepare NIfTI inputs and gradient sidecars,
2. extract ROI signals from each acquisition,
3. convert those signals into canonical long-form experiment tables,
4. optionally rotate brain OGSE data into tensor-informed axes,
5. build matched contrasts between acquisitions,
6. fit physical signal or contrast models,
7. derive correction factors and final cross-experiment summaries.

## Brain and phantom workflows

The repository has two front ends:

- brains: structural-registration workflow based on FreeSurfer, T1-to-DWI alignment, atlas ROI transfer, tensor rotation, contrast fitting, and `t_c`-vs-`t_d` summary fitting
- phantoms: direct DWI-space masking workflow with optional direct-`g` curve assembly, CPMG-vs-Hahn contrast building, contrast fitting, and optional summary stages

The main batch runners are:

- `bash_template/brains/run_brains_pipeline.sh`
- `bash_template/phantoms/run_phantoms_pipeline.sh`
- `bash_template/phantoms/run_phantoms_pipeline_ogse.sh`
- `bash_template/phantoms/run_phantoms_pipeline_nogse.sh`

## Most important code locations

- signal extraction:
  - `src/signal_extraction/coreg_extract_brain.py`
  - `src/signal_extraction/coreg_extract_phantom.py`
  - `src/signal_extraction/extract_roi_tables.py`
- results-to-table conversion:
  - `scripts/process_one_results.py`
  - `src/data_processing/reshape.py`
  - `src/data_processing/schema.py`
- rotation and projection:
  - `scripts/rotate_ogse_tensor.py`
  - `src/signal_rotation/rotation_tensor.py`
- contrast construction:
  - `scripts/make_contrast.py`
  - `src/fitting/contrast.py`
- fitting:
  - `scripts/fit_ogse_signal_vs_g.py`
  - `scripts/fit_nogse_signal_vs_g.py`
  - `scripts/fit_ogse_contrast_vs_g.py`
  - `scripts/fit_nogse_contrast_vs_g.py`
  - `src/ogse_fitting/*`
  - `src/nogse_fitting/*`
- physical model formulas:
  - `src/nogse_models/nogse_model_fitting.py`
- correction and final summaries:
  - `scripts/make_grad_correction_table.py`
  - `scripts/make_alpha_macro_summary.py`
  - `scripts/run_tc_pipeline.py`
  - `scripts/run_tc_vs_td.py`

## External tools

Depending on the stage, the pipeline expects some or all of:

- FreeSurfer: `recon-all`, `mri_convert`
- MRtrix3: `dwiextract`
- FSL: `bet`, `fslmaths`, `fslmeants`, `fslroi`
- ANTs: `antsRegistration`, `antsApplyTransforms`

Python 3.10+ is recommended. The package itself is defined in `pyproject.toml`, and the environment files in the repository document the broader scientific stack used around it.

## Where to start

- For a scientific explanation of how the pipeline works: read [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).
- For the brain batch order: read `bash_template/brains/run_brains_pipeline.sh`.
- For the phantom batch order: read `bash_template/phantoms/run_phantoms_pipeline*.sh`.
