# DWI preprocessing stage

This document defines the upstream DWI preprocessing stage for the brain and phantom OGSE/NOGSE workflows.

The stage is intended to prepare converted NIfTI diffusion data for the existing downstream signal-extraction, table-processing, contrast-construction, fitting, and summary-analysis steps. It must not change the downstream scientific logic or output schemas.

## Scope

The preprocessing stage covers:

1. MRtrix denoising with `dwidenoise`.
2. Residual image generation with `mrcalc`.
3. Gibbs-ringing correction with `mrdegibbs`.
4. FSL acquisition-parameter generation from JSON sidecars.
5. FSL `topup` for valid reverse phase-encoding b0 inputs.
6. Brain-mask generation from topup-unwarped b0 images.
7. FSL `eddy_openmp` plus `eddy_quad` QC.
8. MRtrix/ANTs bias-field correction with `dwibiascorrect ants`.
9. Optional slice-specification handling for eddy.

The implementation is command-oriented: the Python CLI resolves inputs, validates metadata, builds commands, and can either print them with `--dry-run` or execute them.

## Repository integration

The preprocessing stage is integrated as an upstream `0.x` stage inside the existing `bash_template` workflows. It does not create a duplicate pipeline.

Expected locations:

```text
src/preprocessing/
scripts/preprocess_dwi.py
bash_template/brains_ogse/0.5-run_preprocess_dwi.sh
bash_template/phantoms_ogse/0.5-run_preprocess_dwi.sh
notebooks/preprocessing_demo.ipynb
docs/preprocessing.md
```

Files under `bash/` must not be edited. Pipeline runner changes must be made under `bash_template/`.

## Inputs

The stage expects data already converted from DICOM to NIfTI, with diffusion sidecars available when required.

Required inputs for denoising and Gibbs correction:

```text
<subject>_ses-<session>_dwi.nii.gz
```

Required sidecars for eddy and downstream diffusion-gradient consistency:

```text
<subject>_ses-<session>_dwi.bvec
<subject>_ses-<session>_dwi.bval
```

Required JSON metadata for topup acquisition parameters:

```text
<subject>_ses-<session>_dwi.json
<subject>_ses-<session>_dwi_dirPA-b0.json
```

The exact input root is supplied by the CLI. No user-specific storage path is hard-coded.

## Brain workflow

For brain datasets, the default full chain is:

```text
denoise -> degibbs -> topup -> eddy -> bias -> qc
```

`topup` and `eddy` require valid reverse phase-encoding b0 inputs, JSON metadata, b-values, and b-vectors. If these inputs are missing, the CLI must fail with an explicit error unless the user selected only steps that do not require them.

## Phantom workflow

For phantom datasets, the default safe chain is:

```text
denoise -> degibbs -> bias
```

`topup` and `eddy` are opt-in for phantoms. They must run only when valid reverse phase-encoding inputs and metadata are explicitly provided. The CLI must not silently run topup or eddy on invalid 2D phantom inputs.

## Output layout

The output layout is intentionally close to the legacy scripts and to the downstream expectations:

```text
<output-root>/<subject>/ses-<session>/dwi/den/
<output-root>/<subject>/ses-<session>/dwi/den/denoising-der/
<output-root>/<subject>/ses-<session>/dwi/den/preproc-1/topup-in/
<output-root>/<subject>/ses-<session>/dwi/den/preproc-1/topup-out/
<output-root>/<subject>/ses-<session>/dwi/den/preproc-2/eddy-in/
<output-root>/<subject>/ses-<session>/dwi/den/preproc-2/eddy-out/
<output-root>/<subject>/ses-<session>/dwi/den/preproc-2/bias-corr/
```

The intended final preprocessed image for downstream signal extraction is:

```text
<output-root>/<subject>/ses-<session>/dwi/den/preproc-2/bias-corr/<subject>_ses-<session>_dwi_den_grc_tec_preproc-2_bias-corr.nii.gz
```

For workflows that stop before eddy, the latest valid output is the Gibbs-corrected image:

```text
<output-root>/<subject>/ses-<session>/dwi/den/<subject>_ses-<session>_dwi_den_grc.nii.gz
```

## CLI examples

Dry-run brain preprocessing:

```bash
python scripts/preprocess_dwi.py \
  --dataset brains \
  --subjects BRAIN01 \
  --steps denoise degibbs topup eddy bias qc \
  --input-root /path/to/nifti_inputs \
  --output-root /path/to/preprocessed_outputs \
  --session T0 \
  --nthreads 8 \
  --dry-run
```

Dry-run phantom preprocessing without topup or eddy:

```bash
python scripts/preprocess_dwi.py \
  --dataset phantoms \
  --subjects PHANTOM01 \
  --steps denoise degibbs bias \
  --input-root /path/to/nifti_inputs \
  --output-root /path/to/preprocessed_outputs \
  --session T0 \
  --nthreads 8 \
  --dry-run
```

## Validation checks

The CLI must validate:

1. Selected steps are known and allowed for the selected dataset.
2. Required input images exist for each selected step.
3. Required JSON metadata exist before generating acquisition parameters.
4. `TotalReadoutTime` and `PhaseEncodingDirection` exist in JSON metadata.
5. Reverse phase-encoding metadata are consistent before topup.
6. The number of b-values matches the number of DWI volumes before eddy.
7. `index.txt` length is derived from the actual DWI volume count, not hard-coded.
8. External commands are available before execution, or reported clearly in dry-run output.
9. Existing outputs are not overwritten unless `--overwrite` is set.

## External dependencies

The preprocessing stage can require:

```text
MRtrix3: dwidenoise, mrdegibbs, mrcalc, dwibiascorrect, mrinfo
FSL: fslselectvols, fslmerge, topup, fslmaths, bet, eddy_openmp, eddy_quad
ANTs: N4BiasFieldCorrection through dwibiascorrect ants
Python: json, argparse, pathlib, subprocess
```

## Notes

The legacy scripts are used as behavioral references only. Their hard-coded paths, subject IDs, FSL switcher assumptions, b0 volume indices, and fixed eddy index length must be replaced with validated parameters or inferred values.
