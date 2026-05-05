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
