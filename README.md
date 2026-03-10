# Project-Balseiro-Microstructure — DWI ROI Signal Extraction Pipeline

This repository contains a server-side pipeline to:
1) run FreeSurfer recon-all (per subject) to obtain structural segmentations,
2) coregister DWI (b0) to T1 (FreeSurfer),
3) warp selected atlas labels and optional manual masks into diffusion space,
4) extract ROI-wise signal statistics vs b-values and write MATLAB-like Excel tables.

Main scripts:
- `scripts/run_freesurfer.sh` — runs FreeSurfer recon-all for a set of subjects.  
- `scripts/coreg_extract.py` — main pipeline: registration, ROI masks, tables.  
- `scripts/extract_roi_tables.py` — ROI/statistics/Excel helpers imported by `coreg_extract.py`.

---

## Requirements

### External neuroimaging tools (must be available in PATH)
- FreeSurfer: `recon-all`, `mri_convert`
- MRtrix3: `dwiextract`
- FSL: `bet`, `fslmaths`, `fslmeants`
- ANTs: `antsRegistration`, `antsApplyTransforms`

`coreg_extract.py` calls these tools via subprocess. It prints each command before running it. :contentReference[oaicite:1]{index=1}

### Python
Python 3.10+ recommended.

Python packages:
- numpy
- nibabel
- pandas
- openpyxl

---

## Data layout (expected)

Example inputs:
- DWI NIfTI files under `--exp-root` (e.g. `Data-NIFTI-BRAINS-denoised_topup/<SUBJ>/...nii.gz`)
- Matching gradient files `<seq_no_ext>.bval` and `<seq_no_ext>.bvec` under `--grad-root` (defaults to `--exp-root`) :contentReference[oaicite:2]{index=2}

FreeSurfer outputs:
- A FreeSurfer SUBJECTS_DIR that contains folders named `sub-<SUBJ>` with `mri/T1.mgz`, `mri/brain.mgz`, `mri/wmparc.mgz` (or produced by recon-all). 

---

## Step 1 — Run FreeSurfer (once per subject)

`run_freesurfer.sh` is a convenience script that sets SUBJECTS_DIR and calls `recon-all -all` for multiple subjects. Edit it to add/remove subjects and point the `-i` path to the subject’s T1 input. :contentReference[oaicite:4]{index=4}

Example:
```bash
bash scripts/run_freesurfer.sh

