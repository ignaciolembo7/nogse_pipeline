# NOGSE / OGSE Pipeline

End-to-end analysis pipeline for brain and phantom diffusion MRI experiments
using N-pulse Oscillating Gradient Spin Echo (NOGSE) and Oscillating Gradient
Spin Echo (OGSE) sequences.

The pipeline converts per-session ROI-level diffusion signals into:
- clean long-form signal tables,
- OGSE or NOGSE contrast tables (`S(N1, Hz1) − S(N2, Hz2)`),
- monoexponential diffusivity fits (D₀ per ROI/direction/Td),
- physical contrast-model fits (free, tort, rest, mixed, ...),
- α\_macro summaries from D\_proj(Δ),
- tc-vs-Td fits using the pseudo-Huber or linear model.

## Quick start

**For a complete step-by-step guide for all four pipeline cases**
(ogse\_brain, ogse\_phantom, nogse\_brain, nogse\_phantom) **see
[bash\_template/PIPELINE\_GUIDE.md](bash_template/PIPELINE_GUIDE.md).**

All analysis commands use the unified runner from the project root:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest rotate contrast
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_monoexp alpha tc
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest rotate contrast fit_contrast_free tc
```

For the full step reference and environment variables, run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh --help
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse <step> --help
```

## What the pipeline does

1. **Preprocessing** — DICOM → NIfTI → `*_results.xlsx` signal tables.
   Scripts live in `bash_template/steps/preprocessing/`.

2. **Ingest** — Read `*_results.xlsx` into `master.long.parquet`
   (`row_kind=signal`).

3. **Rotate** — Rotate diffusion tensor directions; append `row_kind=signal_rotated`
   rows with `D_proj`.

4. **Contrast** — Subtract two signal groups; append `row_kind=contrast` rows.

5. **Fit signals** — Monoexponential or NOGSE model per ROI/direction/Td;
   append fit params to `master_fit_params.parquet`.

6. **Fit contrasts** — Physical contrast model (OGSE: free/rest/mixed;
   NOGSE: nogse\_free); append fit params.

7. **Summaries** — α\_macro from D\_proj(Δ); tc-vs-Td pseudo-Huber fit.

All steps read and write `master.long.parquet`; nothing is stored twice.

## Directory layout

```
nogse_pipeline/
├── bash_template/
│   ├── run_dataset.sh              unified analysis runner
│   ├── PIPELINE_GUIDE.md           complete per-case walkthrough
│   ├── README.md                   quick reference
│   ├── manifests/                  CSV manifests per dataset
│   └── steps/
│       ├── 01_ingest_results.sh    … 13_fit_global_signals.sh
│       └── preprocessing/          DICOM → Results scripts
├── scripts/
│   ├── data/                       ingest, rotate, contrast scripts
│   ├── fitting/                    signal, contrast, tc fitting scripts
│   └── summary/                    alpha_macro, grad_correction scripts
└── src/
    ├── data_processing/            schema, reshape, features
    ├── fitting/                    model registry, experiment registry
    ├── models/                     physical model formulas
    ├── ogse_fitting/               OGSE contrast fitting loop + registry
    ├── tc_fittings/                tc-vs-Td models and fitting
    └── signal_extraction/          NIfTI → per-ROI signal extraction
```

## Most important code locations

| Task | File |
|------|------|
| Physical model formulas | `src/models/model_fitting.py` |
| OGSE contrast model registry | `src/ogse_fitting/contrast_model_registry.py` |
| tc-vs-Td model registry | `src/tc_fittings/tc_td_registry.py` |
| tc-vs-Td physics functions | `src/tc_fittings/tc_td_models.py` |
| Experiment/model registry | `src/fitting/experiments.py` |
| Results → master table | `scripts/data/process_one_results.py` |
| Tensor rotation | `scripts/data/rotate_ogse_tensor.py` |
| Contrast construction | `scripts/data/make_contrast.py` |
| OGSE signal fitting | `scripts/fitting/fit_ogse_signal_vs_g.py` |
| NOGSE signal fitting | `scripts/fitting/fit_nogse_signal_vs_g.py` |
| OGSE contrast fitting | `scripts/fitting/fit_ogse_contrast_vs_g.py` |
| NOGSE contrast fitting | `scripts/fitting/fit_nogse_contrast_vs_g.py` |
| tc-vs-Td fitting runner | `scripts/fitting/run_tc_vs_td.py` |
| α\_macro summary | `scripts/summary/make_alpha_macro_summary.py` |
| Gradient correction | `scripts/data/make_grad_correction_table.py` |
| Brain signal extraction | `src/signal_extraction/coreg_extract.py` |

## Extending the pipeline

**Adding a new tc-vs-Td fitting model** (e.g., a linear model):
1. Add `tc_mymodel(Td, param1, param2)` in `src/tc_fittings/tc_td_models.py`
2. Add an entry to `METHODS` in `src/tc_fittings/tc_td_registry.py`
3. Run via `TC_METHOD=mymodel bash ... tc`

**Adding a new OGSE contrast model:**
1. Add `_eval_mymodel(td_ms, G1, G2, n_1, n_2, params)` in
   `src/ogse_fitting/contrast_model_registry.py`
2. Add an entry to `OGSE_CONTRAST_FIT_SPECS`
3. The fitting loop, plot panels, and `experiments.py` pick it up automatically

## External tools

Depending on the stage:

- **FreeSurfer**: `recon-all`, `mri_convert` (brain parcellation)
- **FSL**: `bet`, `fslmaths`, `fslmeants`, `fslroi` (brain/phantom masking)
- **ANTs**: `antsRegistration`, `antsApplyTransforms` (T1→DWI registration)
- **MRtrix3**: `dwiextract` (DWI volume selection)
- **dcm2niix**: DICOM to NIfTI conversion

Python 3.10+ with the `nogse_pipe_env` conda environment.
