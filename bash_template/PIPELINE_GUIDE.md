# Pipeline Guide

This document describes how to run the full diffusion MRI pipeline for all four
experimental cases. Each case combines a subject type (`brain` or `phantom`) with
a pulse sequence (`ogse` or `nogse`).

---

## Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Directory layout](#directory-layout)
4. [Phase 1 — Preprocessing](#phase-1--preprocessing)
5. [Phase 2 — Analysis pipeline](#phase-2--analysis-pipeline)
   - [Case: ogse\_brain](#case-ogse_brain)
   - [Case: ogse\_phantom](#case-ogse_phantom)
   - [Case: nogse\_brain](#case-nogse_brain)
   - [Case: nogse\_phantom](#case-nogse_phantom)
6. [Step reference](#step-reference)
7. [Environment variable reference](#environment-variable-reference)
8. [Manifests](#manifests)
9. [Inspecting the master table](#inspecting-the-master-table)

---

## Overview

The pipeline has two phases:

**Phase 1 — Preprocessing** converts raw DICOM images into signal-versus-gradient
tables (Excel workbooks called `*_results.xlsx`). This phase is subject/session
specific and runs once per acquisition session.

**Phase 2 — Analysis** ingests those Excel workbooks into a single master
table (`master.long.parquet`) and then runs fitting, plotting, and summary
steps through the unified runner `run_dataset.sh`.

All Phase 2 commands follow this pattern:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh <type_subj> <type_seq> <step...>
```

Run all commands from the project root (the parent of `nogse_pipeline/`).

---

## Prerequisites

### Python environment

The pipeline requires the `nogse_pipe_env` conda environment:

```bash
conda activate nogse_pipe_env
```

The runner detects it automatically. You can override with:

```bash
PY=/path/to/python bash nogse_pipeline/bash_template/run_dataset.sh ...
```

### Directory tree expected by Phase 2

```
<project_root>/
├── nogse_pipeline/          # this repository
├── Data-signals/
│   ├── Results/             # *_results.xlsx files (output of Phase 1)
│   ├── sequence_parameters_brains.xlsx
│   └── sequence_parameters_phantoms.xlsx
└── analysis/
    ├── brains/
    │   ├── ogse_experiments/    # created by the runner
    │   └── nogse_experiments/   # created by the runner
    └── phantoms/
        ├── ogse_experiments/
        └── nogse_experiments/
```

The runner creates `analysis/` subdirectories automatically.

---

## Phase 1 — Preprocessing

Phase 1 scripts live in `steps/preprocessing/`. They are run directly (not
through `run_dataset.sh`) and must be edited once per acquisition session to
set the correct input paths.

### brains\_ogse

```
steps/preprocessing/brains_ogse/
  0.0-run_dicom2nifti.sh                       DICOM → NIfTI conversion
  1.0-run_BRAINS-denoised_topup_signal_extraction.sh  NIfTI → *_results.xlsx
```

`1.0-run_BRAINS-*` applies:
- Topup distortion correction
- MP2RAGE-derived brain mask + FreeSurfer atlas ROIs
- Signal extraction per ROI and per gradient direction

Edit the script to add or remove `run_case` blocks for each acquisition session.
The `--atlas-roi` flags select FreeSurfer regions (e.g., `4:Left-Lateral-Ventricle`).
The `--syringe-mask-*` flag adds the Syringe reference region when present.

### phantoms\_nogse and phantoms\_ogse

```
steps/preprocessing/phantoms_nogse/   (same structure for phantoms_ogse)
  0.0-run_dicom2nifti.sh              DICOM → NIfTI
  0.1-run_make_gval_gvec.sh           Build gradient magnitude/direction tables
  0.2-prep_phantom_b0.sh              Prepare b=0 reference image
  0.3-copy_selected_files.sh          Copy selected NIfTI volumes to working dir
  1.0-run_PHANTOM-denoised_signal_extraction.sh   NIfTI → *_results.xlsx
```

Edit `1.0-run_PHANTOM-*` to set `EXP_ROOT` and `OUT_SUBJ_REL` for each
acquisition session. Phantoms use `USE_MEAN=1` (multiple repetitions are
averaged into one signal row) and `--phantom-direct` (no brain mask needed).

### dicom\_params (all cases)

```
steps/preprocessing/dicom_params/
  0.0-run_extract_dicom_sequence_metadata.sh    Export sequence params from DICOM headers
  0.1-run_export_one_dicom_parameters.sh        Export params for a single DICOM file
  0.2-run_correlate_dicom_params_with_gradient.sh  Cross-check params vs gradient table
```

Run these when verifying that `sequence_parameters_*.xlsx` matches the actual
acquisition parameters stored in the DICOM headers.

---

## Phase 2 — Analysis pipeline

The analysis pipeline is the same runner for all four cases; the `type_subj`
and `type_seq` arguments select the correct defaults, scripts, and manifests.

### Case: ogse\_brain

**Full pipeline (run in order):**

```bash
# 1. Import Results into master table
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest

# 2. Rotate diffusion tensor directions (adds D_proj to each row)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

# 3. Build contrast rows S(N1,Hz1) - S(N2,Hz2) using contrasts.csv manifest
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast

# 4. Explore data visually
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast

# 5. Fit monoexponential signal model (extracts D0 per ROI/direction/Td)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_monoexp

# 6. Compute alpha_macro from D0 vs Delta (requires fit_signal first)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha

# 7. (Optional) Build gradient-correction table from the Syringe reference
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction

# 8. (Optional) Refit signals with gradient correction applied
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr

# 9. Fit OGSE contrast curves (extracts tc_peak, D0, alpha per ROI/direction/Td)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
# or use the mixed_global model:
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_mixed_global

# 10. Fit tc vs Td (the key biophysical summary: tc(Td) = c + alpha·delta·...)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc

# 11. Diagnostic plots
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_d0_delta
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d
```

**Output locations:**

```
analysis/brains/ogse_experiments/
  master.long.parquet                  master table (all row kinds)
  master_fit_params.parquet            cumulative fit parameters
  data/tables/                         per-file ingested tables
  data-rotated/tables/                 per-file rotated tables
  contrast-data-master/                per-contrast data tables
  plots-master/signal/                 signal plots
  plots-master/contrast/               contrast plots
  plots-master/monoexp_D_vs_time/      D vs td diagnostic plots
  fits/ogse_signal_master/             monoexp signal fit results
  fits/ogse_contrast_master/           contrast fit results
  fits/grad_correction_master/         gradient correction table
  fits/tc_vs_td_master/                tc-vs-td fit results
  alpha_macro/master/                  alpha_macro summaries
```

**Key models for ogse\_brain:**

| Step | Default model | Notes |
|------|--------------|-------|
| `fit_signal_monoexp` | `monoexp` | Extracts D0 per ROI |
| `fit_contrast_free` | `ogse_free` | Free fit: tc, D0 |
| `fit_contrast_mixed_global` | `mixed_global` | Global tc across Td |
| `fit_global_signal` | `ogse_mixed_offset` | Global fit on raw signals |
| `tc` | `pseudohuber_fixed_macro` | Pseudo-Huber tc(Td) |

---

### Case: ogse\_phantom

Same step sequence as ogse\_brain. Key differences:

- Gradient-correction ROI is `water` (not `Syringe`):

  ```bash
  CORR_ROI=water bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse grad_correction
  ```

- No FreeSurfer atlas: ROIs are phantom regions (e.g., `Bundle`, `water`)

- `USE_MEAN=1` during preprocessing: one signal row per sequence

- Ingesting a single phantom session:

  ```bash
  bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse \
    --results-root Data-signals/Results/20260122-PHANTOM_FIBER \
    ingest
  ```

**Full pipeline:**

```bash
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse rotate
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse contrast
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse plot_signal
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse plot_contrast
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_signal_monoexp
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse alpha
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse grad_correction
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_signal_gradcorr
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_contrast_free
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse tc
```

**Output locations:**

```
analysis/phantoms/ogse_experiments/   (same subdirectory structure as brains)
```

---

### Case: nogse\_brain

> **Status:** The analysis pipeline for NOGSE brain data is structurally complete
> but the manifests are empty templates. Before running the contrast and fit steps,
> fill in `manifests/brains_nogse/contrasts.csv` and
> `manifests/brains_nogse/signal_fits.csv`.

There are no dedicated NOGSE brain preprocessing scripts. NOGSE brain data uses
the same NIfTI files produced by `brains_ogse/` preprocessing (the sequences
share the acquisition session); only the `sequence_parameters_brains.xlsx`
workbook tab that specifies NOGSE timings differs.

**Full pipeline:**

```bash
# 1. Ingest NOGSE results (reads the NOGSE-tagged rows from *_results.xlsx)
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse ingest

# 2. Rotate tensor directions
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse rotate

# 3. Build contrasts (requires contrasts.csv to be filled)
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse contrast

# 4. Plot
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse plot_signal
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse plot_contrast

# 5. Fit NOGSE signal model (nogse_free: fits M0, D0, tc simultaneously)
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse fit_signal

# 6. Compute alpha_macro
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse alpha

# 7. Fit NOGSE contrast curves
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse fit_contrast_free

# 8. Fit tc vs Td
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse tc
```

**Key models for nogse\_brain:**

| Step | Default model | Notes |
|------|--------------|-------|
| `fit_signal` | `nogse_free` | Fits M0, D0, tc jointly |
| `fit_contrast_free` | `nogse_free` | NOGSE contrast model |
| `tc` | `pseudohuber_fixed_macro` | Same tc(Td) model |

**Output locations:**

```
analysis/brains/nogse_experiments/
```

**Populating the manifests:**

```
manifests/brains_nogse/contrasts.csv    columns: subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
manifests/brains_nogse/signal_fits.csv  columns: subj,sheet,roi,direction,td_ms,N,Hz,model
```

Example contrast row (N=8 pulses at 50 Hz vs N=4 pulses at 25 Hz, both at td=90 ms):
```
BRAIN,20220622_BRAIN,Left-Lateral-Ventricle,long,90,8,4,50,25
```

---

### Case: nogse\_phantom

> **Status:** Phase 1 preprocessing scripts exist and are ready to use. The Phase 2
> manifests are empty templates. Fill `manifests/phantoms_nogse/contrasts.csv`
> and `manifests/phantoms_nogse/signal_fits.csv` before running contrast/fit steps.

**Full pipeline:**

```bash
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse ingest
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse rotate
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse contrast
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse plot_signal
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse plot_contrast
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse fit_signal
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse alpha
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse fit_contrast_free
bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse tc
```

**Output locations:**

```
analysis/phantoms/nogse_experiments/
```

---

## Step reference

Each step maps to a script in `steps/`. Run any step with `--help` for details:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest --help
```

| Step alias | Script | Description |
|-----------|--------|-------------|
| `ingest` | `01_ingest_results.sh` | Read `*_results.xlsx` → `master.long.parquet` (row\_kind=signal) |
| `rotate` | `02_rotate_signals.sh` | Rotate tensor directions → row\_kind=signal\_rotated, adds D\_proj |
| `contrast` | `03_make_contrasts.sh` | Subtract two signal groups → row\_kind=contrast |
| `plot_signal` | `04_plot_signals.sh` | Plot S vs g curves from master |
| `plot_contrast` | `05_plot_contrasts.sh` | Plot contrast vs g curves from master |
| `fit_signal` | `06_fit_signals.sh` | Fit signal model per manifest row |
| `fit_signal_monoexp` | `06_fit_signals.sh` | Like fit\_signal but forces monoexp/bvalue\_thorsten defaults |
| `fit_signal_gradcorr` | `06_fit_signals.sh` | Like fit\_signal but applies gradient correction |
| `fit_contrast` | `07_fit_contrasts.sh` | Fit all contrast rows in master |
| `fit_contrast_free` | `07_fit_contrasts.sh` | Like fit\_contrast but forces the "free" model |
| `fit_contrast_mixed_global` | `07_fit_contrasts.sh` | Like fit\_contrast but forces mixed\_global model |
| `alpha` | `08_alpha_macro.sh` | Compute α\_macro from D\_proj; writes `summary_alpha_values.xlsx` |
| `tc` | `09_tc_vs_td.sh` | Fit tc(Td) model; reads `master_fit_params.parquet` |
| `grad_correction` | `10_make_grad_correction_table.sh` | Build gradient correction table from Syringe/water fits |
| `plot_d0_delta` | `11_plot_D0_vs_Delta_alpha.sh` | Plot D₀/D\_proj vs Δ\_app |
| `plot_monoexp_d` | `12_plot_monoexp_D_vs_time.sh` | Plot monoexp D vs td |
| `fit_global_signal` | `13_fit_global_signals.sh` | Fit global/mixed signal model on raw signals |

### Step dependencies

```
ingest
  └── rotate
        └── contrast
              ├── plot_contrast
              └── fit_contrast ─────────────┐
        ├── plot_signal                     │
        └── fit_signal                      │
              ├── alpha                     │
              │     └── tc (with fixed α)  ←┘
              ├── grad_correction
              │     └── fit_signal_gradcorr
              ├── plot_d0_delta
              └── plot_monoexp_d
```

`tc` reads `master_fit_params.parquet` which is written by both `fit_signal`
and `fit_contrast`. Run at least one of them first.

---

## Environment variable reference

Variables are set before the command:

```bash
VAR=value bash nogse_pipeline/bash_template/run_dataset.sh brain ogse <step>
```

### Path overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `PY` | auto-detected | Python interpreter |
| `SIGNALS_ROOT` | `<project_root>/Data-signals` | Root of input data |
| `RESULTS_ROOT` | `$SIGNALS_ROOT/Results` | One Results folder |
| `RESULTS_ROOTS` | — | Space-separated list of Results folders |
| `PARAMS_XLSX` | `Data-signals/sequence_parameters_<type_subj>.xlsx` | Sequence parameter workbook |
| `ANALYSIS_ROOT` | `analysis/<type_subj>s/<type_seq>_experiments` | Analysis output root |
| `MASTER_PARQUET` | `$ANALYSIS_ROOT/master.long.parquet` | Master table |
| `MASTER_FIT_PARAMS` | `$ANALYSIS_ROOT/master_fit_params.parquet` | Cumulative fit params |
| `MANIFEST_DIR` | `manifests/<type_subj>s_<type_seq>/` | CSV manifests directory |

### Step-specific variables (most common)

| Variable | Step | Description |
|----------|------|-------------|
| `SIGNAL_FIT_MODEL` | fit\_signal | Model name, e.g. `monoexp`, `nogse_free` |
| `SIGNAL_FIT_G_TYPE` | fit\_signal | Gradient column, e.g. `bvalue_thorsten`, `g` |
| `SIGNAL_FIT_EXTRA_ARGS` | fit\_signal | Extra Python flags |
| `FIT_MODEL` | fit\_contrast | Model name, e.g. `ogse_free`, `nogse_free`, `mixed_global` |
| `FIT_GBASE` | fit\_contrast | Gradient axis, e.g. `g_lin_max`, `g_thorsten_1` |
| `FIT_EXTRA_ARGS` | fit\_contrast | Extra Python flags |
| `CORR_ROI` | grad\_correction, fit\_signal\_gradcorr | Reference ROI (`Syringe` or `water`) |
| `CORR_XLSX` | fit\_signal\_gradcorr | Path to the correction table |
| `ALPHA_N` | alpha | N value used for D₀ extraction (default: 1) |
| `TC_METHOD` | tc | Fitting model: `pseudohuber_fixed_macro`, `pseudohuber_free`, `linear` |
| `TC_Y_COL` | tc | Y column: `tc_peak_ms` (default) |
| `PLOT_ROI` | plot\_signal, plot\_contrast | Filter by ROI |
| `PLOT_DIRECTION` | plot\_signal, plot\_contrast | Filter by direction (`long`, `tra`, `x`, `y`, `z`) |
| `GLOBAL_SIGNAL_MODEL` | fit\_global\_signal | e.g. `ogse_mixed_offset` |
| `GLOBAL_SIGNAL_TC_MODE` | fit\_global\_signal | `fixed`, `free`, `global_td`, `global_contrast` |

Run any step with `--help` to see all variables for that step.

---

## Manifests

Manifests are CSV files that tell the runner which ROI/direction/Td combinations
to process. Edit them to match your dataset.

### contrasts.csv

Specifies which pairs of signals to subtract to form a contrast.

```
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
BRAIN,20220622_BRAIN,Left-Lateral-Ventricle,long,90,8,4,50,25
```

Each row creates one `row_kind=contrast` group in master:
`S(N_1, Hz_1) - S(N_2, Hz_2)` at the given `td_ms`.

### signal\_fits.csv

Specifies which signal groups to fit individually.

```
subj,sheet,roi,direction,td_ms,N,Hz,model
BRAIN,20220622_BRAIN,Left-Lateral-Ventricle,long,90,4,25,monoexp
```

Use `ALL` in any column to match all values for that dimension.

### Manifest locations

```
manifests/brains_ogse/contrasts.csv      # OGSE brain — ready to fill
manifests/brains_ogse/signal_fits.csv
manifests/phantoms_ogse/contrasts.csv    # OGSE phantom — ready to fill
manifests/phantoms_ogse/signal_fits.csv
manifests/brains_nogse/contrasts.csv     # NOGSE brain — empty template
manifests/brains_nogse/signal_fits.csv
manifests/phantoms_nogse/contrasts.csv   # NOGSE phantom — empty template
manifests/phantoms_nogse/signal_fits.csv
```

---

## Inspecting the master table

`master.long.parquet` is never modified in place. All steps append rows.
Each row has a `row_kind` column:

| row\_kind | Written by | Content |
|-----------|-----------|---------|
| `signal` | ingest | Raw signal vs gradient |
| `signal_rotated` | rotate | Rotated signal with D\_proj |
| `contrast` | contrast | S(N1) - S(N2) per gradient |
| `fit_params` | fit\_signal, fit\_contrast | Fitted model parameters |

Export to Excel for inspection:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --out-xlsx /tmp/master_inspect.xlsx
```

Filter by row kind:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --row-kind signal_rotated \
  --out-xlsx /tmp/master_rotated.xlsx
```

Show only the first 5000 rows:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --head 5000 --out-xlsx /tmp/master_head.xlsx
```
