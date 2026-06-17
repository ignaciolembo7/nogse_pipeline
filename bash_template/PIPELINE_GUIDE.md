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
   - [contrasts.csv](#contrastscsv)
   - [signal\_fits.csv](#signal_fitscsv)
   - [grad\_correction.csv](#grad_correctioncsv)
   - [Manifest locations](#manifest-locations)
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

# 1b. (Optional) Keep only first N b-steps per td_ms before analysis
# MASTER_FIRST_POINTS_BY_TD="120=8,210=6" \
#   bash nogse_pipeline/bash_template/run_dataset.sh brain ogse filter_master_points
# Then pass the filtered table to subsequent steps:
# export MASTER_PARQUET=analysis/brains/ogse_experiments/master.first_points.long.parquet

# 2. Rotate diffusion tensor directions (adds D_proj to each row)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

# 3. Build contrast rows S(N1,Hz1) - S(N2,Hz2) using contrasts.csv manifest
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast

# 4. Explore data visually
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast

# 5. Fit monoexponential signal model (extracts D0 per ROI/direction/Td)
#    Set model=monoexp in signal_fits.csv manifest, then:
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

# 6. Compute alpha_macro from D0 vs Delta (requires fit_signal first)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha

# 7. (Optional) Embed gradient-correction factors from the Syringe reference.
#    Reads grad_correction.csv manifest + master parquet directly — no prior
#    fit_signal or contrast step required.
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction

# 8. (Optional) Refit signals with gradient correction applied
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr

# 9. Fit OGSE contrast curves (extracts tc_peak, D0, alpha per ROI/direction/Td)
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
# or use the mixed_global model:
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_mixed_global

# 10. Fit tc vs Td (requires setting TC_FIT_PARAMS to the contrast fit-params parquet)
TC_FIT_PARAMS=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_glinmax_ogse_free/fit_params.ogse_free.glinmax.value_norm.direction_ALL.parquet \
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc

# 11. Diagnostic plots
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_d0_delta
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d
```

**Output locations:**

```
analysis/brains/ogse_experiments/
  master.long.parquet                  master table (all row kinds)
  data/tables/                         per-file ingested tables
  data-rotated/tables/                 per-file rotated tables
  contrast-data-master/                per-contrast data tables
  plots-master/signal/                 signal plots
  plots-master/contrast/               contrast plots
  plots-master/monoexp_D_vs_time/      D vs td diagnostic plots
  fits/<master_name>/<type_seq>_<ycol>_vs_<gtype>_<model>/
                                       fit results (signal and contrast)
    fit_params.<model>.<gtype>.<ycol>.direction_ALL.parquet
                                       accumulated fit-params for this type
    <exp>.N<n>.td<td>.<roi>.<model>.<gtype>.<ycol>.direction_<dir>.png
                                       per-fit plots (signal)
    <analysis_id>/                     per-experiment subdirs (contrast)
  fits/grad_correction/                gradient correction audit outputs
  fits/tc_vs_td_master/                tc-vs-td fit results
  alpha_macro/master/                  alpha_macro summaries
```

> **Note on `<master_name>`**: derived from the `MASTER_PARQUET` filename — `master.long.parquet` → `master`, `master.first_points.long.parquet` → `master.first_points`.

**Key models for ogse\_brain:**

| Step | Default model | Notes |
|------|--------------|-------|
| `fit_signal` | set in manifest | Extracts D0 per ROI (use model=monoexp in manifest) |
| `fit_contrast_free` | `ogse_free` | Free fit: tc, D0 |
| `fit_contrast_mixed_global` | `mixed_global` | Global tc across Td |
| `fit_global_signal` | `ogse_mixed_offset` | Global fit on raw signals |
| `tc` | `pseudohuber_fixed_macro` | Pseudo-Huber tc(Td) |

---

### Case: ogse\_phantom

Same step sequence as ogse\_brain. Key differences:

- Gradient-correction reference ROI is `water` (not `Syringe`). Populate
  `manifests/phantoms_ogse/grad_correction.csv` with the water curves before
  running:

  ```bash
  bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse grad_correction
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
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_signal
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
| `filter_master_points` | `00_filter_master_points.sh` | Write filtered master keeping first N b-steps per td\_ms |
| `fit_signal` | `06_fit_signals.sh` | Fit signal model per manifest row (model set in manifest) |
| `fit_signal_gradcorr` | `06_fit_signals.sh` | Like fit\_signal but applies gradient correction |
| `fit_contrast` | `07_fit_contrasts.sh` | Fit all contrast rows in master |
| `fit_contrast_free` | `07_fit_contrasts.sh` | Like fit\_contrast but forces the "free" model |
| `fit_contrast_mixed_global` | `07_fit_contrasts.sh` | Like fit\_contrast but forces mixed\_global model |
| `fit_global_signal` | `13_fit_global_signals.sh` | Fit global/mixed signal model on raw signals |
| `alpha` | `08_alpha_macro.sh` | Compute α\_macro from D\_proj; writes `summary_alpha_values.xlsx` |
| `tc` | `09_tc_vs_td.sh` | Fit tc(Td) from contrast fit-params (`TC_FIT_PARAMS` required) |
| `grad_correction` | `10_make_grad_correction_table.sh` | Fit each syringe/water curve with NOGSE free + monoexp, compute correction\_factor = √(D0\_nogse/D0\_mono), embed in master for all ROIs |
| `plot_d0_delta` | `11_plot_D0_vs_Delta_alpha.sh` | Plot D₀/D\_proj vs Δ\_app |
| `plot_monoexp_d` | `12_plot_monoexp_D_vs_time.sh` | Plot monoexp D vs td |
| `export_master_xlsx` | `99_export_master_xlsx.sh` | Export master parquet to Excel for inspection |

### Step dependencies

```
[optional] filter_master_points  ← run before rotate/contrast/fit if using first-points filter

ingest
  └── rotate
        ├── plot_signal
        ├── grad_correction ─────────────── (reads grad_correction.csv manifest directly,
        │     └── fit_signal_gradcorr         no contrast or prior signal-fit required)
        ├── fit_signal ──────────────────── alpha
        ├── plot_d0_delta
        ├── plot_monoexp_d
        └── contrast
              ├── plot_contrast
              └── fit_contrast ──────────── tc  (set TC_FIT_PARAMS to the fit_params parquet
                                                  produced by fit_contrast)
```

`tc` requires `TC_FIT_PARAMS` to be set to the accumulated fit-params parquet produced by
`fit_contrast` (e.g. `fits/master/ogse_value_norm_vs_glinmax_ogse_free/fit_params.ogse_free.glinmax.value_norm.direction_ALL.parquet`).
It reads contrast fit results only — `fit_signal` does not contribute to this file.

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
| `TC_FIT_PARAMS` | — | Contrast fit-params parquet for the `tc` step (required, set explicitly) |
| `MANIFEST_DIR` | `manifests/<type_subj>s_<type_seq>/` | CSV manifests directory |

### Step-specific variables (most common)

| Variable | Step | Description |
|----------|------|-------------|
| `MASTER_FIRST_POINTS_BY_TD` | filter\_master\_points | Filter rules, e.g. `"120=8,210=6"` |
| `FILTERED_MASTER_PARQUET` | filter\_master\_points | Output path (alias: `MASTER_FIRST_POINTS_PARQUET`) |
| `SIGNAL_FIT_MODEL` | fit\_signal | Fallback model when manifest column is empty, e.g. `monoexp`, `nogse_free` |
| `SIGNAL_FIT_G_TYPE` | fit\_signal | Gradient column, e.g. `bvalue_thorsten`, `g` |
| `SIGNAL_FIT_OUT_ROOT` | fit\_signal | Output root (derived dynamically; set to override) |
| `SIGNAL_FIT_EXTRA_ARGS` | fit\_signal | Extra Python flags |
| `FIT_MODEL` | fit\_contrast | Model name, e.g. `ogse_free`, `nogse_free`, `mixed_global` |
| `FIT_GBASE` | fit\_contrast | Gradient axis, e.g. `g_lin_max`, `g_thorsten_1` |
| `FIT_OUT_ROOT` | fit\_contrast | Output root (derived dynamically; set to override) |
| `FIT_EXTRA_ARGS` | fit\_contrast | Extra Python flags |
| `GRAD_CORR_MANIFEST` | grad\_correction | Path to the grad-correction manifest CSV (default: `$MANIFEST_DIR/grad_correction.csv`) |
| `GRAD_CORR_OUT_DIR` | grad\_correction | Audit output directory for the xlsx/csv inspection table (D0\_nogse, D0\_monoexp, factor per curve). Primary output is MASTER\_PARQUET. Default: `$ANALYSIS_ROOT/fits/grad_correction` |
| `GRAD_CORR_EXTRA_ARGS` | grad\_correction | Advanced Python flags, e.g. `--gbase g_thorsten`, `--bbase bvalue_g_lin_max`, `--D0-init 2.3e-12`, `--free-M0` |
| `SIGNAL_FITS_ROOT` | plot\_monoexp\_d | Signal fit root (default: `$ANALYSIS_ROOT/fits`; set to specific subfolder) |
| `ALPHA_N` | alpha | N value used for D₀ extraction (default: 1) |
| `ALPHA_EXTRA_ARGS` | alpha | Use `--bvalmax N` and `--roi-bvalmax ROI=N` for per-ROI bvalue selection |
| `TC_FIT_PARAMS` | tc | **Required.** Contrast fit-params parquet (from fit\_contrast output) |
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
to process. They live in `manifests/<type_subj>s_<type_seq>/` and are read by
the corresponding pipeline step. Lines starting with `#` and the header line are
skipped. Edit them to match your dataset before running the pipeline.

### contrasts.csv

Specifies which pairs of signals to subtract to form an OGSE/NOGSE contrast.

**Format:**
```
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
```

**What each row does:**

Each row selects two groups of `signal_rotated` rows from the master table —
side 1 identified by `(N_1, Hz_1)` and side 2 by `(N_2, Hz_2)` — and computes
the point-wise difference:

```
contrast(g) = S(g; N_1, Hz_1) − S(g; N_2, Hz_2)   at the given td_ms
```

The result is appended to the master table as a new `row_kind=contrast` group.
One row in the manifest → one contrast entry per matching (roi, direction) pair.

**Column reference:**

| Column | Required | Description |
|--------|----------|-------------|
| `subj` | yes | Subject label (e.g. `BRAIN`, `LUDG`, `MBBL`, `PHANTOM`). Use `ALL` to match every subject in master. |
| `sheet` | yes | Session name exactly as stored in master (e.g. `20220622_BRAIN`, `20220610-PHANTOM3`). Use `ALL` to match every session. |
| `roi` | yes | Region of interest (e.g. `AntCC`, `Left-Lateral-Ventricle`, `fiber1`). Use `ALL` to process every ROI found for that (subj, sheet, td_ms, N, Hz). |
| `direction` | yes | Gradient direction label (e.g. `long`, `tra`, `1`, `2`, `3`). Use `ALL` to include all directions. |
| `td_ms` | yes | Diffusion time in ms. Selects only rows where `td_ms` matches this value exactly. Must be a number (not `ALL`). |
| `N_1` | yes | Number of OGSE oscillations for **side 1** (typically the higher-N signal, e.g. `8`). Matches the `N` column in master. |
| `N_2` | yes | Number of OGSE oscillations for **side 2** (typically the lower-N reference, e.g. `4`). |
| `Hz_1` | yes | Oscillation frequency in Hz for side 1 (e.g. `50`). Matches the `Hz` column in master. |
| `Hz_2` | yes | Oscillation frequency in Hz for side 2 (e.g. `25`). |

> **`ALL` vs specific value:** When a column is set to `ALL`, the corresponding
> `--flag` is *not* passed to the Python script, so the selection is unconstrained
> for that dimension. A specific value adds an equality filter. `td_ms`, `N_1`,
> `N_2`, `Hz_1`, `Hz_2` do not support `ALL` — leave the cell empty to skip that
> filter (the script will not pass the flag).

**Multiple td_ms per session:**

Each `td_ms` value requires its own row. A session with two diffusion times needs
two rows:

```
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
BRAIN,20230619_BRAIN-3,ALL,ALL,120,8,4,40,20
BRAIN,20230619_BRAIN-3,ALL,ALL,143.4,8,4,30,15
```

**Full example — all brain OGSE sessions:**

```
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
# BRAIN (20220622) — td=90ms
BRAIN,20220622_BRAIN,ALL,ALL,90,8,4,50,25
# BRAIN-3 (20230619) — two diffusion times
BRAIN,20230619_BRAIN-3,ALL,ALL,120,8,4,40,20
BRAIN,20230619_BRAIN-3,ALL,ALL,143.4,8,4,30,15
# BRAIN-4 (20230623)
BRAIN,20230623_BRAIN-4,ALL,ALL,76,8,4,65,35
BRAIN,20230623_BRAIN-4,ALL,ALL,210,8,4,20,10
# LUDG-2 (20230623)
LUDG,20230623_LUDG-2,ALL,ALL,120,8,4,40,20
LUDG,20230623_LUDG-2,ALL,ALL,143.4,8,4,30,15
```

**Full example — PHANTOM3 OGSE:**

```
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
PHANTOM,20220610-PHANTOM3,ALL,ALL,75.1,8,4,65,35
PHANTOM,20220610-PHANTOM3,ALL,ALL,97.1,8,4,50,25
PHANTOM,20220610-PHANTOM3,ALL,ALL,119.1,8,4,40,20
PHANTOM,20220610-PHANTOM3,ALL,ALL,142.5,8,4,30,15
PHANTOM,20220610-PHANTOM3,ALL,ALL,209.1,8,4,20,10
```

**Restricting to specific ROIs or directions:**

If you only want one direction:
```
BRAIN,20220622_BRAIN,ALL,long,90,8,4,50,25
```

If you only want one ROI:
```
BRAIN,20220622_BRAIN,AntCC,ALL,90,8,4,50,25
```

**Advanced: fitted-resampled contrasts**

By default the contrast is computed as a direct point-wise subtraction. To
instead fit each signal curve with a monoexponential model and subtract the
fitted curves on a common gradient grid, pass extra flags via the environment
variable `MAKE_CONTRAST_EXTRA_ARGS`:

```bash
MAKE_CONTRAST_EXTRA_ARGS="--contrast-source fitted_resampled --g_type g_lin_max" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast
```

Key options for `MAKE_CONTRAST_EXTRA_ARGS`:

| Flag | Default | Description |
|------|---------|-------------|
| `--contrast-source` | `direct` | `direct` (point subtraction) or `fitted_resampled` (fit then subtract) |
| `--signal-model` | `monoexp` | Signal model for `fitted_resampled` (e.g. `monoexp`) |
| `--g_type` | `g` | Gradient axis for fitting and the common resampling grid |
| `--fit_points N` | `6` | Number of leading gradient points used per fit |
| `--auto_fit_points` | off | Automatically choose the number of fit points |
| `--no-master-rotated` | off | Use `signal` rows instead of `signal_rotated` |

---

### signal\_fits.csv

Specifies which signal groups to fit individually (one fit per row).

**Format:**
```
subj,sheet,roi,direction,td_ms,N,Hz,model
```

**Column reference:**

| Column | Required | Description |
|--------|----------|-------------|
| `subj` | yes | Subject label. `ALL` = all subjects. |
| `sheet` | yes | Session name. `ALL` = all sessions. |
| `roi` | yes | Region of interest. `ALL` = all ROIs. |
| `direction` | yes | Gradient direction. `ALL` = all directions. |
| `td_ms` | yes | Diffusion time in ms. |
| `N` | yes | Number of OGSE oscillations (e.g. `4`, `8`). |
| `Hz` | yes | Oscillation frequency in Hz. |
| `model` | yes | Fitting model name (e.g. `monoexp`, `nogse_free`). |

**Example:**
```
subj,sheet,roi,direction,td_ms,N,Hz,model
BRAIN,20220622_BRAIN,ALL,ALL,90,1,0,monoexp
BRAIN,20220622_BRAIN,ALL,ALL,90,4,25,monoexp
BRAIN,20220622_BRAIN,ALL,ALL,90,8,50,monoexp
```

Each row fits the signal curve `S(g)` at the specified `(td_ms, N, Hz)` with the
given model, for every matching (roi, direction) combination. `N=1, Hz=0` selects
the b=0 / no-oscillation reference sequence.

---

### grad\_correction.csv

Specifies which Syringe (brain) or water (phantom) signal curves to use for
computing the gradient correction factor. One row per individual curve to fit.

**Format:**
```
subj,sheet,roi,direction,td_ms,N,Hz,model
```

The `Hz` and `model` columns are read but currently ignored by the script —
the model is always NOGSE free + monoexp.

**What the step does:**

For each row the script:

1. Loads the matching `signal_rotated` rows from `master.long.parquet`.
2. Fits the signal curve `S(g)` with the **NOGSE free model** (`M_nogse_free`,
   using the `g_lin_max` gradient column) → `D0_nogse`.
3. Fits the same curve with a **monoexp model** (`exp(-b·D0)`, using the
   `bvalue_thorsten` b-value column) → `D0_monoexp`.
4. Computes `correction_factor = √(D0_nogse / D0_monoexp)`.
5. Writes the factor to `master.long.parquet` for **every ROI** in the master
   that shares the same `(subj, sheet, direction, td_ms, N)` parameters.

**Why this works without a prior contrast or signal-fit step:** the raw
`signal_rotated` rows are already present in the master after `rotate`. The
fitting is done on-the-fly by the grad\_correction script itself.

**Column reference:**

| Column | Description |
|--------|-------------|
| `subj` | Subject label (e.g. `BRAIN`, `LUDG`). |
| `sheet` | Session name exactly as in master (e.g. `20230619_BRAIN-3`). |
| `roi` | Reference ROI to fit (e.g. `Syringe`, `water`). |
| `direction` | Gradient direction (e.g. `long`, `tra`). |
| `td_ms` | Diffusion time in ms. |
| `N` | Number of OGSE oscillations for this acquisition. |
| `Hz` | Oscillation frequency (informational). |
| `model` | Informational only (always `monoexp`). |

**Example (brains\_ogse):**
```
subj,sheet,roi,direction,td_ms,N,Hz,model
BRAIN,20230619_BRAIN-3,Syringe,long,120,1,0,monoexp
BRAIN,20230619_BRAIN-3,Syringe,long,120,4,20,monoexp
BRAIN,20230619_BRAIN-3,Syringe,long,120,8,40,monoexp
BRAIN,20230619_BRAIN-3,Syringe,long,120,12,55,monoexp
BRAIN,20230619_BRAIN-3,Syringe,tra,120,1,0,monoexp
...
```

Each `(td_ms, N, direction)` combination should have its own row. One manifest
entry produces one fitted correction factor that propagates to all ROIs.

---

### Manifest locations

```
manifests/brains_ogse/contrasts.csv          # OGSE brain  — filled
manifests/brains_ogse/signal_fits.csv        # OGSE brain  — fill before fit_signal
manifests/brains_ogse/grad_correction.csv    # OGSE brain  — filled (Syringe curves)
manifests/phantoms_ogse/contrasts.csv        # OGSE phantom — filled
manifests/phantoms_ogse/signal_fits.csv      # OGSE phantom — fill before fit_signal
manifests/phantoms_ogse/grad_correction.csv  # OGSE phantom — fill with water curves
manifests/brains_nogse/contrasts.csv         # NOGSE brain  — empty template
manifests/brains_nogse/signal_fits.csv       # NOGSE brain  — empty template
manifests/phantoms_nogse/contrasts.csv       # NOGSE phantom — empty template
manifests/phantoms_nogse/signal_fits.csv     # NOGSE phantom — empty template
```

To use a different manifest file without editing it in place:

```bash
CONTRAST_MANIFEST=my_custom_contrasts.csv \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast
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
