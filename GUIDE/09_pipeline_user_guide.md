# Complete Pipeline User Guide

This guide documents the unified runnable pipeline under `bash_template/`.
It combines the older acquisition/preprocessing scripts with the master-table
post-Results workflow.

## Canonical Layout

```text
bash_template/
  run_dataset.sh              # master-table post-Results runner
  steps/                      # shared post-Results steps
  manifests/                  # selector CSVs for repeated analyses
  brains_ogse/                # brain OGSE DICOM/preprocessing/extraction drivers only
  phantoms_ogse/              # phantom OGSE DICOM/preprocessing/extraction drivers only
  phantoms_nogse/             # phantom NOGSE DICOM/preprocessing/extraction drivers only
  dicom_params/               # DICOM/Phoenix metadata utilities
  helpers/                    # shared bash helpers
```

Use `bash_template/` as the source of truth. `bash/` is operational material.
There are no dataset-specific post-Results analysis wrappers: after signal
extraction writes `Results/*_results.xlsx`, use `run_dataset.sh`.

## End-To-End Order

1. Convert DICOM to NIfTI.
2. Prepare phantom sidecars/references when needed.
3. Extract ROI signals into `Data-signals/Results`.
4. Ingest `Results/*_results.xlsx` into a master table.
5. Rotate signals when tensor directions are available.
6. Build contrasts from manifest selectors.
7. Fit signals and contrasts.
8. Embed gradient-correction factors in the master table and run corrected fits when needed.
9. Generate summaries and plots.

## Common Environment

These variables are accepted by most bash drivers:

- `PY`: Python interpreter.
- `PROJECT_ROOT`: project root. Defaults to the parent of `nogse_pipeline`.
- `REPO_ROOT`: repository root. Usually `PROJECT_ROOT/nogse_pipeline`.
- `LOG_ROOT`: log directory for batch drivers.
- `PYTHONPATH`: augmented by scripts to include `nogse_pipeline/src`.
- `MPLCONFIGDIR`: defaults to `/tmp/matplotlib` for plotting.

Current shared helpers:

- `helpers/master_table_common.sh`: `run_dataset.sh` setup, argument-derived
  defaults, step dispatch, and embedded step help.
- `helpers/dicom2nifti_batch_lib.sh`: DICOM-to-NIfTI batch functions used by
  `0.0-run_dicom2nifti.sh`.
- `helpers/coreg_batch_lib.sh`: signal-extraction/coregistration batch
  functions used by `1.0-run_*signal_extraction.sh`.
- `helpers/run_extract_dicom_sequence_metadata.sh`: shared implementation for
  the DICOM metadata extraction driver.

## DICOM And Preprocessing Drivers

### Brain OGSE DICOM Conversion

Command:

```bash
bash nogse_pipeline/bash_template/brains_ogse/0.0-run_dicom2nifti.sh
```

Inputs:

- `Data-DICOM/<case>` folders listed in the script.

Outputs:

- `Data-NIFTI/<case>` NIfTI outputs.
- Logs under `nogse_pipeline/logs/brains` unless `LOG_ROOT` is set.

Controls:

- `PROJECT_ROOT`
- `INPUT_ROOT`
- `OUTPUT_ROOT`
- `LOG_ROOT`

The case list is intentionally explicit inside the script through `run_case`.
Comment/uncomment cases to change the batch.

### Phantom OGSE/NOGSE DICOM Conversion

Commands:

```bash
bash nogse_pipeline/bash_template/phantoms_ogse/0.0-run_dicom2nifti.sh
bash nogse_pipeline/bash_template/phantoms_nogse/0.0-run_dicom2nifti.sh
```

Inputs and outputs are the same pattern as brain conversion:

- Input: `Data-DICOM/<case>`.
- Output: `Data-NIFTI/<case>`.
- Logs: `LOG_ROOT` or the script default.

Controls:

- `PROJECT_ROOT`
- `INPUT_ROOT`
- `OUTPUT_ROOT`
- `LOG_ROOT`

### Phantom Gradient Sidecars

Commands:

```bash
bash nogse_pipeline/bash_template/phantoms_ogse/0.1-run_make_gval_gvec.sh
bash nogse_pipeline/bash_template/phantoms_nogse/0.1-run_make_gval_gvec.sh
```

Inputs:

- NIfTI files under `Data-NIFTI/<SUBJ>`.
- Gradients inferred from filenames.

Outputs:

- `.gval` and `.gvec` sidecars next to each matched NIfTI file.

Arguments:

- `--subject SUBJ`: subject/experiment folder under the parent.
- `--exp-parent DIR`: parent directory, default `Data-NIFTI`.
- `--glob PATTERN`: comma-separated NIfTI patterns.
- `--dir GX GY GZ`: common direction written to each `.gvec`.
- `--overwrite`: overwrite existing sidecars.
- `--dry-run`: print planned outputs without writing.
- `-h`, `--help`: print usage.

Example:

```bash
bash nogse_pipeline/bash_template/phantoms_nogse/0.1-run_make_gval_gvec.sh \
  --subject 20260519_PHANTOM/QUALITY_JACK_19800122TMSF \
  --glob '*_001_NOGSE*.nii.gz, *_001_TNOGSE*.nii.gz' \
  --dir 1 0 0
```

### Phantom Reference Preparation

Commands:

```bash
bash nogse_pipeline/bash_template/phantoms_ogse/0.2-prep_phantom_b0.sh
bash nogse_pipeline/bash_template/phantoms_nogse/0.2-prep_phantom_b0.sh
```

Inputs:

- `Data-NIFTI/<SUBJ>` NIfTI files.

Outputs:

- Reference images and prepared folders under `Data-signals/<SUBJ>`.

Arguments:

- `--subject SUBJ`
- `--exp-parent DIR`
- `--out-root DIR`
- `--cut-token TOKEN`
- `--dwi-variant VAR`
- `-h`, `--help`

Internal controls configured in the script:

- `REF_MODE`: `mean` or `b0`.
- `DUMMY_SCANS`: number of initial volumes discarded for mean references.
- `REUSE_REFERENCE`: `1` to reuse existing references, `0` to overwrite.

### Copy Selected Phantom Files

Commands:

```bash
bash nogse_pipeline/bash_template/phantoms_ogse/0.3-copy_selected_files.sh
bash nogse_pipeline/bash_template/phantoms_nogse/0.3-copy_selected_files.sh
```

Inputs:

- `BASE`, `SRC`, and `FILES` configured inside the script.

Outputs:

- Selected masks/files copied from `SRC` into sibling sequence folders.

This step is intentionally manual because it copies curated masks.

### Signal Extraction

Commands:

```bash
bash nogse_pipeline/bash_template/brains_ogse/1.0-run_BRAINS-denoised_topup_signal_extraction.sh
bash nogse_pipeline/bash_template/phantoms_ogse/1.0-run_PHANTOM-denoised_signal_extraction.sh
bash nogse_pipeline/bash_template/phantoms_nogse/1.0-run_PHANTOM-denoised_signal_extraction.sh
```

Inputs:

- Prepared NIfTI experiment folders.
- Brain FreeSurfer/subject folders when required.
- ROI masks or atlas labels.

Outputs:

- Signal extraction folders and `Results/*_results.xlsx` under `Data-signals`.
- Logs under `LOG_ROOT`.

Common controls:

- `OUT_ROOT`
- `LOG_ROOT`
- `DWI_VARIANT`
- `USE_MEAN`
- `DUMMY_SCANS`
- `REUSE_REFERENCE`
- `REQUIRE_SUBJECTS_DIR`

The case/ROI list is explicit in each script via `run_case`.

## DICOM Metadata Utilities

### Extract Sequence Metadata

Command:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.0-run_extract_dicom_sequence_metadata.sh
```

Inputs:

- `DICOM_ROOT`
- optional `NIFTI_ROOT`

Outputs:

- `analysis/dicom_metadata/<experiment>/<name>/sequence_metadata_from_dicom.xlsx`
- long key/value tables, CSV and optionally Parquet.

Controls:

- `EXPERIMENT`
- `NIFTI_EXPERIMENT`
- `NAME`
- `DICOM_ROOT`
- `NIFTI_ROOT`
- `OUT_ROOT`
- `DICOM_GLOB_CSV`
- `NIFTI_GLOB_CSV`
- `RECURSIVE`
- `NIFTI_RECURSIVE`
- `WRITE_STRINGS`
- `SCANNER_GRAD_MAX_MTM`
- `WRITE_PARQUET`
- `OUT_XLSX`

### Export One DICOM Parameter Table

Command:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.1-run_export_one_dicom_parameters.sh
```

Inputs:

- `METADATA_ROOT/dicom_asconv_key_values.long.parquet` or `.csv`.
- `DICOM_FILE`, which can be a full path, basename, stem, or unique substring.

Outputs:

- Per-file parameter table as Parquet, CSV, or XLSX.

Controls:

- `EXPERIMENT`
- `NAME`
- `METADATA_ROOT`
- `DICOM_FILE`
- `KEY_VALUES_PARQUET`
- `KEY_VALUES_CSV`
- `OUT_CSV`
- `OUT_PARQUET`
- `OUT_XLSX`

### Correlate DICOM Parameters With Gradient

Command:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.2-run_correlate_dicom_params_with_gradient.sh
```

Inputs:

- `KEY_VALUES`
- `NIFTI_TABLE`

Outputs:

- Correlation table, CSV/XLSX.

Controls:

- `EXPERIMENT`
- `NAME`
- `METADATA_ROOT`
- `KEY_VALUES`
- `NIFTI_TABLE`
- `OUT_CSV`
- `OUT_XLSX`
- `MIN_OBSERVATIONS`
- `SORT_BY`

## Master-Table Runner

The post-Results runner is:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh <type_subj> <type_seq> <step...>
```

Allowed values:

- `type_subj`: `brain` or `phantom`. Aliases: `brains`, `phantoms`.
- `type_seq`: `ogse` or `nogse`.

Runner arguments:

- `--type-subj brain|phantom`
- `--type_subj brain|phantom`
- `--dataset brain|phantom`
- `--type-seq ogse|nogse`
- `--type_seq ogse|nogse`
- `--results-root PATH`, repeatable
- `-h`, `--help`

Argument meaning:

- `type_subj` selects defaults that depend on the subject class. `brain` uses
  `Data-signals/sequence_parameters_brains.xlsx` and writes under
  `analysis/brains/<type_seq>_experiments`. `phantom` uses
  `Data-signals/sequence_parameters_phantoms.xlsx` and writes under
  `analysis/phantoms/<type_seq>_experiments`.
- `type_seq` selects the sequence family, script defaults, manifests, and
  output root suffix. `ogse` writes under `ogse_experiments`; `nogse` writes
  under `nogse_experiments`.
- `step` is one or more names from the master step list below. Steps run in the
  order supplied on the command line.
- `--results-root PATH` adds an input folder for `ingest`. It can be repeated.
  When omitted, `ingest` searches `RESULTS_ROOT`, defaulting to
  `Data-signals/Results`.
- `--` stops runner option parsing. Use it if a future step name or extra token
  begins with `-`.
- `--help` before any step prints runner help. `--help` after one step prints
  that step's help, for example `brain ogse fit_signal --help`.

Examples:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest rotate contrast
bash nogse_pipeline/bash_template/run_dataset.sh --type-subj brain --type-seq nogse fit_signal
```

You can also run steps directly:

```bash
TYPE_SUBJ=brain TYPE_SEQ=ogse \
  bash nogse_pipeline/bash_template/steps/01_ingest_results.sh
```

Common master-table controls:

- `TYPE_SUBJ`: `brain` or `phantom`.
- `TYPE_SEQ`: `ogse` or `nogse`.
- `PIPELINE_STEPS`: space- or comma-separated step list.
- `PY`
- `SIGNALS_ROOT`
- `RESULTS_ROOT`
- `RESULTS_ROOTS`
- `PARAMS_XLSX`
- `ANALYSIS_ROOT`
- `MASTER_PARQUET`
- `MASTER_FIT_PARAMS`
- `MANIFEST_DIR`

Master table storage:

- `master.long.parquet` is the canonical master table.
- Pipeline steps append only to parquet; they do not maintain
  `master.long.xlsx`.
- Excel files are inspection exports. Regenerate them explicitly from parquet:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --out-xlsx analysis/brains/ogse_experiments/master.inspect.xlsx
```

Export rotated rows after rotation, including the `D_proj` column:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --row-kind signal_rotated \
  --out-xlsx analysis/brains/ogse_experiments/master.rotated.inspect.xlsx
```

How step-specific arguments are passed:

- Fixed arguments such as `type_subj`, `type_seq`, `--results-root`, and step
  names are parsed by `run_dataset.sh`.
- Repeated selectors and model choices are passed as environment variables
  listed in each step below.
- Free-form Python options can be appended through each `*_EXTRA_ARGS`
  variable. Example:

```bash
SIGNAL_FIT_EXTRA_ARGS="--fix_M0 1.0 --auto_fit_tol 0.05" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
```

- Repeated analyses should normally go into CSV manifests instead of more bash
  scripts. `contrasts.csv` controls the `contrast` step and `signal_fits.csv`
  controls `fit_signal`.

Default outputs:

```text
analysis/brains/ogse_experiments/master.long.parquet
analysis/phantoms/ogse_experiments/master.long.parquet
analysis/brains/nogse_experiments/master.long.parquet
analysis/phantoms/nogse_experiments/master.long.parquet
```

## Master Step Reference

### `migrate`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse migrate
```

Inputs:

- Existing legacy analysis root, usually `analysis/<type_subj>s/<type_seq>_experiments`.

Outputs:

- `MASTER_PARQUET`
- `master_migration_report/*`

Controls:

- `ANALYSIS_ROOT`
- `MASTER_PARQUET`
- `MIGRATE_SCRIPT`
- `MIGRATION_REPORT_DIR`
- `MIGRATE_EXTRA_ARGS`

### `ingest`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
```

Inputs:

- `Results/*_results.xlsx`
- `PARAMS_XLSX`

Outputs:

- `ANALYSIS_ROOT/data/tables/*`
- appended `row_kind='signal'` rows in `MASTER_PARQUET`

Controls:

- `RESULTS_ROOT`
- `RESULTS_ROOTS`
- `PARAMS_XLSX`
- `RESULTS_GLOB`
- `PROCESS_SCRIPT`
- `PROCESS_OUT_ROOT`
- `MASTER_PARQUET`

Example for one folder:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
  --results-root Data-signals/Results/20220622_BRAIN \
  ingest
```

### `rotate`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate
```

Inputs:

- `MASTER_PARQUET` with `row_kind='signal'`.
- direction table, default `assets/dirs/dirs_6.txt`.

Outputs:

- `ANALYSIS_ROOT/data-rotated/tables/*`
- appended `row_kind='signal_rotated'` and projected diffusion rows.

Controls:

- `MASTER_PARQUET`
- `MASTER_SUBJ`
- `MASTER_SHEET`
- `DIRS_TXT`
- `ROTATE_SCRIPT`
- `ROTATED_OUT_ROOT`
- `ROTATE_EXTRA_ARGS`

### `contrast`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast
```

Inputs:

- `MASTER_PARQUET` with rotated signal rows.
- `CONTRAST_MANIFEST`, default `bash_template/manifests/<type_subj>s_<type_seq>/contrasts.csv`.

Outputs:

- `ANALYSIS_ROOT/contrast-data-master/*`
- appended `row_kind='contrast'` rows.

Manifest columns:

```csv
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
```

Controls:

- `MAKE_CONTRAST_SCRIPT`
- `CONTRAST_MANIFEST`
- `CONTRAST_OUT_ROOT`
- `MAKE_CONTRAST_EXTRA_ARGS`

### `plot_signal`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
```

Inputs:

- `MASTER_PARQUET`.

Outputs:

- `ANALYSIS_ROOT/plots-master/signal/*`

Controls:

- `PLOT_SIGNAL_SCRIPT`
- `PLOT_OUT_ROOT`
- `PLOT_ROW_KIND`
- `PLOT_SUBJ`
- `PLOT_SHEET`
- `PLOT_ROI`
- `PLOT_DIRECTION`
- `PLOT_TD_MS`
- `PLOT_N`
- `PLOT_SIGNAL_YCOL`
- `PLOT_SIGNAL_XCOL`
- `PLOT_SIGNAL_G_TYPE`
- `PLOT_STAT`
- `PLOT_SIGNAL_EXTRA_ARGS`

### `plot_contrast`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast
```

Inputs:

- `MASTER_PARQUET` with contrast rows.

Outputs:

- `ANALYSIS_ROOT/plots-master/contrast/*`

Controls:

- `PLOT_CONTRAST_SCRIPT`
- `PLOT_OUT_ROOT`
- `PLOT_SUBJ`
- `PLOT_SHEET`
- `PLOT_ROI`
- `PLOT_DIRECTION`
- `PLOT_TD_MS`
- `PLOT_N1`
- `PLOT_N2`
- `PLOT_CONTRAST_YCOL`
- `PLOT_CONTRAST_XCOL`
- `PLOT_STAT`
- `PLOT_CONTRAST_EXTRA_ARGS`

### `fit_signal`, `fit_signal_monoexp`, `fit_signal_gradcorr`

Commands:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_monoexp
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr
```

Inputs:

- `MASTER_PARQUET` with `signal_rotated` rows.
- `SIGNAL_FIT_MANIFEST`, default `bash_template/manifests/<type_subj>s_<type_seq>/signal_fits.csv`.

Outputs:

- `ANALYSIS_ROOT/fits/<type_seq>_signal_master/*`
- appended signal-fit rows in `MASTER_FIT_PARAMS`

Manifest columns:

```csv
subj,sheet,roi,direction,td_ms,N,Hz,model
```

Controls:

- `FIT_SIGNAL_SCRIPT`
- `SIGNAL_FIT_MANIFEST`
- `SIGNAL_FIT_OUT_ROOT`
- `SIGNAL_FIT_MODEL`
- `SIGNAL_FIT_G_TYPE`
- `SIGNAL_FIT_XCOL`
- `SIGNAL_FIT_YCOL`
- `SIGNAL_FIT_EXTRA_ARGS`
- `MASTER_FIT_PARAMS`

`fit_signal_monoexp` sets OGSE monoexponential defaults. `fit_signal_gradcorr`
adds `--apply_grad_corr` and reads embedded `grad_correction_factor` values from `MASTER_PARQUET`.

### `fit_contrast`, `fit_contrast_free`, `fit_contrast_mixed_global`

Commands:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_mixed_global
```

Inputs:

- `MASTER_PARQUET` with contrast rows.

Outputs:

- `ANALYSIS_ROOT/fits/<type_seq>_contrast_master/*`
- appended contrast-fit rows in `MASTER_FIT_PARAMS`

Controls:

- `FIT_CONTRAST_SCRIPT`
- `FIT_OUT_ROOT`
- `FIT_MODEL`
- `FIT_GBASE`
- `FIT_YCOL`
- `FIT_STAT`
- `FIT_EXTRA_ARGS`

`fit_contrast_free` uses the free model for the selected `TYPE_SEQ` when
`FIT_MODEL` is unset. `fit_contrast_mixed_global` uses `mixed_global` when
`FIT_MODEL` is unset. Both call the same script as `fit_contrast`; they are
convenience presets, not separate fitting implementations.

### `fit_global_signal`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal
```

Inputs:

- `MASTER_PARQUET` with signal rows.

Outputs:

- `ANALYSIS_ROOT/fits/<type_seq>_signal_mixed_global_master/*`

Controls:

- `FIT_GLOBAL_SIGNAL_SCRIPT`
- `GLOBAL_SIGNAL_OUT_ROOT`
- `GLOBAL_SIGNAL_ROW_KIND`
- `GLOBAL_SIGNAL_MODEL`
- `GLOBAL_SIGNAL_YCOL`
- `GLOBAL_SIGNAL_G_TYPE`
- `GLOBAL_SIGNAL_STAT`
- `GLOBAL_SIGNAL_MIN_POINTS`
- `GLOBAL_SIGNAL_TC_MODE`
- `GLOBAL_SIGNAL_ALPHA_MODE`
- `GLOBAL_SIGNAL_RN_MODE`
- `GLOBAL_SIGNAL_M0_MODE`
- `GLOBAL_SIGNAL_C_MODE`
- `GLOBAL_SIGNAL_D0_MODE`
- `GLOBAL_SIGNAL_D0_FIXED`
- `GLOBAL_SIGNAL_DIRECTIONS`
- `GLOBAL_SIGNAL_ROIS`
- `GLOBAL_SIGNAL_SUBJS`
- `GLOBAL_SIGNAL_APPLY_GRAD_CORR`
- `GLOBAL_SIGNAL_EXTRA_ARGS`

Parameter modes are `fixed`, `free`, `global_td`, and `global_contrast`.

### `grad_correction`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction
```

Inputs:

- `MASTER_PARQUET` with `signal_rotated` rows
- `GRAD_CORR_MANIFEST` (`$MANIFEST_DIR/grad_correction.csv` by default)

Outputs:

- embedded `grad_correction_factor*` columns in `MASTER_PARQUET`
- audit copies under `ANALYSIS_ROOT/fits/grad_correction/`

Controls:

- `GRAD_CORR_SCRIPT`
- `GRAD_CORR_MANIFEST`
- `GRAD_CORR_OUT_DIR`
- `GRAD_CORR_EXTRA_ARGS`

### `plot_d0_delta`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_d0_delta
```

Inputs:

- `MASTER_PARQUET`
- optional `SUMMARY_ALPHA`

Outputs:

- plots/tables under `ALPHA_OUT_DIR`

Controls:

- `PLOT_D0_SCRIPT`
- `ALPHA_OUT_DIR`
- `SUMMARY_ALPHA`
- `DPROJ_N`
- `DPROJ_HZ`
- `DPROJ_ROIS`
- `DPROJ_DIRS`
- `PLOT_D0_EXTRA_ARGS`

### `plot_monoexp_d`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d
```

Inputs:

- signal fit root.

Outputs:

- `ANALYSIS_ROOT/plots-master/monoexp_D_vs_time/*`

Controls:

- `PLOT_MONOEXP_D_SCRIPT`
- `SIGNAL_FITS_ROOT`
- `MONOEXP_D_OUT_DIR`
- `PLOT_MONOEXP_D_EXTRA_ARGS`

### `alpha`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha
```

Inputs:

- `MASTER_PARQUET`
- `MASTER_FIT_PARAMS`

Outputs:

- `ALPHA_OUT_DIR/summary_alpha_values.xlsx`
- `ALPHA_OUT_DIR/D_vs_delta_app.combined.xlsx`

Controls:

- `ALPHA_MACRO_SCRIPT`
- `ALPHA_OUT_DIR`
- `ALPHA_N`
- `ALPHA_EXTRA_ARGS`

### `tc`

Command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc
```

Inputs:

- `MASTER_FIT_PARAMS`

Outputs:

- `TC_OUT_DIR/<method>/*`

Controls:

- `TC_VS_TD_SCRIPT`
- `TC_OUT_DIR`
- `TC_METHOD`
- `TC_Y_COL`
- `TC_EXTRA_ARGS`

## Example Pipelines

Brain OGSE, post-Results:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
  ingest rotate contrast fit_signal fit_contrast alpha tc
```

Phantom OGSE with gradient correction:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse \
  ingest rotate contrast fit_signal grad_correction fit_signal_gradcorr fit_contrast
```

Brain NOGSE, once master-table inputs and manifests exist:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse \
  ingest plot_signal fit_signal fit_global_signal
```

Use `PIPELINE_STEPS` for long jobs:

```bash
PIPELINE_STEPS="ingest rotate contrast fit_signal fit_contrast alpha tc" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
  > nogse_pipeline/logs/brains_ogse/master_pipeline.log 2>&1 &
```

## Troubleshooting

- Missing `master.long.parquet`: run `ingest` or `migrate`.
- Missing rotated rows: run `rotate` after `ingest`.
- Missing contrast rows: fill `contrasts.csv`, then run `contrast`.
- Empty fits: check selector columns in the manifest against `master.long.parquet`.
- Python import errors: ensure `PY` points to the project environment and that
  `PYTHONPATH` includes `nogse_pipeline/src`.
- Plotting cache errors on shared filesystems: set `MPLCONFIGDIR=/tmp/matplotlib`.
