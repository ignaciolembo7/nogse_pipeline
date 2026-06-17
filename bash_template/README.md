# Unified bash templates

`bash_template` is the canonical runnable batch layer. It has two clean parts:

- numbered preparation drivers for DICOM/preprocessing/signal extraction;
- one shared post-Results runner, `run_dataset.sh`, for all master-table
  analysis steps.

Post-Results analysis no longer lives in dataset-specific numbered wrappers.
Choose the subject kind with `type_subj` and the sequence kind with `type_seq`.

**For a complete end-to-end walkthrough of all four pipeline cases
(ogse\_brain, ogse\_phantom, nogse\_brain, nogse\_phantom) see
[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md). For a detailed argument-by-argument
command reference, see [CLI_REFERENCE.md](CLI_REFERENCE.md).**

## Basic Usage

Run from the project root:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest rotate contrast
bash nogse_pipeline/bash_template/run_dataset.sh brain nogse plot_signal
```

The explicit option form is equivalent:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh \
  --type-subj brain \
  --type-seq ogse \
  ingest rotate contrast
```

Allowed values:

- `type_subj`: `brain` or `phantom` (`brains` and `phantoms` are accepted aliases).
- `type_seq`: `ogse` or `nogse`.

You can also use environment variables, especially when running a step file
directly:

```bash
TYPE_SUBJ=brain TYPE_SEQ=ogse \
  bash nogse_pipeline/bash_template/steps/01_ingest_results.sh
```

## Step List

```text
ingest
rotate
contrast
plot_signal
plot_contrast
fit_signal
fit_signal_gradcorr
fit_contrast
fit_contrast_free
fit_contrast_mixed_global
fit_global_signal
grad_correction
plot_d0_delta
plot_monoexp_d
alpha
tc
```

Preparation drivers that still operate before `Results/*_results.xlsx` exists:

```text
brains_ogse/0.0-run_dicom2nifti.sh
brains_ogse/1.0-run_BRAINS-denoised_topup_signal_extraction.sh
phantoms_ogse/0.0-run_dicom2nifti.sh
phantoms_ogse/0.1-run_make_gval_gvec.sh
phantoms_ogse/0.2-prep_phantom_b0.sh
phantoms_ogse/0.3-copy_selected_files.sh
phantoms_ogse/1.0-run_PHANTOM-denoised_signal_extraction.sh
phantoms_nogse/0.0-run_dicom2nifti.sh
phantoms_nogse/0.1-run_make_gval_gvec.sh
phantoms_nogse/0.2-prep_phantom_b0.sh
phantoms_nogse/0.3-copy_selected_files.sh
phantoms_nogse/1.0-run_PHANTOM-denoised_signal_extraction.sh
dicom_params/*.sh
```

Use `--help` for the full runner help:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh --help
```

Use step help like this:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal --help
```

## Default Paths

The runner derives defaults from `type_subj` and `type_seq`:

```text
analysis/<type_subj>s/<type_seq>_experiments/master.long.parquet
analysis/<type_subj>s/<type_seq>_experiments/fits/<master_name>/<type_seq>_<ycol>_vs_<gtype>_<model>/
nogse_pipeline/bash_template/manifests/<type_subj>s_<type_seq>/
```

Examples:

```text
analysis/brains/ogse_experiments/master.long.parquet
analysis/phantoms/ogse_experiments/master.long.parquet
analysis/brains/nogse_experiments/master.long.parquet
```

The default sequence-parameter workbook still depends on `type_subj`:

```text
Data-signals/sequence_parameters_brains.xlsx
Data-signals/sequence_parameters_phantoms.xlsx
```

## Inspecting The Master Table

`master.long.parquet` is the canonical master table. Pipeline steps append only
to parquet; they do not update `master.long.xlsx`.

Export an Excel copy only when you want to inspect it:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --out-xlsx analysis/brains/ogse_experiments/master.inspect.xlsx
```

Useful filtered exports:

```bash
python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --row-kind signal_rotated \
  --out-xlsx analysis/brains/ogse_experiments/master.rotated.inspect.xlsx

python nogse_pipeline/scripts/data/export_master_table.py \
  analysis/brains/ogse_experiments/master.long.parquet \
  --head 5000 \
  --out-xlsx /tmp/master.head.xlsx
```

## Shared Helpers

Only shared shell code that is still called by current entry points remains in
`helpers/`:

- `master_table_common.sh`: runner setup, `type_subj`/`type_seq` defaults,
  step dispatch, and step help.
- `dicom2nifti_batch_lib.sh`: shared DICOM-to-NIfTI batch utilities used by
  `0.0-run_dicom2nifti.sh`.
- `coreg_batch_lib.sh`: shared signal-extraction/coregistration batch
  utilities used by `1.0-run_*signal_extraction.sh`.
- `run_extract_dicom_sequence_metadata.sh`: shared implementation for
  `dicom_params/0.0-run_extract_dicom_sequence_metadata.sh`.

## Common Overrides

- `PY`: Python interpreter.
- `SIGNALS_ROOT`: root containing `Results/` and sequence parameter Excel files.
- `RESULTS_ROOT`: one Results root.
- `RESULTS_ROOTS`: space-separated list of Results roots.
- `PARAMS_XLSX`: sequence-parameter workbook.
- `ANALYSIS_ROOT`: output root.
- `MASTER_PARQUET`: master table path.
- `TC_FIT_PARAMS`: contrast fit-params parquet for the `tc` step (required, set explicitly).
- `MANIFEST_DIR`: directory containing `contrasts.csv` and `signal_fits.csv`.

The runner also accepts repeated `--results-root` options:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
  --results-root Data-signals/Results/20220622_BRAIN \
  --results-root Data-signals/Results/20220701_BRAIN \
  ingest
```

## Typical OGSE Runs

Ingest all default results for brains:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
```

Ingest one phantom folder:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse \
  --results-root Data-signals/Results/20260122-PHANTOM_FIBER \
  ingest
```

Run the core table-building sequence:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
  ingest rotate contrast
```

The same using `PIPELINE_STEPS`:

```bash
PIPELINE_STEPS="ingest rotate contrast fit_signal fit_contrast alpha tc" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse
```

## Replacements For Numbered Analysis Wrappers

Monoexponential brain OGSE signal fit (set `model=monoexp` in `signal_fits.csv` manifest):

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
```

Gradient-corrected phantom OGSE signal fit:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_signal_gradcorr
```

Contrast fit wrappers:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_mixed_global
```

## Manifests

Routine repeated analyses should be edited in CSV manifests, not by adding file
lists to bash scripts.

OGSE manifests already exist:

```text
bash_template/manifests/brains_ogse/contrasts.csv
bash_template/manifests/brains_ogse/signal_fits.csv
bash_template/manifests/phantoms_ogse/contrasts.csv
bash_template/manifests/phantoms_ogse/signal_fits.csv
```

NOGSE manifests are present as empty templates to be filled when that workflow
is moved to the master-table layout:

```text
bash_template/manifests/brains_nogse/contrasts.csv
bash_template/manifests/brains_nogse/signal_fits.csv
bash_template/manifests/phantoms_nogse/contrasts.csv
bash_template/manifests/phantoms_nogse/signal_fits.csv
```

## Useful Examples

Plot only one ROI/direction selection:

```bash
PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=long \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
```

Build and apply embedded gradient correction (reads `manifests/brains_ogse/grad_correction.csv`
directly from the master — no prior fit\_signal or contrast step required):

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr
```

Run a global signal model from master:

```bash
GLOBAL_SIGNAL_MODEL=ogse_mixed_offset \
GLOBAL_SIGNAL_APPLY_GRAD_CORR=true \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal
```

Run a NOGSE signal fit once `analysis/brains/nogse_experiments/master.long.parquet`
and `manifests/brains_nogse/signal_fits.csv` have real selectors:

```bash
SIGNAL_FIT_MODEL=nogse_free SIGNAL_FIT_XCOL=g \
  bash nogse_pipeline/bash_template/run_dataset.sh brain nogse fit_signal
```
