# Master-Driven OGSE Pipeline

This is the current preferred OGSE workflow for brains and phantoms. From
`Results/` onward, analysis should use explicit table columns in
`master.long.parquet`, not information parsed from filenames.

## Core Tables

- `analysis/brains/ogse_experiments/master.long.parquet`
- `analysis/phantoms/ogse_experiments/master.long.parquet`
- `analysis/brains/ogse_experiments/master_fit_params.parquet`
- `analysis/phantoms/ogse_experiments/master_fit_params.parquet`

`master.long.parquet` stores signal-like data:

- `row_kind='signal'`: processed signal rows from `Results/`
- `row_kind='signal_rotated'`: tensor-rotated signal rows, including `D_proj`
- `row_kind='contrast'`: contrast rows built by selecting two signal groups

`row_kind='dproj'` is accepted for migrated legacy tables, but the current
rotation step stores projected diffusion values in the `D_proj` column of
`signal_rotated` rows instead of appending separate dproj rows.

`master_fit_params.parquet` stores cumulative fit-like parameters:

- signal fit params
- contrast fit params
- `alpha_macro` summaries
- later tc-vs-td derived parameters where applicable

## One Step At A Time

Use `bash_template` for new runs. It has one shared implementation of each
step. Select the subject kind and sequence kind with `type_subj` and `type_seq`.

Brains:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
```

Phantoms:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest
```

The explicit option form is equivalent:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh \
  --type-subj brain \
  --type-seq ogse \
  ingest
```

Available steps:

```text
migrate ingest rotate contrast plot_signal plot_contrast fit_signal fit_signal_monoexp fit_signal_gradcorr fit_contrast fit_contrast_free fit_contrast_mixed_global fit_global_signal grad_correction plot_d0_delta plot_monoexp_d alpha tc
```

You can combine explicit steps:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest rotate
```

Or use `PIPELINE_STEPS`:

```bash
PIPELINE_STEPS="ingest rotate contrast" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse
```

For long runs:

```bash
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate \
  > nogse_pipeline/logs/brains_ogse/rotate.log 2>&1 &
```

## Migration From Existing Analysis Outputs

Use the migrator to rebuild a master table from existing legacy analysis
folders:

```bash
python nogse_pipeline/scripts/data/migrate_analysis_to_master.py \
  analysis/brains/ogse_experiments \
  --drop-exact-duplicates
```

```bash
python nogse_pipeline/scripts/data/migrate_analysis_to_master.py \
  analysis/phantoms/ogse_experiments \
  --drop-exact-duplicates
```

Dry-run with report only:

```bash
python nogse_pipeline/scripts/data/migrate_analysis_to_master.py \
  analysis/brains/ogse_experiments \
  --dry-run
```

Outputs:

- `master.long.parquet`
- `master_migration_report/*_summary.csv`
- `master_migration_report/*_by_row_kind.csv`
- `master_migration_report/*_missing_required.csv`
- `master_migration_report/*_duplicate_keys.csv`
- `master_migration_report/*_migration_report.xlsx`

Use `--strict-duplicate-keys` only after inspecting the report. Use
`--hash-source-files` only for forensic audits because hashing every legacy
table is slower.

## Adding New Results To Master

To add one new experiment folder such as `Data-signals/Results/20220622_BRAIN`,
point `RESULTS_ROOT` at that folder and run ingestion:

```bash
RESULTS_ROOT=Data-signals/Results/20220622_BRAIN \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
```

That step uses `Data-signals/sequence_parameters_brains.xlsx` by default.

For phantoms:

```bash
RESULTS_ROOT=Data-signals/Results/20260122-PHANTOM_FIBER \
  bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest
```

That step uses `Data-signals/sequence_parameters_phantoms.xlsx` by default.

If the sequence-parameter workbook lives somewhere else, override it:

```bash
RESULTS_ROOT=Data-signals/Results/20220622_BRAIN \
PARAMS_XLSX=/path/to/sequence_parameters_brains.xlsx \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
```

For a single Results file, call the ingestion script directly:

```bash
python nogse_pipeline/scripts/data/process_one_results.py \
  Data-signals/Results/20220622_BRAIN/my_results_file_results.xlsx \
  Data-signals/sequence_parameters_brains.xlsx \
  --out_dir analysis/brains/ogse_experiments/data/tables \
  --master-parquet analysis/brains/ogse_experiments/master.long.parquet
```

After that, the new rows live in `master.long.parquet` as `row_kind='signal'`.
The next steps are rotation, contrast, fitting, and plotting.

## Contrast Manifest

Contrasts are no longer filename pairs. They are declarative selectors.

Brains:

```text
nogse_pipeline/bash_template/manifests/brains_ogse/contrasts.csv
```

Phantoms:

```text
nogse_pipeline/bash_template/manifests/phantoms_ogse/contrasts.csv
```

Format:

```csv
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
BRAIN,20220622_BRAIN,Left-Lateral-Ventricle,long,90,8,4,50,25
```

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast
```

## Signal Fit Manifest

Signal fits are also declarative selectors.

```csv
subj,sheet,roi,direction,td_ms,N,Hz,model
BRAIN,20220622_BRAIN,Left-Lateral-Ventricle,long,90,4,25,monoexp
```

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
```

## Fitting Modes

The unified fitting interface supports the common parameter modes:

- `free`
- `fixed`
- `global_td`
- `global_contrast`

Examples:

```bash
python nogse_pipeline/scripts/fitting/fit_ogse_contrast_vs_g.py \
  --master-parquet analysis/brains/ogse_experiments/master.long.parquet \
  --master-fit-params analysis/brains/ogse_experiments/master_fit_params.parquet \
  --model ogse_rest_offset \
  --param-mode C=global_contrast \
  --param-mode alpha=global_td \
  --out_root analysis/brains/ogse_experiments/fits/ogse_contrast_master
```

```bash
python nogse_pipeline/scripts/fitting/fit_ogse_signal_vs_g.py \
  --master-parquet analysis/brains/ogse_experiments/master.long.parquet \
  --row-kind signal_rotated \
  --subj BRAIN \
  --roi Left-Lateral-Ventricle \
  --direction long \
  --td_ms 90 \
  --N 4 \
  --Hz 25 \
  --model ogse_free \
  --param-fixed M0=1 \
  --out_root analysis/brains/ogse_experiments/fits/ogse_signal_master
```

The model registry is canonical in:

```text
src/fitting/model_registry.py
```

Parameter mode validation/configuration is canonical in:

```text
src/fitting/parameter_modes.py
```

Shared CLI glue lives in:

```text
src/fitting/cli_common.py
```

## Alpha Macro And TC

Build `alpha_macro` from master `dproj` rows:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha
```

Fit tc-vs-td from `master_fit_params`:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc
```

Direct command:

```bash
python nogse_pipeline/scripts/fitting/run_tc_vs_td.py \
  --master-fit-params analysis/brains/ogse_experiments/master_fit_params.parquet \
  --method pseudohuber_fixed_macro \
  --y-col tc_peak_ms
```

## Extension Points

Add a new model:

1. Add the math/evaluator in the appropriate fitting module.
2. Register the model/aliases in `src/fitting/model_registry.py`.
3. Reuse `src/fitting/parameter_modes.py` for `fixed/free/global_td/global_contrast`.
4. Expose the model through the existing fit script CLI.
5. Write fit params to `master_fit_params.parquet`.

Add a new analysis table:

1. Add a `row_kind` if it is a new data family.
2. Add validation/selectors in `src/data_processing/master_table.py`.
3. Keep plotting/fitting code accepting DataFrames, not roots/globs.
4. Add a manifest if the analysis requires a repeated set of selectors.

Add a new plotting script:

1. Select rows with `master_table.py`.
2. Pass the filtered DataFrame into plotting functions.
3. Keep legacy parquet inputs optional only as compatibility.
