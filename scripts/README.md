# Command-Line Scripts

The command-line scripts are grouped by the pipeline concept they operate on.
Most reusable logic lives in `src/`; scripts should stay thin and should mainly
parse CLI options, call library code, and write outputs.

## Layout

- `data/`: ingestion, master-table migration, rotation, contrasts, gradient correction tables.
- `fitting/`: signal/contrast fitting and tc-vs-td fitting.
- `plotting/`: analysis plots and diagnostic plots.
- `summary/`: summary-table builders such as alpha macro.
- `publication/`: publication figure entrypoints.
- `dicom/`: DICOM metadata utilities.
- `simulation/`: simulation utilities.

## Adding A New Results Experiment To Master

Use `scripts/data/process_one_results.py` with the experiment Results file and
the sequence-parameter workbook for that family.

Brains example:

```bash
python nogse_pipeline/scripts/data/process_one_results.py \
  Data-signals/Results/20220622_BRAIN/your_results_file_results.xlsx \
  Data-signals/sequence_parameters_brains.xlsx \
  --out_dir analysis/brains/ogse_experiments/data/tables \
  --master-parquet analysis/brains/ogse_experiments/master.long.parquet
```

Phantoms example:

```bash
python nogse_pipeline/scripts/data/process_one_results.py \
  Data-signals/Results/20260122-PHANTOM_FIBER/your_results_file_results.xlsx \
  Data-signals/sequence_parameters_phantoms.xlsx \
  --out_dir analysis/phantoms/ogse_experiments/data/tables \
  --master-parquet analysis/phantoms/ogse_experiments/master.long.parquet
```

To ingest every `*_results.xlsx` currently under `Data-signals/Results`, use the
template step:

```bash
bash nogse_pipeline/bash_template_2/brains_ogse/01-ingest_results_to_master.sh
bash nogse_pipeline/bash_template_2/phantoms_ogse/01-ingest_results_to_master.sh
```

To ingest one new Results folder only:

```bash
RESULTS_ROOT=Data-signals/Results/20220622_BRAIN \
  bash nogse_pipeline/bash_template_2/brains_ogse/01-ingest_results_to_master.sh
```

Override `PARAMS_XLSX` when the matching sequence-parameter workbook is not the
default family workbook.

The important rule is that all metadata needed later must be columns in
`master.long.parquet`. After ingestion, downstream steps should select by
columns such as `subj`, `sheet`, `roi`, `direction`, `td_ms`, `N`, `Hz`, `g`,
`b_step`, and `stat`.
