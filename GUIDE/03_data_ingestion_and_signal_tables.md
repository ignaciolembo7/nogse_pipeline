# Data Ingestion And Signal Tables

The central data product before fitting is a long-form signal table. Most later
modules assume this table shape rather than reading raw ROI spreadsheets
directly.

## ROI Extraction

Brain extraction code is in:

- `src/signal_extraction/coreg_extract_brain.py`
- `src/signal_extraction/coreg_extract.py`
- `src/signal_extraction/extract_roi_tables.py`

Phantom extraction code is in:

- `src/signal_extraction/coreg_extract_phantom.py`
- `src/signal_extraction/extract_roi_tables.py`

The shared extraction layer computes ROI-level statistics and writes
MATLAB-style Excel outputs. The downstream processing stage converts those
tables into the canonical long form.

## Processing ROI Results

The main CLI is:

- `scripts/process_one_results.py`

The reusable modules are:

- `src/data_processing/io.py`
- `src/data_processing/match_params.py`
- `src/data_processing/params.py`
- `src/data_processing/reshape.py`
- `src/data_processing/schema.py`
- `src/data_processing/features.py`
- `src/data_processing/metadata.py`
- `src/data_processing/experiment_tables.py`

`process_one_results.py` performs these operations:

1. reads a ROI result workbook;
2. detects whether the table is organized by `b` or by `g`;
3. parses filename metadata;
4. selects the matching row from the sequence-parameter spreadsheet;
5. reshapes statistics into one row per ROI, direction, and measurement step;
6. attaches acquisition metadata such as `N`, `Hz`, `delta_ms`,
   `Delta_app_ms`, `td_ms`, `TE`, and `TR`;
7. derives gradient and `bvalue_*` columns when possible;
8. adds `S0` and `value_norm`;
9. validates and writes canonical `.long.parquet` and `.xlsx` outputs.

## Long-Form Signal Table

The long table is the contract between data preparation and fitting. It should
carry:

- subject and sheet labels,
- ROI name,
- direction,
- statistic,
- measurement step,
- signal value and normalized signal,
- acquisition timing,
- gradient or `b` axes,
- source-file provenance.

Strict column checking is provided by `src/tools/strict_columns.py`. Final table
normalization is handled by `src/data_processing/schema.py`.

## Gradient And `b` Axes

The shared conversion helper is:

- `src/fitting/b_from_g.py`

Fitters may use different internal axes depending on the model, but the signal
table keeps the axis columns explicit so later stages can choose the appropriate
fit and plot representation.

