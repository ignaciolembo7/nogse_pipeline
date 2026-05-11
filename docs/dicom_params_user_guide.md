# DICOM Parameter Extraction User Guide

This guide explains how to extract Siemens DICOM ASCCONV parameters for NOGSE phantom
studies and how to export the full parameter table for one selected `.IMA` file.

The recommended user-facing entry points are the bash launchers in:

```text
bash_template/dicom_params/
```

## Available Launchers

```text
0.0-run_extract_dicom_sequence_metadata.sh
0.1-run_export_one_dicom_parameters.sh
0.2-run_correlate_dicom_params_with_gradient.sh
```

The default study configured in these scripts is:

```text
EXPERIMENT=20260506_PHANTOM_FIBER
NAME=QUALITY_JACK_19800122TMSF
```

Override those variables on the command line if needed.

## Step 0.0: Extract DICOM Metadata

Run:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.0-run_extract_dicom_sequence_metadata.sh
```

This reads DICOM/IMA files from:

```text
Data-DICOM/<EXPERIMENT>/<NAME>
```

and optional NIfTI files from:

```text
Data-NIFTI/<NIFTI_EXPERIMENT>/<NAME>
```

It writes outputs under:

```text
analysis/dicom_metadata/<EXPERIMENT>/<NAME>
```

The standard outputs are:

```text
sequence_parameters_from_dicom.csv
sequence_parameters_from_dicom.parquet
sequence_parameters_by_nifti_from_dicom.csv
sequence_parameters_by_nifti_from_dicom.parquet
dicom_metadata_summary.csv
dicom_metadata_summary.parquet
dicom_asconv_key_values.long.csv
dicom_asconv_key_values.long.parquet
sequence_metadata_from_dicom.xlsx
```

The `.xlsx` workbook is compact and intended for quick inspection. The full ASCCONV
table can be much larger than Excel's row limit, so the complete long table is stored as
CSV and Parquet.

Important configurable variables:

```bash
EXPERIMENT=20260506_PHANTOM_FIBER
NIFTI_EXPERIMENT=20260506-PHANTOM_FIBER
NAME=QUALITY_JACK_19800122TMSF
DICOM_GLOB_CSV=*.IMA
NIFTI_GLOB_CSV=*_NOGSE*.nii.gz
WRITE_PARQUET=1
WRITE_STRINGS=0
SCANNER_GRAD_MAX_MTM=80
```

## Step 0.1: Export One Selected DICOM Image

Edit `DICOM_FILE` in:

```text
bash_template/dicom_params/0.1-run_export_one_dicom_parameters.sh
```

or pass it on the command line:

```bash
DICOM_FILE=QUALITY_JACK.MR.JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE.0060.0001.2026.05.06.17.21.33.307329.523834585.IMA \
bash nogse_pipeline/bash_template/dicom_params/0.1-run_export_one_dicom_parameters.sh
```

The query can be a full path, a basename, a filename stem, or a unique substring.

By default this writes:

```text
<DICOM_FILE>.dicom_parameters.long.parquet
<DICOM_FILE>.dicom_parameters.long.xlsx
```

The Excel workbook contains:

```text
parameters_long
numeric_parameters
matched_dicoms
```

`parameters_long` is the complete ASCCONV long table for the selected DICOM image. It has
columns:

```text
dicom_file, series, image, source, key, value, value_numeric
```

CSV export is disabled by default to avoid Excel's "features might be lost" warning when
users open and save CSV files. If a CSV is explicitly needed, pass:

```bash
OUT_CSV=/path/to/output.long.csv bash nogse_pipeline/bash_template/dicom_params/0.1-run_export_one_dicom_parameters.sh
```

## Step 0.2: Correlate ASCCONV Parameters With Gradient

Run:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.2-run_correlate_dicom_params_with_gradient.sh
```

This uses:

```text
dicom_asconv_key_values.long.parquet
sequence_parameters_by_nifti_from_dicom.parquet
```

and writes:

```text
dicom_asconv_gradient_correlations.csv
dicom_asconv_gradient_correlations.xlsx
```

The correlation output is sorted by `abs_correlation` by default. For the current NOGSE
phantom data, `sWipMemBlock.alFree[7]` is the strongest match to the non-zero intended
NOGSE `G` values.

## What "Scanned Chunk" Means

Older versions of the export scripts printed a line for every chunk of the large long
table they scanned. A chunk is simply a block of rows read from the Parquet/CSV table so
the code does not load a multi-million-row file all at once.

Chunked scanning is still used internally, but progress messages are now disabled by
default. To print progress from the Python CLI, use:

```bash
--progress-every 1
```

## Common Workflows

Extract metadata for the default 20260506 phantom study:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.0-run_extract_dicom_sequence_metadata.sh
```

Export one selected DICOM image:

```bash
DICOM_FILE=QUALITY_JACK.MR.JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE.0060.0001.2026.05.06.17.21.33.307329.523834585.IMA \
bash nogse_pipeline/bash_template/dicom_params/0.1-run_export_one_dicom_parameters.sh
```

Run the gradient correlation audit:

```bash
bash nogse_pipeline/bash_template/dicom_params/0.2-run_correlate_dicom_params_with_gradient.sh
```

Use another study:

```bash
EXPERIMENT=20260505_PHANTOM_FIBER \
NIFTI_EXPERIMENT=20260505-PHANTOM_FIBER \
NAME=QUALITY_JACK_19800122TMSF \
bash nogse_pipeline/bash_template/dicom_params/0.0-run_extract_dicom_sequence_metadata.sh
```

## Interpretation Limits

These tables expose scanner/protocol metadata recorded in DICOM ASCCONV. They are useful
for auditing what the scanner saved in the DICOM file. They are not, by themselves, an
independent physical measurement of the gradient waveform applied by the scanner
hardware.
