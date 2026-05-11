# DICOM ASCCONV Metadata Audit for NOGSE Phantom Acquisitions

This document describes how the repository extracts Siemens ASCCONV metadata from
DICOM/IMA files, why these tables contain more parameters than a normal `pydicom show`,
and how the dedicated DICOM audit scripts are used to inspect gradient-related protocol
metadata.

The current implementation is intentionally separate from the main image-processing and
fitting pipeline. It is an audit workflow for understanding how NOGSE protocol values are
encoded in DICOM files.

## Why `pydicom show` Is Not Enough

`pydicom show` reports the DICOM data elements that pydicom can parse as standard DICOM
tags and private tags. This is useful for public metadata and visible private fields, but
it does not fully expand the Siemens protocol text embedded inside the file.

Siemens DICOM/IMA files can contain a private protocol block known as ASCCONV. It is a
text serialization of many scanner protocol parameters. In these files it appears inside
the printable byte stream between markers similar to:

```text
### ASCCONV BEGIN
...
### ASCCONV END
```

Inside that block, parameters are stored as text key-value lines:

```text
key = value
```

Examples include:

```text
tProtocolName = "002_NOGSE_CPMG_N2_TN50_G08"
sWipMemBlock.alFree[7] = 8
sWipMemBlock.alFree[5] = 25000
sWipMemBlock.alFree[6] = 25000
```

These lines are not necessarily exposed by a simple `pydicom show` view as individual
fields. The audit scripts therefore read the DICOM file bytes directly, extract printable
strings, locate the ASCCONV block, parse the key-value lines, and write them into normal
tabular outputs.

## What ASCCONV Represents

ASCCONV is scanner/protocol metadata recorded in the DICOM file. It should be interpreted
as recorded protocol state, not as an independent measurement of the physical gradient
waveform applied by the scanner hardware.

For this repository there are three useful levels of information:

| Level | Meaning | Examples |
| --- | --- | --- |
| Physical scanner output | The actual gradient waveform/current produced by hardware. | Requires logs, raw/twix/MDH information, field monitoring, or scanner-side validation. |
| DICOM-recorded scanner/protocol metadata | Values saved in DICOM public/private metadata or ASCCONV. | `tProtocolName`, `sWipMemBlock.alFree[7]`, sequence identifiers, timing values. |
| Filename- or user-derived metadata | Values reconstructed after conversion from filenames or command-line defaults. | `.gval/.gvec` generated from `_G08_` filename tokens and `--direction`. |

The ASCCONV audit workflow improves access to the second level. It does not prove the
first level by itself.

## Existing Extraction Path

The core DICOM extraction script is:

```text
scripts/extract_dicom_sequence_metadata.py
```

It does the following:

1. Reads each DICOM/IMA file as bytes.
2. Extracts printable strings from the byte stream.
3. Finds the ASCCONV block using the `### ASCCONV BEGIN` and `### ASCCONV END` markers.
4. Parses lines matching `key = value`.
5. Stores the parsed ASCCONV dictionary for each DICOM image.
6. Extracts selected convenience fields into summary tables.
7. Writes compact sequence summaries and long audit tables.

The long ASCCONV output has this schema:

```text
dicom_file, series, image, source, key, value
```

where:

- `dicom_file` is the full DICOM/IMA path.
- `series` is parsed from the Siemens filename.
- `image` is parsed from the Siemens filename.
- `source` is currently `ASCCONV`.
- `key` is the ASCCONV parameter name.
- `value` is the raw parsed text value.

For the current study, the main long table is:

```text
analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/dicom_asconv_key_values.long.csv
```

This file is large because it contains every ASCCONV key for every DICOM image.

## Dedicated Audit Code Added for This Study

Reusable DICOM audit code lives under:

```text
src/dicom_params/
```

Thin command-line wrappers live under:

```text
scripts/extract_dicom_sequence_metadata.py
scripts/dicom_correlate_asconv_with_gradient.py
scripts/dicom_export_file_parameters.py
```

User-facing bash launchers live under:

```text
bash_template/dicom_params/
```

They are separate from the main pipeline because this is an exploratory metadata audit,
not a required processing step for every fit. The extraction step now writes CSV and
Parquet outputs in the same pass, so there is no separate CSV-to-Parquet launcher in the
standard workflow.

### `dicom_correlate_asconv_with_gradient.py`

This script searches for numeric ASCCONV parameters that correlate with the gradient
encoded for each NOGSE measurement.

Inputs:

- `dicom_asconv_key_values.long.csv` or `dicom_asconv_key_values.long.parquet`
- `sequence_parameters_by_nifti_from_dicom.csv` or `.parquet`

The NIfTI table provides the mapping from `sequence`/DICOM series to the intended
gradient `G`. The gradient comes from the DICOM-derived sequence table when available,
or from the NIfTI/protocol naming convention used by the extraction workflow.

Example:

```bash
python scripts/dicom_correlate_asconv_with_gradient.py \
  analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/dicom_asconv_key_values.long.parquet \
  --nifti-table analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/sequence_parameters_by_nifti_from_dicom.parquet
```

For each ASCCONV key, the script:

1. Keeps rows whose `source` is `ASCCONV`.
2. Converts `value` to numeric when possible.
3. Maps each row's DICOM `series` to the series gradient.
4. Accumulates Pearson correlation statistics by key.
5. Writes one row per numeric key, sorted by absolute correlation by default.

The output schema is:

```text
key, correlation, abs_correlation, n_observations,
min_value, max_value, min_gradient, max_gradient
```

For the current study the top gradient-correlated parameter is:

```text
sWipMemBlock.alFree[7]
```

with Pearson correlation `1.0` for the non-zero gradient series. Its values range from
`4` to `76`, matching the non-zero `G` values in mT/m encoded in the protocol naming and
DICOM-derived summary table.

The generated outputs are:

```text
dicom_asconv_gradient_correlations.csv
dicom_asconv_gradient_correlations.xlsx
```

### `dicom_export_file_parameters.py`

This script exports the ASCCONV long table for one selected DICOM/IMA image. It is useful
when the full long table is too large and one image needs to be inspected in Excel.

Example:

```bash
python scripts/dicom_export_file_parameters.py \
  analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/dicom_asconv_key_values.long.parquet \
  --dicom-file QUALITY_JACK.MR.JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE.0003.0060.2026.05.05.16.58.45.590234.521849008
```

The script accepts a full DICOM path, a basename, a filename stem, or a unique substring.
It writes a long CSV and an Excel workbook:

```text
<selected-dicom>.dicom_parameters.long.csv
<selected-dicom>.dicom_parameters.long.xlsx
```

The long export has this schema:

```text
dicom_file, series, image, source, key, value, value_numeric
```

`value_numeric` is populated when the text value can be interpreted as a finite number.
The Excel workbook contains:

- `parameters_long`: all ASCCONV rows for the selected DICOM image.
- `numeric_parameters`: only rows with numeric values.
- `matched_dicoms`: the DICOM path, series, and image that matched the query.

The first worksheet is `parameters_long` so the workbook opens directly on the full
parameter table.

## How NOGSE Gradient Values Are Encoded in These DICOMs

For the current `20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF` audit, the strongest
ASCCONV match to NOGSE `G` is:

```text
sWipMemBlock.alFree[7]
```

For non-zero `G` series, this key is present and equals the gradient value:

| Protocol token | ASCCONV key | Example value |
| --- | --- | --- |
| `_G04_` | `sWipMemBlock.alFree[7]` | `4` |
| `_G08_` | `sWipMemBlock.alFree[7]` | `8` |
| `_G76_` | `sWipMemBlock.alFree[7]` | `76` |

The extraction script also records:

```text
G_from_protocol_name
G_source
G_fraction_of_scanner_max
```

where `G_fraction_of_scanner_max` is computed from the configured scanner maximum
gradient value, currently `80 mT/m` by default.

## Special Case: `G00`

The `G00` measurements require special care.

For the current study, the two `G00` NOGSE series are:

```text
002_NOGSE_CPMG_N2_TN50_G00
002_NOGSE_HAHN_N2_TN50_G00
```

In these series:

- `tProtocolName` contains `_G00_`.
- `G_from_protocol_name` is `0.0`.
- `sWipMemBlock.alFree[7]` is absent from the ASCCONV block.
- `PhaseGradientAmplitude`, `ReadoutGradientAmplitude`, and
  `SelectionGradientAmplitude` are `0.0` in the extracted summary fields.
- Other `sWipMemBlock.alFree[...]` values are present, but they do not encode the NOGSE
  `G` axis the way `sWipMemBlock.alFree[7]` does for non-zero series.

The current extraction therefore reports:

```text
G = 0.0
G_source = sWipMemBlock.alFree[7] missing; protocol G00 assumed zero
```

This is a protocol-level inference, not a hardware-level measurement. The DICOM evidence
is consistent with `G=0`: the protocol name says `G00`, the correlated WIP gradient field
is absent, and extracted gradient-amplitude fields are zero. However, proving the
scanner's physically applied gradient waveform would require an independent source such
as raw/twix/MDH data, scanner logs, sequence logs, field monitoring, or scanner-side
waveform validation.

## Current Study Outputs

For the current study folder:

```text
analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF
```

the audit workflow generated:

```text
dicom_asconv_key_values.long.parquet
dicom_metadata_summary.parquet
sequence_parameters_by_nifti_from_dicom.parquet
sequence_parameters_from_dicom.parquet
dicom_asconv_gradient_correlations.csv
dicom_asconv_gradient_correlations.xlsx
QUALITY_JACK.MR.JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE.0003.0060.2026.05.05.16.58.45.590234.521849008.dicom_parameters.long.csv
QUALITY_JACK.MR.JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE.0003.0060.2026.05.05.16.58.45.590234.521849008.dicom_parameters.long.xlsx
```

The single-DICOM export for image `0003.0060` contains 1116 ASCCONV parameter rows.

## Practical Interpretation

Use the ASCCONV audit tables when the question is:

- Which scanner/protocol parameters were recorded in the DICOM file?
- Which private Siemens protocol key appears to encode the intended NOGSE `G` value?
- Which ASCCONV numeric parameters correlate with the acquisition gradient?
- What full protocol key-value table belongs to one selected DICOM image?

Do not use these tables alone to claim:

- direct measurement of scanner hardware output,
- independently validated gradient waveform execution,
- scanner-side safety or waveform validation.

The strongest current DICOM/protocol conclusion is:

```text
sWipMemBlock.alFree[7] encodes the non-zero intended NOGSE G value in these ASCCONV blocks.
For G00, the DICOM/protocol evidence is consistent with zero, but the value is inferred
from protocol naming and absence of the WIP G field rather than read from that same key.
```
