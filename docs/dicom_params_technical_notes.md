# DICOM Parameter Extraction Technical Notes

This document explains how the DICOM parameter extraction code is organized and how the
main functions work internally.

For user-facing commands, see `docs/dicom_params_user_guide.md`.

## Code Layout

Reusable implementation lives in:

```text
src/dicom_params/
```

with these modules:

```text
extraction.py
single_file.py
correlation.py
```

Thin command-line wrappers live in:

```text
scripts/extract_dicom_sequence_metadata.py
scripts/dicom_export_file_parameters.py
scripts/dicom_correlate_asconv_with_gradient.py
```

Bash launchers live in:

```text
bash_template/dicom_params/
```

This layout keeps the reusable logic importable from Python while preserving simple
script entry points for routine use.

## `src/dicom_params/extraction.py`

This module extracts ASCCONV and summary metadata from DICOM/IMA files.

### Printable String Extraction

The function:

```python
extract_printable_strings(path)
```

reads the DICOM file bytes and extracts printable byte ranges using:

```python
PRINTABLE_RE = re.compile(rb"[\x09\x20-\x7E]{4,}")
```

This is intentionally lower-level than `pydicom show`. Siemens private protocol text can
exist as printable text inside private binary payloads, and not every ASCCONV key appears
as an individual parsed DICOM tag.

### ASCCONV Parsing

The function:

```python
parse_asconv(strings)
```

scans the printable strings for:

```text
### ASCCONV BEGIN
### ASCCONV END
```

Between those markers it parses lines matching:

```python
KEY_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.\[\]<>/-]+)\s*=\s*(?P<value>.*)$")
```

The parsed output is a dictionary where keys are ASCCONV parameter names such as:

```text
tProtocolName
sWipMemBlock.alFree[7]
```

### Per-DICOM Metadata Object

The function:

```python
extract_dicom_meta(path, scanner_grad_max_mtm=...)
```

returns a `DicomMeta` dataclass with path, series/image numbers, protocol names, selected
gradient-related fields, the parsed ASCCONV dictionary, and the printable strings.

Series and image numbers are parsed from Siemens filenames of the form:

```text
... .<series>.<image>.<date>...
```

The intended NOGSE gradient is extracted from `sWipMemBlock.alFree[7]` when present. If
that key is absent and the protocol name encodes `G00`, the extractor records:

```text
G = 0.0
G_source = sWipMemBlock.alFree[7] missing; protocol G00 assumed zero
```

That is a DICOM/protocol inference, not a hardware waveform measurement.

### Table Construction

The extraction module builds four main tables:

```python
build_dicom_rows(...)
build_sequence_rows(...)
build_nifti_rows(...)
build_key_value_rows(...)
```

`build_dicom_rows` writes one summary row per DICOM image.

`build_sequence_rows` groups DICOM images by `(series, protocol_name)` and writes one
sequence-level row.

`build_nifti_rows` maps converted NIfTI files back to DICOM series and writes one row per
NIfTI.

`build_key_value_rows` writes the full ASCCONV long table:

```text
dicom_file, series, image, source, key, value
```

### Output Writing

The public orchestration function is:

```python
extract_metadata_tables(...)
```

It writes CSV and, by default, Parquet for each table. The helper:

```python
write_table_pair(...)
```

keeps CSV and Parquet filenames synchronized. For files ending in `.long.csv`, it writes
the matching Parquet path as `.long.parquet`.

The compact Excel workbook contains only summary sheets because the full ASCCONV long
table can exceed Excel's row limit.

## `src/dicom_params/single_file.py`

This module exports the long ASCCONV table for one selected DICOM image.

The main function is:

```python
export_one_dicom_parameters(...)
```

It accepts a full DICOM path, basename, stem, or unique substring. Matching is handled by:

```python
match_dicom_rows(chunk, query)
```

The input key-value table can be CSV or Parquet. Large inputs are read in chunks through:

```python
read_key_value_chunks(...)
```

Chunked reading is why older script output printed messages such as `Scanned chunk 1`.
Each chunk is just a block of rows from the large long table. Progress messages are now
disabled by default and can be enabled with `--progress-every`.

The selected DICOM output is written as:

```text
<DICOM>.dicom_parameters.long.parquet
<DICOM>.dicom_parameters.long.xlsx
```

CSV is optional. The output rows include:

```text
dicom_file, series, image, source, key, value, value_numeric
```

`value_numeric` is populated when the ASCCONV text value can be converted to a finite
number.

## `src/dicom_params/correlation.py`

This module computes Pearson correlations between numeric ASCCONV parameters and the
series gradient.

The main function is:

```python
correlate_asconv_with_gradient(...)
```

It reads:

```text
dicom_asconv_key_values.long.parquet
sequence_parameters_by_nifti_from_dicom.parquet
```

The function:

```python
load_series_gradient_map(...)
```

builds a `series -> gradient` mapping from the NIfTI table. It uses `G` when available,
or falls back to parsing `_Gxx_` tokens from `nifti_base` or `nifti_file`.

The long ASCCONV table is streamed in chunks. For each chunk:

1. Keep rows with `source == ASCCONV`.
2. Convert `value` to numeric when possible.
3. Map `series` to the gradient.
4. Accumulate sufficient statistics by `key`.

The accumulated statistics are:

```text
n, sum_x, sum_y, sum_x2, sum_y2, sum_xy,
min_value, max_value, min_gradient, max_gradient
```

`finalize_stats(...)` converts those sums into Pearson correlations without storing all
numeric values for every key in memory.

## CLI Wrappers

The scripts in `scripts/` do only argument parsing and printing. They import the reusable
functions from `src/dicom_params`.

This keeps the command-line behavior stable while making the implementation easier to
test, inspect, and reuse.

## Bash Launchers

The launchers in `bash_template/dicom_params/` define study paths and call the Python
scripts with repository-standard defaults.

`0.0-run_extract_dicom_sequence_metadata.sh` calls the shared helper:

```text
bash_template/helpers/run_extract_dicom_sequence_metadata.sh
```

The helper passes `--write-parquet` by default, so CSV and Parquet outputs are produced
in the same extraction step.

`0.1-run_export_one_dicom_parameters.sh` uses the full-study long table as input and
writes only the selected DICOM table to Excel and Parquet by default.

`0.2-run_correlate_dicom_params_with_gradient.sh` performs the numeric correlation audit.

## Memory Behavior

Full-study ASCCONV tables can contain millions of rows. The single-DICOM export and
correlation tools therefore read the input table in chunks.

Chunking is an implementation detail. It does not change the output values. It only
prevents loading the entire long table into memory at once.
