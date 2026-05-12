# Phantoms NOGSE: origin of bval/bvec and gval/gvec in steps 0.0 and 0.1

This document explains how these scripts handle gradient metadata:

- `bash_template/phantoms_nogse/0.0-run_dicom2nifti.sh`
- `bash_template/phantoms_nogse/0.1-run_make_gval_gvec.sh`

The practical goal is to distinguish whether the pipeline is using values recorded in
the DICOM files, values entered in the scanner protocol, or values reconstructed later
from filenames and user-provided assumptions.

For the dedicated Siemens ASCCONV audit workflow, including how the repository extracts
private protocol key-value tables that are not expanded by a normal `pydicom show`, see
`docs/dicom_asconv_metadata_audit.md`.

## Short Summary

`0.0-run_dicom2nifti.sh` converts DICOM to NIfTI through `dcm2niix`. If `dcm2niix`
recognizes diffusion metadata in the DICOM series, it writes `.bval` and `.bvec`
sidecars. The script itself does not calculate or invent `bval/bvec`.

`0.1-run_make_gval_gvec.sh` does not read DICOM metadata. It generates `.gval` and
`.gvec` sidecars from already-converted NIfTI files. The `g` value is parsed from the
NIfTI filename, using tokens such as `_G30_` or `_G30p5_`. The direction is taken from
the `--dir GX GY GZ` argument, whose current wrapper default is `1 0 0`.

Therefore:

- `.bval/.bvec` from `dcm2niix` are DICOM-derived scanner/protocol metadata.
- `.gval/.gvec` from step `0.1` are filename-derived and user-assumed metadata.

## What "Used by the Scanner" Means

There are three different levels of truth, and they should not be mixed:

1. **Physically applied by the hardware**: the actual gradient waveform/current produced
   by the scanner hardware. The sidecars in this pipeline do not prove this directly.
   Proving it would require scanner logs, sequence validation, field monitoring, or
   another hardware-level validation source.

2. **Recorded by the scanner/protocol in DICOM**: values saved by the scanner in public
   or private DICOM metadata. These are the best values available from the DICOM files,
   but they are still recorded protocol metadata, not an independent measurement of the
   physical gradient output.

3. **Entered or inferred by the user after conversion**: values reconstructed from file
   names, spreadsheets, script defaults, or command-line arguments. These may match the
   intended protocol, but they are not extracted from DICOM metadata.

In this repository, the distinction is:

| Sidecar source | Meaning in this pipeline |
| --- | --- |
| `.bval/.bvec` written by `dcm2niix` | DICOM-derived scanner/protocol metadata recognized by `dcm2niix`. |
| `.gval/.gvec` written by `0.1-run_make_gval_gvec.sh` | User/inferred metadata: `G` comes from the filename and direction comes from `--dir`. |
| DICOM metadata tables from `extract_dicom_sequence_metadata.py` | DICOM-derived scanner/protocol metadata for audit and provenance. They do not replace `.gval/.gvec` unless a later step explicitly uses them. |

## Step 0.0: DICOM to NIfTI Conversion

`0.0-run_dicom2nifti.sh` is a small driver. It defines paths and calls shared functions
from `bash_template/helpers/dicom2nifti_batch_lib.sh`.

Relevant configuration:

```bash
PROJECT_ROOT="${PROJECT_ROOT:-$(resolve_default_project_root "$REPO_ROOT")}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/Data-DICOM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/Data-NIFTI}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/phantoms-3_nogse}"

run_case "20260505_PHANTOM_FIBER"
```

For each series under:

```text
Data-DICOM/20260505_PHANTOM_FIBER/<series>/
```

the helper creates an output folder:

```text
Data-NIFTI/20260505_PHANTOM_FIBER/<series>/
```

and runs:

```bash
dcm2niix -z y -o "$out_series_dir" "$series_dir"
```

### Where `.bval/.bvec` Come From

`.bval/.bvec` appear only when `dcm2niix` can decode diffusion information from the
DICOM series. In that case:

- `.bval` contains one b-value per volume.
- `.bvec` contains one diffusion direction per volume in FSL format.
- `0.0` does not modify these values.
- `0.0` does not contain an internal b-value or direction table.

If `.bval/.bvec` exist after step `0.0`, their immediate source is DICOM metadata
interpreted by `dcm2niix`.

### When DICOM Does Not Produce `.bval/.bvec`

If the DICOM series does not contain diffusion fields recognized by `dcm2niix`, the
conversion can still produce `.nii.gz` and `.json` files, but no `.bval/.bvec`.

In that case, the pipeline has no DICOM-derived `b` axis for that series. For direct-`G`
NOGSE acquisitions, step `0.1` can create `.gval/.gvec` sidecars from the NIfTI filenames.

## Step 0.1: Generating `.gval/.gvec`

`0.1-run_make_gval_gvec.sh` calls:

```bash
python nogse_pipeline/scripts/make_gval_gvec_from_filenames.py "$EXP_ROOT" \
  --glob "$DWI_GLOB" \
  --direction "$DIR_X" "$DIR_Y" "$DIR_Z"
```

Current wrapper defaults:

```bash
SUBJ="20260505-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF"
EXP_PARENT="Data-NIFTI"
DWI_GLOB="*_002_NOGSE*.nii.gz"
DIR_X="1"
DIR_Y="0"
DIR_Z="0"
OVERWRITE=0
DRY_RUN=0
```

Therefore it searches for NIfTI files in:

```text
Data-NIFTI/20260505-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/
```

matching:

```text
*_002_NOGSE*.nii.gz
```

### Where `.gval` Comes From

The Python generator searches each NIfTI filename for a token:

```text
G<number>
G<number>p<decimal>
```

Examples:

- `_G30_` is parsed as `30`
- `_G30p5_` is parsed as `30.5`

It then loads the NIfTI with `nibabel` to infer the number of volumes:

- if the NIfTI is 4D, the fourth dimension is used as the number of volumes;
- if it is 3D, one volume is assumed.

Finally, it writes a `.gval` file with the same `G` repeated once per volume:

```text
30 30 30 30 ...
```

This means step `0.1` assumes that all volumes in one sequence have the same `G` value.
That value comes from the filename, not from DICOM.

### Where `.gvec` Comes From

The direction comes from `--dir GX GY GZ`, not from DICOM or NIfTI. The current wrapper
default is:

```text
1 0 0
```

The Python generator normalizes the vector before writing it. For example, `--dir 2 0 0`
is written as `1 0 0`.

The `.gvec` file is written as three FSL-style rows, each component repeated once per
volume:

```text
1 1 1 1 ...
0 0 0 0 ...
0 0 0 0 ...
```

### Overwrite Behavior

By default `OVERWRITE=0`. If `.gval/.gvec` already exist, the Python generator refuses
to overwrite them. Use `--overwrite` to regenerate existing sidecars.

## Practical Cases

### Case A: DICOM Has Recognized Diffusion Metadata

Flow:

1. `0.0` runs `dcm2niix`.
2. `dcm2niix` writes `.nii.gz`, `.json`, `.bval`, and `.bvec`.
3. `.bval/.bvec` come from DICOM metadata.

Interpretation:

- These are scanner/protocol-recorded b-values and directions as interpreted by
  `dcm2niix`.
- They are not generated by this repository.
- They are not a direct hardware measurement of the actual gradient waveform.

### Case B: DICOM Does Not Produce `.bval/.bvec`

Flow:

1. `0.0` runs `dcm2niix`.
2. The NIfTI is created, but `.bval/.bvec` are not created.
3. `0.1` creates `.gval/.gvec`.

Interpretation:

- `.gval` comes from the `G...` token in the NIfTI filename.
- `.gvec` comes from the `--dir` argument.
- These values are intended/user-inferred metadata, not DICOM-extracted metadata.
- They may match the protocol naming convention, but they do not prove that the DICOM
  contains those values.

This case is useful for NOGSE acquisitions where the natural analysis axis is `G` and
there is no DICOM b-value axis recognized by `dcm2niix`.

### Case C: Both `.bval/.bvec` and `.gval/.gvec` Exist

In `src/signal_extraction/coreg_extract_phantom.py`, gradient discovery first checks:

```text
<seq_no_ext>.bval + <seq_no_ext>.bvec
```

and only if those files do not exist, it checks:

```text
<seq_no_ext>.gval + <seq_no_ext>.gvec
```

Therefore, if both pairs exist for the same sequence, the phantom extractor uses
`.bval/.bvec` and classifies the sequence as kind `b`. The `.gval/.gvec` pair is ignored
by that stage.

This matters if step `0.1` is run in a folder where `dcm2niix` already produced
`.bval/.bvec`: creating `.gval/.gvec` does not automatically change the analysis axis.

### Case D: Custom NOGSE `G` Values Stored in Siemens Private/WIP DICOM Fields

Some custom Siemens sequences can store user-entered protocol parameters in private
Phoenix/WIP metadata rather than in standard diffusion DICOM fields. Those values are
DICOM-derived scanner/protocol metadata if they are parsed directly from the DICOM.

Step `0.1` still uses the filename as the source of `G`. The dedicated metadata
extractor added for audit/provenance is:

```text
scripts/extract_dicom_sequence_metadata.py
```

It reads the DICOM/Phoenix text available in the files, extracts candidate gradient
fields, and writes compact sequence-parameter-style tables plus long audit tables.
The main NIfTI-level output is:

```text
sequence_parameters_by_nifti_from_dicom.csv
```

This table has one row per NIfTI file when `--nifti-root` is provided. That is the
recommended table to compare against `sequence_parameters` rows.

The extractor also writes:

```text
sequence_parameters_from_dicom.csv
dicom_metadata_summary.csv
dicom_asconv_key_values.long.csv
dicom_printable_strings.long.csv  # only with --write-strings
```

The long ASCCONV and printable-string tables are CSV-only by default because full DICOM
exports can exceed Excel's row limit.

To use WIP/private DICOM values as actual `.gval/.gvec` sidecars reproducibly, the
repository still needs a separate explicit writer that:

1. reads the relevant private DICOM/Phoenix fields;
2. maps each WIP field to a documented physical/protocol parameter;
3. writes `.gval/.gvec` with the same naming and length conventions used elsewhere;
4. records the field names and units used for extraction.

## Inspection of the 2026-01-22 G00 DICOM Series

The inspected series was:

```text
Data-DICOM/20260122-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/
JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE_20260122_125354_246000/
002_NOGSE_CPMG_N2_TN50_G00_0003
```

Running `dcm2niix` on this series in a temporary folder produced:

```text
probe_g00.nii.gz
probe_g00.json
```

It did not produce:

```text
probe_g00.bval
probe_g00.bvec
```

The generated JSON includes:

```json
"Manufacturer": "Siemens",
"ManufacturersModelName": "Prisma",
"ProtocolName": "002_NOGSE_CPMG_N2_TN50_G00",
"PulseSequenceDetails": "%CustomerSeq%\\NOGSE_SE_EPI"
```

The DICOM contains Siemens private/CSA/Phoenix metadata with diffusion-related names,
including `B_value`, `B_matrix`, `MRDiffusion`, and `sDiffusion`. The visible Phoenix
block includes:

```text
sDiffusion.ulMode = 1
sDiffusion.dsScheme = 1
sDiffusion.lQSpaceSteps = 1
sDiffusion.alBValue.__attribute__.size = 128
sDiffusion.alAverages.__attribute__.size = 128
sDiffusion.sFreeDiffusionData.asDiffDirVector.__attribute__.size = 0
```

No explicit `sDiffusion.alBValue[0] = ...` values or free diffusion direction vectors
were found in the inspected strings output. This is consistent with `dcm2niix` not
writing `.bval/.bvec`.

The same DICOM also contains WIP fields:

```text
sWipMemBlock.alFree[1] = 2
sWipMemBlock.alFree[2] = 5
sWipMemBlock.alFree[3] = 4
sWipMemBlock.alFree[4] = 2
sWipMemBlock.alFree[5] = 25000
sWipMemBlock.alFree[6] = 25000
sWipMemBlock.alFree[8] = 700
sWipMemBlock.adFree[0] = 2.0
sWipMemBlock.adFree[1] = 12.0
sWipMemBlock.adFree[2] = 22.0
```

Comparing sibling `002_NOGSE_CPMG_N2_TN50_GXX` series showed that:

```text
sWipMemBlock.alFree[7] = 4   for G04
sWipMemBlock.alFree[7] = 8   for G08
sWipMemBlock.alFree[7] = 12  for G12
...
sWipMemBlock.alFree[7] = 76  for G76
```

For the inspected G00 series, `sWipMemBlock.alFree[7]` was not shown as an explicit
nonzero field, which is consistent with a zero/default value.

Conclusion for this specific series:

- Standard `dcm2niix` extraction does not produce `.bval/.bvec`.
- No `.gval/.gvec` are produced by `dcm2niix`.
- The custom `G` value appears to be present in Siemens private WIP metadata,
  specifically `sWipMemBlock.alFree[7]` for nonzero sibling G-series.
- The current step `0.1` does not use that DICOM field; it gets `G` from the filename.
- A DICOM-based `.gval` extractor looks feasible, but it should be implemented only
  after confirming the WIP field mapping and units from the sequence/protocol author.
- No DICOM-derived `.gvec` source was confirmed from this inspection. The direction is
  still user-assumed in the current workflow.

An additional DICOM value, `Stim_max_ges_norm_online`, can look close to `G / 80` for
high-gradient series. For example, in the 2026-05-05 phantom acquisition, G48 has
`Stim_max_ges_norm_online = 0.60434878`, which is close to `48 / 80 = 0.6`. However,
the same field is approximately `0.59646350` for G00, G04, and G08, so it does not
track the NOGSE gradient amplitude across the full series. The extraction script keeps
this value as audit metadata instead of using it as the `.gval` axis.

The helper script extracts both the WIP-derived `G` value and the
stimulation/safety-related candidate values into CSV/XLSX tables so the provenance can
be inspected before choosing an analysis axis.

Example:

```bash
python nogse_pipeline/scripts/extract_dicom_sequence_metadata.py \
  Data-DICOM/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF \
  --out-root analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF \
  --nifti-root Data-NIFTI/20260505-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF \
  --nifti-glob '*_NOGSE*.nii.gz' \
  --glob '*.IMA' \
  --out-xlsx analysis/dicom_metadata/20260505_PHANTOM_FIBER/QUALITY_JACK_19800122TMSF/sequence_metadata_from_dicom.xlsx
```

## Relation to `0.2-prep_phantom_b0.sh`

`0.2-prep_phantom_b0.sh` has two conceptual reference modes:

- If it uses `REF_MODE="mean"`, it averages volumes and does not need gradients to find
  b0 volumes.
- If it does not use `mean`, it looks specifically for `.bval/.bvec` and calls
  `dwiextract -bzero`.

In the current `0.2` logic, b0 extraction does not use `.gval/.gvec`; it expects
`.bval/.bvec`. For acquisitions with only `.gval/.gvec`, use a mean reference or adapt
the step explicitly if "zero gradient" should be defined from `.gval`.

## How to Check Which Source Is Being Used

For a concrete sequence, inspect the NIfTI folder:

```bash
ls Data-NIFTI/<subject>/<sequence_base>.bval Data-NIFTI/<subject>/<sequence_base>.bvec
ls Data-NIFTI/<subject>/<sequence_base>.gval Data-NIFTI/<subject>/<sequence_base>.gvec
```

If `.bval/.bvec` exist, the phantom extractor prioritizes them.

To inspect values:

```bash
cat Data-NIFTI/<subject>/<sequence_base>.bval
cat Data-NIFTI/<subject>/<sequence_base>.bvec
cat Data-NIFTI/<subject>/<sequence_base>.gval
cat Data-NIFTI/<subject>/<sequence_base>.gvec
```

To check whether `.gval` values came from filenames, verify that the NIfTI name has a
`G...` token:

```text
..._G30_...
..._G30p5_...
```

If the name does not have that token, step `0.1` skips the file with a warning.

## Operational Conclusion

The current behavior is:

- If DICOM contains diffusion metadata that `dcm2niix` recognizes, step `0.0` creates
  `.bval/.bvec` from DICOM.
- If that metadata is unavailable or not recognized, step `0.0` does not create those
  sidecars.
- Step `0.1` creates `.gval/.gvec` from the NIfTI filename and a manually supplied
  direction.

For the inspected 2026-01-22 custom NOGSE series, the nonzero `G` values appear to be
stored in Siemens private WIP metadata, but the current pipeline does not extract them.
Until a dedicated DICOM/WIP extractor is added, `.gval/.gvec` generated by step `0.1`
should be treated as filename-derived/user-assumed metadata.
