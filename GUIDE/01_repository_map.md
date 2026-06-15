# Repository Map

The repository is organized around thin command-line scripts, reusable Python
modules, and canonical batch templates.

## Top-Level Areas

`scripts/`

Command-line entry points. They parse arguments, load input tables, resolve
paths, and call reusable functions from `src/`. They should stay thin; shared
logic belongs in modules.

`src/`

Reusable implementation. The important subpackages are:

- `data_processing/`: reads ROI result tables, matches sequence parameters, and
  writes canonical long-form signal tables.
- `dicom_params/`: extracts and correlates scanner-side metadata from DICOM
  headers.
- `fitting/`: shared fitting helpers, experiment registries, contrast building,
  gradient-axis conversion, and correction lookup logic.
- `monoexp_fitting/`: monoexponential signal fitting and diffusivity summaries.
- `nogse_fitting/`: NOGSE signal and contrast fitting.
- `ogse_fitting/`: OGSE signal and contrast fitting, gradient correction, and
  contrast-fit panels.
- `models/`: mathematical model functions used by fitters.
- `plottings/`: shared plotting primitives and styles.
- `signal_extraction/`: ROI extraction and registration workflows.
- `signal_rotation/`: tensor-based rotation of directional OGSE signals.
- `tc_fittings/`: `t_c` summaries, alpha summaries, and group-level fits.
- `tools/`: strict column validation, label normalization, and fit schema
  standardization.

`bash_template/`

Canonical batch scripts. Preparation drivers stay in dataset folders, while
post-Results analysis is centralized in `run_dataset.sh` and `steps/`.

- `brains_ogse`: brain OGSE DICOM conversion and signal extraction.
- `phantoms_ogse`: phantom OGSE DICOM conversion, sidecars, preparation, and
  signal extraction.
- `phantoms_nogse`: phantom NOGSE DICOM conversion, sidecars, preparation, and
  signal extraction.
- `dicom_params`: scanner metadata utilities.
- `helpers`: shared shell helpers used by the preparation drivers and runner.
- `run_dataset.sh` and `steps/`: master-table post-Results workflow for
  `brain|phantom` and `ogse|nogse`.

`bash/`

Operational copies. Do not edit this folder as source.

`docs/`

Long-form reports and meeting-oriented summaries.

`GUIDE/`

Implementation guide for developers and maintainers.

## Current Experiment Families

The canonical experiment registry is `src/fitting/experiments.py`:

- `ogse_signal_vs_g`
- `ogse_contrast_vs_g`
- `nogse_signal_vs_g`
- `nogse_contrast_vs_g`

The same module defines valid model families and the standard output directory
naming pattern:

```text
{experiment}_{model}
{experiment}_{model}_corr
```

This registry is the first place to check before adding or renaming a fit mode.
