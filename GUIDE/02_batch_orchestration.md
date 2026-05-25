# Batch Orchestration

The batch layer lives in `bash_template/`. It is intentionally explicit: each
script names the inputs, output roots, model choice, axes, ROIs, correction
mode, and Python entry point used for one stage.

## Canonical Workflow Families

`bash_template/brains_ogse/`

Brain OGSE workflow:

- DICOM to NIfTI conversion.
- Brain signal extraction.
- ROI result processing.
- Tensor rotation.
- OGSE contrast construction and plotting.
- Monoexponential OGSE signal fits.
- `D0` summaries and alpha summaries.
- Free, mixed, mixed-global, rest, and corrected contrast fits.
- Final `t_c` vs `t_d` summaries.

`bash_template/phantoms_ogse/`

Phantom OGSE workflow:

- DICOM conversion and gradient sidecar preparation.
- Phantom signal extraction.
- OGSE signal processing and fitting.
- OGSE and NOGSE contrast construction.
- Gradient correction and corrected contrast fits.

`bash_template/phantoms_nogse/`

Phantom NOGSE workflow:

- DICOM conversion and gradient sidecar preparation.
- Phantom signal extraction.
- NOGSE signal plotting and fitting.
- NOGSE contrast construction.
- Gradient correction and corrected contrast fits.

`bash_template/dicom_params/`

DICOM metadata utilities:

- metadata extraction from series,
- single-file parameter export,
- correlation between scanner parameters and gradient settings.

`bash_template/helpers/`

Reusable shell functions and shared batch helpers. These support orchestration
but should not contain Python-level scientific or fitting logic.

## Design Pattern

Batch scripts should:

- set `PROJECT_ROOT` and `REPO_ROOT`;
- export `PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"`;
- call `scripts/*.py`;
- define model and axis choices explicitly;
- write under `analysis/...` or the caller-provided output root;
- continue across selected batch jobs when the workflow is intended to be
  resilient.

Batch scripts should not:

- duplicate Python fitting logic;
- write new output schemas;
- implement table postprocessing that belongs in `src/`;
- edit files under `bash/`.

## Entry-Point Runners

The top-level runners are:

- `bash_template/brains_ogse/run_brains_pipeline_ogse.sh`
- `bash_template/phantoms_ogse/run_phantoms_pipeline_ogse.sh`
- `bash_template/phantoms_nogse/run_phantoms_pipeline_nogse.sh`

These runners are useful for seeing the stage order, but the numbered scripts
are the best reference for exact parameters.

