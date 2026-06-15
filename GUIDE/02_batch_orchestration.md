# Batch Orchestration

The batch layer lives in `bash_template/`. It is intentionally explicit: each
script names the inputs, output roots, model choice, axes, ROIs, correction
mode, and Python entry point used for one stage.

## Canonical Workflow Layers

`bash_template/brains_ogse/`

Brain OGSE preparation:

- DICOM to NIfTI conversion.
- Brain signal extraction.

`bash_template/phantoms_ogse/`

Phantom OGSE preparation:

- DICOM conversion and gradient sidecar preparation.
- Phantom signal extraction.

`bash_template/phantoms_nogse/`

Phantom NOGSE preparation:

- DICOM conversion and gradient sidecar preparation.
- Phantom signal extraction.

`bash_template/run_dataset.sh`

Shared post-Results analysis for all `type_subj` and `type_seq` combinations:

- ROI result processing into `master.long.parquet`.
- Tensor rotation.
- OGSE/NOGSE contrast construction and plotting.
- Signal fits, contrast fits, and global signal fits.
- Gradient correction.
- `D0`, alpha, monoexponential-D, and `t_c` summaries.

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

The current master-table runner is:

- `bash_template/run_dataset.sh`

Use `type_subj` (`brain` or `phantom`) and `type_seq` (`ogse` or `nogse`) to
select the workflow:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest
```

Use `GUIDE/09_pipeline_user_guide.md` for the full step-by-step command
reference, arguments, inputs, and expected outputs.
