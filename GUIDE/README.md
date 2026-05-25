# Technical Guide

This folder documents the implementation of the current `nogse_pipeline`
repository. It is organized by repository area instead of by pipeline-guide
fragments, so each file can be used independently while sharing one common
vocabulary.

## Reading Order

1. [Repository Map](01_repository_map.md)
2. [Batch Orchestration](02_batch_orchestration.md)
3. [Data Ingestion And Signal Tables](03_data_ingestion_and_signal_tables.md)
4. [Signal Rotation And Contrast Tables](04_signal_rotation_and_contrast.md)
5. [Fitting Architecture](05_fitting_architecture.md)
6. [Gradient Correction](06_gradient_correction.md)
7. [Plotting And Summaries](07_plotting_and_summaries.md)
8. [Output Schemas And Validation](08_output_schemas_and_validation.md)

## Current Repository Rules

- `bash_template/` is the canonical source for runnable batch scripts.
- `bash/` is generated or operational material and must not be edited as source
  documentation.
- User-facing scripts live in `scripts/`; reusable implementation lives in
  `src/`.
- Fit output schemas are centralized in `src/tools/fit_params_schema.py`.
- Supported experiment and model families are centralized in
  `src/fitting/experiments.py`.

