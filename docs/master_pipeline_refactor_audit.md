# Master Pipeline Cleanup Status

This is the current cleanup state after moving the OGSE flow to the master-table
pipeline.

## Current Main Path

- `bash_template_2/brains_ogse/*.sh`
- `bash_template_2/phantoms_ogse/*.sh`
- `scripts/data/*`
- `scripts/fitting/*`
- `scripts/plotting/*`
- `scripts/summary/*`
- `src/data_processing/master_table.py`
- `src/fitting/model_registry.py`
- `src/fitting/parameter_modes.py`

From `Results/` onward, the intended path is `master.long.parquet` plus
explicit selectors. Routine OGSE analysis should not depend on filename parsing
or long lists of parquet files.

## Removed

The old OGSE post-Results bash wrappers under these directories were removed:

- `bash_template/brains_ogse`
- `bash_template/phantoms_ogse`

Only pre-Results scripts remain there for DICOM conversion, phantom prep, and
signal extraction.

## Kept For Now

- NOGSE bash templates remain because the NOGSE flow has not yet been rebuilt
  around `bash_template_2`.
- Publication figure scripts remain separate because they are figure-specific
  entrypoints, not routine pipeline stages.
- Some plotting CLIs still have family-specific names. They now live under
  `scripts/plotting/`, and the next reduction would be to merge OGSE/NOGSE
  signal plotting and OGSE/NOGSE contrast plotting into generic CLIs.

## Extension Points

- Add ingestion/rotation/contrast behavior in `scripts/data/` and reusable code
  under `src/data_processing/`.
- Add models in `src/fitting/model_registry.py`.
- Add parameter sharing rules in `src/fitting/parameter_modes.py`.
- Add fitting CLI behavior in `scripts/fitting/`, keeping reusable logic in
  `src/fitting/`.
- Add plots in `scripts/plotting/`, with row selection handled by
  `src/data_processing/master_table.py` where possible.

## Remaining Cleanup Candidates

- Merge duplicated OGSE/NOGSE plotting CLIs.
- Move NOGSE bash templates to the same master-table structure as OGSE.
- Gradually remove legacy parquet/root inputs from CLIs once all current
  analyses are reproducible from `master.long.parquet`.
