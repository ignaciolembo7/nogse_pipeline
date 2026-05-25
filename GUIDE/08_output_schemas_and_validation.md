# Output Schemas And Validation

The repository relies on strict, explicit schemas to keep different workflows
compatible.

## Fit Parameter Schemas

Central schema module:

- `src/tools/fit_params_schema.py`

This module defines ordered columns for:

- monoexponential fits,
- OGSE signal fits,
- NOGSE signal fits,
- OGSE contrast fits,
- NOGSE contrast fits.

It also standardizes:

- `fit_kind`,
- `model`,
- `source_file`,
- `D0` units and error columns,
- missing columns,
- output column ordering.

Fitters should call schema helpers instead of writing ad hoc fit tables.

## Strict Column Checking

Strict validation lives in:

- `src/tools/strict_columns.py`

Important call pattern:

```python
raise_on_unrecognized_column_names(df.columns, context="module_or_function")
```

Use strict validation at boundaries:

- reading processed signal tables,
- reading correction tables,
- building contrasts,
- loading fit parameters for downstream summaries.

## Output Writers

General table writers live in:

- `src/data_processing/io.py`

Common outputs are written as paired `.xlsx` and `.csv` or `.parquet` and
`.xlsx` artifacts depending on the stage. Prefer existing writer helpers over
direct `to_excel` or `to_parquet` calls when a stage already has a standard.

## Standard Output Families

Common output families include:

- `*.long.parquet`: canonical long-form signal tables;
- `*.rot_tensor.long.parquet`: tensor-rotated brain OGSE signal tables;
- `*.Dproj.long.parquet`: projected diffusivity tables;
- `fit_params*.parquet` / `fit_params*.xlsx`: standardized fit-parameter
  tables;
- `fit_points.parquet` / `fit_points.xlsx`: points used or evaluated by a fit;
- `summary_plots/`: tables and plots derived from many fit outputs;
- `grad_correction*.xlsx` / `.csv`: side-specific correction tables.

## Adding A New Output

Before adding a new output:

1. check whether an existing schema already covers the table;
2. add columns to the centralized schema if the table is a fit output;
3. use existing naming conventions from `src/fitting/experiments.py`;
4. validate columns at read boundaries;
5. document the new artifact in this guide if it becomes part of the canonical
   workflow.

