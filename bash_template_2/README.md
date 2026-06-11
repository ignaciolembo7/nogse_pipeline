# Master-driven OGSE pipeline templates

`bash_template_2` is the compact, master-table-first replacement for the older
OGSE bash templates. It keeps one shared implementation of each pipeline step and
only thin dataset entrypoints for brains and phantoms.

Run one step at a time:

```bash
bash nogse_pipeline/bash_template_2/brains_ogse/run.sh ingest
bash nogse_pipeline/bash_template_2/brains_ogse/run.sh rotate
bash nogse_pipeline/bash_template_2/brains_ogse/run.sh contrast
```

Run a selected sequence:

```bash
bash nogse_pipeline/bash_template_2/brains_ogse/run.sh ingest rotate contrast fit_contrast
bash nogse_pipeline/bash_template_2/phantoms_ogse/run.sh migrate plot_signal
```

The same can be written with `PIPELINE_STEPS`:

```bash
PIPELINE_STEPS="ingest rotate contrast fit_signal fit_contrast alpha tc" \
  bash nogse_pipeline/bash_template_2/brains_ogse/run.sh
```

Common environment overrides:

- `PY`: Python interpreter.
- `SIGNALS_ROOT`: root containing `Results/` and sequence parameter Excel files.
- `RESULTS_ROOT`: explicit Results root.
- `ANALYSIS_ROOT`: output root, defaulting to `analysis/<dataset>/ogse_experiments`.
- `MASTER_PARQUET`: master table path.
- `MASTER_FIT_PARAMS`: master fit params path.
- `MANIFEST_DIR`: contrast and signal-fit manifest directory.

Step names:

- `migrate`: build a master table from legacy `data`, `data-rotated`, and `contrast-data`.
- `ingest`: parse `Results/*_results.xlsx` and append signal rows to master.
- `rotate`: rotate signal rows and append `row_kind=signal_rotated`.
- `contrast`: build declarative contrasts and append `row_kind=contrast`.
- `plot_signal`: plot signal curves from master selectors.
- `plot_contrast`: plot contrast curves from master selectors.
- `fit_signal`: fit signal curves from manifest selectors.
- `fit_contrast`: fit contrast curves from master.
- `fit_global_signal`: fit mixed/global signal models directly from master.
- `alpha`: build alpha macro summaries and alpha-vs-ROI plots.
- `tc`: run tc-vs-td summaries from `master_fit_params`.
- `grad_correction`: build the gradient-correction table used by corrected fits.
- `plot_d0_delta`: plot D0/D_proj vs Delta and reuse the alpha summary.
- `plot_monoexp_d`: plot monoexp D vs `td_ms` and `Delta_app_ms` from signal fits.

Edit only the CSV manifests for routine analyses. Avoid adding filename lists to
the bash scripts.

If you prefer the older "one visible bash file per step" workflow, use the
numbered scripts inside `brains_ogse/` or `phantoms_ogse/`. Those wrappers call
the shared steps but keep the run surface explicit.

## Explicit Step Files

Brains and phantoms expose the same numbered files:

```text
00-migrate_legacy_to_master.sh
01-ingest_results_to_master.sh
02-rotate_signals_to_master.sh
03-make_contrasts_to_master.sh
04-fit_signals_monoexp_pgse.sh
05-make_grad_correction_table.sh
06-fit_signals_monoexp_pgse_gradcorr.sh
07-fit_signals_mixed_global.sh
08-fit_contrasts_free.sh
09-fit_contrasts_mixed_global.sh
10-alpha_macro_summary_and_plot.sh
11-plot_D0_vs_Delta_alpha.sh
12-plot_monoexp_D_vs_td.sh
13-tc_vs_td.sh
```

Example:

```bash
bash nogse_pipeline/bash_template_2/brains_ogse/02-rotate_signals_to_master.sh
bash nogse_pipeline/bash_template_2/brains_ogse/04-fit_signals_monoexp_pgse.sh
bash nogse_pipeline/bash_template_2/brains_ogse/05-make_grad_correction_table.sh
bash nogse_pipeline/bash_template_2/brains_ogse/06-fit_signals_monoexp_pgse_gradcorr.sh
```

Gradient correction has two pieces:

- `05-make_grad_correction_table.sh` builds
  `analysis/<dataset>/ogse_experiments/fits/grad_correction_master/*.xlsx`.
- `06-fit_signals_monoexp_pgse_gradcorr.sh` applies that table through
  `fit_ogse_signal_vs_g.py --apply_grad_corr --corr_xlsx ...`.

`07-fit_signals_mixed_global.sh` calls `fit_global_signal.py --master-parquet`,
so the global signal fit no longer needs a legacy root with many input files.
