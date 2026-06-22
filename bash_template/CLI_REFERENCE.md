# Pipeline CLI Reference

This document is the command-line reference for the Phase 2 NOGSE/OGSE analysis
pipeline. It is written for a new user who has never run this repository before.

The reference is based on the current `--help` output from:

- `bash_template/run_dataset.sh`
- every pipeline step under `bash_template/steps/`
- the Python scripts called by those steps

Run all examples from the project root, which is the parent directory of
`nogse_pipeline/`.

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse --help
```

## Command Pattern

All analysis steps are run through one wrapper:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh <type_subj> <type_seq> <step...>
```

`<type_subj>` chooses the dataset family:

| Value | Meaning | Aliases |
|---|---|---|
| `brain` | brain acquisitions | `brains` |
| `phantom` | phantom acquisitions | `phantoms` |

`<type_seq>` chooses the sequence family:

| Value | Meaning |
|---|---|
| `ogse` | OGSE analysis defaults |
| `nogse` | NOGSE analysis defaults |

The same command can be written with named runner arguments:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh \
  --type-subj brain \
  --type-seq ogse \
  rotate contrast
```

Accepted runner arguments:

| Argument | How to write it | What it does |
|---|---|---|
| `--type-subj` | `--type-subj brain` | Selects `brain` or `phantom`. |
| `--type_subj` | `--type_subj phantom` | Same as `--type-subj`. |
| `--dataset` | `--dataset brain` | Same as `--type-subj`. |
| `--type-seq` | `--type-seq ogse` | Selects `ogse` or `nogse`. |
| `--type_seq` | `--type_seq nogse` | Same as `--type-seq`. |
| `--results-root` | `--results-root Data-signals/Results/20220622_BRAIN` | Adds one Results folder for `ingest`. Repeat for multiple input folders. |
| `-h`, `--help` | `--help` | Shows global help. After a step name, shows help for that step. |

Step-specific settings are environment variables written before the command:

```bash
PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=long \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
```

Extra Python flags are passed through a step-specific `*_EXTRA_ARGS` variable:

```bash
SIGNAL_FIT_EXTRA_ARGS="--fix_M0 1.0 --auto_fit_tol 0.05" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
```

You can run multiple steps in one command:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate contrast fit_signal
```

Or with `PIPELINE_STEPS`:

```bash
PIPELINE_STEPS="rotate contrast fit_signal fit_contrast alpha tc" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse
```

## Defaults

The runner derives paths from `<type_subj>` and `<type_seq>`.

For `brain ogse`, defaults are:

```text
SIGNALS_ROOT    = <project_root>/Data-signals
RESULTS_ROOT    = <project_root>/Data-signals/Results
PARAMS_XLSX     = <project_root>/Data-signals/sequence_parameters_brains.xlsx
ANALYSIS_ROOT   = <project_root>/analysis/brains/ogse_experiments
MASTER_PARQUET  = <project_root>/analysis/brains/ogse_experiments/master.long.parquet
MANIFEST_DIR    = <repo_root>/bash_template/manifests/brains_ogse
```

Common variables:

| Variable | Used by | Meaning |
|---|---|---|
| `PY` | all steps | Python interpreter. Defaults to the active conda Python, `nogse_pipe_env`, or `python3`. |
| `SIGNALS_ROOT` | runner defaults | Root containing `Results/` and sequence parameter workbooks. |
| `RESULTS_ROOT` | `ingest` | One Results root if `RESULTS_ROOTS` is not set. |
| `RESULTS_ROOTS` | `ingest` | Space-separated Results roots. |
| `PARAMS_XLSX` | `ingest` | Sequence-parameter workbook. |
| `ANALYSIS_ROOT` | all steps | Case-specific analysis output root. |
| `MASTER_PARQUET` | most steps | Canonical master table. Some steps read it, others append/update rows in it. |
| `MANIFEST_DIR` | manifest-driven steps | Directory containing `contrasts.csv`, `signal_fits.csv`, and `grad_correction.csv`. |
| `TC_FIT_PARAMS` | `tc` | Required contrast fit-params parquet for the `tc` step. |

## Step Map

| Step | Script | Main output |
|---|---|---|
| `ingest` | `01_ingest_results.sh` | `row_kind=signal` rows in `master.long.parquet` |
| `rotate` | `02_rotate_signals.sh` | `row_kind=signal_rotated`, with `D_proj` |
| `contrast` | `03_make_contrasts.sh` | `row_kind=contrast` |
| `plot_signal` | `04_plot_signals.sh` | signal plots from master |
| `plot_contrast` | `05_plot_contrasts.sh` | contrast plots from master |
| `filter_master_points` | `00_filter_master_points.sh` | filtered master parquet |
| `fit_signal` | `06_fit_signals.sh` | signal fit tables and plots |
| `fit_signal_gradcorr` | `06_fit_signals.sh` | signal fits using embedded gradient correction |
| `fit_contrast` | `07_fit_contrasts.sh` | contrast fit tables and plots |
| `fit_contrast_free` | `07_fit_contrasts.sh` | contrast fits with free-model defaults |
| `fit_contrast_mixed_global` | `07_fit_contrasts.sh` | contrast fits with `mixed_global` |
| `fit_global_signal` | `13_fit_global_signals.sh` | pooled/global signal fits |
| `alpha` | `08_alpha_macro.sh` | `summary_alpha_values.xlsx` |
| `tc` | `09_tc_vs_td.sh` | tc-vs-td fit summaries |
| `grad_correction` | `10_make_grad_correction_table.sh` | embedded `grad_correction_factor*` columns plus audit tables |
| `plot_d0_delta` | `11_plot_D0_vs_Delta_alpha.sh` | D0/Dproj vs Delta plots |
| `plot_monoexp_d` | `12_plot_monoexp_D_vs_time.sh` | monoexp D vs time plots |
| `export_master_xlsx` | `99_export_master_xlsx.sh` | Excel copy of the master table |

## Manifests

Manifests live in:

```text
bash_template/manifests/<type_subj>s_<type_seq>/
```

The runner skips blank lines, comment lines starting with `#`, and the header
line.

### `contrasts.csv`

Format:

```csv
subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2
BRAIN,20220622_BRAIN,ALL,ALL,90,8,4,50,25
```

Each row selects two signal groups from `signal_rotated` rows and appends:

```text
contrast(g) = S(g; N_1, Hz_1) - S(g; N_2, Hz_2)
```

Column meanings:

| Column | Meaning |
|---|---|
| `subj` | Subject label, such as `BRAIN`, `LUDG`, `MBBL`, `PHANTOM`; `ALL` leaves it unconstrained. |
| `sheet` | Session/sheet name stored in master; `ALL` leaves it unconstrained. |
| `roi` | ROI to process; `ALL` processes every matching ROI. |
| `direction` | Direction label such as `long`, `tra`, `x`, `y`, `z`, `1`, `2`, `3`; `ALL` processes all. |
| `td_ms` | Diffusion time in ms. |
| `N_1`, `Hz_1` | Side-1 selector. |
| `N_2`, `Hz_2` | Side-2 selector. |

### `signal_fits.csv`

Format:

```csv
subj,sheet,roi,direction,td_ms,N,Hz,model
BRAIN,20220622_BRAIN,ALL,ALL,90,4,25,monoexp
```

Each row selects signal rows from master and fits one signal model for every
matching ROI/direction group.

Column meanings:

| Column | Meaning |
|---|---|
| `subj`, `sheet`, `roi`, `direction` | Selectors. Use `ALL` to keep the selector unconstrained. |
| `td_ms` | Diffusion time in ms. |
| `N` | Number of oscillations. |
| `Hz` | Oscillation frequency. |
| `model` | Model name. OGSE examples: `monoexp`, `ogse_free`, `ogse_rest`, `ogse_rest_offset`. NOGSE examples: `nogse_free`, `nogse_free_cpmg`, `nogse_mixed_global`. |

If `model` is empty, the step uses `SIGNAL_FIT_MODEL`.

### `grad_correction.csv`

Format:

```csv
subj,sheet,roi,direction,td_ms,N,Hz,model
BRAIN,20230619_BRAIN-3,Syringe,long,120,8,40,monoexp
```

Each row identifies one reference curve, usually `Syringe` for brain data or
`water` for phantom data. The step fits the selected curve twice:

- NOGSE free fit using `--gbase` (default `g_lin_max`) -> `D0_nogse`
- monoexp fit using `--bbase` (default `bvalue_thorsten`) -> `D0_monoexp`

Then it computes:

```text
correction_factor = sqrt(D0_nogse / D0_monoexp)
```

The factor is written into `MASTER_PARQUET` for all ROIs with the same
`subj`, `sheet`, `direction`, `td_ms`, and `N`.

## Step Reference

### `ingest`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
```

With one explicit Results folder:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
  --results-root Data-signals/Results/20220622_BRAIN \
  ingest
```

What it does:

Reads `*_results.xlsx`, combines each workbook with the sequence-parameter
workbook, writes per-file long tables, and appends `row_kind=signal` rows to
`MASTER_PARQUET`.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `RESULTS_ROOT` | `$SIGNALS_ROOT/Results` | Input Results root when `RESULTS_ROOTS` is unset. |
| `RESULTS_ROOTS` | empty | Space-separated list of Results roots. |
| `RESULTS_GLOB` | `*_results.xlsx` | File pattern inside each Results root. |
| `PARAMS_XLSX` | dataset workbook | Sequence-parameter workbook. |
| `MASTER_PARQUET` | `$ANALYSIS_ROOT/master.long.parquet` | Master table to append. |
| `PROCESS_SCRIPT` | `scripts/data/process_one_results.py` | Python script override. |
| `PROCESS_OUT_ROOT` | `$ANALYSIS_ROOT/data/tables` | Per-file long table output root. |

Python CLI called by this step:

```bash
python scripts/data/process_one_results.py RESULTS_FILE PARAMS_XLSX \
  --out_dir OUT_DIR \
  --master-parquet MASTER_PARQUET
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `results_file` | One `*_results.xlsx` workbook to ingest. |
| `params_xlsx` | Sequence-parameter workbook. |
| `--out_dir DIR` | Directory for per-file parquet/xlsx tables. |
| `--gamma VALUE` | Gyromagnetic ratio in `1/(ms*mT)`. |
| `--oneg` | Treat one-g-per-sequence result files as points in one direct-g curve. |
| `--strip-output-token TOKEN` | Remove a token from generated output stems. Repeatable; alias `--output-stem-strip`. |
| `--master-parquet PATH` | Append processed signal rows to this master table. |

### `filter_master_points`

Run:

```bash
MASTER_FIRST_POINTS_BY_TD="120=8,210=6,90=ALL" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse filter_master_points
```

What it does:

Writes a new master parquet containing only the first requested number of
`b_step` values for each `td_ms`. It does not modify the original master table.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `MASTER_FIRST_POINTS_BY_TD` | required | Rules such as `120=8,210=6,90=ALL`. |
| `FILTERED_MASTER_PARQUET` | `$ANALYSIS_ROOT/master.first_points.long.parquet` | Output path. Alias: `MASTER_FIRST_POINTS_PARQUET`. |
| `MASTER_FIRST_POINTS_DIR` | `$ANALYSIS_ROOT` | Output directory override. |
| `FILTER_MASTER_SCRIPT` | `scripts/data/filter_master_table.py` | Python script override. |

Use the filtered table in later steps:

```bash
MASTER_PARQUET=analysis/brains/ogse_experiments/master.first_points.long.parquet \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate contrast fit_signal
```

Python CLI:

```bash
python scripts/data/filter_master_table.py MASTER_PARQUET \
  --out-parquet OUT_PARQUET \
  --first-points-by-td "120=8,210=6"
```

### `rotate`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate
```

What it does:

Selects `row_kind=signal` rows from the master, rotates tensor directions,
computes `D_proj`, appends `row_kind=signal_rotated`, and updates original
signal rows with `D_proj`.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input/output master table. |
| `MASTER_SUBJ` | empty | Optional `subj` selector. |
| `MASTER_SHEET` | empty | Optional `sheet` selector. |
| `DIRS_TXT` | `assets/dirs/dirs_6.txt` | Direction table. |
| `ROTATE_SCRIPT` | `scripts/data/rotate_ogse_tensor.py` | Python script override. |
| `ROTATED_OUT_ROOT` | `$ANALYSIS_ROOT/data-rotated/tables` | Legacy output root. |
| `ROTATE_EXTRA_ARGS` | empty | Extra Python flags. |

Examples:

```bash
MASTER_SUBJ=BRAIN MASTER_SHEET=20220622_BRAIN \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate
```

```bash
ROTATE_EXTRA_ARGS="--s0_mode mean --no-legacy-output" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate
```

Python CLI:

```bash
python scripts/data/rotate_ogse_tensor.py \
  --master-parquet MASTER_PARQUET \
  --row-kind signal \
  --dirs_txt assets/dirs/dirs_6.txt \
  --out_dir OUT_DIR
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `long_parquet` | Legacy positional input, used when not reading from master. |
| `--master-parquet PATH` | Read signal rows from master and write rotated rows back. |
| `--row-kind KIND` | Master row kind selector. |
| `--analysis-id`, `--subj`, `--sheet`, `--roi`, `--direction`, `--source-file` | Master selectors. Can be repeated or comma-separated. |
| `--stat`, `--td_ms`, `--N`, `--Hz` | Numeric/stat selectors. |
| `--no-legacy-output` | Skip legacy rotated parquet/xlsx output files. |
| `--solver lstsq|solve` | Tensor solve method. |
| `--s0_mode dir1|mean` | How to estimate S0 for tensor rotation. |
| `--b_col COL` | B-value column used by the rotation. |
| `--dirs_txt PATH` | Direction matrix, no header. Defaults from number of directions. |

### `contrast`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast
```

What it does:

Reads `CONTRAST_MANIFEST`, selects two signal groups from `signal_rotated` rows,
subtracts them, writes contrast tables, and appends `row_kind=contrast` rows.
By default this is a direct point-by-point subtraction. Fitted/resampled
contrasts are also built here, not in `fit_contrast`.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `CONTRAST_MANIFEST` | `$MANIFEST_DIR/contrasts.csv` | Contrast manifest. |
| `MASTER_PARQUET` | current master | Input/output master table. |
| `MAKE_CONTRAST_SCRIPT` | `scripts/data/make_contrast.py` | Python script override. |
| `CONTRAST_OUT_ROOT` | `$ANALYSIS_ROOT/contrast-data-master` | Per-contrast output root. |
| `MAKE_CONTRAST_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/data/make_contrast.py \
  --master-parquet MASTER_PARQUET \
  --append-master \
  --subj BRAIN \
  --sheet 20220622_BRAIN \
  --td_ms 90 \
  --N_1 8 --Hz_1 50 \
  --N_2 4 --Hz_2 25 \
  --out_root OUT_ROOT
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `ref_parquet`, `cmp_parquet` | Legacy side-1 and side-2 input tables. |
| `--master-parquet PATH` | Select both sides from master. |
| `--append-master` | Append contrast rows back to master. |
| `--master-rotated` / `--no-master-rotated` | Use `signal_rotated` rows by default, or raw `signal` rows. |
| `--subj`, `--sheet`, `--roi`, `--direction`, `--stat`, `--source-file` | Shared selectors for both sides. |
| `--N_1`, `--Hz_1`, `--g_1` | Side-1 selectors. |
| `--N_2`, `--Hz_2`, `--g_2` | Side-2 selectors. |
| `--g_pair_col COL` | Column used by `--g_1` and `--g_2`. |
| `--td_ms VALUE` | Optional td_ms selector for master table rows. |

### `plot_signal`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
```

What it does:

Plots signal curves from master rows. The wrapper chooses the OGSE or NOGSE
plotting script based on `TYPE_SEQ`.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `PLOT_SIGNAL_SCRIPT` | sequence-specific | Python script override. |
| `PLOT_OUT_ROOT` | `$ANALYSIS_ROOT/plots-master/signal` | Output root. |
| `PLOT_ROW_KIND` | `signal_rotated` | `signal_rotated` or `signal`. |
| `PLOT_SUBJ`, `PLOT_SHEET`, `PLOT_ROI`, `PLOT_DIRECTION` | empty | Optional selectors. |
| `PLOT_TD_MS`, `PLOT_N` | empty | Optional numeric selectors. |
| `PLOT_SIGNAL_YCOL` | `value_norm` | Signal column. |
| `PLOT_SIGNAL_XCOL` | OGSE: `g_thorsten`; NOGSE: `g` | X-axis column. Alias: `PLOT_SIGNAL_G_TYPE`. |
| `PLOT_STAT` | `avg` | Statistic row to plot. |
| `PLOT_SIGNAL_EXTRA_ARGS` | empty | Extra Python flags. |

Examples:

```bash
PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=long \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
```

Python CLI:

```bash
python scripts/plotting/plot_ogse_signal_vs_g.py \
  --master-parquet MASTER_PARQUET \
  --row-kind signal_rotated \
  --xcol g_thorsten \
  --ycol value_norm \
  --out_root OUT_ROOT
```

Common Python arguments:

| Argument | Meaning |
|---|---|
| `--master-parquet PATH` | Read signal rows from master. |
| `--row-kind signal|signal_rotated` | Row kind to plot. |
| `--analysis-id`, `--subj`, `--sheet`, `--roi`, `--direction`, `--td_ms`, `--N`, `--Hz` | Selectors. |
| `--out_root DIR` | Plot output root. OGSE also accepts `--out_dir`. |
| `--xcol COL` | X-axis column. |
| `--ycol COL` | Y-axis column. OGSE also accepts `--y_col`. |
| `--stat STAT` | Statistic row to plot. |
| `--no_ylim` | OGSE only; disables default y-axis limits. |
| `--rois`, `--directions` | NOGSE only; optional subset selectors for extra plots. |

### `plot_contrast`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast
```

What it does:

Plots contrast curves from `row_kind=contrast` rows in master.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `PLOT_CONTRAST_SCRIPT` | sequence-specific | Python script override. |
| `PLOT_OUT_ROOT` | `$ANALYSIS_ROOT/plots-master/contrast` | Output root. |
| `PLOT_SUBJ`, `PLOT_SHEET`, `PLOT_ROI`, `PLOT_DIRECTION` | empty | Optional selectors. |
| `PLOT_TD_MS`, `PLOT_N1`, `PLOT_N2` | empty | Optional numeric selectors. |
| `PLOT_CONTRAST_YCOL` | `value_norm` | Contrast column. |
| `PLOT_CONTRAST_XCOL` | `g_thorsten_1` | X-axis column. |
| `PLOT_STAT` | `avg` | Statistic row. |
| `PLOT_CONTRAST_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/plotting/plot_ogse_contrast_vs_g.py \
  --master-parquet MASTER_PARQUET \
  --xcol g_thorsten_1 \
  --y value_norm \
  --out_root OUT_ROOT
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `contrast_parquet` | Legacy positional contrast table. |
| `--master-parquet PATH` | Read contrast rows from master. |
| `--analysis-id`, `--subj`, `--sheet`, `--roi`, `--direction`, `--td_ms`, `--N_1`, `--N_2`, `--Hz_1`, `--Hz_2` | Selectors. |
| `--xcol COL` | X-axis column, for example `g_lin_max_1`, `g_max_1`, `g_thorsten_1`. |
| `--y`, `--ycol` | Contrast column, usually `value` or `value_norm`. |
| `--out_root DIR` | Output root. |
| `--exp NAME` | Optional experiment folder name. |
| `--directions`, `--dirs`, `--axes` | Directions to plot. |
| `--stat STAT` | Statistic row. Use `ALL` to skip filtering. |
| `--rois` | Optional ROI subset for extra plots. |

### `fit_signal` and `fit_signal_gradcorr`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
```

Gradient-corrected run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr
```

What it does:

Reads `SIGNAL_FIT_MANIFEST`, selects signal groups from master, and fits the
model listed in each manifest row. `fit_signal_gradcorr` adds
`--apply_grad_corr`; the fitters then read embedded `grad_correction_factor`
values from selected master rows.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `FIT_SIGNAL_SCRIPT` | sequence-specific | Python script override. |
| `SIGNAL_FIT_MANIFEST` | `$MANIFEST_DIR/signal_fits.csv` | Signal-fit manifest. |
| `SIGNAL_FIT_OUT_ROOT` | derived | Output root. |
| `SIGNAL_FIT_MODEL` | OGSE: `monoexp`; NOGSE: `nogse_free` | Fallback model when manifest `model` is empty. |
| `SIGNAL_FIT_G_TYPE` | OGSE: `bvalue_thorsten`; NOGSE: `g` | Gradient/b-value axis. |
| `SIGNAL_FIT_XCOL` | `SIGNAL_FIT_G_TYPE` | NOGSE x-axis override. |
| `SIGNAL_FIT_YCOL` | `value_norm` | Signal column. |
| `SIGNAL_FIT_EXTRA_ARGS` | empty | Extra Python flags. |

Derived output root:

```text
$ANALYSIS_ROOT/fits/<master_name>/<type_seq>_<ycol>_vs_<gtype>_<model>
```

OGSE Python CLI:

```bash
python scripts/fitting/fit_ogse_signal_vs_g.py \
  --master-parquet MASTER_PARQUET \
  --row-kind signal_rotated \
  --model monoexp \
  --out_root OUT_ROOT \
  --ycol value_norm \
  --g_type bvalue_thorsten \
  --auto_fit_points
```

NOGSE Python CLI:

```bash
python scripts/fitting/fit_nogse_signal_vs_g.py \
  --master-parquet MASTER_PARQUET \
  --row-kind signal_rotated \
  --model nogse_free \
  --out_root OUT_ROOT \
  --xcol g \
  --ycol value_norm
```

Common fitter arguments:

| Argument | Meaning |
|---|---|
| positional parquet(s) | Legacy input signal table(s), used when not reading from master. |
| `--master-parquet PATH` | Read input rows from master. |
| `--row-kind KIND` | Row kind selector. |
| `--analysis-id`, `--subj`, `--sheet`, `--roi`, `--direction`, `--source-file`, `--td_ms`, `--N`, `--Hz` | Selectors. |
| `--model MODEL` | Model to fit. |
| `--out_root DIR` | Output directory. |
| `--ycol COL` | Signal column to fit. |
| `--stat STAT` | Statistic row to fit. |
| `--apply_grad_corr` / `--no_grad_corr` | Use or disable embedded correction factors. |
| `--param-mode PARAM=MODE` | Unified parameter mode: `fixed`, `free`, `global_td`, or `global_contrast`. Repeatable. |
| `--param-init PARAM=VALUE` | Initial value. Repeatable. |
| `--param-fixed PARAM=VALUE` | Fixed value; implies fixed mode if no mode is set. Repeatable. |
| `--param-bounds PARAM=LOW:HIGH` | Bounds. Repeatable. |
| `--append-fit-params-to-master` | Also append fit parameter rows into `MASTER_PARQUET` as `row_kind=fit_params`. |

OGSE-only signal arguments:

| Argument | Meaning |
|---|---|
| `--g_type COL` | Fit axis: `bvalue`, `g`, `bvalue_g`, `g_max`, `g_lin_max`, `bvalue_g_lin_max`, `g_thorsten`, `bvalue_thorsten`. |
| `--plot_xcol COL` | Separate plot x-axis. |
| `--fit_points N` / `--auto_fit_points` | Fixed or automatic number of leading points. |
| `--auto_fit_tol`, `--auto_fit_err_floor`, `--auto_fit_min_points`, `--auto_fit_max_points` | Automatic point-selection controls. |
| `--gamma`, `--delta_ms`, `--Delta_app_ms`, `--D0_init` | Physical/model controls. |
| `--fix_M0 VALUE` / `--free_M0` | Fix or fit M0. |
| `--out_dproj_root DIR` | Optional synthetic Dproj output root from fitted reference D0 values. |

NOGSE-only signal arguments:

| Argument | Meaning |
|---|---|
| `--xcol COL` | Fit x-axis: `bvalue`, `bvalue_g`, `bvalue_g_lin_max`, `bvalue_thorsten`, `g`, `g_lin_max`, `g_max`, `g_thorsten`. |
| `--plot_xcol COL` | Plot x-axis. |
| `--fix_M0 VALUE` / `--free_M0` | Fix or fit M0. |
| `--fix_D0 VALUE` / `--free_D0` | Fix or fit D0. |
| `--M0_bounds MIN MAX`, `--D0_bounds MIN MAX` | Parameter bounds. |
| `--tc_init VALUE`, `--tc_bounds MIN MAX` | `mixed_global` tc controls. |
| `--alpha_table PATH`, `--alpha_col COL`, `--alpha_td_col COL`, `--alpha_td_tol_ms VALUE` | Fixed alpha lookup table controls. |
| `--no_plots` | Skip fit plots. |

### `fit_contrast`, `fit_contrast_free`, `fit_contrast_mixed_global`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
```

What it does:

Fits all selected `row_kind=contrast` rows from master. `fit_contrast_free`
sets the free-model defaults. `fit_contrast_mixed_global` sets
`FIT_MODEL=mixed_global`.

`fit_contrast_free` and `fit_contrast_mixed_global` are convenience presets over
the same `fit_contrast` script:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
FIT_MODEL=ogse_free bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast
```

Those two commands are equivalent as long as `FIT_MODEL` resolves to the same
model and all other variables (`FIT_GBASE`, `FIT_YCOL`, `FIT_STAT`,
`FIT_EXTRA_ARGS`, `FIT_OUT_ROOT`) are also the same.

Resampled contrasts are handled by step `contrast_resampled` (07b). Run that step
first with signal fits from step `fit_signal` or `fit_global_signal`, then run
any `fit_contrast*` command on the master contrast rows.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `FIT_CONTRAST_SCRIPT` | sequence-specific | Python script override. |
| `FIT_OUT_ROOT` | derived | Output root. |
| `FIT_MODEL` | OGSE: `ogse_free`; NOGSE: `nogse_free` | Contrast model. |
| `FIT_GBASE` | `g_lin_max` | Gradient axis base. |
| `FIT_YCOL` | `value_norm` | Contrast column. |
| `FIT_STAT` | `avg` | Statistic row. |
| `FIT_EXTRA_ARGS` | empty | Extra Python flags. |

Derived output root:

```text
$ANALYSIS_ROOT/fits/<master_name>/<type_seq>_<ycol>_vs_<gtype>_<model>
```

Python CLI:

```bash
python scripts/fitting/fit_ogse_contrast_vs_g.py \
  --master-parquet MASTER_PARQUET \
  --model ogse_free \
  --out_root OUT_ROOT \
  --gbase g_lin_max \
  --ycol value_norm
```

Common contrast fitter arguments:

| Argument | Meaning |
|---|---|
| positional `contrast_parquet` | Legacy input contrast table(s). |
| `--master-parquet PATH` | Read contrast rows from master. |
| `--row-kind KIND` | Row kind selector. |
| `--analysis-id`, `--subj`, `--sheet`, `--source-file`, `--td_ms`, `--Hz` | Selectors. |
| `--model MODEL` | Model to fit. |
| `--gbase COL` | Base gradient/b-value column. |
| `--plot_xcol COL` | Plot x-axis column. |
| `--ycol COL` | Contrast column, usually `value` or `value_norm`. |
| `--directions`, `--direction` | Direction filter. |
| `--subjs`, `--rois` | Subject and ROI filters; use `ALL` to keep all. |
| `--stat STAT` | Statistic filter; use `ALL` to skip. |
| `--oneg` | Allow one-g-per-sequence contrast tables with sequence ranges. |
| `--out_root DIR` | Output root. |
| `--no_plots` | Skip plots. |
| `--apply_grad_corr` / `--no_grad_corr` | Use embedded `grad_correction_factor_1/2` columns. |
| `--corr_td_ms VALUE` | Optional td override retained for compatibility. |
| `--param-mode`, `--param-init`, `--param-fixed`, `--param-bounds` | Unified parameter controls. |
| `--append-fit-params-to-master` | Also append fit params as `row_kind=fit_params`. |
| `--n_fit N` | Use first N points after sorting by x. |
| `--peak_grid_n N` | Number of points for fitted peak search. |
| `--peak_D0_fix VALUE`, `--peak_gamma VALUE` | Convert peak gradient to `tc_peak_ms`. |

OGSE contrast model and parameter flags:

| Argument | Meaning |
|---|---|
| `--fix_M0 VALUE` / `--free_M0 [VALUE]` | Fix or fit M0. |
| `--fix_D0 VALUE` / `--free_D0 [VALUE]` | Fix or fit D0 in m2/ms. |
| `--fix_tc VALUE` / `--free_tc [VALUE]` / `--tc_init VALUE` | tc controls in ms. |
| `--fix_C VALUE` / `--free_C [VALUE]` | Offset parameter for rest-offset models. |
| `--tc_bounds`, `--M0_bounds`, `--D0_bounds`, `--C_bounds` | Bounds. Hyphen aliases are accepted. |
| `--alpha_table`, `--alpha_col`, `--alpha_td_col`, `--alpha_td_tol_ms` | Fixed-alpha table for `mixed_global`. |
| `--global_params PARAM...` | Jointly fit all td curves per subject/ROI/direction sharing listed parameters. |
| `--peak_g_max_mTm VALUE` | Raw gradient max for fitted peak search. |
| `--peak_resample_gradient` | Also compute `tc_peak_resampled_ms`. |
| `--peak_resample_g_max_corr_mTm VALUE` | Corrected common-gradient max for resampled peak. |

NOGSE contrast-only flags:

| Argument | Meaning |
|---|---|
| `--fix_g0 VALUE` / `--free_g0 [VALUE]` | Offset-gradient parameter for `nogse_free_grad_offset`. |
| `--g0_bounds MIN MAX` | Bounds for g0 in mT/m. |

### `fit_global_signal`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal
```

What it does:

Fits global/mixed signal models directly on raw signal rows. Parameters can be
fixed, free per curve, shared per `td_ms`, or shared per
`(td_ms, roi, direction)` contrast.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `FIT_GLOBAL_SIGNAL_SCRIPT` | `scripts/fitting/fit_global_signal.py` | Python script override. |
| `GLOBAL_SIGNAL_OUT_ROOT` | `$ANALYSIS_ROOT/fits/<type_seq>_signal_<model>` | Output root. |
| `GLOBAL_SIGNAL_ROW_KIND` | `signal_rotated` | Row kind to fit. |
| `GLOBAL_SIGNAL_MODEL` | OGSE: `ogse_mixed_offset`; NOGSE: `nogse_mixed_offset` | Model. |
| `GLOBAL_SIGNAL_YCOL` | `value` | Signal column. |
| `GLOBAL_SIGNAL_G_TYPE` | OGSE brain: `g_thorsten`; OGSE phantom/NOGSE: `g` | Gradient axis. |
| `GLOBAL_SIGNAL_STAT` | `avg` | Statistic row. |
| `GLOBAL_SIGNAL_MIN_POINTS` | `4` | Minimum points per curve. |
| `GLOBAL_SIGNAL_DIRECTIONS` | OGSE brain: `long tra`; others: `ALL` | Direction subset. |
| `GLOBAL_SIGNAL_ROIS`, `GLOBAL_SIGNAL_SUBJS` | `ALL` | ROI and subject subsets. |
| `GLOBAL_SIGNAL_APPLY_GRAD_CORR` | `true` | Use embedded correction factors. |
| `GLOBAL_SIGNAL_EXTRA_ARGS` | empty | Extra Python flags. |

Parameter mode variables:

| Variable pair | Default mode | Meaning |
|---|---|---|
| `GLOBAL_SIGNAL_TC_MODE`, `GLOBAL_SIGNAL_TC_FIXED` | `global_td` | tc mode/value. |
| `GLOBAL_SIGNAL_ALPHA_MODE`, `GLOBAL_SIGNAL_ALPHA_FIXED` | `global_td` | alpha mode/value. |
| `GLOBAL_SIGNAL_RN_MODE`, `GLOBAL_SIGNAL_RN_FIXED` | `global_td` | RN mode/value. |
| `GLOBAL_SIGNAL_M0_MODE`, `GLOBAL_SIGNAL_M0_FIXED` | `global_contrast` | M0 mode/value. |
| `GLOBAL_SIGNAL_C_MODE`, `GLOBAL_SIGNAL_C_FIXED` | `global_contrast` | offset C mode/value. |
| `GLOBAL_SIGNAL_D0_MODE`, `GLOBAL_SIGNAL_D0_FIXED` | `fixed`; brain `3.2e-12`, phantom `2.3e-12` | D0 mode/value in m2/ms. |

Example:

```bash
GLOBAL_SIGNAL_RN_MODE=fixed GLOBAL_SIGNAL_RN_FIXED=10 \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal
```

Python CLI:

```bash
python scripts/fitting/fit_global_signal.py \
  --master-parquet MASTER_PARQUET \
  --row-kind signal_rotated \
  --out_root OUT_ROOT \
  --type-seq ogse \
  --model ogse_mixed_offset
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `--master-parquet PATH` | Master table containing signal rows. |
| `--row-kind KIND` | Row kind to fit. |
| `--out_root DIR` | Output root. |
| `--type-seq`, `--type_seq`, `--family` | `ogse` or `nogse`. |
| `--model MODEL` | One of `ogse_free`, `ogse_rest`, `ogse_rest_offset`, `ogse_mixed`, `ogse_mixed_offset`, `nogse_free`, `nogse_free_offset`, `nogse_rest`, `nogse_rest_offset`, `nogse_mixed`, `nogse_mixed_offset`. |
| `--ycol`, `--g_type` | Signal column and gradient axis. |
| `--subjs`, `--sheets`, `--td_ms`, `--N`, `--Hz`, `--directions`, `--rois`, `--stat` | Selectors. |
| `--n_fit N` | Use first N points after sorting by G. |
| `--min_points N` | Minimum points per group. |
| `--tc_mode`, `--alpha_mode`, `--RN_mode`, `--M0_mode`, `--C_mode`, `--D0_mode` | Parameter sharing modes. |
| `--D0_fixed`, `--tc_init`, `--tc_fixed`, `--alpha_init`, `--alpha_fixed`, `--M0_init`, `--M0_fixed`, `--C_init`, `--C_fixed`, `--RN_init`, `--RN_fixed` | Parameter values/seeds. |
| `--tc_bounds`, `--alpha_bounds`, `--M0_bounds`, `--C_bounds`, `--RN_bounds`, `--D0_bounds` | Bounds. |
| `--max_nfev N` | Maximum optimizer evaluations. |
| `--apply_grad_corr` / `--no_grad_corr` | Use or disable embedded correction factors. |
| `--corr_missing error|identity|skip` | What to do when correction is missing. |
| `--no_plots`, `--write_csv`, `--write_signal_tables` | Output controls. |

### `grad_correction`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction
```

What it does:

Reads `GRAD_CORR_MANIFEST`, fits each reference curve with NOGSE free and
monoexp models, writes audit tables, and updates `MASTER_PARQUET` in place with
gradient-correction columns.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `GRAD_CORR_SCRIPT` | `scripts/data/make_grad_correction_table.py` | Python script override. |
| `GRAD_CORR_MANIFEST` | `$MANIFEST_DIR/grad_correction.csv` | Reference-curve manifest. |
| `MASTER_PARQUET` | current master | Input/output master table. |
| `GRAD_CORR_OUT_DIR` | `$ANALYSIS_ROOT/fits/grad_correction` | Audit output directory. |
| `GRAD_CORR_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/data/make_grad_correction_table.py \
  --manifest MANIFEST \
  --master-parquet MASTER_PARQUET \
  --out-xlsx OUT.xlsx \
  --out-csv OUT.csv
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `--manifest PATH` | CSV with `subj,sheet,roi,direction,td_ms,N[,Hz,model]`. |
| `--master-parquet PATH` | Master table to read and update. |
| `--out-xlsx PATH` | Audit workbook path. |
| `--out-csv PATH` | Optional audit CSV. |
| `--stat STAT` | Statistic row to use; default `avg`. |
| `--row-kind KIND` | Input row kind; default `signal_rotated`. |
| `--gbase COL` | Gradient column for NOGSE free fit; default `g_lin_max`. |
| `--bbase COL` | B-value column for monoexp fit; default `bvalue_thorsten`. |
| `--D0-init VALUE` | NOGSE free D0 seed in m2/ms; default `2.3e-12`. |
| `--tol-ms VALUE` | td matching tolerance; default `1e-3`. |
| `--fix-M0 VALUE` | Fix M0 in both fits; default `1.0`. |
| `--free-M0 [VALUE]` | Fit M0 in both fits with optional seed. |

After this step:

- signal rows use `grad_correction_factor`
- contrast rows use `grad_correction_factor_1` and `grad_correction_factor_2`
- corrected fit steps only need `--apply_grad_corr`; they no longer need a
  correction xlsx path

### `alpha`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha
```

What it does:

Computes alpha summaries from `D_proj` values in `signal_rotated` rows.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input master table. |
| `ALPHA_MACRO_SCRIPT` | `scripts/summary/make_alpha_macro_summary.py` | Python script override. |
| `ALPHA_N` | `1` | N selector. |
| `ALPHA_OUT_DIR` | `$ANALYSIS_ROOT/alpha_macro/master` | Output directory. |
| `ALPHA_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/summary/make_alpha_macro_summary.py \
  --master-parquet MASTER_PARQUET \
  --no-master-fit-params \
  --N 1 \
  --out-summary summary_alpha_values.xlsx \
  --out-avg D_vs_delta_app.combined.xlsx
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `--combined-table PATH` | Use a combined table produced by `plot_D0_vs_Delta.py`. |
| `--master-parquet PATH` | Read Dproj values from master. |
| `--no-master-fit-params` | Do not append alpha rows to a fit-params table. |
| `--dproj-root DIR`, `--pattern GLOB` | Legacy Dproj table loading. |
| `--analysis-id`, `--subjs`, `--sheets`, `--rois`, `--dirs` | Selectors. |
| `--N N` / `--Hz HZ` | Select by N or Hz. |
| `--bvalue-decimals N` | Round b-values before grouping. |
| `--bvalmax N` | Use the N-th b-value, 1-based after ascending sort. |
| `--roi-bvalmax ROI=N` | Per-ROI b-value override. Repeatable. |
| `--reference-D0`, `--reference-D0-error` | Reference D0 and uncertainty. |
| `--direction-alias raw=grouped` | Direction grouping alias. Repeatable. |
| `--out-summary`, `--out-avg`, `--out-plot` | Output files. |
| `--plot-rois`, `--plot-directions` | Plot subsets. |

### `tc`

Run:

```bash
TC_FIT_PARAMS=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_glinmax_ogse_free/fit_params.ogse_free.glinmax.value_norm.direction_ALL.parquet \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc
```

What it does:

Fits `tc(Td)` summaries from contrast fit-params. `TC_FIT_PARAMS` is required
and should point to the accumulated contrast fit-params parquet produced by
`fit_contrast`.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `TC_FIT_PARAMS` | required | Contrast fit-params parquet. |
| `TC_VS_TD_SCRIPT` | `scripts/fitting/run_tc_vs_td.py` | Python script override. |
| `TC_OUT_DIR` | `$ANALYSIS_ROOT/fits/tc_vs_td_master` | Output root. |
| `TC_METHOD` | `pseudohuber_fixed_macro` | Fit method. |
| `TC_Y_COL` | `tc_peak_ms` | y column fitted vs `td_ms`. |
| `TC_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/fitting/run_tc_vs_td.py \
  --master-fit-params TC_FIT_PARAMS \
  --method pseudohuber_fixed_macro \
  --y-col tc_peak_ms \
  --out-dir OUT_DIR
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `--method linear|pseudohuber_fixed_macro|pseudohuber_free` | tc-vs-td model. |
| `--master-fit-params PATH` | Cumulative fit-params table. |
| `--master-parquet PATH` | Alternative source with `row_kind=fit_params`. |
| `--fits PATH...`, `--pattern GLOB` | Load fit_params files or roots directly. |
| `--only-fitresamp` | Use only fitted/resampled contrast fits. |
| `--models`, `--subjs`, `--directions`, `--rois` | Filters. |
| `--y-col COL` | Column to fit, e.g. `tc_peak_ms`, `tc_peak_resampled_ms`, `tc_fit_ms`. |
| `--add-resampled-data-peaks`, `--contrast-root`, `--peak-D0-fix`, `--peak-gamma` | Add peaks from resampled contrast tables. |
| `--exclude-td-ms`, `--exclude-match` | Exclude td values or specific rows. |
| `--no-errorbars`, `--td-min-ms`, `--td-max-ms` | Plot controls. |
| `--c-fixed`, `--c-min`, `--c-max`, `--delta-fixed`, `--delta-min`, `--delta-max` | Pseudohuber parameter controls. |
| `--alpha-macro-fixed`, `--alpha-macro-min`, `--alpha-macro-max`, `--summary-alpha` | Alpha controls. |
| `--include-failed` | Include `ok=False` rows. |
| `--out-dir DIR` | Output directory. |

### `plot_d0_delta`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_d0_delta
```

What it does:

Plots `D_proj`-derived D0/Dproj curves versus `Delta_app_ms` and can reuse
`summary_alpha_values.xlsx`.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Must contain `signal_rotated` rows with `D_proj`. |
| `PLOT_D0_SCRIPT` | `scripts/plotting/plot_D0_vs_Delta.py` | Python script override. |
| `ALPHA_OUT_DIR` | `$ANALYSIS_ROOT/alpha_macro/master` | Output directory. |
| `SUMMARY_ALPHA` | empty | Optional summary alpha file. |
| `DPROJ_N`, `DPROJ_HZ` | empty | N or Hz selector. |
| `DPROJ_ROIS`, `DPROJ_DIRS` | empty | Space-separated subsets. |
| `PLOT_D0_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/plotting/plot_D0_vs_Delta.py \
  --master-parquet MASTER_PARQUET \
  --out-dir OUT_DIR
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `--dproj-root DIR`, `--pattern GLOB` | Legacy Dproj table loading. |
| `--master-parquet PATH` | Read Dproj values from master. |
| `--analysis-id`, `--sheets`, `--subjs`, `--rois`, `--dirs` | Selectors. |
| `--out-dir DIR` | Output folder. |
| `--N N` / `--Hz HZ` | Select by N or Hz. |
| `--bvalue-decimals N`, `--bvalmax N` | Grouping and b-step controls. |
| `--reference-D0`, `--reference-D0-error` | Plot annotation values. |
| `--summary-alpha PATH` | Optional alpha summary table. |

### `plot_monoexp_d`

Run:

```bash
SIGNAL_FITS_ROOT=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_monoexp \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d
```

What it does:

Builds monoexp D vs `td_ms` and `Delta_app_ms` plots from signal fit outputs.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `PLOT_MONOEXP_D_SCRIPT` | `scripts/plotting/plot_monoexp_D_vs_time.py` | Python script override. |
| `SIGNAL_FITS_ROOT` | `$ANALYSIS_ROOT/fits` | Root scanned for signal fit parquets. |
| `MONOEXP_D_OUT_DIR` | `$ANALYSIS_ROOT/plots-master/monoexp_D_vs_time` | Output directory. |
| `PLOT_MONOEXP_D_EXTRA_ARGS` | empty | Extra Python flags. |

Python CLI:

```bash
python scripts/plotting/plot_monoexp_D_vs_time.py \
  --fits-root FITS_ROOT \
  --out-dir OUT_DIR
```

Important Python arguments:

| Argument | Meaning |
|---|---|
| `--fits-root DIR` | Root folder with `fit_params*.parquet` files. |
| `--out-dir DIR` | Output folder. |
| `--pattern GLOB` | Relative glob inside fits root. |
| `--subjs`, `--rois`, `--dirs`, `--Ns` | Filters. |
| `--stat STAT`, `--ycol COL` | Keep only matching fit rows. |

### `export_master_xlsx`

Run:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse export_master_xlsx
```

What it does:

Exports a master parquet to an Excel workbook for inspection. It does not
modify the parquet file.

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `MASTER_PARQUET` | current master | Input table. |
| `MASTER_XLSX` | `MASTER_PARQUET` with `.xlsx` suffix | Output workbook. |
| `EXPORT_MASTER_ROW_KIND` | empty | Optional row-kind filter; space- or comma-separated. |
| `EXPORT_MASTER_HEAD` | empty | Optional number of first rows to export. |
| `EXPORT_MASTER_SCRIPT` | `scripts/data/export_master_table.py` | Python script override. |

Python CLI:

```bash
python scripts/data/export_master_table.py MASTER_PARQUET \
  --out-xlsx MASTER.xlsx \
  --row-kind signal \
  --head 100
```

## Practical Recipes

Full OGSE brain core pipeline:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate contrast
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal plot_contrast
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
```

Build correction factors, then run corrected fits:

```bash
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction
bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr
FIT_EXTRA_ARGS="--apply_grad_corr" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free
```

Fit `tc(Td)` with fixed alpha from the alpha summary:

```bash
TC_FIT_PARAMS=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_glinmax_ogse_free/fit_params.ogse_free.glinmax.value_norm.direction_ALL.parquet \
TC_METHOD=pseudohuber_fixed_macro \
TC_EXTRA_ARGS="--summary-alpha analysis/brains/ogse_experiments/alpha_macro/master/summary_alpha_values.xlsx" \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc
```

Export a small Excel preview:

```bash
EXPORT_MASTER_ROW_KIND="signal signal_rotated" EXPORT_MASTER_HEAD=500 \
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse export_master_xlsx
```
