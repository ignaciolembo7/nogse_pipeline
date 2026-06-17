#!/usr/bin/env bash

pipeline_setup_common() {
    MASTER_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ "$(basename "$(dirname "$MASTER_HELPER_DIR")")" == "manifests" ]]; then
        TEMPLATE_ROOT="$(cd "$MASTER_HELPER_DIR/../.." && pwd)"
    else
        TEMPLATE_ROOT="$(cd "$MASTER_HELPER_DIR/.." && pwd)"
    fi
    REPO_ROOT="$(cd "$TEMPLATE_ROOT/.." && pwd)"
    PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

    export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

    local default_py="python"
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        default_py="${CONDA_PREFIX}/bin/python"
    elif [[ -x "$HOME/.conda/envs/nogse_pipe_env/bin/python" ]]; then
        default_py="$HOME/.conda/envs/nogse_pipe_env/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        default_py="$(command -v python3)"
    fi
    PY="${PY:-$default_py}"
}

pipeline_normalize_type_subj() {
    case "$1" in
        brain|brains) echo "brains" ;;
        phantom|phantoms) echo "phantoms" ;;
        *)
            echo "ERROR: unknown type_subj '$1' (use brain or phantom)" >&2
            return 2
            ;;
    esac
}

pipeline_normalize_type_seq() {
    case "$1" in
        ogse|nogse) echo "$1" ;;
        *)
            echo "ERROR: unknown type_seq '$1' (use ogse or nogse)" >&2
            return 2
            ;;
    esac
}

pipeline_set_dataset_defaults() {
    local dataset
    local type_seq
    dataset="$(pipeline_normalize_type_subj "$1")" || exit 2
    type_seq="$(pipeline_normalize_type_seq "${2:-${TYPE_SEQ:-ogse}}")" || exit 2

    DATASET="$dataset"
    TYPE_SUBJ="${dataset%s}"
    TYPE_SEQ="$type_seq"
    EXPERIMENT_ROOT_NAME="${type_seq}_experiments"
    SIGNALS_ROOT="${SIGNALS_ROOT:-$PROJECT_ROOT/Data-signals}"
    RESULTS_ROOT="${RESULTS_ROOT:-$SIGNALS_ROOT/Results}"

    case "$dataset" in
        brains)
            PARAMS_XLSX="${PARAMS_XLSX:-$SIGNALS_ROOT/sequence_parameters_brains.xlsx}"
            ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/brains/$EXPERIMENT_ROOT_NAME}"
            ;;
        phantoms)
            PARAMS_XLSX="${PARAMS_XLSX:-$SIGNALS_ROOT/sequence_parameters_phantoms.xlsx}"
            ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms/$EXPERIMENT_ROOT_NAME}"
            ;;
    esac

    MASTER_PARQUET="${MASTER_PARQUET:-$ANALYSIS_ROOT/master.long.parquet}"
    MANIFEST_DIR="${MANIFEST_DIR:-$TEMPLATE_ROOT/manifests/${dataset}_${type_seq}}"

    export DATASET TYPE_SUBJ TYPE_SEQ EXPERIMENT_ROOT_NAME
}

pipeline_require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $label not found: $path" >&2
        exit 1
    fi
}

pipeline_apply_master_first_points_by_td() {
    local spec="${MASTER_FIRST_POINTS_BY_TD:-}"
    if [[ -z "${spec// }" || "${spec^^}" == "ALL" ]]; then
        return 0
    fi
    if [[ -n "${MASTER_FIRST_POINTS_APPLIED:-}" ]]; then
        return 0
    fi

    local filter_script="$REPO_ROOT/scripts/data/filter_master_table.py"
    pipeline_require_file "$MASTER_PARQUET" "master table"
    pipeline_require_file "$filter_script" "master filter script"

    local out_dir
    out_dir="${MASTER_FIRST_POINTS_DIR:-$ANALYSIS_ROOT}"
    mkdir -p "$out_dir"

    MASTER_PARQUET_ORIGINAL="${MASTER_PARQUET_ORIGINAL:-$MASTER_PARQUET}"
    MASTER_PARQUET="${MASTER_FIRST_POINTS_PARQUET:-$out_dir/master.first_points.long.parquet}"

    "$PY" "$filter_script" "$MASTER_PARQUET_ORIGINAL" \
        --out-parquet "$MASTER_PARQUET" \
        --first-points-by-td "$spec"

    MASTER_FIRST_POINTS_APPLIED=1
    export MASTER_PARQUET MASTER_PARQUET_ORIGINAL MASTER_FIRST_POINTS_APPLIED
    echo "Using filtered master table: $MASTER_PARQUET"
    echo "First points by td_ms: $spec"
}

pipeline_usage() {
    cat <<'EOF'
Usage:
  bash nogse_pipeline/bash_template/run_dataset.sh <type_subj> <type_seq> <step...>

Run from:
  /mnt/storage/tier2/MUMI-EXT-001/mumi-data/Project-Balseiro-Microstructure

Subject and sequence:
  <type_subj>  brain | phantom
               Aliases accepted: brains, phantoms
  <type_seq>   ogse | nogse

Equivalent option form:
  bash nogse_pipeline/bash_template/run_dataset.sh --type-subj brain --type-seq ogse <step...>

Runner arguments:
  --type-subj brain|phantom   Select subject type.
  --type_subj brain|phantom   Select subject type, underscore spelling.
  --dataset brain|phantom     Alias for --type-subj.
  --type-seq ogse|nogse       Select sequence type.
  --type_seq ogse|nogse       Select sequence type, underscore spelling.
  --results-root DIR          Add one Results input folder for ingest. Repeatable.
  -h, --help                  Show this help, or step help after a step name.

How step arguments work:
  The runner only parses <type_subj>, <type_seq>, step names, and --results-root.
  Step-specific settings are environment variables placed before the command:

    VAR=value OTHER_VAR=value \
      bash nogse_pipeline/bash_template/run_dataset.sh brain ogse <step>

  Extra Python flags go through each step's *_EXTRA_ARGS variable:

    SIGNAL_FIT_EXTRA_ARGS="--fix_M0 1.0 --auto_fit_tol 0.05" \
      bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

Available steps:
  Data import:
    ingest           Read Results/*_results.xlsx into master.long.parquet.

  Master-table construction:
    filter_master_points
                     Write an intermediate master table with first points by td_ms.
    rotate           Rotate signal tensor directions.
    contrast         Build contrast rows using manifests/*/contrasts.csv.
    export_master_xlsx
                     Export the selected master parquet to an Excel workbook.

  Plots:
    plot_signal      Plot signal curves from master.long.parquet.
    plot_contrast    Plot contrast curves from master.long.parquet.
    plot_d0_delta    Plot D0/Dproj vs Delta_app_ms.
    plot_monoexp_d   Plot monoexponential D vs td_ms/Delta_app_ms.

  Fits:
    fit_signal                 Fit signal curves using signal_fits.csv.
    fit_signal_gradcorr        fit_signal with gradient correction enabled.
    fit_contrast               Fit contrast rows.
    fit_contrast_free          fit_contrast with free-model defaults.
    fit_contrast_mixed_global  fit_contrast with mixed_global model.
    fit_global_signal          Fit mixed/global signal models directly.

  Summaries:
    grad_correction  Build and embed gradient-correction factors.
    alpha            Build alpha_macro summaries.
    tc               Fit tc-vs-td summaries.

Help for one step:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest --help
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate --help
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal --help

Common environment variables:
  PY                 Python interpreter.
  SIGNALS_ROOT       Root containing Results/ and sequence parameter workbooks.
  PARAMS_XLSX        Sequence-parameter workbook.
  ANALYSIS_ROOT      Output analysis root.
  MASTER_PARQUET     Master table path.
  MANIFEST_DIR       Directory with contrasts.csv, signal_fits.csv, and grad_correction.csv.
  MASTER_FIRST_POINTS_BY_TD
                    Optional TD=POINTS rules applied before post-ingest steps.
                    Example: MASTER_FIRST_POINTS_BY_TD="120=8,210=6"
                    Unlisted td_ms values keep all points.
  MASTER_FIRST_POINTS_PARQUET
                    Optional filtered master output path. Default:
                    $ANALYSIS_ROOT/master.first_points.long.parquet.
  TC_FIT_PARAMS     Fit-params parquet used by the tc step. Must point to
                    the accumulated contrast fit-params file produced by
                    fit_contrast (e.g. fits/master/ogse_.../fit_params.*.parquet).

Master table format:
  master.long.parquet is the canonical master table and the pipeline only
  appends to parquet. To inspect it in Excel, export a copy explicitly:

    python nogse_pipeline/scripts/data/export_master_table.py \
      analysis/brains/ogse_experiments/master.long.parquet \
      --out-xlsx analysis/brains/ogse_experiments/master.inspect.xlsx

Examples:
  # Ingest one Results folder.
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
    --results-root Data-signals/Results/20220622_BRAIN \
    ingest

  # Run the next core steps after ingest.
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate contrast

  # Filter a rotate step by subject/sheet.
  MASTER_SUBJ=BRAIN MASTER_SHEET=20220622_BRAIN \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

  # Plot one ROI/direction.
  PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=long \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal

  # Fit signals with extra Python options.
  SIGNAL_FIT_MODEL=monoexp \
  SIGNAL_FIT_EXTRA_ARGS="--fix_M0 1.0" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

  # Run several steps from PIPELINE_STEPS instead of positional step names.
  PIPELINE_STEPS="rotate contrast fit_signal fit_contrast alpha tc" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse
EOF
}

pipeline_step_help() {
    local step="$1"
    case "$step" in
        ingest)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> ingest
  bash run_dataset.sh <type_subj> <type_seq> --results-root RESULTS_ROOT ingest

What it does:
  Reads Results/*_results.xlsx, uses the sequence-parameter workbook, writes long signal rows,
  and appends row_kind='signal' rows to master.long.parquet.

Runner arguments:
  --results-root DIR  Add one input folder through run_dataset.sh. Repeat for multiple folders.

Variables for this step:
  RESULTS_ROOT        One Results folder/root if RESULTS_ROOTS is unset.
  RESULTS_ROOTS       Space-separated Results roots.
  PARAMS_XLSX         Sequence-parameter workbook.
  RESULTS_GLOB        Results filename pattern. Default: *_results.xlsx
  MASTER_PARQUET      Master table output.
  PROCESS_SCRIPT      Python script override.
  PROCESS_OUT_ROOT    Per-file table output root. Default: $ANALYSIS_ROOT/data/tables

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse \
    --results-root Data-signals/Results/20220622_BRAIN \
    ingest

  RESULTS_ROOTS="Data-signals/Results/20220622_BRAIN Data-signals/Results/20230619_BRAIN-3" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest

  PARAMS_XLSX=Data-signals/sequence_parameters_brains.xlsx \
  RESULTS_GLOB="*_results.xlsx" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest
EOF
            ;;
        filter_master_points)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> filter_master_points

What it does:
  Writes an intermediate master table that keeps only the requested first b_step
  values for each td_ms. The original master table is not modified.

Variables for this step:
  MASTER_PARQUET              Input master table.
  MASTER_FIRST_POINTS_BY_TD   TD=POINTS rules. Example: "120=8,210=6,90=ALL".
                              Unlisted td_ms values keep all points.
  FILTERED_MASTER_PARQUET     Output filtered master table path.
                              Also accepted as MASTER_FIRST_POINTS_PARQUET.
                              Default: $ANALYSIS_ROOT/master.first_points.long.parquet
  MASTER_FIRST_POINTS_DIR     Output directory override. Default: $ANALYSIS_ROOT
  FILTER_MASTER_SCRIPT        Python script override.

Note:
  After running filter_master_points, pass the output path as MASTER_PARQUET
  to subsequent steps:

Examples:
  MASTER_FIRST_POINTS_BY_TD="120=8,210=6,90=ALL" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse filter_master_points

  MASTER_PARQUET=analysis/brains/ogse_experiments/master.first_points.long.parquet \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate contrast fit_signal
EOF
            ;;
        export_master_xlsx)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> export_master_xlsx

What it does:
  Exports MASTER_PARQUET to an inspection .xlsx workbook. It does not modify the
  parquet master table.

Variables for this step:
  MASTER_PARQUET           Input master table.
  MASTER_XLSX              Output Excel path. Default: MASTER_PARQUET with .xlsx suffix.
  EXPORT_MASTER_ROW_KIND   Optional row_kind filter. Space- or comma-separated.
  EXPORT_MASTER_HEAD       Optional number of first rows to export.
  EXPORT_MASTER_SCRIPT     Python script override.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse export_master_xlsx

  MASTER_PARQUET=analysis/brains/ogse_experiments/master.first_points.long.parquet \
  MASTER_XLSX=analysis/brains/ogse_experiments/master.first_points.xlsx \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse export_master_xlsx
EOF
            ;;
        rotate)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> rotate

What it does:
  Selects row_kind='signal' rows from master.long.parquet, rotates tensor directions,
  and appends row_kind='signal_rotated' rows. The rotated rows include D_proj.

Variables for this step:
  MASTER_PARQUET      Input/output master table.
  MASTER_SUBJ         Optional selector for the master column 'subj'.
                      Example values in brain data may be BRAIN, LUDG, MBBL.
  MASTER_SHEET        Optional selector for the master column 'sheet'.
                      Example values may be 20220622_BRAIN or 20230619_BRAIN-3.
  DIRS_TXT            Direction table. Default: assets/dirs/dirs_6.txt
  ROTATE_SCRIPT       Python script override.
  ROTATED_OUT_ROOT    Legacy rotated output root. Default: $ANALYSIS_ROOT/data-rotated/tables
  ROTATE_EXTRA_ARGS   Extra rotate_ogse_tensor.py options.

Useful ROTATE_EXTRA_ARGS:
  --solver lstsq|solve
  --s0_mode dir1|mean
  --b_col bvalue
  --no-legacy-output

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

  MASTER_SUBJ=BRAIN \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

  MASTER_SHEET=20220622_BRAIN \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

  MASTER_SUBJ=BRAIN \
  MASTER_SHEET=20220622_BRAIN \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate

  ROTATE_EXTRA_ARGS="--s0_mode mean --no-legacy-output" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate
EOF
            ;;
        contrast)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> contrast

What it does:
  Reads declarative contrast selectors from manifests/<type_subj>_<type_seq>/contrasts.csv,
  selects two signal_rotated groups from master, subtracts them, and appends row_kind='contrast'.

  By default contrasts are direct point-by-point subtractions. To build fitted/resampled
  contrasts instead, pass --contrast-source fitted_resampled through MAKE_CONTRAST_EXTRA_ARGS.
  Fit steps do not resample contrasts themselves; they fit whatever contrast rows already
  exist in MASTER_PARQUET.

Variables for this step:
  CONTRAST_MANIFEST       CSV contrast manifest.
  MASTER_PARQUET          Input/output master table.
  MAKE_CONTRAST_SCRIPT    Python script override.
  CONTRAST_OUT_ROOT       Output root. Default: $ANALYSIS_ROOT/contrast-data-master
  MAKE_CONTRAST_EXTRA_ARGS Extra make_contrast.py options.

Manifest columns:
  subj,sheet,roi,direction,td_ms,N_1,N_2,Hz_1,Hz_2

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast

  CONTRAST_MANIFEST=nogse_pipeline/bash_template/manifests/brains_ogse/contrasts.csv \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast

  MAKE_CONTRAST_EXTRA_ARGS="--contrast-source fitted_resampled --signal-model monoexp --g_type g_lin_max --auto_fit_points" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast
EOF
            ;;
        plot_signal)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> plot_signal

What it does:
  Plots signal curves selected from master.long.parquet using the script for TYPE_SEQ.

Variables for this step:
  MASTER_PARQUET          Input master table.
  PLOT_SIGNAL_SCRIPT      Python script override.
  PLOT_OUT_ROOT           Output root. Default: $ANALYSIS_ROOT/plots-master/signal
  PLOT_ROW_KIND           signal_rotated|signal. Default: signal_rotated
  PLOT_SUBJ               Optional subj selector.
  PLOT_SHEET              Optional sheet selector.
  PLOT_ROI                Optional ROI selector.
  PLOT_DIRECTION          Optional direction selector, e.g. long|tra|x|y|z.
  PLOT_TD_MS              Optional td_ms selector.
  PLOT_N                  Optional N selector.
  PLOT_SIGNAL_YCOL        value|value_norm. Default: value_norm
  PLOT_SIGNAL_XCOL        Gradient column (x axis).
                          OGSE: g|g_max|g_lin_max|g_thorsten|bvalue|bvalue_g|bvalue_thorsten. Default: g_thorsten
                          NOGSE: g|g_max|g_lin_max|g_thorsten|bvalue|bvalue_g|bvalue_thorsten. Default: g
  PLOT_SIGNAL_G_TYPE      Backward-compatible alias for PLOT_SIGNAL_XCOL.
  PLOT_STAT               avg|std. Default: avg
  PLOT_SIGNAL_EXTRA_ARGS  Extra plot_<type_seq>_signal_vs_g.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal

  PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=long \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal

  PLOT_SUBJ=20220622_BRAIN PLOT_DIRECTION="long" PLOT_SIGNAL_XCOL=g_thorsten \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal
EOF
            ;;
        plot_contrast)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> plot_contrast

What it does:
  Plots contrast curves selected from master.long.parquet using the script for TYPE_SEQ.

Variables for this step:
  MASTER_PARQUET            Input master table.
  PLOT_CONTRAST_SCRIPT      Python script override.
  PLOT_OUT_ROOT             Output root. Default: $ANALYSIS_ROOT/plots-master/contrast
  PLOT_SUBJ                 Optional subj selector.
  PLOT_SHEET                Optional sheet selector.
  PLOT_ROI                  Optional ROI selector.
  PLOT_DIRECTION            Optional direction selector, e.g. long|tra|x|y|z.
  PLOT_TD_MS                Optional td_ms selector.
  PLOT_N1                   Optional N_1 selector.
  PLOT_N2                   Optional N_2 selector.
  PLOT_CONTRAST_YCOL        value|value_norm. Default: value_norm
  PLOT_CONTRAST_XCOL        Gradient column for side 1 (x axis).
                            g_1|g_max_1|g_lin_max_1|g_thorsten_1|bvalue_1|bvalue_thorsten_1. Default: g_thorsten_1
  PLOT_STAT                 avg|std. Default: avg
  PLOT_CONTRAST_EXTRA_ARGS  Extra plot_<type_seq>_contrast_vs_g.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast

  PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=tra \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast
EOF
            ;;
        fit_signal|fit_signal_gradcorr)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> fit_signal
  bash run_dataset.sh <type_subj> <type_seq> fit_signal_gradcorr

What it does:
  Fits signal curves selected from master according to manifests/<type_subj>_<type_seq>/signal_fits.csv.
  The model to use per row is set in the manifest's "model" column.
  fit_signal_gradcorr adds --apply_grad_corr to every fit.

Variables for this step:
  MASTER_PARQUET          Input master table.
  FIT_SIGNAL_SCRIPT       Python script override.
  SIGNAL_FIT_MANIFEST     CSV signal-fit manifest.
  SIGNAL_FIT_OUT_ROOT     Output root. Default: $ANALYSIS_ROOT/fits/<master_name>/<type_seq>_<ycol>_vs_<gtype>_<model>
  SIGNAL_FIT_MODEL        Fallback model when the manifest "model" column is empty.
                          OGSE: monoexp|ogse_free|ogse_rest|ogse_rest_offset. Default: monoexp
                          NOGSE: free_cpmg|nogse_free|mixed_global. Default: nogse_free
  SIGNAL_FIT_G_TYPE       Gradient column.
                          OGSE: g|g_max|g_lin_max|g_thorsten|bvalue|bvalue_g|bvalue_thorsten. Default: bvalue_thorsten
                          NOGSE: g|g_max|g_lin_max|g_thorsten|bvalue|bvalue_g|bvalue_thorsten. Default: g
  SIGNAL_FIT_XCOL         NOGSE x-axis override. Defaults to SIGNAL_FIT_G_TYPE.
                          Has no effect for OGSE (OGSE uses --g_type / SIGNAL_FIT_G_TYPE directly).
  SIGNAL_FIT_YCOL         value|value_norm. Default: value_norm
  SIGNAL_FIT_EXTRA_ARGS   Extra fit_<type_seq>_signal_vs_g.py options.

Manifest columns:
  subj,sheet,roi,direction,td_ms,N,Hz,model

Useful SIGNAL_FIT_EXTRA_ARGS:
  OGSE monoexp:  --fix_M0 1.0 --auto_fit_tol 0.05 --auto_fit_min_points 3 --auto_fit_max_points 9
  NOGSE/OGSE free: see scripts/fitting/fit_{nogse,ogse}_signal_vs_g.py --help

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

  SIGNAL_FIT_MANIFEST=nogse_pipeline/bash_template/manifests/brains_ogse/signal_fits.csv \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

  SIGNAL_FIT_EXTRA_ARGS="--fix_M0 1.0 --auto_fit_tol 0.05" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr
EOF
            ;;
        fit_contrast|fit_contrast_free|fit_contrast_mixed_global)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> fit_contrast
  bash run_dataset.sh <type_subj> <type_seq> fit_contrast_free
  bash run_dataset.sh <type_subj> <type_seq> fit_contrast_mixed_global

What it does:
  Fits contrast rows from master.long.parquet. Saves per-experiment tables to subdirs and
  accumulates all results in a descriptively named fit_params parquet in FIT_OUT_ROOT.

  fit_contrast_free and fit_contrast_mixed_global are convenience presets. They call
  the same step script as fit_contrast and only set FIT_MODEL when FIT_MODEL is not
  already provided:
    fit_contrast_free         -> ogse_free (OGSE) / nogse_free (NOGSE)
    fit_contrast_mixed_global -> mixed_global

  Therefore, these are equivalent:
    bash run_dataset.sh brain ogse fit_contrast_free
    FIT_MODEL=ogse_free bash run_dataset.sh brain ogse fit_contrast

  Contrast resampling is controlled by the contrast step, not by fit_contrast. If you
  need resampled contrasts, first run contrast with MAKE_CONTRAST_EXTRA_ARGS containing
  --contrast-source fitted_resampled, then run the desired fit_contrast* step.

Variables for this step:
  MASTER_PARQUET        Input master table.
  FIT_CONTRAST_SCRIPT   Python script override.
  FIT_OUT_ROOT          Output root. Default: $ANALYSIS_ROOT/fits/<master_name>/<type_seq>_<ycol>_vs_<gtype>_<model>
  FIT_MODEL             OGSE: ogse_free|ogse_tort|ogse_rest|ogse_rest_offset|ogse_mixed. Default: ogse_free
                        NOGSE: nogse_free|nogse_free_grad_offset|nogse_tort|nogse_rest. Default: nogse_free
                        fit_contrast_free  → presets ogse_free (OGSE) / nogse_free (NOGSE).
                        fit_contrast_mixed_global → presets mixed_global.
  FIT_GBASE             g|g_lin_max|g_max|g_thorsten|bvalue|bvalue_g|bvalue_thorsten. Default: g_lin_max
  FIT_YCOL              value|value_norm. Default: value_norm
  FIT_STAT              avg|std. Default: avg
  FIT_EXTRA_ARGS        Extra fit_<type_seq>_contrast_vs_g.py options.

Useful FIT_EXTRA_ARGS:
  --apply_grad_corr         Apply embedded gradient correction when fitting.
  Use the Python script help for model-specific flags:
    python scripts/fitting/fit_ogse_contrast_vs_g.py --help
    python scripts/fitting/fit_nogse_contrast_vs_g.py --help

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free

  FIT_MODEL=ogse_free \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast

  FIT_MODEL=ogse_mixed FIT_GBASE=g_lin_max FIT_YCOL=value_norm \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast

  FIT_EXTRA_ARGS="--apply_grad_corr" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast
EOF
            ;;
        fit_global_signal)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> fit_global_signal

What it does:
  Fits a global/mixed signal model pooling all subjects from master.long.parquet.
  Parameters that are "global_td" are shared across subjects at the same td_ms;
  "global_contrast" parameters are shared across subjects at the same (td_ms, roi, direction).

Available models (set via GLOBAL_SIGNAL_MODEL):
  OGSE: ogse_free | ogse_rest | ogse_rest_offset | ogse_mixed | ogse_mixed_offset
  NOGSE: nogse_free | nogse_free_offset | nogse_rest | nogse_rest_offset | nogse_mixed | nogse_mixed_offset
  Parameter coverage per model:
    ogse_free / nogse_free*          →  M0, D0
    ogse_rest / nogse_rest           →  tc, M0, D0
    ogse_rest_offset                 →  tc, M0, D0, C
    ogse_mixed / nogse_mixed         →  tc, alpha, M0, D0
    ogse_mixed_offset / nogse_mixed_offset → tc, alpha, RN, M0, D0, C
  Mode variables for parameters not in the chosen model are silently ignored.

Variables for this step:
  MASTER_PARQUET                 Input master table.
  FIT_GLOBAL_SIGNAL_SCRIPT       Python script override.
  GLOBAL_SIGNAL_OUT_ROOT         Output root.
                                 Default: $ANALYSIS_ROOT/fits/<type_seq>_signal_<model>
                                 Override explicitly when running multiple models or correction modes
                                 so outputs don't overwrite each other.
  GLOBAL_SIGNAL_ROW_KIND         signal_rotated|signal. Default: signal_rotated
  GLOBAL_SIGNAL_MODEL            See model list above.
                                 Default: ogse_mixed_offset (OGSE) | nogse_mixed_offset (NOGSE)
  GLOBAL_SIGNAL_YCOL             value|value_norm. Default: value
  GLOBAL_SIGNAL_G_TYPE           Gradient column.
                                 OGSE brain: g_thorsten (default) | g_lin_max | bvalue_thorsten | ...
                                 OGSE phantom / NOGSE: g (default) | g_lin_max | g_thorsten | ...
  GLOBAL_SIGNAL_STAT             avg|std. Default: avg
  GLOBAL_SIGNAL_MIN_POINTS       Minimum points per group to attempt a fit. Default: 4
  Parameter modes and fixed values
  ---------------------------------
  Each parameter has a _MODE variable and a _FIXED variable. They work as a pair:
    _MODE controls how the parameter is treated across subjects and td_ms values.
    _FIXED sets the numeric value when _MODE=fixed.
  Mode and fixed vars for parameters not present in the chosen model are silently ignored.

  GLOBAL_SIGNAL_TC_MODE          fixed|free|global_td|global_contrast. Default: global_td
  GLOBAL_SIGNAL_TC_FIXED         Fixed tc value [ms]. Required when TC_MODE=fixed.
                                 (tc: rest, rest_offset, mixed, mixed_offset only)
  GLOBAL_SIGNAL_ALPHA_MODE       fixed|free|global_td|global_contrast. Default: global_td
  GLOBAL_SIGNAL_ALPHA_FIXED      Fixed alpha value [0–1]. Required when ALPHA_MODE=fixed.
                                 (alpha: mixed, mixed_offset only)
  GLOBAL_SIGNAL_RN_MODE          fixed|free|global_td|global_contrast. Default: global_td
  GLOBAL_SIGNAL_RN_FIXED         Fixed RN value. Required when RN_MODE=fixed.
                                 (RN: mixed_offset, nogse_mixed_offset only)
  GLOBAL_SIGNAL_M0_MODE          fixed|free|global_td|global_contrast. Default: global_contrast
  GLOBAL_SIGNAL_M0_FIXED         Fixed M0 value. Required when M0_MODE=fixed.
  GLOBAL_SIGNAL_C_MODE           fixed|free|global_td|global_contrast. Default: global_contrast
  GLOBAL_SIGNAL_C_FIXED          Fixed C value. Required when C_MODE=fixed.
                                 (C: rest_offset, mixed_offset only)
  GLOBAL_SIGNAL_D0_MODE          fixed|free|global_td|global_contrast. Default: fixed
  GLOBAL_SIGNAL_D0_FIXED         Fixed D0 in m²/s. Default: 3.2e-12 (brain) | 2.3e-12 (phantom)
  GLOBAL_SIGNAL_DIRECTIONS       Space- or comma-separated directions, or ALL.
                                 Default: long tra (OGSE brain) | ALL (others)
  GLOBAL_SIGNAL_ROIS             Space- or comma-separated ROIs, or ALL. Default: ALL
  GLOBAL_SIGNAL_SUBJS            Space- or comma-separated subjects, or ALL. Default: ALL
  GLOBAL_SIGNAL_APPLY_GRAD_CORR  true|false. Default: true
  GLOBAL_SIGNAL_EXTRA_ARGS       Extra fit_global_signal.py options (e.g. --tc_init 5.0).

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal

  # ogse_mixed_offset with RN fixed at 10 (grad correction on by default):
  GLOBAL_SIGNAL_RN_MODE=fixed \
  GLOBAL_SIGNAL_RN_FIXED=10 \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal

  # ogse_rest (no alpha, no RN, no C): only TC and M0 modes apply
  GLOBAL_SIGNAL_MODEL=ogse_rest \
  GLOBAL_SIGNAL_TC_MODE=global_td \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal

  # run without grad correction
  GLOBAL_SIGNAL_APPLY_GRAD_CORR=false \
  GLOBAL_SIGNAL_OUT_ROOT="analysis/brains/ogse_experiments/fits/ogse_signal_ogse_mixed_offset_raw" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal

  # restrict to specific ROIs/directions
  GLOBAL_SIGNAL_DIRECTIONS="long tra" GLOBAL_SIGNAL_ROIS="Left-Lateral-Ventricle AntCC" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal
EOF
            ;;
        grad_correction)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> grad_correction

What it does:
  Computes gradient-correction factors from the curves listed in grad_correction.csv
  and embeds them in MASTER_PARQUET. For each manifest row, the Python script fits
  the same reference curve with NOGSE free and monoexp models, computes
  correction_factor = sqrt(D0_nogse / D0_monoexp), and writes the factor back to
  all master rows that share the same subj, sheet, direction, td_ms, and N.

Variables for this step:
  GRAD_CORR_SCRIPT      Python script override.
  GRAD_CORR_MANIFEST    CSV manifest. Default: $MANIFEST_DIR/grad_correction.csv
                        Columns: subj,sheet,roi,direction,td_ms,N,Hz,model
  MASTER_PARQUET        Input/output master table. Must contain signal_rotated rows.
  GRAD_CORR_OUT_DIR     Audit output directory. Default: $ANALYSIS_ROOT/fits/grad_correction
  GRAD_CORR_EXTRA_ARGS  Extra make_grad_correction_table.py options.

Useful GRAD_CORR_EXTRA_ARGS:
  --stat avg               Statistic row to fit. Default: avg
  --row-kind signal_rotated Input row kind. Default: signal_rotated
  --gbase g_lin_max        Gradient column for the NOGSE free fit.
  --bbase bvalue_thorsten  B-value column for the monoexp fit.
  --D0-init 2.3e-12        NOGSE free D0 seed in m2/ms.
  --tol-ms 1e-3            Matching tolerance for td_ms.
  --fix-M0 1.0             Fix M0 in both fits.
  --free-M0 1.0            Fit M0 in both fits with optional seed.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction

  GRAD_CORR_MANIFEST=nogse_pipeline/bash_template/manifests/brains_ogse/grad_correction.csv \
  GRAD_CORR_EXTRA_ARGS="--bbase bvalue_thorsten --fix-M0 1.0" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction
EOF
            ;;
        plot_d0_delta)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> plot_d0_delta

What it does:
  Plots Dproj/D0 vs Delta_app_ms from master and can reuse summary_alpha_values.

Variables for this step:
  MASTER_PARQUET       Input master table. Must contain signal_rotated rows with D_proj.
  PLOT_D0_SCRIPT       Python script override.
  ALPHA_OUT_DIR        Output directory. Default: $ANALYSIS_ROOT/alpha_macro/master
  SUMMARY_ALPHA        Optional summary_alpha_values.xlsx path.
  DPROJ_N              Optional N selector.
  DPROJ_HZ             Optional Hz selector.
  DPROJ_ROIS           Space-separated ROI list.
  DPROJ_DIRS           Space-separated direction list, e.g. "long tra x y z".
  PLOT_D0_EXTRA_ARGS   Extra plot_D0_vs_Delta.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_d0_delta

  DPROJ_N=1 DPROJ_DIRS="long tra x y z" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_d0_delta
EOF
            ;;
        plot_monoexp_d)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> plot_monoexp_d

What it does:
  Builds monoexp D vs td_ms and Delta_app_ms plots from signal fit outputs.

Variables for this step:
  PLOT_MONOEXP_D_SCRIPT     Python script override.
  SIGNAL_FITS_ROOT          Root scanned for signal fit parquets.
                            Default: $ANALYSIS_ROOT/fits (scans all subdirs).
                            Set explicitly to the specific fit folder, e.g.:
                            $ANALYSIS_ROOT/fits/master/ogse_value_norm_vs_bvaluethorsten_monoexp
  MONOEXP_D_OUT_DIR         Output plot/table directory. Default: $ANALYSIS_ROOT/plots-master/monoexp_D_vs_time
  PLOT_MONOEXP_D_EXTRA_ARGS Extra plot_monoexp_D_vs_time.py options.

Examples:
  SIGNAL_FITS_ROOT=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_monoexp \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d
EOF
            ;;
        alpha)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> alpha

What it does:
  Computes alpha_macro summaries from D_proj values in signal_rotated rows.

Variables for this step:
  MASTER_PARQUET       Input master table. Must contain signal_rotated rows with D_proj.
  ALPHA_MACRO_SCRIPT   Python script override.
  ALPHA_N              N selector. Default: 1
  ALPHA_OUT_DIR        Output directory. Default: $ANALYSIS_ROOT/alpha_macro/master
  ALPHA_EXTRA_ARGS     Extra make_alpha_macro_summary.py options.

Note:
  This step always passes --no-master-fit-params, so alpha_macro results are NOT
  appended to any cumulative fit-params table. The outputs are the xlsx files below.

Useful ALPHA_EXTRA_ARGS:
  --bvalmax N              Use the N-th bvalue (1-based ascending) for all ROIs.
                           Default: highest bvalue.
  --roi-bvalmax ROI=N      Per-ROI bvalue override. Repeatable.
                           Example: --roi-bvalmax AntCC=7 --roi-bvalmax CSF=3
  --dirs long tra          Restrict to specific directions.
  --rois ROI1 ROI2         Restrict to specific ROIs.

Outputs:
  $ALPHA_OUT_DIR/summary_alpha_values.xlsx
  $ALPHA_OUT_DIR/D_vs_delta_app.combined.xlsx
  $ALPHA_OUT_DIR/alpha_macro_vs_roi.png

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha

  ALPHA_N=1 \
  ALPHA_EXTRA_ARGS="--bvalmax 5 --roi-bvalmax AntCC=7 --roi-bvalmax CSF=3" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha
EOF
            ;;
        tc)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> tc

What it does:
  Fits tc-vs-td summaries from a contrast fit-params parquet.

Variables for this step:
  TC_FIT_PARAMS      Required. Path to the accumulated contrast fit-params parquet
                     produced by fit_contrast (e.g. fits/master/ogse_.../fit_params.*.parquet).
  TC_VS_TD_SCRIPT    Python script override.
  TC_OUT_DIR         Output root. Default: $ANALYSIS_ROOT/fits/tc_vs_td_master
                     Each method writes to its own subdirectory: $TC_OUT_DIR/$TC_METHOD
  TC_METHOD          pseudohuber_free|pseudohuber_fixed_macro|linear. Default: pseudohuber_fixed_macro
                     pseudohuber_free: fits c, delta, alpha_macro freely.
                     pseudohuber_fixed_macro: fits c and delta; alpha_macro fixed from summary.
                       Requires: TC_EXTRA_ARGS="--summary-alpha path/to/summary_alpha_values.xlsx"
                     linear: fits tc(Td) = c + slope * Td. Useful for the large-Td regime.
  TC_Y_COL           tc_peak_ms|tc_peak_resampled_ms. Default: tc_peak_ms
                     Override TC_OUT_DIR when changing TC_Y_COL to avoid output conflicts.
  TC_EXTRA_ARGS      Extra run_tc_vs_td.py options.

Examples:
  # pseudohuber with fixed alpha_macro from summary
  TC_FIT_PARAMS="fits/master/ogse_.../fit_params.*.parquet" \
  TC_METHOD=pseudohuber_fixed_macro \
  TC_EXTRA_ARGS="--summary-alpha analysis/brains/ogse_experiments/alpha_macro/master/summary_alpha_values.xlsx" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc

  # linear fit on resampled tc_peak
  TC_FIT_PARAMS="fits/master/ogse_.../fit_params.*.parquet" \
  TC_METHOD=linear \
  TC_Y_COL=tc_peak_resampled_ms \
  TC_OUT_DIR="analysis/brains/ogse_experiments/fits/tc_vs_td_resampled_master" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc
EOF
            ;;
        *)
            echo "ERROR: no help available for unknown step '$step'" >&2
            return 2
            ;;
    esac
}

pipeline_maybe_step_help() {
    local step="$1"
    shift || true
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        pipeline_step_help "$step"
        exit 0
    fi
}

pipeline_step_script() {
    case "$1" in
        ingest) echo "$TEMPLATE_ROOT/steps/01_ingest_results.sh" ;;
        filter_master_points) echo "$TEMPLATE_ROOT/steps/00_filter_master_points.sh" ;;
        export_master_xlsx) echo "$TEMPLATE_ROOT/steps/99_export_master_xlsx.sh" ;;
        rotate) echo "$TEMPLATE_ROOT/steps/02_rotate_signals.sh" ;;
        contrast) echo "$TEMPLATE_ROOT/steps/03_make_contrasts.sh" ;;
        plot_signal) echo "$TEMPLATE_ROOT/steps/04_plot_signals.sh" ;;
        plot_contrast) echo "$TEMPLATE_ROOT/steps/05_plot_contrasts.sh" ;;
        fit_signal|fit_signal_gradcorr) echo "$TEMPLATE_ROOT/steps/06_fit_signals.sh" ;;
        fit_contrast|fit_contrast_free|fit_contrast_mixed_global) echo "$TEMPLATE_ROOT/steps/07_fit_contrasts.sh" ;;
        fit_global_signal) echo "$TEMPLATE_ROOT/steps/13_fit_global_signals.sh" ;;
        alpha) echo "$TEMPLATE_ROOT/steps/08_alpha_macro.sh" ;;
        tc) echo "$TEMPLATE_ROOT/steps/09_tc_vs_td.sh" ;;
        grad_correction) echo "$TEMPLATE_ROOT/steps/10_make_grad_correction_table.sh" ;;
        plot_d0_delta) echo "$TEMPLATE_ROOT/steps/11_plot_D0_vs_Delta_alpha.sh" ;;
        plot_monoexp_d) echo "$TEMPLATE_ROOT/steps/12_plot_monoexp_D_vs_time.sh" ;;
        *) return 1 ;;
    esac
}

pipeline_prepare_step_env() {
    case "$1" in
        fit_signal_gradcorr)
            if [[ "$TYPE_SEQ" == "nogse" ]]; then
                SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-nogse_free}"
                SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-g}"
            else
                SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-monoexp}"
                SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-bvalue_thorsten}"
            fi
            SIGNAL_FIT_EXTRA_ARGS="${SIGNAL_FIT_EXTRA_ARGS:-} --apply_grad_corr"
            export SIGNAL_FIT_MODEL SIGNAL_FIT_G_TYPE SIGNAL_FIT_EXTRA_ARGS
            ;;
        fit_contrast_free)
            if [[ "$TYPE_SEQ" == "nogse" ]]; then
                FIT_MODEL="${FIT_MODEL:-nogse_free}"
            else
                FIT_MODEL="${FIT_MODEL:-ogse_free}"
            fi
            export FIT_MODEL
            ;;
        fit_contrast_mixed_global)
            FIT_MODEL="${FIT_MODEL:-mixed_global}"
            export FIT_MODEL
            ;;
    esac
}

pipeline_run_steps() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        pipeline_usage
        return 0
    fi
    if [[ $# -eq 2 && ( "${2:-}" == "-h" || "${2:-}" == "--help" ) ]]; then
        pipeline_step_help "$1"
        return 0
    fi
    local raw_steps="${PIPELINE_STEPS:-$*}"
    raw_steps="${raw_steps//,/ }"
    if [[ -z "${raw_steps// }" ]]; then
        pipeline_usage
        return 0
    fi

    local step script
    for step in $raw_steps; do
        script="$(pipeline_step_script "$step")" || {
            echo "ERROR: unknown step '$step'" >&2
            pipeline_usage >&2
            exit 2
        }
        if [[ "$step" != "ingest" && "$step" != "filter_master_points" && "$step" != "export_master_xlsx" ]]; then
            pipeline_apply_master_first_points_by_td
        fi
        pipeline_prepare_step_env "$step"
        echo
        echo "==> [$TYPE_SUBJ/$TYPE_SEQ] $step"
        DATASET="$DATASET" TYPE_SUBJ="$TYPE_SUBJ" TYPE_SEQ="$TYPE_SEQ" bash "$script"
    done
}
