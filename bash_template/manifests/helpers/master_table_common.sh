#!/usr/bin/env bash

pipeline_setup_common() {
    MASTER_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TEMPLATE_ROOT="$(cd "$MASTER_HELPER_DIR/.." && pwd)"
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
    MASTER_FIT_PARAMS="${MASTER_FIT_PARAMS:-$ANALYSIS_ROOT/master_fit_params.parquet}"
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
    MASTER_FIT_PARAMS_ORIGINAL="${MASTER_FIT_PARAMS_ORIGINAL:-$MASTER_FIT_PARAMS}"
    MASTER_PARQUET="${MASTER_FIRST_POINTS_PARQUET:-$out_dir/master.first_points.long.parquet}"
    MASTER_FIT_PARAMS="${MASTER_FIRST_POINTS_FIT_PARAMS:-$out_dir/master.first_points_fit_params.parquet}"

    "$PY" "$filter_script" "$MASTER_PARQUET_ORIGINAL" \
        --out-parquet "$MASTER_PARQUET" \
        --first-points-by-td "$spec"

    MASTER_FIRST_POINTS_APPLIED=1
    export MASTER_PARQUET MASTER_FIT_PARAMS MASTER_PARQUET_ORIGINAL MASTER_FIT_PARAMS_ORIGINAL MASTER_FIRST_POINTS_APPLIED
    echo "Using filtered master table: $MASTER_PARQUET"
    echo "First points by td_ms: $spec"
}

pipeline_usage() {
    cat <<'EOF'
Usage:
  bash nogse_pipeline/bash_template/run_dataset.sh <type_subj> <type_seq> <step...>

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

  Plots:
    plot_signal      Plot signal curves from master.long.parquet.
    plot_contrast    Plot contrast curves from master.long.parquet.
    plot_d0_delta    Plot D0/Dproj vs Delta_app_ms.
    plot_monoexp_d   Plot monoexponential D vs td_ms/Delta_app_ms.

  Fits:
    fit_signal                 Fit signal curves using signal_fits.csv.
    fit_signal_monoexp         fit_signal with OGSE monoexp defaults.
    fit_signal_gradcorr        fit_signal with gradient correction enabled.
    fit_contrast               Fit contrast rows.
    fit_contrast_free          fit_contrast with free-model defaults.
    fit_contrast_mixed_global  fit_contrast with mixed_global model.
    fit_global_signal          Fit mixed/global signal models directly.

  Summaries:
    grad_correction  Build gradient-correction table.
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
  MASTER_FIT_PARAMS  Cumulative fit-params table.
  MANIFEST_DIR       Directory with contrasts.csv and signal_fits.csv.
  MASTER_FIRST_POINTS_BY_TD
                    Optional TD=POINTS rules applied before post-ingest steps.
                    Example: MASTER_FIRST_POINTS_BY_TD="120=8,210=6"
                    Unlisted td_ms values keep all points.
  MASTER_FIRST_POINTS_PARQUET
                    Optional filtered master output path. Default:
                    $ANALYSIS_ROOT/master.first_points.long.parquet.

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
  FILTERED_MASTER_PARQUET     Output intermediate master table. Default:
                              $ANALYSIS_ROOT/master.first_points.long.parquet
  FILTER_MASTER_SCRIPT        Python script override.

Examples:
  MASTER_FIRST_POINTS_BY_TD="120=8,210=6,90=ALL" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse filter_master_points

  MASTER_PARQUET=analysis/brains/ogse_experiments/master.first_points.long.parquet \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate contrast fit_signal
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
  PLOT_ROW_KIND           Rows to plot. Default: signal_rotated
  PLOT_SUBJ               Optional subj selector.
  PLOT_SHEET              Optional sheet selector.
  PLOT_ROI                Optional ROI selector.
  PLOT_DIRECTION          Optional direction selector, e.g. long, tra, x, y, z.
  PLOT_TD_MS              Optional td_ms selector.
  PLOT_N                  Optional N selector.
  PLOT_SIGNAL_YCOL        Y column. Default: value_norm
  PLOT_SIGNAL_XCOL        X column. Default: g_thorsten for OGSE, g for NOGSE
  PLOT_SIGNAL_G_TYPE      Backward-compatible alias for PLOT_SIGNAL_XCOL.
  PLOT_STAT               Statistic selector. Default: avg
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
  PLOT_DIRECTION            Optional direction selector.
  PLOT_TD_MS                Optional td_ms selector.
  PLOT_N1                  Optional N_1 selector.
  PLOT_N2                  Optional N_2 selector.
  PLOT_CONTRAST_YCOL        Y column. Default: value_norm
  PLOT_CONTRAST_XCOL        X column. Default: g_thorsten_1
  PLOT_STAT                 Statistic selector. Default: avg
  PLOT_CONTRAST_EXTRA_ARGS  Extra plot_<type_seq>_contrast_vs_g.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast

  PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=tra \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_contrast
EOF
            ;;
        fit_signal|fit_signal_monoexp|fit_signal_gradcorr)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> fit_signal
  bash run_dataset.sh <type_subj> <type_seq> fit_signal_monoexp
  bash run_dataset.sh <type_subj> <type_seq> fit_signal_gradcorr

What it does:
  Fits signal curves selected from master according to manifests/<type_subj>_<type_seq>/signal_fits.csv.

Variables for this step:
  MASTER_PARQUET          Input master table.
  MASTER_FIT_PARAMS       Cumulative fit-params table.
  FIT_SIGNAL_SCRIPT       Python script override.
  SIGNAL_FIT_MANIFEST     CSV signal-fit manifest.
  SIGNAL_FIT_OUT_ROOT     Output root. Default: $ANALYSIS_ROOT/fits/<type_seq>_signal_master
  SIGNAL_FIT_MODEL        Default: monoexp for OGSE, nogse_free for NOGSE.
  SIGNAL_FIT_G_TYPE       Default: bvalue_thorsten for OGSE, g for NOGSE.
  SIGNAL_FIT_XCOL         NOGSE x-axis override. Defaults to SIGNAL_FIT_G_TYPE.
  SIGNAL_FIT_YCOL         Y column. Default: value_norm
  SIGNAL_FIT_EXTRA_ARGS   Extra fit_<type_seq>_signal_vs_g.py options.
  CORR_ROI                ROI used by fit_signal_gradcorr. Default: Syringe for brains, water for phantoms.
  CORR_XLSX               Correction table used by fit_signal_gradcorr.

Manifest columns:
  subj,sheet,roi,direction,td_ms,N,Hz,model

Useful SIGNAL_FIT_EXTRA_ARGS:
  OGSE examples: --fix_M0 1.0 --auto_fit_tol 0.05 --auto_fit_min_points 3 --auto_fit_max_points 9
  NOGSE examples depend on scripts/fitting/fit_nogse_signal_vs_g.py --help.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_monoexp

  SIGNAL_FIT_MANIFEST=nogse_pipeline/bash_template/manifests/brains_ogse/signal_fits.csv \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

  SIGNAL_FIT_MODEL=monoexp SIGNAL_FIT_G_TYPE=bvalue_thorsten \
  SIGNAL_FIT_EXTRA_ARGS="--fix_M0 1.0 --auto_fit_tol 0.05" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal

  CORR_ROI=Syringe \
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
  Fits contrast rows from master.long.parquet and appends useful params to master_fit_params.parquet.

Variables for this step:
  MASTER_PARQUET        Input master table.
  MASTER_FIT_PARAMS     Cumulative fit-params table.
  FIT_CONTRAST_SCRIPT   Python script override.
  FIT_OUT_ROOT          Output root. Default: $ANALYSIS_ROOT/fits/<type_seq>_contrast_master
  FIT_MODEL             Default: ogse_free for OGSE, nogse_free for NOGSE.
  FIT_GBASE             Gradient axis. Default: g_lin_max
  FIT_YCOL              Y column. Default: value_norm
  FIT_STAT              Statistic selector. Default: avg
  FIT_EXTRA_ARGS        Extra fit_<type_seq>_contrast_vs_g.py options.

Useful FIT_EXTRA_ARGS:
  Use the Python script help for model-specific flags:
    python scripts/fitting/fit_ogse_contrast_vs_g.py --help
    python scripts/fitting/fit_nogse_contrast_vs_g.py --help

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free

  FIT_MODEL=mixed_global FIT_GBASE=g_thorsten_1 FIT_YCOL=value_norm \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast

  FIT_EXTRA_ARGS="--apply_grad_corr --corr_roi Syringe --corr_xlsx analysis/brains/ogse_experiments/fits/grad_correction_master/Syringe.grad_correction.xlsx" \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast
EOF
            ;;
        fit_global_signal)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> fit_global_signal

What it does:
  Fits mixed/global signal models directly from master.long.parquet.

Variables for this step:
  MASTER_PARQUET                 Input master table.
  FIT_GLOBAL_SIGNAL_SCRIPT       Python script override.
  GLOBAL_SIGNAL_OUT_ROOT         Output root. Default: $ANALYSIS_ROOT/fits/<type_seq>_signal_mixed_global_master
  GLOBAL_SIGNAL_ROW_KIND         Row kind to fit. Default: signal_rotated
  GLOBAL_SIGNAL_MODEL            Default: ogse_mixed_offset for OGSE, nogse_mixed_offset for NOGSE.
  GLOBAL_SIGNAL_YCOL             Y column. Default: value
  GLOBAL_SIGNAL_G_TYPE           Gradient column. Default depends on type_subj/type_seq.
  GLOBAL_SIGNAL_STAT             Statistic selector. Default: avg
  GLOBAL_SIGNAL_MIN_POINTS       Minimum points per group. Default: 4
  GLOBAL_SIGNAL_TC_MODE          fixed|free|global_td|global_contrast. Default: global_td
  GLOBAL_SIGNAL_ALPHA_MODE       fixed|free|global_td|global_contrast. Default: global_td
  GLOBAL_SIGNAL_RN_MODE          fixed|free|global_td|global_contrast. Default: global_td
  GLOBAL_SIGNAL_M0_MODE          fixed|free|global_td|global_contrast. Default: global_contrast
  GLOBAL_SIGNAL_C_MODE           fixed|free|global_td|global_contrast. Default: global_contrast
  GLOBAL_SIGNAL_D0_MODE          fixed|free|global_td|global_contrast. Default: fixed
  GLOBAL_SIGNAL_D0_FIXED         Fixed D0 value. Default: brain 3.2e-12, phantom 2.3e-12
  GLOBAL_SIGNAL_DIRECTIONS       Space- or comma-separated directions, or ALL.
  GLOBAL_SIGNAL_ROIS             Space- or comma-separated ROIs, or ALL.
  GLOBAL_SIGNAL_SUBJS            Space- or comma-separated subjects, or ALL.
  GLOBAL_SIGNAL_APPLY_GRAD_CORR  true|false. Default: false
  GLOBAL_SIGNAL_CORR_XLSX        Correction table path.
  GLOBAL_SIGNAL_CORR_ROI         Correction ROI. Default: Syringe for brains, water for phantoms.
  GLOBAL_SIGNAL_EXTRA_ARGS       Extra fit_global_signal.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal

  GLOBAL_SIGNAL_DIRECTIONS="long tra" GLOBAL_SIGNAL_ROIS=Left-Lateral-Ventricle \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal

  GLOBAL_SIGNAL_APPLY_GRAD_CORR=true GLOBAL_SIGNAL_CORR_ROI=Syringe \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal
EOF
            ;;
        grad_correction)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> grad_correction

What it does:
  Builds the gradient-correction table used by corrected signal/contrast fits.

Variables for this step:
  GRAD_CORR_SCRIPT      Python script override.
  GRAD_CORR_ROI         Default: Syringe for brains, water for phantoms.
  SIGNAL_FITS_ROOT      Signal fit root. Default: $ANALYSIS_ROOT/fits/<type_seq>_signal_master
  CONTRAST_FITS_ROOT    Contrast fit root. Default: $ANALYSIS_ROOT/fits/<type_seq>_contrast_master
  CONTRAST_DATA_ROOT    Contrast data root. Default: $ANALYSIS_ROOT/contrast-data-master
  GRAD_CORR_OUT_DIR     Output directory. Default: $ANALYSIS_ROOT/fits/grad_correction_master
  GRAD_CORR_EXTRA_ARGS  Extra make_grad_correction_table.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction

  GRAD_CORR_ROI=Syringe \
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
  SIGNAL_FITS_ROOT          Signal fit root. Default: $ANALYSIS_ROOT/fits/<type_seq>_signal_master
  MONOEXP_D_OUT_DIR         Output plot/table directory. Default: $ANALYSIS_ROOT/plots-master/monoexp_D_vs_time
  PLOT_MONOEXP_D_EXTRA_ARGS Extra plot_monoexp_D_vs_time.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d

  SIGNAL_FITS_ROOT=analysis/brains/ogse_experiments/fits/ogse_signal_master \
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
  MASTER_FIT_PARAMS    Cumulative fit-params table.
  ALPHA_MACRO_SCRIPT   Python script override.
  ALPHA_N              N selector. Default: 1
  ALPHA_OUT_DIR        Output directory. Default: $ANALYSIS_ROOT/alpha_macro/master
  ALPHA_EXTRA_ARGS     Extra make_alpha_macro_summary.py options.

Outputs:
  $ALPHA_OUT_DIR/summary_alpha_values.xlsx
  $ALPHA_OUT_DIR/D_vs_delta_app.combined.xlsx

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha

  ALPHA_N=1 ALPHA_OUT_DIR=analysis/brains/ogse_experiments/alpha_macro/master \
    bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha
EOF
            ;;
        tc)
            cat <<'EOF'
Usage:
  bash run_dataset.sh <type_subj> <type_seq> tc

What it does:
  Fits tc-vs-td summaries from master_fit_params.parquet.

Variables for this step:
  MASTER_FIT_PARAMS  Input cumulative fit-params table.
  TC_VS_TD_SCRIPT    Python script override.
  TC_OUT_DIR         Output root. Default: $ANALYSIS_ROOT/fits/tc_vs_td_master
  TC_METHOD          Method. Default: pseudohuber_fixed_macro
  TC_Y_COL           Y column. Default: tc_peak_ms
  TC_EXTRA_ARGS      Extra run_tc_vs_td.py options.

Examples:
  bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc

  TC_METHOD=pseudohuber_fixed_macro TC_Y_COL=tc_peak_ms \
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
        rotate) echo "$TEMPLATE_ROOT/steps/02_rotate_signals.sh" ;;
        contrast) echo "$TEMPLATE_ROOT/steps/03_make_contrasts.sh" ;;
        plot_signal) echo "$TEMPLATE_ROOT/steps/04_plot_signals.sh" ;;
        plot_contrast) echo "$TEMPLATE_ROOT/steps/05_plot_contrasts.sh" ;;
        fit_signal|fit_signal_monoexp|fit_signal_gradcorr) echo "$TEMPLATE_ROOT/steps/06_fit_signals.sh" ;;
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
        fit_signal_monoexp)
            SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-monoexp}"
            SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-bvalue_thorsten}"
            export SIGNAL_FIT_MODEL SIGNAL_FIT_G_TYPE
            ;;
        fit_signal_gradcorr)
            if [[ "$TYPE_SEQ" == "nogse" ]]; then
                SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-nogse_free}"
                SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-g}"
            else
                SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-monoexp}"
                SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-bvalue_thorsten}"
            fi
            if [[ "$DATASET" == "brains" ]]; then
                CORR_ROI="${CORR_ROI:-Syringe}"
            else
                CORR_ROI="${CORR_ROI:-water}"
            fi
            CORR_XLSX="${CORR_XLSX:-$ANALYSIS_ROOT/fits/grad_correction_master/${CORR_ROI}.grad_correction.xlsx}"
            SIGNAL_FIT_EXTRA_ARGS="${SIGNAL_FIT_EXTRA_ARGS:-} --apply_grad_corr --corr_xlsx $CORR_XLSX --corr_roi $CORR_ROI"
            export SIGNAL_FIT_MODEL SIGNAL_FIT_G_TYPE CORR_ROI CORR_XLSX SIGNAL_FIT_EXTRA_ARGS
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
        if [[ "$step" != "ingest" && "$step" != "filter_master_points" ]]; then
            pipeline_apply_master_first_points_by_td
        fi
        pipeline_prepare_step_env "$step"
        echo
        echo "==> [$TYPE_SUBJ/$TYPE_SEQ] $step"
        DATASET="$DATASET" TYPE_SUBJ="$TYPE_SUBJ" TYPE_SEQ="$TYPE_SEQ" bash "$script"
    done
}
