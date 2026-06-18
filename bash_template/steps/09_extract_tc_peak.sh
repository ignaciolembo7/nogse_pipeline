#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/manifests/helpers/master_table_common.sh"
pipeline_maybe_step_help extract_tc_peak "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

# Derive FIT_OUT_ROOT using the same defaults as step 08 (fit_contrast).
if [[ "$TYPE_SEQ" == "nogse" ]]; then
    DEFAULT_FIT_MODEL="${DEFAULT_FIT_MODEL:-nogse_free}"
else
    DEFAULT_FIT_MODEL="${DEFAULT_FIT_MODEL:-ogse_free}"
fi
_master_name=$(basename "${MASTER_PARQUET:-master.long.parquet}" | sed 's/\.long\.parquet$//' | sed 's/\.parquet$//')
_ycol="${FIT_YCOL:-value_norm}"
_gtype="${FIT_GBASE:-g_lin_max}"
_gtype_clean="${_gtype//_/}"
_model="${FIT_MODEL:-$DEFAULT_FIT_MODEL}"
FIT_OUT_ROOT="${FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/$_master_name/${TYPE_SEQ}_${_ycol}_vs_${_gtype_clean}_${_model}}"

TC_PIPELINE_SCRIPT="${TC_PIPELINE_SCRIPT:-$REPO_ROOT/scripts/fitting/run_tc_pipeline.py}"
TC_PEAK_DIR="${TC_PEAK_DIR:-$FIT_OUT_ROOT/tc_peak}"

pipeline_require_file "$TC_PIPELINE_SCRIPT" "tc pipeline script"
pipeline_require_file "$MASTER_PARQUET" "master table"
if [[ ! -d "$FIT_OUT_ROOT" ]]; then
    echo "ERROR: contrast fits directory not found: $FIT_OUT_ROOT" >&2
    echo "  Run step 08 (fit_contrast) first." >&2
    exit 1
fi
mkdir -p "$TC_PEAK_DIR"

# Consolidate fit_params from step 08 into a canonical tc_peak table.
"$PY" "$TC_PIPELINE_SCRIPT" "$FIT_OUT_ROOT" \
    --out-xlsx "$TC_PEAK_DIR/tc_peak_table.xlsx" \
    --out-parquet "$TC_PEAK_DIR/tc_peak_table.parquet" \
    ${TC_PEAK_MODELS:+--models ${TC_PEAK_MODELS}} \
    ${TC_PEAK_SUBJS:+--subjs ${TC_PEAK_SUBJS}} \
    ${TC_PEAK_ROIS:+--rois ${TC_PEAK_ROIS}} \
    ${TC_PEAK_DIRECTIONS:+--directions ${TC_PEAK_DIRECTIONS}} \
    ${TC_PIPELINE_EXTRA_ARGS:-}

# Plot tc_peak panels (OGSE only).
if [[ "$TYPE_SEQ" == "ogse" ]]; then
    PLOT_TC_PEAKS_SCRIPT="${PLOT_TC_PEAKS_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_ogse-contrast_tc_peak_panels.py}"
    TC_PEAKS_OUT_DIR="${TC_PEAKS_OUT_DIR:-$TC_PEAK_DIR/panels}"
    TC_PEAKS_CONTRAST_ROOT="${TC_PEAKS_CONTRAST_ROOT:-$ANALYSIS_ROOT/contrast-data-resampled}"
    if [[ -f "$PLOT_TC_PEAKS_SCRIPT" ]]; then
        mkdir -p "$TC_PEAKS_OUT_DIR"
        _tc_args=("$FIT_OUT_ROOT"
            --contrast-root "$TC_PEAKS_CONTRAST_ROOT"
            --out-dir "$TC_PEAKS_OUT_DIR"
            --contrast-source "${TC_PEAKS_CONTRAST_SOURCE:-fitted_resampled}"
            --peak-source "${TC_PEAKS_PEAK_SOURCE:-resampled}"
        )
        [[ -n "${TC_PEAKS_MODELS:-}" ]] && _tc_args+=(--models ${TC_PEAKS_MODELS})
        [[ -n "${TC_PEAKS_SUBJS:-}" ]] && _tc_args+=(--subjs ${TC_PEAKS_SUBJS})
        [[ -n "${TC_PEAKS_ROIS:-}" ]] && _tc_args+=(--rois ${TC_PEAKS_ROIS})
        [[ -n "${TC_PEAKS_DIRECTIONS:-}" ]] && _tc_args+=(--directions ${TC_PEAKS_DIRECTIONS})
        [[ -n "${TC_PEAKS_N1:-}" ]] && _tc_args+=(--n1 "$TC_PEAKS_N1")
        [[ -n "${TC_PEAKS_N2:-}" ]] && _tc_args+=(--n2 "$TC_PEAKS_N2")
        [[ -n "${TC_PEAKS_X_VARS:-}" ]] && _tc_args+=(--x-vars ${TC_PEAKS_X_VARS})
        [[ "${TC_PEAKS_SHOW_RESAMPLED_FIT:-0}" == "1" ]] && _tc_args+=(--show-resampled-fit)
        [[ "${TC_PEAKS_HIDE_DATA_POINTS:-0}" == "1" ]] && _tc_args+=(--hide-data-points)
        [[ "${TC_PEAKS_EXCLUDE_FITRESAMP:-0}" == "1" ]] && _tc_args+=(--exclude-fitresamp)
        "$PY" "$PLOT_TC_PEAKS_SCRIPT" \
            "${_tc_args[@]}" \
            ${TC_PEAKS_EXTRA_ARGS:-}
    fi
fi
