#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-python}"
PIPELINE_SCRIPT="$REPO_ROOT/scripts/run_tc_pipeline.py"
PLOT_FIT_PANELS_SCRIPT="$REPO_ROOT/bash_template/helpers/run_plot_ogse_contrast_fit_panels.sh"
PLOT_TC_PEAK_PANELS_SCRIPT="$REPO_ROOT/scripts/plot_ogse-contrast_tc_peak_panels.py"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms-3/ogse_experiments}"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}" # direct or fitted_resampled
SIGNAL_MODEL="${SIGNAL_MODEL:-}"
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g}"
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    SIGNAL_MODEL="${SIGNAL_MODEL:-rest_offset}"
    FITS_INPUT_ROOT="${FITS_INPUT_ROOT:-$ANALYSIS_ROOT/contrast-data/tables}"
    SUMMARY_ROOT="${SUMMARY_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_gresampled_${SIGNAL_MODEL}_corr}"
    CONTRAST_ROOT="${CONTRAST_ROOT:-$ANALYSIS_ROOT/contrast-data}"
    FIT_PARAMS_PATTERN="${FIT_PARAMS_PATTERN:-**/fit_params.${SIGNAL_MODEL}.g.value_norm.direction_*.parquet}"
else
    SIGNAL_MODEL="${SIGNAL_MODEL:-rest}"
    FITS_INPUT_ROOT="${FITS_INPUT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_g_rest_corr}"
    SUMMARY_ROOT="${SUMMARY_ROOT:-$FITS_INPUT_ROOT}"
    CONTRAST_ROOT="${CONTRAST_ROOT:-$ANALYSIS_ROOT/contrast-data}"
    FIT_PARAMS_PATTERN="${FIT_PARAMS_PATTERN:-**/fit_params.*}"
fi
OUT_XLSX="$SUMMARY_ROOT/groupfits_rest.xlsx"
OUT_PARQUET="$SUMMARY_ROOT/groupfits_rest.parquet"
MODELS="${MODELS:-$SIGNAL_MODEL}"
SUBJS="ALL"
ROIS="${ROIS:-fiber1,fiber2}"
DIRECTIONS="ALL"
EXCLUDE_TD_MS="209.1"
FIT_PANELS_OUT_DIR="$SUMMARY_ROOT/contrast_fit_panels"
TC_PEAK_PANELS_OUT_DIR="$SUMMARY_ROOT/tc_peak_panels"
X_VARS="g,Ld,lcf,Lcf,tc"
TC_PEAK_MARKER_X_VARS="tc" # NONE ALL
RESAMPLED_GRID_N="${RESAMPLED_GRID_N:-1000}"
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    PLOT_FIT_PANELS="${PLOT_FIT_PANELS:-false}"
else
    PLOT_FIT_PANELS="${PLOT_FIT_PANELS:-true}"
fi
PEAK_D0_FIX="2.3e-12"
PEAK_GAMMA="267.5221900"
TC_PEAK_XLIMS=(
    "g 0 80"
    "Ld 0 4"
    "lcf 2.5 12.5"
    "lcf_a 0 2"
    "tc 0 100"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------

EXCLUDE_TD_MS="${EXCLUDE_TD_MS:-}"

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PIPELINE_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_FIT_PANELS_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_FIT_PANELS_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_TC_PEAK_PANELS_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_TC_PEAK_PANELS_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$FITS_INPUT_ROOT" ]]; then
    echo "Fit-parameter input root not found: $FITS_INPUT_ROOT. Skipping grouped summaries."
    exit 0
fi

if [[ ! -d "$CONTRAST_ROOT" ]]; then
    echo "Contrast root not found: $CONTRAST_ROOT. Skipping grouped summaries."
    exit 0
fi

if [[ -z "$(find "$FITS_INPUT_ROOT" -type f -name 'fit_params*.parquet' -print -quit)" ]]; then
    echo "No fit_params were found in $FITS_INPUT_ROOT. Skipping grouped summaries."
    exit 0
fi

mkdir -p "$SUMMARY_ROOT"

echo "============================================================"
echo "Dataset       : phantoms-3"
echo "Fit input root: $FITS_INPUT_ROOT"
echo "Summary root  : $SUMMARY_ROOT"
echo "Contrast root : $CONTRAST_ROOT"
echo "Contrast source: $CONTRAST_SOURCE"
echo "Models        : $MODELS"
echo "Subjs         : $SUBJS"
echo "ROIs          : $ROIS"
echo "Directions    : $DIRECTIONS"
echo "Output XLSX   : $OUT_XLSX"
echo "Output Parquet: $OUT_PARQUET"
echo "Fit panels    : $FIT_PANELS_OUT_DIR"
echo "tc_peak panels: $TC_PEAK_PANELS_OUT_DIR"
echo "tc_peak x vars: $X_VARS"
echo "tc_peak marker x vars: $TC_PEAK_MARKER_X_VARS"

"$PY" "$PIPELINE_SCRIPT" \
    "$FITS_INPUT_ROOT" \
    --pattern "$FIT_PARAMS_PATTERN" \
    --models "$MODELS" \
    --out-xlsx "$OUT_XLSX" \
    --out-parquet "$OUT_PARQUET"

if [[ "${PLOT_FIT_PANELS,,}" == "true" ]]; then
    echo
    echo "Generating fit panels..."
    PY="$PY" \
    FITS_ROOT="$FITS_INPUT_ROOT" \
    CONTRAST_ROOT="$CONTRAST_ROOT" \
    OUT_DIR="$FIT_PANELS_OUT_DIR" \
    MODELS="$MODELS" \
    SUBJS="$SUBJS" \
    ROIS="$ROIS" \
    DIRECTIONS="$DIRECTIONS" \
    EXCLUDE_TD_MS="$EXCLUDE_TD_MS" \
    bash "$PLOT_FIT_PANELS_SCRIPT"
fi

echo
echo "Generating tc_peak panels..."
tc_peak_args=()
if [[ "$MODELS" != "ALL" ]]; then
    read -r -a model_list <<< "${MODELS//,/ }"
    if (( ${#model_list[@]} > 0 )); then
        tc_peak_args+=(--models "${model_list[@]}")
    fi
fi
if [[ "$SUBJS" != "ALL" ]]; then
    read -r -a subj_list <<< "${SUBJS//,/ }"
    if (( ${#subj_list[@]} > 0 )); then
        tc_peak_args+=(--subjs "${subj_list[@]}")
    fi
fi
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        tc_peak_args+=(--rois "${roi_list[@]}")
    fi
fi
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        tc_peak_args+=(--directions "${dir_list[@]}")
    fi
fi
if [[ "$X_VARS" != "ALL" ]]; then
    read -r -a xvar_list <<< "${X_VARS//,/ }"
    if (( ${#xvar_list[@]} > 0 )); then
        tc_peak_args+=(--x-vars "${xvar_list[@]}")
    fi
fi
if [[ -n "${TC_PEAK_MARKER_X_VARS// }" ]]; then
    read -r -a peak_marker_xvar_list <<< "${TC_PEAK_MARKER_X_VARS//,/ }"
    if (( ${#peak_marker_xvar_list[@]} > 0 )); then
        tc_peak_args+=(--peak-marker-x-vars "${peak_marker_xvar_list[@]}")
    fi
fi
if [[ -n "${EXCLUDE_TD_MS// }" ]]; then
    read -r -a exclude_td_list <<< "${EXCLUDE_TD_MS//,/ }"
    if (( ${#exclude_td_list[@]} > 0 )); then
        tc_peak_args+=(--exclude-td-ms "${exclude_td_list[@]}")
    fi
fi
for xlim_spec in "${TC_PEAK_XLIMS[@]}"; do
    read -r xvar xmin xmax <<< "$xlim_spec"
    tc_peak_args+=(--xlim "$xvar" "$xmin" "$xmax")
done
resampled_panel_args=()
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    resampled_panel_args=(
        --show-resampled-fit
        --hide-data-points
        --peak-source resampled
        --resampled-curve-n "$RESAMPLED_GRID_N"
    )
fi

"$PY" "$PLOT_TC_PEAK_PANELS_SCRIPT" \
    "$FITS_INPUT_ROOT" \
    --contrast-root "$CONTRAST_ROOT" \
    --out-dir "$TC_PEAK_PANELS_OUT_DIR" \
    --pattern "$FIT_PARAMS_PATTERN" \
    --peak-D0-fix "$PEAK_D0_FIX" \
    --peak-gamma "$PEAK_GAMMA" \
    "${resampled_panel_args[@]}" \
    "${tc_peak_args[@]}"

echo
echo "Finished."
