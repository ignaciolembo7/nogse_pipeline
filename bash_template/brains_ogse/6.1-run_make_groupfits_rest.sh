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
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments}"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}" # direct or fitted_resampled
TC_ANALYSIS="${TC_ANALYSIS:-tc_fit}" # tc_peak or tc_fit
CONTRAST_MODEL="${CONTRAST_MODEL:-${MODEL:-${SIGNAL_MODEL:-rest_offset_globC}}}" # free, rest, rest_offset, rest_offset_globC, mixed_global
SIGNAL_MODEL="$CONTRAST_MODEL"
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g_thorsten}"
FIT_CORR="${FIT_CORR:-${USE_CORR:-true}}"
MODELS="${MODELS:-$CONTRAST_MODEL}"
SUBJS="BRAIN,LUDG,MBBL"
ROIS="${ROIS:-AntCC,MidAntCC,CentralCC,MidPostCC,PostCC}"
# ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Left-Lateral-Ventricle,Right-Lateral-Ventricle,Syringe"
DIRECTIONS="long,tra"
EXCLUDE_TD_MS="76"
X_VARS="g,Ld,lcf,Lcf,tc"
N1="${N1:-8}"
N2="${N2:-4}"
TC_PEAK_MARKER_X_VARS="tc"
RESAMPLED_GRID_N="${RESAMPLED_GRID_N:-1000}"
PEAK_D0_FIX="3.2e-12"
PEAK_GAMMA="267.5221900"
TC_PEAK_XLIMS=(
    "Ld 0 4"
    "lcf 2.5 14"
    "lcf_a 0.25 1.250"
    "tc 0 50"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
while (( $# > 0 )); do
    case "$1" in
        --corr)
            FIT_CORR=true
            shift
            ;;
        --no-corr|--sin-corr|--uncorr)
            FIT_CORR=false
            shift
            ;;
        *)
            echo "ERROR: Unknown argument for $0: $1" >&2
            echo "Use --corr or --no-corr. Other settings are controlled with environment variables." >&2
            exit 1
            ;;
    esac
done
if [[ "$CONTRAST_SOURCE" != "direct" && "$CONTRAST_SOURCE" != "fitted_resampled" ]]; then
    echo "ERROR: CONTRAST_SOURCE must be 'direct' or 'fitted_resampled'. Got: $CONTRAST_SOURCE" >&2
    exit 1
fi
if [[ "$TC_ANALYSIS" != "tc_peak" && "$TC_ANALYSIS" != "tc_fit" ]]; then
    echo "ERROR: TC_ANALYSIS must be 'tc_peak' or 'tc_fit'. Got: $TC_ANALYSIS" >&2
    exit 1
fi
if [[ -z "${FITS_ROOT_SUFFIX+x}" ]]; then
    if [[ "$CONTRAST_MODEL" == "mixed_global" ]]; then
        FITS_ROOT_SUFFIX=""
    else
        case "${FIT_CORR,,}" in
            1|true|yes|y|corr)
                FITS_ROOT_SUFFIX="_corr"
                ;;
            0|false|no|n|none|uncorr|no-corr|sin-corr)
                FITS_ROOT_SUFFIX=""
                ;;
            *)
                echo "ERROR: FIT_CORR must be true/false. Got: $FIT_CORR" >&2
                exit 1
                ;;
        esac
    fi
fi

if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    PLOT_FIT_PANELS="${PLOT_FIT_PANELS:-false}"
else
    PLOT_FIT_PANELS="${PLOT_FIT_PANELS:-true}"
fi
if [[ -z "${PLOT_TC_PEAK_PANELS+x}" ]]; then
    if [[ "$TC_ANALYSIS" == "tc_fit" ]]; then
        PLOT_TC_PEAK_PANELS=false
    else
        PLOT_TC_PEAK_PANELS=true
    fi
fi

if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    FITS_INPUT_ROOT="${FITS_INPUT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_gresampled_${CONTRAST_MODEL}${FITS_ROOT_SUFFIX}}"
    SUMMARY_ROOT="${SUMMARY_ROOT:-$FITS_INPUT_ROOT}"
    CONTRAST_ROOT="${CONTRAST_ROOT:-$FITS_INPUT_ROOT/contrast}"
else
    FITS_INPUT_ROOT="${FITS_INPUT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_g_${CONTRAST_MODEL}${FITS_ROOT_SUFFIX}}"
    SUMMARY_ROOT="${SUMMARY_ROOT:-$FITS_INPUT_ROOT}"
    CONTRAST_ROOT="${CONTRAST_ROOT:-$ANALYSIS_ROOT/contrast-data-rotated}"
fi
if [[ "$CONTRAST_MODEL" == "ALL" ]]; then
    FIT_PARAMS_PATTERN="${FIT_PARAMS_PATTERN:-**/fit_params.*}"
else
    FIT_PARAMS_PATTERN="${FIT_PARAMS_PATTERN:-**/fit_params.${CONTRAST_MODEL}.${SIGNAL_G_TYPE}.value_norm.direction_*.parquet}"
fi


GROUPFITS_TAG=""
if [[ "$TC_ANALYSIS" == "tc_fit" ]]; then
    GROUPFITS_TAG="_tcfit"
fi
OUT_XLSX="$SUMMARY_ROOT/groupfits_rest${GROUPFITS_TAG}.xlsx"
OUT_PARQUET="$SUMMARY_ROOT/groupfits_rest${GROUPFITS_TAG}.parquet"
FIT_PANELS_OUT_DIR="$SUMMARY_ROOT/contrast_fit_panels"
TC_PEAK_PANELS_OUT_DIR="$SUMMARY_ROOT/tc_peak_panels"

# Allow EXCLUDE_TD_MS to remain commented out or empty without breaking `set -u`.
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
    echo "ERROR: Fit-parameter input root not found: $FITS_INPUT_ROOT" >&2
    exit 1
fi

NEEDS_CONTRAST_ROOT=false
if [[ "${PLOT_FIT_PANELS,,}" == "true" || "${PLOT_TC_PEAK_PANELS,,}" == "true" ]]; then
    NEEDS_CONTRAST_ROOT=true
fi
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" && "$TC_ANALYSIS" == "tc_peak" ]]; then
    NEEDS_CONTRAST_ROOT=true
fi
if [[ "$NEEDS_CONTRAST_ROOT" == "true" && ! -d "$CONTRAST_ROOT" ]]; then
    echo "ERROR: Contrast root not found: $CONTRAST_ROOT" >&2
    exit 1
fi

mkdir -p "$SUMMARY_ROOT"

echo "============================================================"
echo "Dataset       : brains"
echo "Fit input root: $FITS_INPUT_ROOT"
echo "Fit pattern   : $FIT_PARAMS_PATTERN"
echo "Summary root  : $SUMMARY_ROOT"
echo "Contrast root : $CONTRAST_ROOT"
echo "Contrast source: $CONTRAST_SOURCE"
echo "TC analysis    : $TC_ANALYSIS"
echo "Contrast model : $CONTRAST_MODEL"
echo "Fit corr      : $FIT_CORR"
echo "Fits suffix    : ${FITS_ROOT_SUFFIX:-<none>}"
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

pipeline_args=()
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    if [[ "$TC_ANALYSIS" == "tc_fit" ]]; then
        pipeline_args+=(--only-fitresamp)
    else
        pipeline_args+=(--exclude-fitresamp)
        pipeline_args+=(
            --add-resampled-data-peaks
            --contrast-root "$CONTRAST_ROOT"
            --peak-D0-fix "$PEAK_D0_FIX"
            --peak-gamma "$PEAK_GAMMA"
        )
    fi
fi
if [[ "$MODELS" != "ALL" ]]; then
    read -r -a pipeline_model_list <<< "${MODELS//,/ }"
    if (( ${#pipeline_model_list[@]} > 0 )); then
        pipeline_args+=(--models "${pipeline_model_list[@]}")
    fi
fi

"$PY" "$PIPELINE_SCRIPT" \
    "$FITS_INPUT_ROOT" \
    --pattern "$FIT_PARAMS_PATTERN" \
    --out-xlsx "$OUT_XLSX" \
    --out-parquet "$OUT_PARQUET" \
    "${pipeline_args[@]}"

if [[ "${PLOT_FIT_PANELS,,}" == "true" ]]; then
    echo
    echo "Generating fit panels..."
    PY="$PY" \
    FITS_ROOT="$FITS_INPUT_ROOT" \
    CONTRAST_ROOT="$CONTRAST_ROOT" \
    OUT_DIR="$FIT_PANELS_OUT_DIR" \
    MODELS="$MODELS" \
    FIT_PARAMS_PATTERN="$FIT_PARAMS_PATTERN" \
    SUBJS="$SUBJS" \
    ROIS="$ROIS" \
    DIRECTIONS="$DIRECTIONS" \
    EXCLUDE_TD_MS="$EXCLUDE_TD_MS" \
    bash "$PLOT_FIT_PANELS_SCRIPT"
fi

if [[ "${PLOT_TC_PEAK_PANELS,,}" != "true" ]]; then
    echo
    echo "Skipping tc_peak panels."
    echo
    echo "Finished."
    exit 0
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
        --contrast-source fitted_resampled
        --exclude-fitresamp
        --n1 "$N1"
        --n2 "$N2"
        --peak-source resampled
        --resampled-curve-n "$RESAMPLED_GRID_N"
    )
else
    resampled_panel_args=(
        --contrast-source direct
        --n1 "$N1"
        --n2 "$N2"
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
