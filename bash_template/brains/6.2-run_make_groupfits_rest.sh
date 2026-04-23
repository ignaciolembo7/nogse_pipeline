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

FIT_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/nogse_contrast_vs_g_rest_corr"
CONTRAST_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/contrast-data-rotated"
OUT_XLSX="$FIT_ROOT/groupfits_rest.xlsx"
OUT_PARQUET="$FIT_ROOT/groupfits_rest.parquet"
MODELS="rest"
SUBJS="BRAIN,LUDG,MBBL"
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Left-Lateral-Ventricle,Right-Lateral-Ventricle,Syringe"
DIRECTIONS="long,tra"
EXCLUDE_TD_MS=""
FIT_PANELS_OUT_DIR="$FIT_ROOT/contrast_fit_panels"
TC_PEAK_PANELS_OUT_DIR="$FIT_ROOT/tc_peak_panels"
X_VARS="g,Ld,lcf,lcf_a,tc"
PEAK_D0_FIX="2.3e-12"
PEAK_GAMMA="267.5221900"
TC_PEAK_XLIMS=(
    "g 0 80"
    "Ld 0 4"
    "lcf 0 20"
    "lcf_a 0.25 1.250"
    "tc 0 50"
)

# ------------------------------------------------------------------
# ------------------------------------------------------------------

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

if [[ ! -d "$FIT_ROOT" ]]; then
    echo "ERROR: Fit root not found: $FIT_ROOT" >&2
    exit 1
fi

if [[ ! -d "$CONTRAST_ROOT" ]]; then
    echo "ERROR: Contrast root not found: $CONTRAST_ROOT" >&2
    exit 1
fi

mkdir -p "$FIT_ROOT"

echo "============================================================"
echo "Dataset       : brains"
echo "Fit root      : $FIT_ROOT"
echo "Contrast root : $CONTRAST_ROOT"
echo "Models        : $MODELS"
echo "Subjs         : $SUBJS"
echo "ROIs          : $ROIS"
echo "Directions    : $DIRECTIONS"
echo "Output XLSX   : $OUT_XLSX"
echo "Output Parquet: $OUT_PARQUET"
echo "Fit panels    : $FIT_PANELS_OUT_DIR"
echo "tc_peak panels: $TC_PEAK_PANELS_OUT_DIR"
echo "tc_peak x vars: $X_VARS"

"$PY" "$PIPELINE_SCRIPT" \
    "$FIT_ROOT" \
    --models "$MODELS" \
    --out-xlsx "$OUT_XLSX" \
    --out-parquet "$OUT_PARQUET"

echo
echo "Generating fit panels..."
PY="$PY" \
FITS_ROOT="$FIT_ROOT" \
CONTRAST_ROOT="$CONTRAST_ROOT" \
OUT_DIR="$FIT_PANELS_OUT_DIR" \
MODELS="$MODELS" \
SUBJS="$SUBJS" \
ROIS="$ROIS" \
DIRECTIONS="$DIRECTIONS" \
EXCLUDE_TD_MS="$EXCLUDE_TD_MS" \
bash "$PLOT_FIT_PANELS_SCRIPT"

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

"$PY" "$PLOT_TC_PEAK_PANELS_SCRIPT" \
    "$FIT_ROOT" \
    --contrast-root "$CONTRAST_ROOT" \
    --out-dir "$TC_PEAK_PANELS_OUT_DIR" \
    --peak-D0-fix "$PEAK_D0_FIX" \
    --peak-gamma "$PEAK_GAMMA" \
    "${tc_peak_args[@]}"

echo
echo "Finished."
