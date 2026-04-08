#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
PIPELINE_SCRIPT="$REPO_ROOT/scripts/run_tc_pipeline.py"
PLOT_FIT_PANELS_SCRIPT="$REPO_ROOT/bash/run_plot_ogse_contrast_fit_panels.sh"
PLOT_TC_PEAK_PANELS_SCRIPT="$REPO_ROOT/bash/run_plot_ogse_tc_peak_panels.sh"

FIT_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr"
CONTRAST_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/contrast-data-rotated"
OUT_XLSX="$FIT_ROOT/groupfits_rest.xlsx"
OUT_PARQUET="$FIT_ROOT/groupfits_rest.parquet"
MODELS="rest"
SUBJS="BRAIN,LUDG,MBBL"
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Left-Lateral-Ventricle,Right-Lateral-Ventricle"
DIRECTIONS="long,tra"
FIT_PANELS_OUT_DIR="$FIT_ROOT/contrast_fit_panels"
TC_PEAK_PANELS_OUT_DIR="$FIT_ROOT/tc_peak_panels"

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
bash "$PLOT_FIT_PANELS_SCRIPT"

echo
echo "Generating tc_peak panels..."
PY="$PY" \
FITS_ROOT="$FIT_ROOT" \
CONTRAST_ROOT="$CONTRAST_ROOT" \
OUT_DIR="$TC_PEAK_PANELS_OUT_DIR" \
MODELS="$MODELS" \
SUBJS="$SUBJS" \
ROIS="$ROIS" \
DIRECTIONS="$DIRECTIONS" \
X_VARS="g,Ld,lcf,Lcf" \
bash "$PLOT_TC_PEAK_PANELS_SCRIPT"

echo
echo "Finished."
