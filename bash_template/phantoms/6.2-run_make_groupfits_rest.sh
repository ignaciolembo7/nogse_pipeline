#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
PIPELINE_SCRIPT="$REPO_ROOT/scripts/run_tc_pipeline.py"
PLOT_FIT_PANELS_SCRIPT="$REPO_ROOT/bash/helpers/run_plot_ogse_contrast_fit_panels.sh"
PLOT_TC_PEAK_PANELS_SCRIPT="$REPO_ROOT/bash/helpers/run_plot_ogse_tc_peak_panels.sh"

FIT_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_rest_ogse_contrast_corr"
CONTRAST_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/contrast-data"
OUT_XLSX="$FIT_ROOT/groupfits_rest.xlsx"
OUT_PARQUET="$FIT_ROOT/groupfits_rest.parquet"
MODELS="rest"
SUBJS="ALL"
ROIS="ALL"
DIRECTIONS="ALL"
EXCLUDE_TD_MS="209.1"
FIT_PANELS_OUT_DIR="$FIT_ROOT/contrast_fit_panels"
TC_PEAK_PANELS_OUT_DIR="$FIT_ROOT/tc_peak_panels"

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
    echo "Fit root not found: $FIT_ROOT. Skipping grouped summaries."
    exit 0
fi

if [[ ! -d "$CONTRAST_ROOT" ]]; then
    echo "Contrast root not found: $CONTRAST_ROOT. Skipping grouped summaries."
    exit 0
fi

if [[ -z "$(find "$FIT_ROOT" -type f -name 'fit_params.parquet' -print -quit)" ]]; then
    echo "No rest contrast fits were found in $FIT_ROOT. Skipping grouped summaries."
    exit 0
fi

mkdir -p "$FIT_ROOT"

echo "============================================================"
echo "Dataset       : phantoms"
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
    EXCLUDE_TD_MS="$EXCLUDE_TD_MS" \
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
    EXCLUDE_TD_MS="$EXCLUDE_TD_MS" \
    bash "$PLOT_TC_PEAK_PANELS_SCRIPT"

echo
echo "Finished."
