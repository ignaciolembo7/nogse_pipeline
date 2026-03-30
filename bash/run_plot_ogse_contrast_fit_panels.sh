#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

PY="${PY:-python}"

# Configuracion
FITS_ROOT="$REPO_ROOT/analysis/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr"
CONTRAST_ROOT="$REPO_ROOT/analysis/ogse_experiments/contrast-data-rotated"
PLOT_SCRIPT="$REPO_ROOT/nogse_pipeline/scripts/plot_ogse-contrast_fit_panels.py"
OUT_DIR="$FITS_ROOT/contrast_fit_panels"
MODELS="rest"
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC"
DIRECTIONS="long,tra"

if [[ ! -d "$FITS_ROOT" ]]; then
    echo "ERROR: Fits root not found: $FITS_ROOT" >&2
    exit 1
fi

if [[ ! -d "$CONTRAST_ROOT" ]]; then
    echo "ERROR: Contrast root not found: $CONTRAST_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

extra_args=()

if [[ "$MODELS" != "ALL" ]]; then
    read -r -a model_list <<< "${MODELS//,/ }"
    if (( ${#model_list[@]} > 0 )); then
        extra_args+=(--models "${model_list[@]}")
    fi
fi

if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi

if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        extra_args+=(--directions "${dir_list[@]}")
    fi
fi

echo "============================================================"
echo "Plotting contrast-fit panels"
echo "  Fits root    : $FITS_ROOT"
echo "  Contrast root: $CONTRAST_ROOT"
echo "  Models       : $MODELS"
echo "  ROIs         : $ROIS"
echo "  Directions   : $DIRECTIONS"
echo "  Out dir      : $OUT_DIR"

"$PY" "$PLOT_SCRIPT" \
    "$FITS_ROOT" \
    --contrast-root "$CONTRAST_ROOT" \
    --out-dir "$OUT_DIR" \
    "${extra_args[@]}"

echo
echo "Finished."
echo "  Output dir: $OUT_DIR"
