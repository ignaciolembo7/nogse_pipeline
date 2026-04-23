#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PY="${PY:-python}"

# Configuration
FITS_ROOT="${FITS_ROOT:-$PROJECT_ROOT/analysis/ogse_experiments/fits/nogse_contrast_vs_g_rest_corr}"
CONTRAST_ROOT="${CONTRAST_ROOT:-$PROJECT_ROOT/analysis/ogse_experiments/contrast-data-rotated}"
PLOT_SCRIPT="${PLOT_SCRIPT:-$REPO_ROOT/scripts/plot_ogse-contrast_fit_panels.py}"
OUT_DIR="${OUT_DIR:-$FITS_ROOT/contrast_fit_panels}"
MODELS="${MODELS:-rest}"
SUBJS="${SUBJS:-ALL}"
ROIS="${ROIS:-ALL}"
DIRECTIONS="${DIRECTIONS:-ALL}"
EXCLUDE_TD_MS="${EXCLUDE_TD_MS:-}"

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

if [[ "$SUBJS" != "ALL" ]]; then
    read -r -a subj_list <<< "${SUBJS//,/ }"
    if (( ${#subj_list[@]} > 0 )); then
        extra_args+=(--subjs "${subj_list[@]}")
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

if [[ -n "${EXCLUDE_TD_MS// }" ]]; then
    read -r -a exclude_td_list <<< "${EXCLUDE_TD_MS//,/ }"
    if (( ${#exclude_td_list[@]} > 0 )); then
        extra_args+=(--exclude-td-ms "${exclude_td_list[@]}")
    fi
fi

echo "============================================================"
echo "Plotting contrast-fit panels"
echo "  Fits root    : $FITS_ROOT"
echo "  Contrast root: $CONTRAST_ROOT"
echo "  Models       : $MODELS"
echo "  Subjs        : $SUBJS"
echo "  ROIs         : $ROIS"
echo "  Directions   : $DIRECTIONS"
echo "  Exclude td_ms: ${EXCLUDE_TD_MS:-<none>}"
echo "  Out dir      : $OUT_DIR"

"$PY" "$PLOT_SCRIPT" \
    "$FITS_ROOT" \
    --contrast-root "$CONTRAST_ROOT" \
    --out-dir "$OUT_DIR" \
    "${extra_args[@]}"

echo
echo "Finished."
echo "  Output dir: $OUT_DIR"
