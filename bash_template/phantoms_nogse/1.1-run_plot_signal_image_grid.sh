#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"
PLOT_HELPER="$REPO_ROOT/bash_template/helpers/run_plot_signal_image_grid.sh"

PY="${PY:-python}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SIGNAL_ROOT="${SIGNAL_ROOT:-$PROJECT_ROOT/Data-signals}"
EXPERIMENT="${EXPERIMENT:-20260122-PHANTOM_NISO4}"
NAME="${NAME:-QUALITY_JACK_19800122TMSF}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/phantoms/nogse_experiments/signal_image_grids}"
IMAGE_NAME="${IMAGE_NAME:-mean.nii.gz}"

GRADIENT_TYPE="${GRADIENT_TYPE:-g}"
GRADIENT_VALUES="${GRADIENT_VALUES:-0 8 16 24 32 40 48 56 64 72}"

# Use comma-separated row labels. Difference rows use " - " between labels.
ROWS_CSV="${ROWS_CSV:-CPMG,HAHN,CPMG - HAHN}"

# Use this to disambiguate acquisitions that share type and gradient.
# For example, use "TN50,002" or "TN65,003".
INCLUDE_TOKENS_CSV="${INCLUDE_TOKENS_CSV:-TN50,002}"
EXCLUDE_TOKENS_CSV="${EXCLUDE_TOKENS_CSV:-}"

TITLE="${TITLE:-exp=$EXPERIMENT - name=$NAME - tokens=$INCLUDE_TOKENS_CSV}"
OUT_STEM="${OUT_STEM:-}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$PLOT_HELPER" ]]; then
    echo "ERROR: plot helper script not found: $PLOT_HELPER" >&2
    exit 1
fi

PY="$PY" \
SIGNAL_ROOT="$SIGNAL_ROOT" \
EXPERIMENT="$EXPERIMENT" \
NAME="$NAME" \
OUT_ROOT="$OUT_ROOT" \
IMAGE_NAME="$IMAGE_NAME" \
GRADIENT_TYPE="$GRADIENT_TYPE" \
GRADIENT_VALUES="$GRADIENT_VALUES" \
ROWS_CSV="$ROWS_CSV" \
INCLUDE_TOKENS_CSV="$INCLUDE_TOKENS_CSV" \
EXCLUDE_TOKENS_CSV="$EXCLUDE_TOKENS_CSV" \
TITLE="$TITLE" \
OUT_STEM="$OUT_STEM" \
ALLOW_MISSING="$ALLOW_MISSING" \
bash "$PLOT_HELPER"
