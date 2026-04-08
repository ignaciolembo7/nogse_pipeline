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
FITS_ROOT="${1:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_monoexp_ogse-signal}"
OUT_ROOT="${2:-$FITS_ROOT/summary_plots}"
PLOT_SCRIPT="${3:-$REPO_ROOT/scripts/plot_monoexp_D_vs_time.py}"
ROIS=(fiber1 fiber2 water water1 water2)
DIRECTIONS=(1 2 3)

if [[ ! -d "$FITS_ROOT" ]]; then
    echo "ERROR: Fits root not found: $FITS_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

"$PY" "$PLOT_SCRIPT" \
    --fits-root "$FITS_ROOT" \
    --out-dir "$OUT_ROOT" \
    --rois "${ROIS[@]}" \
    --dirs "${DIRECTIONS[@]}"
