#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-python}"
PLOT_SCRIPT="$REPO_ROOT/scripts/plot_D0_vs_Delta.py"

DPROJ_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data"
SUBJS="PHANTOM3"
ROIS="fiber1 fiber2 water water1 water2"
DIRS="1 2 3"
N_VALUE="1"
BVALMAX="7"
OUT_DIR="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DPROJ_ROOT" ]]; then
    echo "ERROR: Dproj root not found: $DPROJ_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Dataset    : phantoms"
echo "Dproj root : $DPROJ_ROOT"
echo "Subjs      : $SUBJS"
echo "ROIs       : $ROIS"
echo "Dirs       : $DIRS"
echo "N          : $N_VALUE"
echo "Bstep alpha: $BVALMAX"
echo "Output dir : $OUT_DIR"

# Legacy CLI alias accepted by the plotting script.
"$PY" "$PLOT_SCRIPT" \
    --dproj-root "$DPROJ_ROOT" \
    --brains $SUBJS \
    --rois $ROIS \
    --dirs $DIRS \
    --N "$N_VALUE" \
    --bvalmax "$BVALMAX" \
    --out-dir "$OUT_DIR" \
    --reference-D0 0.0023

echo
echo "Finished."
