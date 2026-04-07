#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
PLOT_SCRIPT="$REPO_ROOT/scripts/plot_D0_vs_Delta.py"

DPROJ_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated"
SUBJS="BRAIN LUDG MBBL"
ROIS="AntCC MidAntCC CentralCC MidPostCC PostCC Left-Lateral-Ventricle Right-Lateral-Ventricle"
DIRS="x y z"
N_VALUE="1"
OUT_DIR="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1"

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
echo "Dataset    : brains"
echo "Dproj root : $DPROJ_ROOT"
echo "Subjs      : $SUBJS"
echo "ROIs       : $ROIS"
echo "Dirs       : $DIRS"
echo "N          : $N_VALUE"
echo "Output dir : $OUT_DIR"

"$PY" "$PLOT_SCRIPT" \
    --dproj-root "$DPROJ_ROOT" \
    --brains $SUBJS \
    --rois $ROIS \
    --dirs $DIRS \
    --N "$N_VALUE" \
    --out-dir "$OUT_DIR"

echo
echo "Finished."
