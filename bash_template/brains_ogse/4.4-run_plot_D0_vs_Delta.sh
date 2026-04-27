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
PLOT_SCRIPT="$REPO_ROOT/scripts/plot_D0_vs_Delta.py"
DPROJ_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated"
SUBJS="BRAIN LUDG MBBL"
ROIS="AntCC MidAntCC CentralCC MidPostCC PostCC Left-Lateral-Ventricle Right-Lateral-Ventricle Syringe"
DIRS="x y z"
N_VALUE="1"
OUT_DIR="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1"
SUMMARY_ALPHA="$OUT_DIR/summary_alpha_values.xlsx"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

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
if [[ -f "$SUMMARY_ALPHA" ]]; then
    echo "Summary    : $SUMMARY_ALPHA"
else
    echo "Summary    : <not found; using fallback bstep selection>"
fi
echo "Output dir : $OUT_DIR"

cmd=(
    "$PY" "$PLOT_SCRIPT"
    --dproj-root "$DPROJ_ROOT"
    --brains $SUBJS
    --rois $ROIS
    --dirs $DIRS
    --N "$N_VALUE"
    --out-dir "$OUT_DIR"
)

if [[ -f "$SUMMARY_ALPHA" ]]; then
    cmd+=(--summary-alpha "$SUMMARY_ALPHA")
fi

"${cmd[@]}"

echo
echo "Finished."
