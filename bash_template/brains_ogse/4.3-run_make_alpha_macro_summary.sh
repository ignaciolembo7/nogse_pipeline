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
SUMMARY_SCRIPT="$REPO_ROOT/scripts/make_alpha_macro_summary.py"
DPROJ_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated"
SUBJS="BRAIN LUDG MBBL"
ROIS="AntCC MidAntCC CentralCC MidPostCC PostCC Left-Lateral-Ventricle Right-Lateral-Ventricle Syringe"
DIRS="x y z"
N_VALUE="1"
PLOT_ROIS="$ROIS"
PLOT_DIRECTIONS="$DIRS"
BVALMAX=""
ROI_BVALMAX_OVERRIDES=(
    "AntCC=10"
    "MidAntCC=10"
    "CentralCC=10"
    "MidPostCC=10"
    "PostCC=10"
    "Left-Lateral-Ventricle=6"
    "Right-Lateral-Ventricle=6"
    "Syringe=6"
)
OUT_SUMMARY="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "ERROR: Script not found: $SUMMARY_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DPROJ_ROOT" ]]; then
    echo "ERROR: Dproj root not found: $DPROJ_ROOT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_SUMMARY")"

echo "============================================================"
echo "Dataset       : brains"
echo "Dproj root    : $DPROJ_ROOT"
echo "Subjs         : $SUBJS"
echo "ROIs          : $ROIS"
echo "Dirs          : $DIRS"
echo "N             : $N_VALUE"
echo "Plot ROIs     : $PLOT_ROIS"
echo "Plot dirs     : $PLOT_DIRECTIONS"
if [[ -n "$BVALMAX" ]]; then
    echo "Bstep alpha   : $BVALMAX"
fi
if [[ ${#ROI_BVALMAX_OVERRIDES[@]} -gt 0 ]]; then
    echo "ROI bstep map : ${ROI_BVALMAX_OVERRIDES[*]}"
fi
echo "Out summary   : $OUT_SUMMARY"

cmd=(
    "$PY" "$SUMMARY_SCRIPT"
    --dproj-root "$DPROJ_ROOT"
    --subj $SUBJS
    --rois $ROIS
    --dirs $DIRS
    --N "$N_VALUE"
    --plot-rois $PLOT_ROIS
    --plot-directions $PLOT_DIRECTIONS
    --out-summary "$OUT_SUMMARY"
)

if [[ -n "$BVALMAX" ]]; then
    cmd+=(--bvalmax "$BVALMAX")
fi

if [[ ${#ROI_BVALMAX_OVERRIDES[@]} -gt 0 ]]; then
    for roi_bstep in "${ROI_BVALMAX_OVERRIDES[@]}"; do
        cmd+=(--roi-bvalmax "$roi_bstep")
    done
fi

"${cmd[@]}"

echo
echo "Finished."
