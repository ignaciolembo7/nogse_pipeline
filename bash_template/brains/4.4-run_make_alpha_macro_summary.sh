#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
SUMMARY_SCRIPT="$REPO_ROOT/scripts/make_alpha_macro_summary.py"

COMBINED_TABLE="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1/D_vs_delta_app.combined.xlsx"
SUBJS="BRAIN LUDG MBBL"
PLOT_ROIS="AntCC MidAntCC CentralCC MidPostCC PostCC Left-Lateral-Ventricle Right-Lateral-Ventricle"
PLOT_DIRECTIONS="x y z"
OUT_SUMMARY="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"

if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "ERROR: Script not found: $SUMMARY_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$COMBINED_TABLE" ]]; then
    echo "ERROR: Combined table not found: $COMBINED_TABLE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_SUMMARY")"

echo "============================================================"
echo "Dataset       : brains"
echo "Combined table: $COMBINED_TABLE"
echo "Subjs         : $SUBJS"
echo "Plot ROIs     : $PLOT_ROIS"
echo "Plot dirs     : $PLOT_DIRECTIONS"
echo "Out summary   : $OUT_SUMMARY"

"$PY" "$SUMMARY_SCRIPT" \
    --combined-table "$COMBINED_TABLE" \
    --subj $SUBJS \
    --plot-rois $PLOT_ROIS \
    --plot-directions $PLOT_DIRECTIONS \
    --out-summary "$OUT_SUMMARY"

echo
echo "Finished."
