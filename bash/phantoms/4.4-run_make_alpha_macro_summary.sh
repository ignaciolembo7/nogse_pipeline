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
SUMMARY_SCRIPT="$REPO_ROOT/scripts/make_alpha_macro_summary.py"

COMBINED_TABLE="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/D_vs_delta_app.combined.xlsx"
SUBJS="PHANTOM3 PHANTOM3DDE"
PLOT_ROIS="fiber1 fiber2 water2 water3"
PLOT_DIRECTIONS="1 2 3"
OUT_SUMMARY="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
OUT_AVG="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/alpha_macro_D0_avg.xlsx"

if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "ERROR: Script not found: $SUMMARY_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$COMBINED_TABLE" ]]; then
    echo "ERROR: Combined table not found: $COMBINED_TABLE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_SUMMARY")"
mkdir -p "$(dirname "$OUT_AVG")"

echo "============================================================"
echo "Dataset       : phantoms"
echo "Combined table: $COMBINED_TABLE"
echo "Subjs         : $SUBJS"
echo "Plot ROIs     : $PLOT_ROIS"
echo "Plot dirs     : $PLOT_DIRECTIONS"
echo "Out summary   : $OUT_SUMMARY"
echo "Out avg       : $OUT_AVG"

"$PY" "$SUMMARY_SCRIPT" \
    --combined-table "$COMBINED_TABLE" \
    --subj $SUBJS \
    --plot-rois $PLOT_ROIS \
    --plot-directions $PLOT_DIRECTIONS \
    --out-summary "$OUT_SUMMARY" \
    --out-avg "$OUT_AVG"

echo
echo "Finished."
