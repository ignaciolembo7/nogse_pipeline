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
TC_SCRIPT="$REPO_ROOT/scripts/run_tc_vs_td.py"

METHOD="pseudohuber_fixed_macro"
GROUPFITS="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_rest_ogse_contrast_corr/groupfits_rest.parquet"
SUMMARY_ALPHA="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
YCOL="tc_peak_ms"
OUT_DIR="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_rest_ogse_contrast_corr/tc_vs_td/$METHOD/$YCOL"

if [[ ! -f "$TC_SCRIPT" ]]; then
    echo "ERROR: Script not found: $TC_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$GROUPFITS" ]]; then
    echo "ERROR: Groupfits file not found: $GROUPFITS" >&2
    exit 1
fi

if [[ ! -f "$SUMMARY_ALPHA" ]]; then
    echo "ERROR: Summary alpha file not found: $SUMMARY_ALPHA" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Dataset       : phantoms"
echo "Method        : $METHOD"
echo "Groupfits     : $GROUPFITS"
echo "Summary alpha : $SUMMARY_ALPHA"
echo "Y column      : $YCOL"
echo "Output dir    : $OUT_DIR"

"$PY" "$TC_SCRIPT" \
    --method "$METHOD" \
    --groupfits "$GROUPFITS" \
    --summary-alpha "$SUMMARY_ALPHA" \
    --y-col "$YCOL" \
    --out-dir "$OUT_DIR"

echo
echo "Finished."
