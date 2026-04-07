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
PIPELINE_SCRIPT="$REPO_ROOT/scripts/run_tc_pipeline.py"

FIT_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_rest_ogse_contrast_corr"
OUT_XLSX="$FIT_ROOT/groupfits_rest.xlsx"
OUT_PARQUET="$FIT_ROOT/groupfits_rest.parquet"
MODELS="rest"

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PIPELINE_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$FIT_ROOT" ]]; then
    echo "ERROR: Fit root not found: $FIT_ROOT" >&2
    exit 1
fi

mkdir -p "$FIT_ROOT"

echo "============================================================"
echo "Dataset       : phantoms"
echo "Fit root      : $FIT_ROOT"
echo "Models        : $MODELS"
echo "Output XLSX   : $OUT_XLSX"
echo "Output Parquet: $OUT_PARQUET"

"$PY" "$PIPELINE_SCRIPT" \
    "$FIT_ROOT" \
    --models "$MODELS" \
    --out-xlsx "$OUT_XLSX" \
    --out-parquet "$OUT_PARQUET"

echo
echo "Finished."
