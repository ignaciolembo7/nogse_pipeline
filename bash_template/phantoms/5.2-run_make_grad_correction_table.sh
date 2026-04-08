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
MAKE_SCRIPT="$REPO_ROOT/scripts/make_grad_correction_table.py"

ROI="water"
EXP_FITS_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_monoexp_ogse-signal"
NOGSE_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit-free_ogse-contrast"
OUT_XLSX="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/grad_correction/water.grad_correction.xlsx"
OUT_CSV="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/grad_correction/water.grad_correction.csv"

if [[ ! -f "$MAKE_SCRIPT" ]]; then
    echo "ERROR: Script not found: $MAKE_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$EXP_FITS_ROOT" ]]; then
    echo "ERROR: Monoexp fits root not found: $EXP_FITS_ROOT" >&2
    exit 1
fi

if [[ ! -d "$NOGSE_ROOT" ]]; then
    echo "ERROR: NOGSE fits root not found: $NOGSE_ROOT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_XLSX")"
mkdir -p "$(dirname "$OUT_CSV")"

echo "============================================================"
echo "Dataset       : phantoms"
echo "ROI           : $ROI"
echo "Monoexp root  : $EXP_FITS_ROOT"
echo "NOGSE root    : $NOGSE_ROOT"
echo "Output XLSX   : $OUT_XLSX"
echo "Output CSV    : $OUT_CSV"

"$PY" "$MAKE_SCRIPT" \
    --roi "$ROI" \
    --exp-fits-root "$EXP_FITS_ROOT" \
    --nogse-root "$NOGSE_ROOT" \
    --out-xlsx "$OUT_XLSX" \
    --out-csv "$OUT_CSV"

echo
echo "Finished."
