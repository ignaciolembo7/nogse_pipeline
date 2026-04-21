#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
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
    echo "Monoexp fits root not found: $EXP_FITS_ROOT. Skipping gradient correction."
    exit 0
fi

if [[ ! -d "$NOGSE_ROOT" ]]; then
    echo "NOGSE fits root not found: $NOGSE_ROOT. Skipping gradient correction."
    exit 0
fi

if [[ -z "$(find "$EXP_FITS_ROOT" -type f -name 'fit_params.parquet' -print -quit)" ]]; then
    echo "No monoexp fits were found in $EXP_FITS_ROOT. Skipping gradient correction."
    exit 0
fi

if [[ -z "$(find "$NOGSE_ROOT" -type f -name 'fit_params.parquet' -print -quit)" ]]; then
    echo "No NOGSE contrast fits were found in $NOGSE_ROOT. Skipping gradient correction."
    exit 0
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
