#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

PY="${PY:-python}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/brains_ogse}"
INPUT_ROOT="${PREPROC_INPUT_ROOT:-$PROJECT_ROOT/Data-NIFTI-BRAINS}"
OUTPUT_ROOT="${PREPROC_OUTPUT_ROOT:-$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup}"
SUBJECTS="${PREPROC_SUBJECTS:-}"
STEPS="${PREPROC_STEPS:-denoise degibbs topup eddy eddy_qc bias}"
NTHREADS="${PREPROC_NTHREADS:-8}"
DRY_RUN="${PREPROC_DRY_RUN:-true}"
OVERWRITE="${PREPROC_OVERWRITE:-false}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$LOG_ROOT"

if [[ -z "$SUBJECTS" ]]; then
    echo "ERROR: set PREPROC_SUBJECTS to one or more subject IDs before running preprocessing." >&2
    exit 1
fi

ARGS=(
    "$REPO_ROOT/scripts/preprocess_dwi.py"
    --dataset brains
    --subjects $SUBJECTS
    --steps $STEPS
    --input-root "$INPUT_ROOT"
    --output-root "$OUTPUT_ROOT"
    --nthreads "$NTHREADS"
)

if [[ "${DRY_RUN,,}" == "true" ]]; then
    ARGS+=(--dry-run)
fi

if [[ "${OVERWRITE,,}" == "true" ]]; then
    ARGS+=(--overwrite)
fi

echo "Running brain DWI preprocessing"
echo "Input root : $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Subjects   : $SUBJECTS"
echo "Steps      : $STEPS"
echo "Dry run    : $DRY_RUN"

"$PY" "${ARGS[@]}" 2>&1 | tee "$LOG_ROOT/0.5-run_preprocess_dwi.log"
