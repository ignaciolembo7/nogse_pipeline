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
ROTATE_DRIVER="$REPO_ROOT/bash/run_rotate_all_signals.sh"
DATA_ROOT="${1:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data}"
DIRS_CSV="${2:-$REPO_ROOT/assets/dirs/dirs_6.csv}"
ROTATE_SCRIPT="${3:-$REPO_ROOT/scripts/rotate_ogse_tensor.py}"
FILE_PATTERN="${4:-*_results.long.parquet}"
OUT_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated"

if [[ ! -f "$ROTATE_DRIVER" ]]; then
    echo "ERROR: rotate driver not found: $ROTATE_DRIVER" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "Dataset      : brains"
echo "Data root    : $DATA_ROOT"
echo "Dirs CSV     : $DIRS_CSV"
echo "Rotate script: $ROTATE_SCRIPT"
echo "File pattern : $FILE_PATTERN"
echo "Output root  : $OUT_ROOT"

PY="$PY" bash "$ROTATE_DRIVER" "$DATA_ROOT" "$DIRS_CSV" "$ROTATE_SCRIPT" "$FILE_PATTERN"

echo
echo "Finished."
