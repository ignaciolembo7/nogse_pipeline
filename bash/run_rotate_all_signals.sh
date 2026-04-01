#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
PY="${PY:-python}"

DATA_ROOT="${1:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data}"
DIRS_CSV="${2:-$REPO_ROOT/assets/dirs/dirs_6.csv}"
ROTATE_SCRIPT="${3:-$REPO_ROOT/scripts/rotate_ogse_tensor.py}"
FILE_PATTERN="${4:-*_results.long.parquet}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: Data root not found: $DATA_ROOT" >&2
    exit 1
fi

if [[ ! -f "$DIRS_CSV" ]]; then
    echo "ERROR: dirs_csv file not found: $DIRS_CSV" >&2
    exit 1
fi

if [[ ! -f "$ROTATE_SCRIPT" ]]; then
    echo "ERROR: Python script not found: $ROTATE_SCRIPT" >&2
    exit 1
fi

total=0
ok=0
failed=0
declare -a failed_files=()

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    base_name="$(basename "$file")"

    echo "Processing: $base_name"
    echo "  File: $file"

    if "$PY" "$ROTATE_SCRIPT" "$file" --dirs_csv "$DIRS_CSV" --out_dir "$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated"; then
        ok=$((ok + 1))
        echo "  OK: $base_name"
    else
        status=$?
        failed=$((failed + 1))
        failed_files+=("$file")
        echo "  WARNING: failed file: $base_name (exit code: $status)" >&2
        echo "  Continuing with next file..." >&2
    fi

done < <(find "$DATA_ROOT" -type f -name "$FILE_PATTERN" | sort)

echo
echo "Finished."
echo "  Total files   : $total"
echo "  Successful    : $ok"
echo "  Failed        : $failed"

if (( failed > 0 )); then
    echo
    echo "Failed files:"
    for f in "${failed_files[@]}"; do
        echo "  - $f"
    done
fi
