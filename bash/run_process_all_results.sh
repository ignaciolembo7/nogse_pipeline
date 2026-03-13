#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

RESULTS_ROOT="${1:-$REPO_ROOT/Data-signals/Results}"
PARAMS="${2:-$REPO_ROOT/Data-signals/sequence_parameters.xlsx}"
OUT_DIR="${3:-$REPO_ROOT/analysis/ogse_experiments/data}"
PROCESS_SCRIPT="${4:-$REPO_ROOT/nogse_pipeline/scripts/process_one_results.py}"

if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "ERROR: Results root not found: $RESULTS_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PARAMS" ]]; then
    echo "ERROR: Parameters file not found: $PARAMS" >&2
    exit 1
fi

if [[ ! -f "$PROCESS_SCRIPT" ]]; then
    echo "ERROR: Python script not found: $PROCESS_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

total=0
ok=0
failed=0
declare -a failed_sequences=()

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    seq_name="$(basename "$file" "_results.xlsx")"

    echo "Processing: $seq_name"
    echo "  File: $file"

    if python "$PROCESS_SCRIPT" "$file" "$PARAMS" --out_dir "$OUT_DIR"; then
        ok=$((ok + 1))
        echo "  OK: $seq_name"
    else
        status=$?
        failed=$((failed + 1))
        failed_sequences+=("$seq_name")
        echo "  WARNING: failed sequence: $seq_name (exit code: $status)" >&2
        echo "  Continuing with next sequence..." >&2
    fi

done < <(find "$RESULTS_ROOT" -type f -name "*_results.xlsx" | sort)

echo
echo "Finished."
echo "  Total files   : $total"
echo "  Successful    : $ok"
echo "  Failed        : $failed"

if (( failed > 0 )); then
    echo
    echo "Failed sequences:"
    for seq in "${failed_sequences[@]}"; do
        echo "  - $seq"
    done
fi
