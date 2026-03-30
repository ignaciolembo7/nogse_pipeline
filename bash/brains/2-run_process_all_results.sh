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
SIGNALS_ROOT="$PROJECT_ROOT/Data-signals"
ANALYSIS_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments"
DEFAULT_RESULTS_ROOT="$SIGNALS_ROOT/Results"
DEFAULT_PARAMS="$SIGNALS_ROOT/sequence_parameters.xlsx"
DEFAULT_OUT_DIR="$ANALYSIS_ROOT/data"
DEFAULT_PROCESS_SCRIPT="$REPO_ROOT/scripts/process_one_results.py"

# Grouped subject tags written to the output tables.
# These are the only explicit subj assignments needed in the bash pipeline.
declare -A SHEET_TO_SUBJ=(
    ["20220622_BRAIN"]="BRAIN"
    ["20230619_BRAIN-3"]="BRAIN"
    ["20230623_BRAIN-4"]="BRAIN"
    ["20230623_LUDG-2"]="LUDG"
    ["20230629_MBBL-2"]="MBBL"
    ["20230630_MBBL-3"]="MBBL"
    ["20230710_LUDG-3"]="LUDG"
)

RESULTS_ROOT="${1:-$DEFAULT_RESULTS_ROOT}"
PARAMS="${2:-$DEFAULT_PARAMS}"
OUT_DIR="${3:-$DEFAULT_OUT_DIR}"
PROCESS_SCRIPT="${4:-$DEFAULT_PROCESS_SCRIPT}"

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

subj_for_file() {
    local file="$1"
    local sheet_name
    sheet_name="$(basename "$(dirname "$file")")"

    if [[ -n "${SHEET_TO_SUBJ[$sheet_name]:-}" ]]; then
        printf '%s\n' "${SHEET_TO_SUBJ[$sheet_name]}"
        return 0
    fi

    echo "ERROR: no subj mapping configured for sheet: $sheet_name" >&2
    return 1
}

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    seq_name="$(basename "$file" "_results.xlsx")"
    if ! subj="$(subj_for_file "$file")"; then
        failed=$((failed + 1))
        failed_sequences+=("$seq_name")
        echo "  WARNING: failed sequence: $seq_name (missing subj mapping)" >&2
        echo "  Continuing with next sequence..." >&2
        continue
    fi

    echo "Processing: $seq_name"
    echo "  File: $file"
    echo "  Subj: $subj"

    if "$PY" "$PROCESS_SCRIPT" "$file" "$PARAMS" --out_dir "$OUT_DIR" --subj "$subj"; then
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
