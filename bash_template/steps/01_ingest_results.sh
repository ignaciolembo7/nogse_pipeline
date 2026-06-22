#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help ingest "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

PROCESS_SCRIPT="${PROCESS_SCRIPT:-$REPO_ROOT/scripts/data/process_one_results.py}"
PROCESS_OUT_ROOT="${PROCESS_OUT_ROOT:-$ANALYSIS_ROOT/data/tables}"
RESULTS_GLOB="${RESULTS_GLOB:-*_results.xlsx}"

pipeline_require_file "$PROCESS_SCRIPT" "process script"
pipeline_require_file "$PARAMS_XLSX" "sequence params"
mkdir -p "$PROCESS_OUT_ROOT" "$(dirname "$MASTER_PARQUET")"

echo "Results root : $RESULTS_ROOT"
if [[ -n "${RESULTS_ROOTS:-}" ]]; then
    echo "Results roots: $RESULTS_ROOTS"
fi
echo "Master       : $MASTER_PARQUET"

count=0
roots="${RESULTS_ROOTS:-$RESULTS_ROOT}"
for root in $roots; do
    if [[ ! -d "$root" ]]; then
        echo "ERROR: Results root not found: $root" >&2
        exit 1
    fi
    while read -r file; do
        [[ -z "$file" ]] && continue
        count=$((count + 1))
        "$PY" "$PROCESS_SCRIPT" "$file" "$PARAMS_XLSX" \
            --out_dir "$PROCESS_OUT_ROOT" \
            --master-parquet "$MASTER_PARQUET"
    done < <(find "$root" -type f -name "$RESULTS_GLOB" | sort)
done

echo "Done ingest. Files processed: $count"
