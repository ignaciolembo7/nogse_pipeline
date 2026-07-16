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

mapfile -t sheet_names < <("$PY" -c '
import sys
import pandas as pd
for name in pd.ExcelFile(sys.argv[1]).sheet_names:
    print(name)
' "$PARAMS_XLSX")

if [[ ${#sheet_names[@]} -eq 0 ]]; then
    echo "ERROR: No sheets found in PARAMS_XLSX: $PARAMS_XLSX" >&2
    exit 1
fi
echo "Sheets       : ${sheet_names[*]}"

is_known_sheet() {
    local candidate="$1" s
    for s in "${sheet_names[@]}"; do
        [[ "$candidate" == "$s" ]] && return 0
    done
    return 1
}

count=0
roots="${RESULTS_ROOTS:-$RESULTS_ROOT}"
for root in $roots; do
    if [[ ! -d "$root" ]]; then
        echo "ERROR: Results root not found: $root" >&2
        exit 1
    fi

    # Only process dataset subfolders whose name matches a sheet in
    # PARAMS_XLSX. If $root itself is already a dataset folder (e.g. a
    # RESULTS_ROOTS override pointing straight at one), use it as-is.
    target_dirs=()
    if is_known_sheet "$(basename "$root")"; then
        target_dirs+=("$root")
    else
        while IFS= read -r d; do
            [[ -z "$d" ]] && continue
            if is_known_sheet "$(basename "$d")"; then
                target_dirs+=("$d")
            fi
        done < <(find "$root" -mindepth 1 -maxdepth 1 -type d | sort)
    fi

    if [[ ${#target_dirs[@]} -eq 0 ]]; then
        echo "WARNING: no subfolders under $root match a sheet name in $PARAMS_XLSX" >&2
        continue
    fi

    for d in "${target_dirs[@]}"; do
        echo "Dataset dir  : $d"
        while read -r file; do
            [[ -z "$file" ]] && continue
            count=$((count + 1))
            "$PY" "$PROCESS_SCRIPT" "$file" "$PARAMS_XLSX" \
                --out_dir "$PROCESS_OUT_ROOT" \
                --master-parquet "$MASTER_PARQUET"
        done < <(find "$d" -type f -name "$RESULTS_GLOB" | sort)
    done
done

echo "Done ingest. Files processed: $count"
