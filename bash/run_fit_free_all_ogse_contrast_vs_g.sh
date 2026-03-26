#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

PY="${PY:-python}"

TABLES_ROOT="${1:-$REPO_ROOT/analysis/ogse_experiments/contrast-data-rotated/tables}"
OUT_ROOT="${2:-$REPO_ROOT/analysis/ogse_experiments/fits/fit-free_ogse-contrast-rotated}"
FIT_SCRIPT="${3:-$REPO_ROOT/nogse_pipeline/scripts/fit_ogse-contrast_vs_g.py}"
FILE_PATTERN="${4:-*.long.parquet}"

MODEL="${5:-free}"
GBASE="${6:-g_thorsten_1}"
YCOL="${7:-value_norm}"

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "ERROR: Tables root not found: $TABLES_ROOT" >&2
    exit 1
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: Fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

total=0
ok=0
failed=0
declare -a failed_files=()

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    base_name="$(basename "$file")"

    echo "============================================================"
    echo "Job $total"
    echo "  File: $base_name"

    if "$PY" "$FIT_SCRIPT" \
        "$file" \
        --model "$MODEL" \
        --gbase "$GBASE" \
        --ycol "$YCOL" \
        --directions long tra \
        --out_root "$OUT_ROOT" \
        --no_grad_corr \
        --fix_M0 1.0 \
        --rois Syringe Right-Lateral-Ventricle Left-Lateral-Ventricle; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_files+=("$file")
        echo "  WARNING: failed file: $base_name (exit code: $status)" >&2
        echo "  Continuing with next file..." >&2
    fi

done < <(find "$TABLES_ROOT" -type f -name "$FILE_PATTERN" | sort)

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
