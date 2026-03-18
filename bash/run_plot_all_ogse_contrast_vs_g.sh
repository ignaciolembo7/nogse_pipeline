#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

TABLES_ROOT="${1:-$REPO_ROOT/analysis/ogse_experiments/contrast-data-rotated/tables}"
OUT_ROOT="${2:-$REPO_ROOT/analysis/ogse_experiments/contrast-data-rotated/plots}"
PLOT_SCRIPT="${3:-$REPO_ROOT/nogse_pipeline/scripts/plot_ogse-contrast_vs_g.py}"
FILE_PATTERN="${4:-*.long.parquet}"
XCOL="${5:-gthorsten_1}"
YCOL="${6:-contrast_norm}"

shift $(( $# >= 6 ? 6 : $# )) || true
AXES=("$@")
if [[ ${#AXES[@]} -eq 0 ]]; then
    AXES=(long tra)
fi

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "ERROR: Tables root not found: $TABLES_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_SCRIPT" >&2
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

    echo "Processing: $base_name"
    echo "  File: $file"

    if python "$PLOT_SCRIPT" \
        "$file" \
        --xcol "$XCOL" \
        --y "$YCOL" \
        --out_root "$OUT_ROOT" \
	--rois AntCC MidAntCC CentralCC MidPostCC PostCC \
        --axes "${AXES[@]}"; then
        ok=$((ok + 1))
        echo "  OK: $base_name"
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
