#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-python}"
DATA_ROOT="${1:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data/tables}"
DIRS_TXT="${2:-$REPO_ROOT/assets/dirs/dirs_6.txt}"
ROTATE_SCRIPT="${3:-$REPO_ROOT/scripts/rotate_ogse_tensor.py}"
FILE_PATTERN="${4:-*_results.long.parquet}"
OUT_ROOT="${5:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated}"
PLOT_SCRIPT="${PLOT_SCRIPT:-$REPO_ROOT/scripts/plot_signal_tables_vs_g.py}"
OUT_TABLES_DIR="$OUT_ROOT/tables"
OUT_PLOTS_DIR="$OUT_ROOT/plots"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

if [[ ! -f "$DIRS_TXT" ]]; then
    echo "ERROR: dirs TXT not found: $DIRS_TXT" >&2
    exit 1
fi

if [[ ! -f "$ROTATE_SCRIPT" ]]; then
    echo "ERROR: rotate script not found: $ROTATE_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_TABLES_DIR" "$OUT_PLOTS_DIR"

echo "============================================================"
echo "Dataset      : brains"
echo "Data root    : $DATA_ROOT"
echo "Dirs TXT     : $DIRS_TXT"
echo "Rotate script: $ROTATE_SCRIPT"
echo "File pattern : $FILE_PATTERN"
echo "Output root  : $OUT_ROOT"
echo "Tables root  : $OUT_TABLES_DIR"
echo "Plots root   : $OUT_PLOTS_DIR"

total=0
ok=0
failed=0
declare -a failed_files=()

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    base_name="$(basename "$file")"

    echo
    echo "Processing: $base_name"
    echo "  File: $file"

    if "$PY" "$ROTATE_SCRIPT" "$file" --dirs_txt "$DIRS_TXT" --out_dir "$OUT_TABLES_DIR"; then
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
    for file in "${failed_files[@]}"; do
        echo "  - $file"
    done
fi

if (( ok > 0 )); then
    echo
    echo "Plotting rotated signal tables versus g_thorsten..."
    "$PY" "$PLOT_SCRIPT" "$OUT_TABLES_DIR" --out_root "$OUT_PLOTS_DIR" --xcol g_thorsten --ycols value value_norm
fi
