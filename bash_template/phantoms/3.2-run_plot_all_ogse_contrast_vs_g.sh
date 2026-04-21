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
DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
ANALYSIS_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments"
TABLES_ROOT="$ANALYSIS_ROOT/contrast-data/tables"
OUT_ROOT="$ANALYSIS_ROOT/contrast-data/plots"
PLOT_SCRIPT="$REPO_ROOT/scripts/plot_ogse-contrast_vs_g.py"
FILE_PATTERN="*.long.parquet"
XCOL="g_1"
YCOL="value_norm"
STAT="avg"
DIRECTIONS="ALL"
ROIS="ALL"

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "Contrast tables root not found: $TABLES_ROOT. Skipping contrast plots."
    exit 0
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

extra_args=()
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        extra_args+=(--directions "${dir_list[@]}")
    fi
fi
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi

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
    echo "  File       : $base_name"
    echo "  X column   : $XCOL"
    echo "  Y column   : $YCOL"
    echo "  Stat       : $STAT"
    echo "  Directions : ${DIRECTIONS[*]}"

    if "$PY" "$PLOT_SCRIPT" \
        "$file" \
        --xcol "$XCOL" \
        --y "$YCOL" \
        --stat "$STAT" \
        --out_root "$OUT_ROOT" \
        "${extra_args[@]}"; then
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
