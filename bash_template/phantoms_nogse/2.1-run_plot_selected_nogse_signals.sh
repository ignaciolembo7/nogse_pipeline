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

PLOT_SCRIPT="$REPO_ROOT/scripts/plot_nogse_signal_vs_g.py"
DATA_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data/20260122-PHANTOM_FIBER"
OUT_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/signal-plots/nogse_signal_vs_g"
XCOL="g"
YCOL="value_norm"
STAT="avg"
ROIS="ALL"
DIRECTIONS="ALL"

declare -a FILES=(
  # Add the signal parquet files manually.
  # Example:
  "$DATA_ROOT/QUALITY_JACK_19800122TMSF_001_NOGSE_CPMG_N2_TN50_results.long.parquet"
  "$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_CPMG_N2_TN50_results.long.parquet"
  "$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_HAHN_N2_TN50_results.long.parquet"
  "$DATA_ROOT/QUALITY_JACK_19800122TMSF_003_NOGSE_CPMG_N2_TN65_results.long.parquet"
  "$DATA_ROOT/QUALITY_JACK_19800122TMSF_003_NOGSE_HAHN_N2_TN65_results.long.parquet"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------


if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

extra_args=()
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        extra_args+=(--directions "${dir_list[@]}")
    fi
fi

total=0
ok=0
failed=0
declare -a failed_files=()

for file_path in "${FILES[@]}"; do
    total=$((total + 1))
    base_name="$(basename "$file_path")"

    echo "============================================================"
    echo "Job $total"
    echo "  File : $base_name"

    if [[ ! -f "$file_path" ]]; then
        failed=$((failed + 1))
        failed_files+=("$file_path")
        echo "  ERROR: missing file: $file_path" >&2
        echo "  Continuing with next job..." >&2
        continue
    fi

    if "$PY" "$PLOT_SCRIPT" \
        "$file_path" \
        --out_root "$OUT_ROOT" \
        --xcol "$XCOL" \
        --ycol "$YCOL" \
        --stat "$STAT" \
        "${extra_args[@]}"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_files+=("$file_path")
        echo "  WARNING: command failed with exit code $status" >&2
        echo "  Continuing with next job..." >&2
    fi
done

echo
echo "Finished."
echo "  Total jobs  : $total"
echo "  Successful  : $ok"
echo "  Failed      : $failed"

if (( total == 0 )); then
    echo "  Notes       : FILES is empty. Add the signal parquets manually in this script."
fi

if (( failed > 0 )); then
    echo
    echo "Failed files:"
    for item in "${failed_files[@]}"; do
        echo "  - $item"
    done
fi
