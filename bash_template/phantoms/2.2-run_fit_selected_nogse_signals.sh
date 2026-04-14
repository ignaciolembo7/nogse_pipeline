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
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

FIT_SCRIPT="$REPO_ROOT/scripts/fit_nogse-signal_vs_g.py"
DATA_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data/20220610-PHANTOM3"
OUT_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_nogse-signal_vs_g"
XCOL="g"
YCOL="value_norm"
STAT="avg"
ROIS="ALL"
DIRECTIONS="ALL"
FIX_M0="1.0"

declare -a JOBS=(
  # Add jobs manually as: "path|model"
  # Supported models: free_cpmg | free_hahn
  # Example:
  # "$DATA_ROOT/20260122-PHANTOM_NISO4_Exp01_N2_TN50_NiSO_phantom.long.parquet|free_cpmg"
)

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
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
if [[ -n "${FIX_M0// }" ]]; then
    extra_args+=(--fix_M0 "$FIX_M0")
fi

total=0
ok=0
failed=0
declare -a failed_jobs=()

for job in "${JOBS[@]}"; do
    total=$((total + 1))
    file_path="${job%%|*}"
    model_name="${job##*|}"
    base_name="$(basename "$file_path")"

    echo "============================================================"
    echo "Job $total"
    echo "  File  : $base_name"
    echo "  Model : $model_name"

    if [[ ! -f "$file_path" ]]; then
        failed=$((failed + 1))
        failed_jobs+=("missing :: $file_path")
        echo "  ERROR: missing file: $file_path" >&2
        echo "  Continuing with next job..." >&2
        continue
    fi

    if "$PY" "$FIT_SCRIPT" \
        "$file_path" \
        --model "$model_name" \
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
        failed_jobs+=("exit $status :: $file_path :: $model_name")
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
    echo "  Notes       : JOBS is empty. Add the signal parquets and models manually in this script."
fi

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
