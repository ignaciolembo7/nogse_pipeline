#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

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
SIGNALS_ROOT="$PROJECT_ROOT/Data-signals"
ANALYSIS_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments"
PHANTOM_SUBJ_REL="20260122-PHANTOM_NISO4/QUALITY_JACK_19800122TMSF"
DEFAULT_RESULTS_ROOT="$SIGNALS_ROOT/Results/$PHANTOM_SUBJ_REL"
DEFAULT_PARAMS="$SIGNALS_ROOT/sequence_parameters_phantoms.xlsx"
DEFAULT_OUT_DIR="$ANALYSIS_ROOT/data"
DEFAULT_PROCESS_SCRIPT="$REPO_ROOT/scripts/process_one_results.py"

# The matched row in DEFAULT_PARAMS must include a populated "subj" column.

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

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    seq_name="$(basename "$file" "_results.xlsx")"

    echo "Processing: $seq_name"
    echo "  File: $file"

    if "$PY" "$PROCESS_SCRIPT" "$file" "$PARAMS" --out_dir "$OUT_DIR"; then
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
    exit 1
fi
