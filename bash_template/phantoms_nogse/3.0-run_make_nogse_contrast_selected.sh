#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

MAKE_CONTRAST_SCRIPT="$REPO_ROOT/scripts/make_contrast.py"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_ROOT="$PROJECT_ROOT/analysis/phantoms/nogse_experiments/data/20260122-PHANTOM_FIBER"
OUT_ROOT="$PROJECT_ROOT/analysis/phantoms/nogse_experiments/contrast_data"
DIRECTIONS=(1)
ONEG="${ONEG:-true}"

# Add the contrast pairs manually.
declare -a PAIRS=(
# "$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_CPMG_N2_TN50_20260122092639_results.long.parquet|$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_HAHN_N2_TN50_20260122092639_results.long.parquet"
"$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_CPMG_N2_TN50_20260122125354_results.long.parquet|$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_HAHN_N2_TN50_20260122125354_results.long.parquet"
"$DATA_ROOT/QUALITY_JACK_19800122TMSF_003_NOGSE_CPMG_N2_TN65_20260122125354_results.long.parquet|$DATA_ROOT/QUALITY_JACK_19800122TMSF_003_NOGSE_HAHN_N2_TN65_20260122125354_results.long.parquet"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast.py not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

MAKE_CONTRAST_ARGS=()
if [[ "${ONEG,,}" == "true" ]]; then
    MAKE_CONTRAST_ARGS+=(--oneg)
fi

total=0
ok=0
failed=0
declare -a failed_jobs=()

for pair in "${PAIRS[@]}"; do
    total=$((total + 1))

    file_a="${pair%%|*}"
    file_b="${pair##*|}"

    base_a="$(basename "$file_a")"
    base_b="$(basename "$file_b")"

    echo "============================================================"
    echo "Job $total"
    echo "  A: $base_a"
    echo "  B: $base_b"

    if [[ ! -f "$file_a" ]]; then
        failed=$((failed + 1))
        failed_jobs+=("missing A :: $file_a")
        echo "  ERROR: missing file A: $file_a" >&2
        echo "  Continuing with next job..." >&2
        continue
    fi

    if [[ ! -f "$file_b" ]]; then
        failed=$((failed + 1))
        failed_jobs+=("missing B :: $file_b")
        echo "  ERROR: missing file B: $file_b" >&2
        echo "  Continuing with next job..." >&2
        continue
    fi

    if "$PY" "$MAKE_CONTRAST_SCRIPT" \
        "$file_a" \
        "$file_b" \
        --direction "${DIRECTIONS[@]}" \
        "${MAKE_CONTRAST_ARGS[@]}" \
        --out_root "$OUT_ROOT"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_jobs+=("exit $status :: $file_a :: $file_b")
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
    echo "  Notes       : PAIRS is empty. Add the contrast pairs manually in this script."
fi

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
