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
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

MAKE_CONTRAST_SCRIPT="${MAKE_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/make_contrast.py}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/nogse-contrast-data}"
DIRECTIONS="${DIRECTIONS:-ALL}"
ONEG="${ONEG:-true}"
SUBJS="${SUBJS:-ALL}"
CPMG_TOKEN="${CPMG_TOKEN:-_NOGSE_CPMG_}"
HAHN_TOKEN="${HAHN_TOKEN:-_NOGSE_HAHN_}"
FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast script not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data root not found: $DATA_ROOT. Skipping NOGSE contrast build."
    exit 0
fi

mkdir -p "$OUT_ROOT"

make_args=()
if [[ "${ONEG,,}" == "true" ]]; then
    make_args+=(--oneg)
fi

if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        make_args+=(--direction "${dir_list[@]}")
    fi
fi

if [[ "$SUBJS" != "ALL" ]]; then
    read -r -a subj_list <<< "${SUBJS//,/ }"
    if (( ${#subj_list[@]} > 0 )); then
        make_args+=(--subjs "${subj_list[@]}")
    fi
fi

# Collect CPMG files and infer the matching HAHN file by token replacement.
mapfile -t cpmg_files < <(
    find "$DATA_ROOT" -type f -name "$FILE_PATTERN" | sort | while read -r f; do
        base="$(basename "$f")"
        if [[ "$base" == *"$CPMG_TOKEN"* ]]; then
            echo "$f"
        fi
    done
)

if (( ${#cpmg_files[@]} == 0 )); then
    echo "No CPMG files were found in $DATA_ROOT with token $CPMG_TOKEN."
    echo "Nothing to do."
    exit 0
fi

total=0
ok=0
failed=0
declare -a failed_jobs=()

for cpmg_file in "${cpmg_files[@]}"; do
    total=$((total + 1))

    hahn_file="${cpmg_file/$CPMG_TOKEN/$HAHN_TOKEN}"
    cpmg_base="$(basename "$cpmg_file")"
    hahn_base="$(basename "$hahn_file")"

    echo "============================================================"
    echo "Job $total"
    echo "  CPMG: $cpmg_base"
    echo "  HAHN: $hahn_base"

    if [[ "$hahn_file" == "$cpmg_file" ]]; then
        failed=$((failed + 1))
        failed_jobs+=("no-token-replacement :: $cpmg_file")
        echo "  ERROR: could not infer HAHN pair by token replacement." >&2
        echo "  Continuing with next job..." >&2
        continue
    fi

    if [[ ! -f "$hahn_file" ]]; then
        failed=$((failed + 1))
        failed_jobs+=("missing HAHN :: $hahn_file")
        echo "  ERROR: missing inferred HAHN file: $hahn_file" >&2
        echo "  Continuing with next job..." >&2
        continue
    fi

    # NOGSE contrast definition: CPMG - HAHN.
    if "$PY" "$MAKE_CONTRAST_SCRIPT" \
        "$cpmg_file" \
        "$hahn_file" \
        "${make_args[@]}" \
        --out_root "$OUT_ROOT"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_jobs+=("exit $status :: $cpmg_file :: $hahn_file")
        echo "  WARNING: command failed with exit code $status" >&2
        echo "  Continuing with next job..." >&2
    fi

done

echo
echo "Finished NOGSE contrast build."
echo "  Total jobs  : $total"
echo "  Successful  : $ok"
echo "  Failed      : $failed"

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
