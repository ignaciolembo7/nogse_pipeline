#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
MAKE_CONTRAST_SCRIPT="${MAKE_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/make_contrast.py}"
HELPER_SCRIPT="$REPO_ROOT/bash_template/helpers/run_make_nogse_contrast_auto.sh"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/nogse-contrast-data}"
DIRECTIONS="${DIRECTIONS:-1}"
ONEG="${ONEG:-true}"
SUBJS="${SUBJS:-ALL}"

# Add pairs here to run selected NOGSE contrasts manually.
# Format: "CPMG parquet|HAHN parquet". Entries may be full paths or
# filenames found under DATA_ROOT. If PAIRS is empty, the script uses
# the automatic CPMG -> HAHN pairing helper.
declare -a PAIRS=(
"$DATA_ROOT/20260122-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF_002_NOGSE_CPMG_N2_TN50_results.long.parquet|$DATA_ROOT/20260122-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF_002_NOGSE_HAHN_N2_TN50_results.long.parquet"
"QUALITY_JACK_19800122TMSF_003_NOGSE_CPMG_N2_TN65_results.long.parquet|QUALITY_JACK_19800122TMSF_003_NOGSE_HAHN_N2_TN65_results.long.parquet"
)

if (( ${#PAIRS[@]} == 0 )); then
    if [[ ! -f "$HELPER_SCRIPT" ]]; then
        echo "ERROR: helper script not found: $HELPER_SCRIPT" >&2
        exit 1
    fi

    PY="$PY" \
    DATA_ROOT="$DATA_ROOT" \
    OUT_ROOT="$OUT_ROOT" \
    DIRECTIONS="$DIRECTIONS" \
    ONEG="$ONEG" \
    SUBJS="$SUBJS" \
    bash "$HELPER_SCRIPT"
    exit $?
fi

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast.py not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

make_contrast_args=()
if [[ "${ONEG,,}" == "true" ]]; then
    make_contrast_args+=(--oneg)
fi
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        make_contrast_args+=(--direction "${dir_list[@]}")
    fi
fi
if [[ "$SUBJS" != "ALL" ]]; then
    read -r -a subj_list <<< "${SUBJS//,/ }"
    if (( ${#subj_list[@]} > 0 )); then
        make_contrast_args+=(--subjs "${subj_list[@]}")
    fi
fi

resolve_data_file() {
    local token="$1"

    if [[ -f "$token" ]]; then
        printf '%s\n' "$token"
        return 0
    fi

    local direct="$DATA_ROOT/$token"
    if [[ -f "$direct" ]]; then
        printf '%s\n' "$direct"
        return 0
    fi

    local -a matches=()
    while read -r path; do
        [[ -n "$path" ]] && matches+=("$path")
    done < <(find "$DATA_ROOT" -type f -name "$(basename "$token")" | sort)

    if (( ${#matches[@]} == 1 )); then
        printf '%s\n' "${matches[0]}"
        return 0
    fi

    if (( ${#matches[@]} > 1 )); then
        echo "ERROR: multiple matches found for $token" >&2
        printf '  %s\n' "${matches[@]}" >&2
        return 1
    fi

    echo "ERROR: missing file: $token" >&2
    return 1
}

total=0
ok=0
failed=0
declare -a failed_jobs=()

for pair in "${PAIRS[@]}"; do
    total=$((total + 1))

    token_a="${pair%%|*}"
    token_b="${pair##*|}"

    base_a="$(basename "$token_a")"
    base_b="$(basename "$token_b")"

    echo "============================================================"
    echo "Job $total"
    echo "  CPMG: $base_a"
    echo "  HAHN: $base_b"

    if ! file_a="$(resolve_data_file "$token_a")"; then
        failed=$((failed + 1))
        failed_jobs+=("missing CPMG :: $token_a")
        echo "  Continuing with next job..." >&2
        continue
    fi

    if ! file_b="$(resolve_data_file "$token_b")"; then
        failed=$((failed + 1))
        failed_jobs+=("missing HAHN :: $token_b")
        echo "  Continuing with next job..." >&2
        continue
    fi

    # NOGSE contrast definition: CPMG - HAHN.
    if "$PY" "$MAKE_CONTRAST_SCRIPT" \
        "$file_a" \
        "$file_b" \
        "${make_contrast_args[@]}" \
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
echo "Finished NOGSE contrast build."
echo "  Mode        : manual"
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
