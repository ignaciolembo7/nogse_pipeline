#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
FIT_SCRIPT="$REPO_ROOT/scripts/fit_nogse_signal_vs_g.py"
DATA_ROOT="$PROJECT_ROOT/analysis/phantoms/nogse_experiments/data/20260122-PHANTOM_FIBER"
FITS_DIR="$PROJECT_ROOT/analysis/phantoms/nogse_experiments/fits"
EXPERIMENT="nogse_signal_vs_g"
APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-false}"
CORR_XLSX="${CORR_XLSX:-$PROJECT_ROOT/analysis/phantoms/nogse_experiments/fits/grad_correction/water.grad_correction.xlsx}"
CORR_ROI="${CORR_ROI:-water}"
CORR_TD_MS="${CORR_TD_MS:-}"
CORR_SHEET="${CORR_SHEET:-}"
CORR_N1="${CORR_N1:-}"
CORR_N2="${CORR_N2:-}"
ROOT_SUFFIX=""
XCOL="g"
YCOL="value_norm"
STAT="avg"
ROIS="ALL"
DIRECTIONS="ALL"
DEFAULT_MODEL="free_cpmg"
AUTO_DISCOVER_JOBS="true"
# Use "fix" or "free" for each fitted parameter.
M0_MODE="fix"
M0_VALUE="1.0"
M0_MIN="0.0"
M0_MAX="2.0"
D0_MODE="free"
D0_VALUE="2.3e-12"
D0_MIN="1e-13"
D0_MAX="1e-10"

# Optional manual jobs, formatted as: "path|model"
# Supported models: free_cpmg | free_hahn
# If this list is empty and AUTO_DISCOVER_JOBS is true, all *.long.parquet
# files under DATA_ROOT are fitted with DEFAULT_MODEL.
declare -a JOBS=(
# "$DATA_ROOT/QUALITY_JACK_19800122TMSF_001_NOGSE_CPMG_N2_TN50_20260122092639_results.long.parquet|free_cpmg"
# "$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_CPMG_N2_TN50_20260122092639_results.long.parquet|free_cpmg"
# "$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_HAHN_N2_TN50_20260122092639_results.long.parquet|free_hahn"
"$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_CPMG_N2_TN50_20260122125354_results.long.parquet|free_cpmg"
"$DATA_ROOT/QUALITY_JACK_19800122TMSF_002_NOGSE_HAHN_N2_TN50_20260122125354_results.long.parquet|free_hahn"
"$DATA_ROOT/QUALITY_JACK_19800122TMSF_003_NOGSE_CPMG_N2_TN65_20260122125354_results.long.parquet|free_cpmg"
"$DATA_ROOT/QUALITY_JACK_19800122TMSF_003_NOGSE_HAHN_N2_TN65_20260122125354_results.long.parquet|free_hahn"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    ROOT_SUFFIX="_corr"
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$FITS_DIR"

if (( ${#JOBS[@]} == 0 )) && [[ "${AUTO_DISCOVER_JOBS,,}" == "true" ]]; then
    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
        exit 1
    fi
    while IFS= read -r file; do
        JOBS+=("$file|$DEFAULT_MODEL")
    done < <(find "$DATA_ROOT" -type f -name "*.long.parquet" | sort)
fi

extra_args=()
extra_args+=(--M0_bounds "$M0_MIN" "$M0_MAX")
extra_args+=(--D0_bounds "$D0_MIN" "$D0_MAX")
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
case "${M0_MODE,,}" in
    fix)
        extra_args+=(--fix_M0 "$M0_VALUE")
        ;;
    free)
        extra_args+=(--free_M0)
        ;;
    *)
        echo "ERROR: M0_MODE must be 'fix' or 'free'." >&2
        exit 1
        ;;
esac
case "${D0_MODE,,}" in
    fix)
        extra_args+=(--fix_D0 "$D0_VALUE")
        ;;
    free)
        extra_args+=(--free_D0)
        ;;
    *)
        echo "ERROR: D0_MODE must be 'fix' or 'free'." >&2
        exit 1
        ;;
esac

corr_args=(--no_grad_corr)
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    corr_args=(
        --apply_grad_corr
        --corr_xlsx "$CORR_XLSX"
        --corr_roi "$CORR_ROI"
    )
    if [[ -n "${CORR_TD_MS// }" ]]; then
        corr_args+=(--corr_td_ms "$CORR_TD_MS")
    fi
    if [[ -n "${CORR_SHEET// }" ]]; then
        corr_args+=(--corr_sheet "$CORR_SHEET")
    fi
fi

total=0
ok=0
failed=0
declare -a failed_jobs=()

for job in "${JOBS[@]}"; do
    total=$((total + 1))
    file_path="${job%%|*}"
    model_name="${job##*|}"
    out_root_job="$FITS_DIR/${EXPERIMENT}_${model_name}${ROOT_SUFFIX}"
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
        --out_root "$out_root_job" \
        --xcol "$XCOL" \
        --ycol "$YCOL" \
        --stat "$STAT" \
        "${corr_args[@]}" \
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
    echo "  ERROR       : no jobs were found. Check DATA_ROOT or add entries to JOBS." >&2
    exit 1
fi

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
