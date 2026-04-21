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

ANALYSIS_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments"
TABLES_ROOT="$ANALYSIS_ROOT/contrast-data/tables"
OUT_ROOT="$ANALYSIS_ROOT/fits/fit_rest_ogse_contrast_corr"
FIT_SCRIPT="$REPO_ROOT/scripts/fit_ogse-contrast_vs_g.py"
GRAD_CORR_XLSX="$ANALYSIS_ROOT/fits/grad_correction/water.grad_correction.xlsx"
FILE_PATTERN="*.long.parquet"

MODEL="rest"
GBASE="g"
YCOL="value_norm"
DIRECTIONS="ALL"
CORR_ROI="water"
ROIS="ALL"
PEAK_D0_FIX="2.3e-12"
FREE_M0="1.0"
FIX_D0="2.3e-12"

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "Contrast tables root not found: $TABLES_ROOT. Skipping corrected rest fit."
    exit 0
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: Fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$GRAD_CORR_XLSX" ]]; then
    echo "Gradient correction file not found: $GRAD_CORR_XLSX. Skipping corrected rest fit."
    exit 0
fi

mkdir -p "$OUT_ROOT"

roi_args=()
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        roi_args+=(--rois "${roi_list[@]}")
    fi
fi

direction_args=()
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        direction_args+=(--directions "${dir_list[@]}")
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
    echo "  File: $base_name"
    echo "  ROIs  : $ROIS"

    cmd=(
        "$PY" "$FIT_SCRIPT"
        "$file"
        --model "$MODEL"
        --gbase "$GBASE"
        --ycol "$YCOL"
        "${direction_args[@]}"
        --peak_D0_fix "$PEAK_D0_FIX"
        --apply_grad_corr
        --corr_xlsx "$GRAD_CORR_XLSX"
        --corr_roi "$CORR_ROI"
        "${roi_args[@]}"
        --out_root "$OUT_ROOT"
    )

    if [[ -n "${FREE_M0// }" ]]; then
        cmd+=(--free_M0 "$FREE_M0")
    fi

    if [[ -n "${FIX_D0// }" ]]; then
        cmd+=(--fix_D0 "$FIX_D0")
    fi

    if "${cmd[@]}"; then
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
