#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHANTOMS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PHANTOMS_ROOT/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"


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

ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms/nogse_experiments}"
TABLES_ROOT="${TABLES_ROOT:-$ANALYSIS_ROOT/contrast_data/tables}"
FIT_SCRIPT="${FIT_SCRIPT:-$REPO_ROOT/scripts/fit_nogse_contrast_vs_g.py}"
FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"

EXPERIMENT="nogse_contrast_vs_g"
MODEL="${MODEL:-free}"
APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-false}"
CORR_XLSX="${CORR_XLSX:-$ANALYSIS_ROOT/fits/grad_correction/water.grad_correction.xlsx}"
CORR_ROI="${CORR_ROI:-water}"
CORR_TD_MS="${CORR_TD_MS:-}"
CORR_SHEET="${CORR_SHEET:-}"
CORR_TOL_MS="${CORR_TOL_MS:-1e-3}"
GBASE="${GBASE:-g}"
YCOL="${YCOL:-value_norm}"
DIRECTIONS="${DIRECTIONS:-ALL}"
ROIS="${ROIS:-ALL}"
# ROIS="water,water1"
ONEG="${ONEG:-true}"
FIX_M0="${FIX_M0:-1.0}"
FREE_M0="${FREE_M0:-}"
M0_MIN="${M0_MIN:-}"
M0_MAX="${M0_MAX:-}"
FIX_D0="${FIX_D0:-}"
FREE_D0="${FREE_D0:-}"
D0_MIN="${D0_MIN:-}"
D0_MAX="${D0_MAX:-}"
TC_MIN="${TC_MIN:-}"
TC_MAX="${TC_MAX:-}"
PEAK_D0_FIX="${PEAK_D0_FIX:-2.3e-12}"

# ------------------------------------------------------------------
# ------------------------------------------------------------------

ROOT_SUFFIX=""
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    ROOT_SUFFIX="_corr"
fi
OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/fits/${EXPERIMENT}_${MODEL}${ROOT_SUFFIX}}"

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "Contrast tables root not found: $TABLES_ROOT. Skipping contrast fit."
    exit 0
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: Fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

corr_args=(--no_grad_corr)
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    corr_args=(
        --apply_grad_corr
        --corr_xlsx "$CORR_XLSX"
        --corr_roi "$CORR_ROI"
        --corr_tol_ms "$CORR_TOL_MS"
    )
    if [[ -n "${CORR_TD_MS// }" ]]; then
        corr_args+=(--corr_td_ms "$CORR_TD_MS")
    fi
    if [[ -n "${CORR_SHEET// }" ]]; then
        corr_args+=(--corr_sheet "$CORR_SHEET")
    fi
fi

m0_args=()
if [[ -n "${FREE_M0// }" ]]; then
    m0_args+=(--free_M0 "$FREE_M0")
elif [[ -n "${FIX_M0// }" ]]; then
    m0_args+=(--fix_M0 "$FIX_M0")
fi

d0_args=()
if [[ -n "${FREE_D0// }" ]]; then
    d0_args+=(--free_D0 "$FREE_D0")
elif [[ -n "${FIX_D0// }" ]]; then
    d0_args+=(--fix_D0 "$FIX_D0")
fi

bounds_args=()
if [[ -n "${M0_MIN// }" || -n "${M0_MAX// }" ]]; then
    if [[ -z "${M0_MIN// }" || -z "${M0_MAX// }" ]]; then
        echo "ERROR: M0_MIN and M0_MAX must be provided together." >&2
        exit 1
    fi
    bounds_args+=(--M0_bounds "$M0_MIN" "$M0_MAX")
fi
if [[ -n "${D0_MIN// }" || -n "${D0_MAX// }" ]]; then
    if [[ -z "${D0_MIN// }" || -z "${D0_MAX// }" ]]; then
        echo "ERROR: D0_MIN and D0_MAX must be provided together." >&2
        exit 1
    fi
    bounds_args+=(--D0_bounds "$D0_MIN" "$D0_MAX")
fi
if [[ -n "${TC_MIN// }" || -n "${TC_MAX// }" ]]; then
    if [[ -z "${TC_MIN// }" || -z "${TC_MAX// }" ]]; then
        echo "ERROR: TC_MIN and TC_MAX must be provided together." >&2
        exit 1
    fi
    bounds_args+=(--tc_bounds "$TC_MIN" "$TC_MAX")
fi

oneg_args=()
if [[ "${ONEG,,}" == "true" ]]; then
    oneg_args+=(--oneg)
fi

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

    if "$PY" "$FIT_SCRIPT" \
        "$file" \
        --model "$MODEL" \
        --gbase "$GBASE" \
        --ycol "$YCOL" \
        "${direction_args[@]}" \
        --out_root "$OUT_ROOT" \
        --peak_D0_fix "$PEAK_D0_FIX" \
        "${oneg_args[@]}" \
        "${corr_args[@]}" \
        "${m0_args[@]}" \
        "${d0_args[@]}" \
        "${bounds_args[@]}" \
        "${roi_args[@]}"; then
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
