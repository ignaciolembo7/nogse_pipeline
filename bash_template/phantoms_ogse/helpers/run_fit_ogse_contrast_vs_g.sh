#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHANTOMS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PHANTOMS_ROOT/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PY="${PY:-python}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms-3/ogse_experiments}"
FIT_SCRIPT="${FIT_SCRIPT:-$REPO_ROOT/scripts/fit_ogse_contrast_vs_g.py}"
EXPORT_RESAMPLED_SCRIPT="${EXPORT_RESAMPLED_SCRIPT:-$REPO_ROOT/scripts/export_ogse_resampled_contrasts_from_fits.py}"
FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"

EXPERIMENT="ogse_contrast_vs_g"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-direct}" # direct or fitted_resampled
SIGNAL_MODEL="${SIGNAL_MODEL:-monoexp}"
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g}"
MODEL="${MODEL:-free}"
APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-false}"
CORR_XLSX="${CORR_XLSX:-$ANALYSIS_ROOT/fits/grad_correction/water.grad_correction.xlsx}"
CORR_ROI="${CORR_ROI:-water}"
CORR_TD_MS="${CORR_TD_MS:-}"
CORR_SHEET="${CORR_SHEET:-}"
CORR_TOL_MS="${CORR_TOL_MS:-1e-3}"
GBASE="${GBASE:-g}"
YCOL="${YCOL:-value_norm}"
DEFAULT_DIRECTIONS=(1 2 3)
if [[ -n "${DIRECTIONS:-}" ]]; then
    read -r -a DIRECTIONS <<< "${DIRECTIONS//,/ }"
else
    DIRECTIONS=("${DEFAULT_DIRECTIONS[@]}")
fi
ROIS="${ROIS:-ALL}"
ONEG="${ONEG:-true}"
FIX_M0="${FIX_M0:-1.0}"
FREE_M0="${FREE_M0:-}"
FIX_D0="${FIX_D0:-}"
FREE_D0="${FREE_D0:-}"
FIX_TC="${FIX_TC:-}"
FREE_TC="${FREE_TC:-}"
TC_INIT="${TC_INIT:-}"
M0_BOUNDS="${M0_BOUNDS:-}"
D0_BOUNDS="${D0_BOUNDS:-}"
TC_BOUNDS="${TC_BOUNDS:-}"
PEAK_D0_FIX="${PEAK_D0_FIX:-2.3e-12}"
PEAK_G_MAX_MTM="${PEAK_G_MAX_MTM:-}"
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    PEAK_RESAMPLE_GRADIENT="${PEAK_RESAMPLE_GRADIENT:-true}"
else
    PEAK_RESAMPLE_GRADIENT="${PEAK_RESAMPLE_GRADIENT:-false}"
fi
RESAMPLED_GRID_MIN_MTM="${RESAMPLED_GRID_MIN_MTM:-0}"
RESAMPLED_GRID_MAX_MTM="${RESAMPLED_GRID_MAX_MTM:-90}"
RESAMPLED_GRID_N="${RESAMPLED_GRID_N:-1000}"
EXPORT_FIT_PARAMS_PATTERN="${EXPORT_FIT_PARAMS_PATTERN:-**/fit_params.*}"
EXPORT_RESAMPLED_CONTRASTS="${EXPORT_RESAMPLED_CONTRASTS:-}"

if [[ -z "${TABLES_ROOT+x}" ]]; then
    if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
        TABLES_ROOT="$ANALYSIS_ROOT/contrast-data/tables"
    else
        TABLES_ROOT="$ANALYSIS_ROOT/contrast-data/tables"
    fi
fi

ROOT_SUFFIX=""
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    ROOT_SUFFIX="_corr"
fi
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/fits/${EXPERIMENT}resampled_${MODEL}${ROOT_SUFFIX}}"
else
    OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/fits/${EXPERIMENT}_${MODEL}${ROOT_SUFFIX}}"
fi

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "ERROR: Tables root not found: $TABLES_ROOT" >&2
    exit 1
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: Fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ "$CONTRAST_SOURCE" == "fitted_resampled" && ! -f "$EXPORT_RESAMPLED_SCRIPT" ]]; then
    echo "ERROR: Resampled contrast export script not found: $EXPORT_RESAMPLED_SCRIPT" >&2
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

tc_args=()
if [[ -n "${FREE_TC// }" ]]; then
    tc_args+=(--free_tc "$FREE_TC")
elif [[ -n "${FIX_TC// }" ]]; then
    tc_args+=(--fix_tc "$FIX_TC")
elif [[ -n "${TC_INIT// }" ]]; then
    tc_args+=(--tc_init "$TC_INIT")
fi

bound_args=()
if [[ -n "${M0_BOUNDS// }" ]]; then
    read -r -a m0_bound_values <<< "${M0_BOUNDS//,/ }"
    bound_args+=(--M0_bounds "${m0_bound_values[@]}")
fi
if [[ -n "${D0_BOUNDS// }" ]]; then
    read -r -a d0_bound_values <<< "${D0_BOUNDS//,/ }"
    bound_args+=(--D0_bounds "${d0_bound_values[@]}")
fi
if [[ -n "${TC_BOUNDS// }" ]]; then
    read -r -a tc_bound_values <<< "${TC_BOUNDS//,/ }"
    bound_args+=(--tc_bounds "${tc_bound_values[@]}")
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

peak_args=(--peak_D0_fix "$PEAK_D0_FIX")
if [[ -n "${PEAK_G_MAX_MTM// }" ]]; then
    peak_args+=(--peak_g_max_mTm "$PEAK_G_MAX_MTM")
fi
if [[ "${PEAK_RESAMPLE_GRADIENT,,}" == "true" ]]; then
    peak_args+=(--peak_resample_gradient)
    if [[ -n "${RESAMPLED_GRID_MAX_MTM// }" ]]; then
        peak_args+=(--peak_resample_g_max_corr_mTm "$RESAMPLED_GRID_MAX_MTM")
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
    echo "  Contrast source: $CONTRAST_SOURCE"

    if "$PY" "$FIT_SCRIPT" \
        "$file" \
        --model "$MODEL" \
        --gbase "$GBASE" \
        --ycol "$YCOL" \
        --directions "${DIRECTIONS[@]}" \
        --out_root "$OUT_ROOT" \
        "${peak_args[@]}" \
        "${oneg_args[@]}" \
        "${corr_args[@]}" \
        "${m0_args[@]}" \
        "${d0_args[@]}" \
        "${tc_args[@]}" \
        "${bound_args[@]}" \
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

if [[ "$CONTRAST_SOURCE" == "fitted_resampled" && "${EXPORT_RESAMPLED_CONTRASTS:-true}" != "false" ]]; then
    echo
    echo "Exporting fitted/resampled model contrasts..."
    "$PY" "$EXPORT_RESAMPLED_SCRIPT" \
        "$OUT_ROOT" \
        --contrast-root "$ANALYSIS_ROOT/contrast-data" \
        --out-dir "$OUT_ROOT/contrast" \
        --pattern "$EXPORT_FIT_PARAMS_PATTERN" \
        --grid-min-mTm "$RESAMPLED_GRID_MIN_MTM" \
        --grid-max-mTm "$RESAMPLED_GRID_MAX_MTM" \
        --grid-n "$RESAMPLED_GRID_N" \
        --models "$MODEL" \
        --rois "$ROIS" \
        --directions "$(IFS=,; echo "${DIRECTIONS[*]}")" \
        --peak-D0-fix "$PEAK_D0_FIX"
fi
