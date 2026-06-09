#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-python}"
MAKE_CONTRAST_SCRIPT="$REPO_ROOT/scripts/make_contrast.py"
ANALYSIS_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments"
DATA_ROOT="$ANALYSIS_ROOT/data-rotated/tables"
DIRECTIONS=(long tra)
ONEG="${ONEG:-false}"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}" # direct or fitted_resampled
SIGNAL_MODEL="${SIGNAL_MODEL:-rest_offset_globC}" #monoexp, rest, rest_offset o rest_offset_globC
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g_thorsten}"
SIGNAL_YCOL="${SIGNAL_YCOL:-value_norm}"
RESAMPLE_GRID_MIN_MTM="${RESAMPLE_GRID_MIN_MTM:-0}"
RESAMPLE_GRID_MAX_MTM="${RESAMPLE_GRID_MAX_MTM:-90}"
RESAMPLE_GRID_N="${RESAMPLE_GRID_N:-1000}"
SIGNAL_FIT_POINTS="${SIGNAL_FIT_POINTS:-6}"
SIGNAL_AUTO_FIT_POINTS="${SIGNAL_AUTO_FIT_POINTS:-false}"
SIGNAL_AUTO_FIT_TOL="${SIGNAL_AUTO_FIT_TOL:-0.05}"
SIGNAL_AUTO_FIT_ERR_FLOOR="${SIGNAL_AUTO_FIT_ERR_FLOOR:-0.005}"
SIGNAL_AUTO_FIT_MIN_POINTS="${SIGNAL_AUTO_FIT_MIN_POINTS:-3}"
SIGNAL_AUTO_FIT_MAX_POINTS="${SIGNAL_AUTO_FIT_MAX_POINTS:-9}"
SIGNAL_FIX_M0="${SIGNAL_FIX_M0:-1.0}"
SIGNAL_FREE_M0="${SIGNAL_FREE_M0:-false}"
SIGNAL_D0_INIT="${SIGNAL_D0_INIT:-0.0023}"
PEAK_D0_FIX="${PEAK_D0_FIX:-3.2e-12}"
if [[ -z "${OUT_ROOT+x}" ]]; then
    if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
        OUT_ROOT="$ANALYSIS_ROOT/contrast-data-fitresampled-rotated-${SIGNAL_MODEL}-${SIGNAL_G_TYPE}"
    else
        OUT_ROOT="$ANALYSIS_ROOT/contrast-data-rotated"
    fi
fi
# ------------------------------------------------------------------
# ------------------------------------------------------------------

declare -a PAIRS=(
"20220622_BRAIN_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz050_b0250_19800122XXXX_20220622170141_7_results.rot_tensor.long.parquet|20220622_BRAIN_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz025_b1075_19800122XXXX_20220622170141_6_results.rot_tensor.long.parquet"

"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230619152408_12_results.rot_tensor.long.parquet|20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz020_b1770_QC-ROUTINE_20230619152408_11_results.rot_tensor.long.parquet"

"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz030_b1175_QC-ROUTINE_20230619152408_7_results.rot_tensor.long.parquet|20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz015_b3210_QC-ROUTINE_20230619152408_6_results.rot_tensor.long.parquet"

"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d33_Hz065_b0105_Anonymous_20230623084632_12_results.rot_tensor.long.parquet|20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d33_Hz035_b0380_Anonymous_20230623084632_11_results.rot_tensor.long.parquet"

"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz020_b3020_Anonymous_20230623084632_7_results.rot_tensor.long.parquet|20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz010_b3020_Anonymous_20230623084632_6_results.rot_tensor.long.parquet"

"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_Anonymous_20230623105657_13_results.rot_tensor.long.parquet|20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz020_b1770_Anonymous_20230623105657_12_results.rot_tensor.long.parquet"

"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz030_b1175_Anonymous_20230623105657_8_results.rot_tensor.long.parquet|20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz015_b3210_Anonymous_20230623105657_7_results.rot_tensor.long.parquet"

"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230629150947_13_results.rot_tensor.long.parquet|20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz020_b1770_QC-ROUTINE_20230629150947_12_results.rot_tensor.long.parquet"

"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz030_b1175_QC-ROUTINE_20230629150947_8_results.rot_tensor.long.parquet|20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz015_b3210_QC-ROUTINE_20230629150947_7_results.rot_tensor.long.parquet"

"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz050_b0250_19760622MBBL_20230630131548_5_results.rot_tensor.long.parquet|20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz025_b1075_19760622MBBL_20230630131548_4_results.rot_tensor.long.parquet"

"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz020_b3020_19760622MBBL_20230630131548_10_results.rot_tensor.long.parquet|20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz010_b3020_19760622MBBL_20230630131548_9_results.rot_tensor.long.parquet"

"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz050_b0250_QC-ROUTINE_20230710145211_5_results.rot_tensor.long.parquet|20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz025_b1075_QC-ROUTINE_20230710145211_4_results.rot_tensor.long.parquet"

"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz020_b3020_QC-ROUTINE_20230710145211_10_results.rot_tensor.long.parquet|20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz010_b3020_QC-ROUTINE_20230710145211_9_results.rot_tensor.long.parquet"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast.py not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

MAKE_CONTRAST_ARGS=()
if [[ "${ONEG,,}" == "true" ]]; then
    MAKE_CONTRAST_ARGS+=(--oneg)
fi
MAKE_CONTRAST_ARGS+=(--contrast-source "$CONTRAST_SOURCE")

if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    MAKE_CONTRAST_ARGS+=(
        --signal-model "$SIGNAL_MODEL"
        --g_type "$SIGNAL_G_TYPE"
        --ycol "$SIGNAL_YCOL"
        --resample_grid_min_mTm "$RESAMPLE_GRID_MIN_MTM"
        --resample_grid_max_mTm "$RESAMPLE_GRID_MAX_MTM"
        --auto_fit_tol "$SIGNAL_AUTO_FIT_TOL"
        --auto_fit_err_floor "$SIGNAL_AUTO_FIT_ERR_FLOOR"
        --auto_fit_min_points "$SIGNAL_AUTO_FIT_MIN_POINTS"
        --auto_fit_max_points "$SIGNAL_AUTO_FIT_MAX_POINTS"
        --fix_M0 "$SIGNAL_FIX_M0"
        --D0_init "$SIGNAL_D0_INIT"
        --peak_D0_fix "$PEAK_D0_FIX"
    )
    if [[ -n "${RESAMPLE_GRID_N// }" ]]; then
        MAKE_CONTRAST_ARGS+=(--resample_grid_n "$RESAMPLE_GRID_N")
    fi
    if [[ "${SIGNAL_AUTO_FIT_POINTS,,}" == "true" ]]; then
        MAKE_CONTRAST_ARGS+=(--auto_fit_points)
    else
        MAKE_CONTRAST_ARGS+=(--fit_points "$SIGNAL_FIT_POINTS")
    fi
    if [[ "${SIGNAL_FREE_M0,,}" == "true" ]]; then
        MAKE_CONTRAST_ARGS+=(--free_M0)
    fi
fi

resolve_data_file() {
    local fname="$1"
    local direct="$DATA_ROOT/$fname"
    if [[ -f "$direct" ]]; then
        printf '%s\n' "$direct"
        return 0
    fi

    local -a matches=()
    while read -r path; do
        [[ -n "$path" ]] && matches+=("$path")
    done < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type f -name "$fname" | sort)

    if (( ${#matches[@]} == 1 )); then
        printf '%s\n' "${matches[0]}"
        return 0
    fi

    if (( ${#matches[@]} > 1 )); then
        echo "ERROR: multiple matches found for $fname" >&2
        printf '  %s\n' "${matches[@]}" >&2
        return 1
    fi

    echo "ERROR: missing file: $direct" >&2
    return 1
}

total=0
ok=0
failed=0
declare -a failed_jobs=()

for pair in "${PAIRS[@]}"; do
    total=$((total + 1))

    fname_a="${pair%%|*}"
    fname_b="${pair##*|}"

    base_a="$(basename "$fname_a")"
    base_b="$(basename "$fname_b")"

    echo "============================================================"
    echo "Job $total"
    echo "  A: $base_a"
    echo "  B: $base_b"
    echo "  Mode: $CONTRAST_SOURCE"
    if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
        echo "  Signal model  : $SIGNAL_MODEL"
        echo "  Signal axis   : $SIGNAL_G_TYPE"
        echo "  Resampled grid: $RESAMPLE_GRID_MIN_MTM..$RESAMPLE_GRID_MAX_MTM mT/m, n=$RESAMPLE_GRID_N"
    fi

    if ! file_a="$(resolve_data_file "$fname_a")"; then
        failed=$((failed + 1))
        failed_jobs+=("missing A :: $fname_a")
        echo "  Continuing with next job..." >&2
        continue
    fi

    if ! file_b="$(resolve_data_file "$fname_b")"; then
        failed=$((failed + 1))
        failed_jobs+=("missing B :: $fname_b")
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

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
