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
FIT_SCRIPT="${1:-$REPO_ROOT/scripts/fit_ogse_signal_vs_g.py}"
DATA_ROOT="${2:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated}"
FITS_DIR="$PROJECT_ROOT/analysis/brains/ogse_experiments/fits"
EXPERIMENT="ogse_signal_vs_g"
MODEL="monoexp"
APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-false}"
CORR_XLSX="${CORR_XLSX:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/grad_correction_rotated/Syringe.grad_correction_rotated.xlsx}"
CORR_ROI="${CORR_ROI:-Syringe}"
CORR_TD_MS="${CORR_TD_MS:-}"
CORR_SHEET="${CORR_SHEET:-}"
CORR_N1="${CORR_N1:-}"
CORR_N2="${CORR_N2:-}"
DEFAULT_OUT_ROOT="$FITS_DIR/${EXPERIMENT}_${MODEL}"
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    DEFAULT_OUT_ROOT="${DEFAULT_OUT_ROOT}_corr"
fi
OUT_ROOT="${3:-$DEFAULT_OUT_ROOT}"
YCOL="value_norm"
G_TYPE="bvalue_thorsten"
DIRECTIONS=(long tra)
FIX_M0="1.0"
AUTO_FIT_MIN_POINTS="3"
AUTO_FIT_MAX_POINTS="11"
AUTO_FIT_ERR_FLOOR="0.05"
AUTO_FIT_TOL="0.15"

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

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

declare -a FILES=(
"20220622_BRAIN_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz000_b2000_19800122XXXX_20220622170141_5_results.rot_tensor.long.parquet"
"20220622_BRAIN_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz025_b1075_19800122XXXX_20220622170141_6_results.rot_tensor.long.parquet"
"20220622_BRAIN_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz050_b0250_19800122XXXX_20220622170141_7_results.rot_tensor.long.parquet"
"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz000_b2000_QC-ROUTINE_20230619152408_10_results.rot_tensor.long.parquet"
"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz020_b1770_QC-ROUTINE_20230619152408_11_results.rot_tensor.long.parquet"
"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230619152408_12_results.rot_tensor.long.parquet"
"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz000_b2000_QC-ROUTINE_20230619152408_5_results.rot_tensor.long.parquet"
"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz015_b3210_QC-ROUTINE_20230619152408_6_results.rot_tensor.long.parquet"
"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d33_Hz000_b2000_Anonymous_20230623084632_10_results.rot_tensor.long.parquet"
"20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz030_b1175_QC-ROUTINE_20230619152408_7_results.rot_tensor.long.parquet"
"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d33_Hz035_b0380_Anonymous_20230623084632_11_results.rot_tensor.long.parquet"
"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d33_Hz065_b0105_Anonymous_20230623084632_12_results.rot_tensor.long.parquet"
"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz000_b2000_Anonymous_20230623084632_5_results.rot_tensor.long.parquet"
"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz010_b3020_Anonymous_20230623084632_6_results.rot_tensor.long.parquet"
"20230623_BRAIN-4_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz020_b3020_Anonymous_20230623084632_7_results.rot_tensor.long.parquet"
"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz000_b2000_Anonymous_20230623105657_11_results.rot_tensor.long.parquet"
"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz020_b1770_Anonymous_20230623105657_12_results.rot_tensor.long.parquet"
"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_Anonymous_20230623105657_13_results.rot_tensor.long.parquet"
"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz000_b2000_Anonymous_20230623105657_6_results.rot_tensor.long.parquet"
"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz015_b3210_Anonymous_20230623105657_7_results.rot_tensor.long.parquet"
"20230623_LUDG-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz030_b1175_Anonymous_20230623105657_8_results.rot_tensor.long.parquet"
"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz000_b2000_QC-ROUTINE_20230629150947_11_results.rot_tensor.long.parquet"
"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz020_b1770_QC-ROUTINE_20230629150947_12_results.rot_tensor.long.parquet"
"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230629150947_13_results.rot_tensor.long.parquet"
"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz000_b2000_QC-ROUTINE_20230629150947_6_results.rot_tensor.long.parquet"
"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz015_b3210_QC-ROUTINE_20230629150947_7_results.rot_tensor.long.parquet"
"20230629_MBBL-2_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d66p7_Hz030_b1175_QC-ROUTINE_20230629150947_8_results.rot_tensor.long.parquet"
"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz000_b2000_19760622MBBL_20230630131548_3_results.rot_tensor.long.parquet"
"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz025_b1075_19760622MBBL_20230630131548_4_results.rot_tensor.long.parquet"
"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz050_b0250_19760622MBBL_20230630131548_5_results.rot_tensor.long.parquet"
"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz000_b2000_19760622MBBL_20230630131548_8_results.rot_tensor.long.parquet"
"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz010_b3020_19760622MBBL_20230630131548_9_results.rot_tensor.long.parquet"
"20230630_MBBL-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz020_b3020_19760622MBBL_20230630131548_10_results.rot_tensor.long.parquet"
"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz000_b2000_QC-ROUTINE_20230710145211_3_results.rot_tensor.long.parquet"
"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz025_b1075_QC-ROUTINE_20230710145211_4_results.rot_tensor.long.parquet"
"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d40_Hz050_b0250_QC-ROUTINE_20230710145211_5_results.rot_tensor.long.parquet"
"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz000_b2000_QC-ROUTINE_20230710145211_8_results.rot_tensor.long.parquet"
"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz010_b3020_QC-ROUTINE_20230710145211_9_results.rot_tensor.long.parquet"
"20230710_LUDG-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d100_Hz020_b3020_QC-ROUTINE_20230710145211_10_results.rot_tensor.long.parquet"
)

roi_args_for_file() {
    local fname="$1"
    if [[ "$fname" == 20220622_BRAIN_* ]]; then
        echo "Left-Lateral-Ventricle Right-Lateral-Ventricle"
    else
        echo "Left-Lateral-Ventricle Right-Lateral-Ventricle Syringe"
    fi
}

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

for fname in "${FILES[@]}"; do
    total=$((total + 1))

    echo "============================================================"
    echo "Job $total"
    echo "  File: $fname"

    if ! file_path="$(resolve_data_file "$fname")"; then
        failed=$((failed + 1))
        failed_jobs+=("missing :: $fname")
        echo "  Continuing with next job..." >&2
        continue
    fi

    read -r -a ROI_ARGS <<< "$(roi_args_for_file "$fname")"

    if "$PY" "$FIT_SCRIPT" \
        "$file_path" \
        --model "$MODEL" \
        --out_root "$OUT_ROOT" \
        --directions "${DIRECTIONS[@]}" \
        --ycol "$YCOL" \
        --g_type "$G_TYPE" \
        --fix_M0 "$FIX_M0" \
        --rois "${ROI_ARGS[@]}" \
        "${corr_args[@]}" \
        --auto_fit_points \
        --auto_fit_min_points "$AUTO_FIT_MIN_POINTS" \
        --auto_fit_max_points "$AUTO_FIT_MAX_POINTS" \
        --auto_fit_err_floor "$AUTO_FIT_ERR_FLOOR" \
        --auto_fit_tol "$AUTO_FIT_TOL"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_jobs+=("exit $status :: $file_path")
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
