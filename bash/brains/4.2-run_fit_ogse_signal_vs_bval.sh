#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
FIT_SCRIPT="${1:-$REPO_ROOT/scripts/fit_ogse-signal_vs_bval.py}"
DATA_ROOT="${2:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated}"
OUT_ROOT="${3:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/fit_monoexp_ogse-signal-rotated}"

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

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
        --out_root "$OUT_ROOT" \
        --directions long tra \
        --ycol value_norm \
        --g_type bvalue_thorsten \
        --fix_M0 1.0 \
        --rois "${ROI_ARGS[@]}" \
        --auto_fit_points \
        --auto_fit_min_points 3 \
	--auto_fit_max_points 11 \
	--auto_fit_err_floor 0.05 \
	--auto_fit_tol 0.15; then 
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
