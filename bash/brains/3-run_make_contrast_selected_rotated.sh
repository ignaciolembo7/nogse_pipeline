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
DATA_ROOT="$ANALYSIS_ROOT/data-rotated"
OUT_ROOT="$ANALYSIS_ROOT/contrast-data-rotated"
DIRECTIONS=(long tra)

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

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast.py not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

total=0
ok=0
failed=0
declare -a failed_jobs=()

for pair in "${PAIRS[@]}"; do
    total=$((total + 1))

    file_a="$DATA_ROOT/${pair%%|*}"
    file_b="$DATA_ROOT/${pair##*|}"

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
