#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
FIT_SCRIPT="${1:-$REPO_ROOT/scripts/fit_ogse-signal_vs_bval.py}"
DATA_ROOT="${2:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data}"
OUT_ROOT="${3:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_monoexp_ogse-signal}"
DPROJ_ROOT="${4:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data}"

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
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d33_Hz000_b2000_DMRIPHANTOM_20220609151744_51_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d33_Hz035_b0380_DMRIPHANTOM_20220609151744_52_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d33_Hz065_b0105_DMRIPHANTOM_20220609151744_53_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d44_Hz000_b2000_DMRIPHANTOM_20220609151744_59_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d44_Hz025_b1075_DMRIPHANTOM_20220609151744_60_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d44_Hz050_b0250_DMRIPHANTOM_20220609151744_61_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d55_Hz000_b2000_DMRIPHANTOM_20220609151744_21_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d55_Hz020_b1755_DMRIPHANTOM_20220609151744_22_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d55_Hz040_b0505_DMRIPHANTOM_20220609151744_23_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d66p7_Hz000_b2000_DMRIPHANTOM_20220609151744_12_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d66p7_Hz015_b2000_DMRIPHANTOM_20220609151744_13_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d66p7_Hz030_b1165_DMRIPHANTOM_20220609151744_14_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d100_Hz000_b2000_DMRIPHANTOM_20220609151744_68_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d100_Hz010_b2000_DMRIPHANTOM_20220609151744_69_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d100_Hz020_b2000_DMRIPHANTOM_20220609151744_70_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d28p6_Hz000_b2000_DMRIPHANTOM_20220609151744_120_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d28p6_Hz035_b0190_DMRIPHANTOM_20220609151744_121_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d28p6_Hz070_b0040_DMRIPHANTOM_20220609151744_122_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d40_Hz000_b2000_DMRIPHANTOM_20220609151744_112_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d40_Hz025_b0535_DMRIPHANTOM_20220609151744_113_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d40_Hz050_b0125_DMRIPHANTOM_20220609151744_114_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d50_Hz000_b2000_DMRIPHANTOM_20220609151744_103_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d50_Hz020_b0885_DMRIPHANTOM_20220609151744_104_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d50_Hz040_b0250_DMRIPHANTOM_20220609151744_105_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d66p7_Hz000_b2000_DMRIPHANTOM_20220609151744_93_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d66p7_Hz015_b1605_DMRIPHANTOM_20220609151744_94_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d66p7_Hz030_b0530_DMRIPHANTOM_20220609151744_95_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d100_Hz000_b2000_DMRIPHANTOM_20220609151744_82_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d100_Hz010_b2000_DMRIPHANTOM_20220609151744_83_results.long.parquet"
"20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d100_Hz020_b1245_DMRIPHANTOM_20220609151744_84_results.long.parquet"
)

roi_args_for_file() {
    local fname="$1"
    echo "fiber1 fiber2 water2 water3"
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
        --out_dproj_root "$DPROJ_ROOT" \
        --directions 1 2 3  \
        --ycol value_norm \
        --g_type bvalue \
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
