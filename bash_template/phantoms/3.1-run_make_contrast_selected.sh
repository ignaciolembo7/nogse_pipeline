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
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

MAKE_CONTRAST_SCRIPT="$REPO_ROOT/scripts/make_contrast.py"
DATA_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data/20220610-PHANTOM3"
OUT_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/contrast-data"
DIRECTIONS=(1 2 3)


# Add the contrast pairs manually.
declare -a PAIRS=(
  "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d33_Hz065_b0105_DMRIPHANTOM_20220609151744_53_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d33_Hz035_b0380_DMRIPHANTOM_20220609151744_52_results.long.parquet"
  "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d44_Hz050_b0250_DMRIPHANTOM_20220609151744_61_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d44_Hz025_b1075_DMRIPHANTOM_20220609151744_60_results.long.parquet"
  "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d55_Hz040_b0505_DMRIPHANTOM_20220609151744_23_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d55_Hz020_b1755_DMRIPHANTOM_20220609151744_22_results.long.parquet"
  "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d66p7_Hz030_b1165_DMRIPHANTOM_20220609151744_14_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d66p7_Hz015_b2000_DMRIPHANTOM_20220609151744_13_results.long.parquet"
  "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d100_Hz020_b2000_DMRIPHANTOM_20220609151744_70_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d100_Hz010_b2000_DMRIPHANTOM_20220609151744_69_results.long.parquet"
#   "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d28p6_Hz070_b0040_DMRIPHANTOM_20220609151744_122_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d28p6_Hz035_b0190_DMRIPHANTOM_20220609151744_121_results.long.parquet"
#   "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d40_Hz050_b0125_DMRIPHANTOM_20220609151744_114_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d40_Hz025_b0535_DMRIPHANTOM_20220609151744_113_results.long.parquet"
#   "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d50_Hz040_b0250_DMRIPHANTOM_20220609151744_105_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d50_Hz020_b0885_DMRIPHANTOM_20220609151744_104_results.long.parquet"
#   "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d66p7_Hz030_b0530_DMRIPHANTOM_20220609151744_95_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d66p7_Hz015_b1605_DMRIPHANTOM_20220609151744_94_results.long.parquet"
#   "$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d100_Hz020_b1245_DMRIPHANTOM_20220609151744_84_results.long.parquet|$DATA_ROOT/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_DDE_10bval_3orthodir_d100_Hz010_b2000_DMRIPHANTOM_20220609151744_83_results.long.parquet"
 )

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast.py not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

total=0
ok=0
failed=0
declare -a failed_jobs=()

for pair in "${PAIRS[@]}"; do
    total=$((total + 1))

    file_a="${pair%%|*}"
    file_b="${pair##*|}"

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

if (( total == 0 )); then
    echo "  Notes       : PAIRS is empty. Add the contrast pairs manually in this script."
fi

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
