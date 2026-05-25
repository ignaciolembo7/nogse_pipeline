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

MAKE_CONTRAST_SCRIPT="$REPO_ROOT/scripts/make_contrast.py"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms-3/ogse_experiments}"
DATA_ROOT="${DATA_ROOT:-$ANALYSIS_ROOT/data/tables/20220610-PHANTOM3}"
DIRECTIONS=(1 2 3)
ONEG="${ONEG:-false}"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}" # direct or fitted_resampled
SIGNAL_MODEL="${SIGNAL_MODEL:-rest_offset}"
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g}"
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
PEAK_D0_FIX="${PEAK_D0_FIX:-2.3e-12}"
if [[ -z "${OUT_ROOT+x}" ]]; then
    OUT_ROOT="$ANALYSIS_ROOT/contrast-data"
fi


# Add the contrast pairs manually.
# Example:
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
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$MAKE_CONTRAST_SCRIPT" ]]; then
    echo "ERROR: make_contrast.py not found: $MAKE_CONTRAST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
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
    echo "  Mode: $CONTRAST_SOURCE"
    if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
        echo "  Signal model  : $SIGNAL_MODEL"
        echo "  Signal axis   : $SIGNAL_G_TYPE"
        echo "  Resampled grid: $RESAMPLE_GRID_MIN_MTM..$RESAMPLE_GRID_MAX_MTM mT/m, n=$RESAMPLE_GRID_N"
    fi

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
