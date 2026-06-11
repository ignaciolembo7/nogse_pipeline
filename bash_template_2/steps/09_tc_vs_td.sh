#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help tc "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

TC_VS_TD_SCRIPT="${TC_VS_TD_SCRIPT:-$REPO_ROOT/scripts/fitting/run_tc_vs_td.py}"
TC_OUT_DIR="${TC_OUT_DIR:-$ANALYSIS_ROOT/fits/tc_vs_td_master}"

bt2_require_file "$TC_VS_TD_SCRIPT" "tc-vs-td script"
bt2_require_file "$MASTER_FIT_PARAMS" "master fit params"
mkdir -p "$TC_OUT_DIR"

"$PY" "$TC_VS_TD_SCRIPT" \
    --master-fit-params "$MASTER_FIT_PARAMS" \
    --method "${TC_METHOD:-pseudohuber_fixed_macro}" \
    --y-col "${TC_Y_COL:-tc_peak_ms}" \
    --out-dir "$TC_OUT_DIR/${TC_METHOD:-pseudohuber_fixed_macro}" \
    ${TC_EXTRA_ARGS:-}
