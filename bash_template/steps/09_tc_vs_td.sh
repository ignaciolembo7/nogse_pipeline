#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/manifests/helpers/master_table_common.sh"
pipeline_maybe_step_help tc "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

TC_VS_TD_SCRIPT="${TC_VS_TD_SCRIPT:-$REPO_ROOT/scripts/fitting/run_tc_vs_td.py}"
TC_OUT_DIR="${TC_OUT_DIR:-$ANALYSIS_ROOT/fits/tc_vs_td_master}"
TC_FIT_PARAMS="${TC_FIT_PARAMS:?TC_FIT_PARAMS must be set to the contrast fit-params parquet (e.g. fits/master/ogse_value_norm_vs_glinmax_ogse_free/fit_params.ogse_free.glinmax.value_norm.direction_ALL.parquet)}"

pipeline_require_file "$TC_VS_TD_SCRIPT" "tc-vs-td script"
pipeline_require_file "$TC_FIT_PARAMS" "tc fit params"
mkdir -p "$TC_OUT_DIR"

"$PY" "$TC_VS_TD_SCRIPT" \
    --master-fit-params "$TC_FIT_PARAMS" \
    --method "${TC_METHOD:-pseudohuber_fixed_macro}" \
    --y-col "${TC_Y_COL:-tc_peak_ms}" \
    --out-dir "$TC_OUT_DIR/${TC_METHOD:-pseudohuber_fixed_macro}" \
    ${TC_EXTRA_ARGS:-}
