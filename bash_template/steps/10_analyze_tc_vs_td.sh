#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help tc "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

TC_VS_TD_SCRIPT="${TC_VS_TD_SCRIPT:-$REPO_ROOT/scripts/fitting/run_tc_vs_td.py}"
TC_OUT_DIR="${TC_OUT_DIR:-$ANALYSIS_ROOT/fits/tc_vs_td_master}"

# Auto-derive tc_peak table from step 09 output using the same FIT_OUT_ROOT logic.
if [[ "$TYPE_SEQ" == "nogse" ]]; then
    _tc_default_model="${DEFAULT_FIT_MODEL:-nogse_free}"
else
    _tc_default_model="${DEFAULT_FIT_MODEL:-ogse_free}"
fi
_tc_master_name=$(basename "${MASTER_PARQUET:-master.long.parquet}" | sed 's/\.long\.parquet$//' | sed 's/\.parquet$//')
_tc_ycol="${FIT_YCOL:-value_norm}"
_tc_gtype="${FIT_GBASE:-g_lin_max}"
_tc_gtype_clean="${_tc_gtype//_/}"
_tc_model="${FIT_MODEL:-$_tc_default_model}"
_tc_fit_root="${FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/$_tc_master_name/${TYPE_SEQ}_${_tc_ycol}_vs_${_tc_gtype_clean}_${_tc_model}}"
TC_PEAK_DIR="${TC_PEAK_DIR:-$_tc_fit_root/tc_peak}"
TC_FIT_PARAMS="${TC_FIT_PARAMS:-$TC_PEAK_DIR/tc_peak_table.parquet}"

pipeline_require_file "$TC_VS_TD_SCRIPT" "tc-vs-td script"
pipeline_require_file "$TC_FIT_PARAMS" "tc fit params"
mkdir -p "$TC_OUT_DIR"

"$PY" "$TC_VS_TD_SCRIPT" \
    --master-fit-params "$TC_FIT_PARAMS" \
    --method "${TC_METHOD:-pseudohuber_fixed_macro}" \
    --y-col "${TC_Y_COL:-tc_peak_ms}" \
    --out-dir "$TC_OUT_DIR/${TC_METHOD:-pseudohuber_fixed_macro}" \
    ${TC_EXTRA_ARGS:-}
