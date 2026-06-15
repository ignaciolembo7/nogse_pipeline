#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help alpha "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

ALPHA_MACRO_SCRIPT="${ALPHA_MACRO_SCRIPT:-$REPO_ROOT/scripts/summary/make_alpha_macro_summary.py}"
ALPHA_OUT_DIR="${ALPHA_OUT_DIR:-$ANALYSIS_ROOT/alpha_macro/master}"

pipeline_require_file "$ALPHA_MACRO_SCRIPT" "alpha macro script"
pipeline_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$ALPHA_OUT_DIR"

"$PY" "$ALPHA_MACRO_SCRIPT" \
    --master-parquet "$MASTER_PARQUET" \
    --master-fit-params "$MASTER_FIT_PARAMS" \
    --N "${ALPHA_N:-1}" \
    --out-summary "$ALPHA_OUT_DIR/summary_alpha_values.xlsx" \
    --out-avg "$ALPHA_OUT_DIR/D_vs_delta_app.combined.xlsx" \
    ${ALPHA_EXTRA_ARGS:-}
