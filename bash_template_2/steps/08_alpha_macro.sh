#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help alpha "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

ALPHA_MACRO_SCRIPT="${ALPHA_MACRO_SCRIPT:-$REPO_ROOT/scripts/summary/make_alpha_macro_summary.py}"
ALPHA_OUT_DIR="${ALPHA_OUT_DIR:-$ANALYSIS_ROOT/alpha_macro/master}"

bt2_require_file "$ALPHA_MACRO_SCRIPT" "alpha macro script"
bt2_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$ALPHA_OUT_DIR"

"$PY" "$ALPHA_MACRO_SCRIPT" \
    --master-parquet "$MASTER_PARQUET" \
    --master-fit-params "$MASTER_FIT_PARAMS" \
    --N "${ALPHA_N:-1}" \
    --out-summary "$ALPHA_OUT_DIR/summary_alpha_values.xlsx" \
    --out-avg "$ALPHA_OUT_DIR/D_vs_delta_app.combined.xlsx" \
    ${ALPHA_EXTRA_ARGS:-}
