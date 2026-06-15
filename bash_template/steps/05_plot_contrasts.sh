#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help plot_contrast "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

if [[ "$TYPE_SEQ" == "nogse" ]]; then
    PLOT_CONTRAST_SCRIPT="${PLOT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_nogse_contrast_vs_g.py}"
else
    PLOT_CONTRAST_SCRIPT="${PLOT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_ogse_contrast_vs_g.py}"
fi
PLOT_OUT_ROOT="${PLOT_OUT_ROOT:-$ANALYSIS_ROOT/plots-master/contrast}"

pipeline_require_file "$PLOT_CONTRAST_SCRIPT" "plot contrast script"
pipeline_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$PLOT_OUT_ROOT"

args=(--master-parquet "$MASTER_PARQUET" --out_root "$PLOT_OUT_ROOT")
[[ "${PLOT_SUBJ:-ALL}" != "ALL" ]] && args+=(--subj "$PLOT_SUBJ")
[[ "${PLOT_SHEET:-ALL}" != "ALL" ]] && args+=(--sheet "$PLOT_SHEET")
[[ "${PLOT_ROI:-ALL}" != "ALL" ]] && args+=(--roi "$PLOT_ROI")
[[ "${PLOT_DIRECTION:-ALL}" != "ALL" ]] && args+=(--direction "$PLOT_DIRECTION")
[[ -n "${PLOT_TD_MS:-}" ]] && args+=(--td_ms "$PLOT_TD_MS")
[[ -n "${PLOT_N1:-}" ]] && args+=(--N_1 "$PLOT_N1")
[[ -n "${PLOT_N2:-}" ]] && args+=(--N_2 "$PLOT_N2")

"$PY" "$PLOT_CONTRAST_SCRIPT" \
    "${args[@]}" \
    --ycol "${PLOT_CONTRAST_YCOL:-value_norm}" \
    --xcol "${PLOT_CONTRAST_XCOL:-g_thorsten_1}" \
    --stat "${PLOT_STAT:-avg}" \
    ${PLOT_CONTRAST_EXTRA_ARGS:-}
