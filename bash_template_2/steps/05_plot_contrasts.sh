#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help plot_contrast "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

PLOT_CONTRAST_SCRIPT="${PLOT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_ogse_contrast_vs_g.py}"
PLOT_OUT_ROOT="${PLOT_OUT_ROOT:-$ANALYSIS_ROOT/plots-master/contrast}"

bt2_require_file "$PLOT_CONTRAST_SCRIPT" "plot contrast script"
bt2_require_file "$MASTER_PARQUET" "master table"
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
