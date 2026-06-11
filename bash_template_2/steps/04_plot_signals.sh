#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help plot_signal "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

PLOT_SIGNAL_SCRIPT="${PLOT_SIGNAL_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_ogse_signal_vs_g.py}"
PLOT_OUT_ROOT="${PLOT_OUT_ROOT:-$ANALYSIS_ROOT/plots-master/signal}"

bt2_require_file "$PLOT_SIGNAL_SCRIPT" "plot signal script"
bt2_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$PLOT_OUT_ROOT"

args=(--master-parquet "$MASTER_PARQUET" --out_root "$PLOT_OUT_ROOT" --row-kind "${PLOT_ROW_KIND:-signal_rotated}")
[[ "${PLOT_SUBJ:-ALL}" != "ALL" ]] && args+=(--subj "$PLOT_SUBJ")
[[ "${PLOT_SHEET:-ALL}" != "ALL" ]] && args+=(--sheet "$PLOT_SHEET")
[[ "${PLOT_ROI:-ALL}" != "ALL" ]] && args+=(--roi "$PLOT_ROI")
[[ "${PLOT_DIRECTION:-ALL}" != "ALL" ]] && args+=(--direction "$PLOT_DIRECTION")
[[ -n "${PLOT_TD_MS:-}" ]] && args+=(--td_ms "$PLOT_TD_MS")
[[ -n "${PLOT_N:-}" ]] && args+=(--N "$PLOT_N")

"$PY" "$PLOT_SIGNAL_SCRIPT" \
    "${args[@]}" \
    --ycol "${PLOT_SIGNAL_YCOL:-value_norm}" \
    --xcol "${PLOT_SIGNAL_XCOL:-g_thorsten}" \
    --stat "${PLOT_STAT:-avg}" \
    ${PLOT_SIGNAL_EXTRA_ARGS:-}
