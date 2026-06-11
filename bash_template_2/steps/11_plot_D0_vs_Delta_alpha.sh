#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help plot_d0_delta "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

PLOT_D0_SCRIPT="${PLOT_D0_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_D0_vs_Delta.py}"
ALPHA_OUT_DIR="${ALPHA_OUT_DIR:-$ANALYSIS_ROOT/alpha_macro/master}"
SUMMARY_ALPHA="${SUMMARY_ALPHA:-$ALPHA_OUT_DIR/summary_alpha_values.xlsx}"

bt2_require_file "$PLOT_D0_SCRIPT" "D0-vs-Delta plot script"
bt2_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$ALPHA_OUT_DIR"

args=(--master-parquet "$MASTER_PARQUET" --out-dir "$ALPHA_OUT_DIR")
if [[ -f "$SUMMARY_ALPHA" ]]; then
    args+=(--summary-alpha "$SUMMARY_ALPHA")
fi
[[ -n "${DPROJ_N:-}" ]] && args+=(--N "$DPROJ_N")
[[ -n "${DPROJ_HZ:-}" ]] && args+=(--Hz "$DPROJ_HZ")
[[ -n "${DPROJ_ROIS:-}" ]] && args+=(--rois ${DPROJ_ROIS})
[[ -n "${DPROJ_DIRS:-}" ]] && args+=(--dirs ${DPROJ_DIRS})

"$PY" "$PLOT_D0_SCRIPT" "${args[@]}" ${PLOT_D0_EXTRA_ARGS:-}
