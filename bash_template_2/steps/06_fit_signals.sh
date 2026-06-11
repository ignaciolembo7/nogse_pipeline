#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help fit_signal "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

FIT_SIGNAL_SCRIPT="${FIT_SIGNAL_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_ogse_signal_vs_g.py}"
SIGNAL_FIT_MANIFEST="${SIGNAL_FIT_MANIFEST:-$MANIFEST_DIR/signal_fits.csv}"
SIGNAL_FIT_OUT_ROOT="${SIGNAL_FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/ogse_signal_master}"

bt2_require_file "$FIT_SIGNAL_SCRIPT" "fit signal script"
bt2_require_file "$MASTER_PARQUET" "master table"
bt2_require_file "$SIGNAL_FIT_MANIFEST" "signal fit manifest"
mkdir -p "$SIGNAL_FIT_OUT_ROOT"

count=0
while IFS=, read -r subj sheet roi direction td_ms n hz model; do
    [[ -z "${subj// }" || "${subj:0:1}" == "#" || "$subj" == "subj" ]] && continue
    count=$((count + 1))
    args=(--master-parquet "$MASTER_PARQUET" --master-fit-params "$MASTER_FIT_PARAMS" --row-kind signal_rotated)
    [[ "$subj" != "ALL" ]] && args+=(--subj "$subj")
    [[ "$sheet" != "ALL" ]] && args+=(--sheet "$sheet")
    [[ "$roi" != "ALL" ]] && args+=(--roi "$roi")
    [[ "$direction" != "ALL" ]] && args+=(--direction "$direction")
    [[ -n "${td_ms// }" ]] && args+=(--td_ms "$td_ms")
    [[ -n "${n// }" ]] && args+=(--N "$n")
    [[ -n "${hz// }" ]] && args+=(--Hz "$hz")
    "$PY" "$FIT_SIGNAL_SCRIPT" \
        "${args[@]}" \
        --model "${model:-${SIGNAL_FIT_MODEL:-monoexp}}" \
        --out_root "$SIGNAL_FIT_OUT_ROOT" \
        --ycol "${SIGNAL_FIT_YCOL:-value_norm}" \
        --g_type "${SIGNAL_FIT_G_TYPE:-bvalue_thorsten}" \
        --auto_fit_points \
        ${SIGNAL_FIT_EXTRA_ARGS:-}
done < "$SIGNAL_FIT_MANIFEST"

echo "Done signal fits. Manifest rows processed: $count"
