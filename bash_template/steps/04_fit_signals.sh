#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help fit_signal "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

if [[ "$TYPE_SEQ" == "nogse" ]]; then
    FIT_SIGNAL_SCRIPT="${FIT_SIGNAL_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_nogse_signal_vs_g.py}"
    DEFAULT_SIGNAL_FIT_MODEL="${DEFAULT_SIGNAL_FIT_MODEL:-nogse_free}"
    DEFAULT_SIGNAL_FIT_G_TYPE="${DEFAULT_SIGNAL_FIT_G_TYPE:-g}"
else
    FIT_SIGNAL_SCRIPT="${FIT_SIGNAL_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_ogse_signal_vs_g.py}"
    DEFAULT_SIGNAL_FIT_MODEL="${DEFAULT_SIGNAL_FIT_MODEL:-ogse_rest_offset}"
    DEFAULT_SIGNAL_FIT_G_TYPE="${DEFAULT_SIGNAL_FIT_G_TYPE:-bvalue_thorsten}"
fi
SIGNAL_FIT_MANIFEST="${SIGNAL_FIT_MANIFEST:-$MANIFEST_DIR/signal_fits.csv}"

# Derive output folder: fits/{master_name}/{type_seq}_{ycol}_vs_{gtype}_{model}
_master_name=$(basename "${MASTER_PARQUET:-master.long.parquet}" | sed 's/\.long\.parquet$//' | sed 's/\.parquet$//')
_ycol="${SIGNAL_FIT_YCOL:-value_norm}"
_gtype="${SIGNAL_FIT_G_TYPE:-$DEFAULT_SIGNAL_FIT_G_TYPE}"
_gtype_clean="${_gtype//_/}"
_model="${SIGNAL_FIT_MODEL:-$DEFAULT_SIGNAL_FIT_MODEL}"
SIGNAL_FIT_OUT_ROOT="${SIGNAL_FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/$_master_name/${TYPE_SEQ}_${_ycol}_vs_${_gtype_clean}_${_model}}"

pipeline_require_file "$FIT_SIGNAL_SCRIPT" "fit signal script"
pipeline_require_file "$MASTER_PARQUET" "master table"
pipeline_require_file "$SIGNAL_FIT_MANIFEST" "signal fit manifest"
mkdir -p "$SIGNAL_FIT_OUT_ROOT"

count=0
while IFS=, read -r subj sheet roi direction td_ms n hz model; do
    [[ -z "${subj// }" || "${subj:0:1}" == "#" || "$subj" == "subj" ]] && continue
    count=$((count + 1))
    args=(--master-parquet "$MASTER_PARQUET" --row-kind signal_rotated)
    [[ "$subj" != "ALL" ]] && args+=(--subj "$subj")
    [[ "$sheet" != "ALL" ]] && args+=(--sheet "$sheet")
    [[ "$roi" != "ALL" ]] && args+=(--roi "$roi")
    [[ "$direction" != "ALL" ]] && args+=(--direction "$direction")
    [[ -n "${td_ms// }" ]] && args+=(--td_ms "$td_ms")
    [[ -n "${n// }" ]] && args+=(--N "$n")
    [[ -n "${hz// }" ]] && args+=(--Hz "$hz")
    if [[ "$TYPE_SEQ" == "nogse" ]]; then
        "$PY" "$FIT_SIGNAL_SCRIPT" \
            "${args[@]}" \
            --model "${model:-${SIGNAL_FIT_MODEL:-$DEFAULT_SIGNAL_FIT_MODEL}}" \
            --out_root "$SIGNAL_FIT_OUT_ROOT" \
            --ycol "${SIGNAL_FIT_YCOL:-value_norm}" \
            --xcol "${SIGNAL_FIT_XCOL:-${SIGNAL_FIT_G_TYPE:-$DEFAULT_SIGNAL_FIT_G_TYPE}}" \
            ${SIGNAL_FIT_EXTRA_ARGS:-}
    else
        "$PY" "$FIT_SIGNAL_SCRIPT" \
            "${args[@]}" \
            --model "${model:-${SIGNAL_FIT_MODEL:-$DEFAULT_SIGNAL_FIT_MODEL}}" \
            --out_root "$SIGNAL_FIT_OUT_ROOT" \
            --ycol "${SIGNAL_FIT_YCOL:-value_norm}" \
            --g_type "${SIGNAL_FIT_G_TYPE:-$DEFAULT_SIGNAL_FIT_G_TYPE}" \
            --auto_fit_points \
            ${SIGNAL_FIT_EXTRA_ARGS:-}
    fi
done < "$SIGNAL_FIT_MANIFEST"

echo "Done signal fits. Manifest rows processed: $count"
