#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/manifests/helpers/master_table_common.sh"
pipeline_maybe_step_help fit_contrast "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

if [[ "$TYPE_SEQ" == "nogse" ]]; then
    FIT_CONTRAST_SCRIPT="${FIT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_nogse_contrast_vs_g.py}"
    DEFAULT_FIT_MODEL="${DEFAULT_FIT_MODEL:-nogse_free}"
else
    FIT_CONTRAST_SCRIPT="${FIT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_ogse_contrast_vs_g.py}"
    DEFAULT_FIT_MODEL="${DEFAULT_FIT_MODEL:-ogse_free}"
fi
# Standalone contrast table produced by step 07 or 07b.
CONTRAST_PARQUET="${CONTRAST_PARQUET:-$ANALYSIS_ROOT/contrast-data-master/master_contrast.parquet}"

# Derive output folder: fits/{master_name}/{type_seq}_{ycol}_vs_{gtype}_{model}
_master_name=$(basename "${MASTER_PARQUET:-master.long.parquet}" | sed 's/\.long\.parquet$//' | sed 's/\.parquet$//')
_ycol="${FIT_YCOL:-value_norm}"
_gtype="${FIT_GBASE:-g_lin_max}"
_gtype_clean="${_gtype//_/}"
_model="${FIT_MODEL:-$DEFAULT_FIT_MODEL}"
FIT_OUT_ROOT="${FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/$_master_name/${TYPE_SEQ}_${_ycol}_vs_${_gtype_clean}_${_model}}"

pipeline_require_file "$FIT_CONTRAST_SCRIPT" "fit contrast script"
pipeline_require_file "$CONTRAST_PARQUET" "contrast table (run step 07 or 07b first)"
mkdir -p "$FIT_OUT_ROOT"

"$PY" "$FIT_CONTRAST_SCRIPT" \
    --master-parquet "$CONTRAST_PARQUET" \
    --model "${FIT_MODEL:-$DEFAULT_FIT_MODEL}" \
    --out_root "$FIT_OUT_ROOT" \
    --gbase "${FIT_GBASE:-g_lin_max}" \
    --ycol "${FIT_YCOL:-value_norm}" \
    --stat "${FIT_STAT:-avg}" \
    ${FIT_EXTRA_ARGS:-}
