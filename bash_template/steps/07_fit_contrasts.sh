#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
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
FIT_OUT_ROOT="${FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/${TYPE_SEQ}_contrast_master}"

pipeline_require_file "$FIT_CONTRAST_SCRIPT" "fit contrast script"
pipeline_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$FIT_OUT_ROOT"

matches="$("$PY" - "$MASTER_PARQUET" "${FIT_STAT:-avg}" <<'PY'
import pandas as pd
import sys

df = pd.read_parquet(sys.argv[1])
stat = sys.argv[2]
sel = df[df["row_kind"].astype(str).eq("contrast")]
if stat != "ALL" and "stat" in sel:
    sel = sel[sel["stat"].astype(str).eq(stat)]
print(len(sel))
PY
)"
if [[ "$matches" == "0" ]]; then
    echo "No contrast rows found in master. Run the contrast step first or provide CONTRAST_MANIFEST."
    exit 0
fi

"$PY" "$FIT_CONTRAST_SCRIPT" \
    --master-parquet "$MASTER_PARQUET" \
    --master-fit-params "$MASTER_FIT_PARAMS" \
    --model "${FIT_MODEL:-$DEFAULT_FIT_MODEL}" \
    --out_root "$FIT_OUT_ROOT" \
    --gbase "${FIT_GBASE:-g_lin_max}" \
    --ycol "${FIT_YCOL:-value_norm}" \
    --stat "${FIT_STAT:-avg}" \
    ${FIT_EXTRA_ARGS:-}
