#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help fit_contrast "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

FIT_CONTRAST_SCRIPT="${FIT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_ogse_contrast_vs_g.py}"
FIT_OUT_ROOT="${FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_master}"

bt2_require_file "$FIT_CONTRAST_SCRIPT" "fit contrast script"
bt2_require_file "$MASTER_PARQUET" "master table"
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
    --model "${FIT_MODEL:-ogse_free}" \
    --out_root "$FIT_OUT_ROOT" \
    --gbase "${FIT_GBASE:-g_lin_max}" \
    --ycol "${FIT_YCOL:-value_norm}" \
    --stat "${FIT_STAT:-avg}" \
    ${FIT_EXTRA_ARGS:-}
