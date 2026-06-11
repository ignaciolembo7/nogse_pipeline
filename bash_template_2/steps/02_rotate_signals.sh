#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help rotate "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

ROTATE_SCRIPT="${ROTATE_SCRIPT:-$REPO_ROOT/scripts/data/rotate_ogse_tensor.py}"
DIRS_TXT="${DIRS_TXT:-$REPO_ROOT/assets/dirs/dirs_6.txt}"
ROTATED_OUT_ROOT="${ROTATED_OUT_ROOT:-$ANALYSIS_ROOT/data-rotated/tables}"

bt2_require_file "$ROTATE_SCRIPT" "rotate script"
bt2_require_file "$MASTER_PARQUET" "master table"
bt2_require_file "$DIRS_TXT" "dirs txt"
mkdir -p "$ROTATED_OUT_ROOT"

"$PY" - "$MASTER_PARQUET" "${MASTER_SUBJ:-ALL}" "${MASTER_SHEET:-ALL}" <<'PY' | while IFS=$'\t' read -r subj sheet td_ms n hz; do
import pandas as pd
import sys

path, subj_sel, sheet_sel = sys.argv[1:4]
df = pd.read_parquet(path)
sig = df[df["row_kind"].astype(str).eq("signal")].copy()
if subj_sel != "ALL":
    sig = sig[sig["subj"].astype(str).eq(subj_sel)]
if sheet_sel != "ALL":
    sig = sig[sig["sheet"].astype(str).eq(sheet_sel)]
cols = [c for c in ["subj", "sheet", "td_ms", "N", "Hz"] if c in sig.columns]
for row in sig[cols].drop_duplicates().sort_values(cols).itertuples(index=False):
    print("\t".join(str(x) for x in row))
PY
    "$PY" "$ROTATE_SCRIPT" \
        --master-parquet "$MASTER_PARQUET" \
        --subj "$subj" \
        --sheet "$sheet" \
        --td_ms "$td_ms" \
        --N "$n" \
        --Hz "$hz" \
        --dirs_txt "$DIRS_TXT" \
        --out_dir "$ROTATED_OUT_ROOT" \
        ${ROTATE_EXTRA_ARGS:-}
done
