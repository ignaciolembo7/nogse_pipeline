#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/manifests/helpers/master_table_common.sh"
pipeline_maybe_step_help contrast "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

MAKE_CONTRAST_SCRIPT="${MAKE_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/data/make_contrast.py}"
CONTRAST_MANIFEST="${CONTRAST_MANIFEST:-$MANIFEST_DIR/contrasts.csv}"
CONTRAST_OUT_ROOT="${CONTRAST_OUT_ROOT:-$ANALYSIS_ROOT/contrast-data-master}"

pipeline_require_file "$MAKE_CONTRAST_SCRIPT" "make_contrast script"
pipeline_require_file "$MASTER_PARQUET" "master table"
pipeline_require_file "$CONTRAST_MANIFEST" "contrast manifest"
mkdir -p "$CONTRAST_OUT_ROOT"

count=0
while IFS=, read -r subj sheet roi direction td_ms n1 n2 hz1 hz2; do
    [[ -z "${subj// }" || "${subj:0:1}" == "#" || "$subj" == "subj" ]] && continue
    count=$((count + 1))
    args=(--master-parquet "$MASTER_PARQUET" --append-master --out_root "$CONTRAST_OUT_ROOT")
    [[ "$subj" != "ALL" ]] && args+=(--subj "$subj")
    [[ "$sheet" != "ALL" ]] && args+=(--sheet "$sheet")
    [[ "$roi" != "ALL" ]] && args+=(--roi "$roi")
    [[ "$direction" != "ALL" ]] && args+=(--direction "$direction")
    [[ -n "${td_ms// }" ]] && args+=(--td_ms "$td_ms")
    [[ -n "${n1// }" ]] && args+=(--N_1 "$n1")
    [[ -n "${n2// }" ]] && args+=(--N_2 "$n2")
    [[ -n "${hz1// }" ]] && args+=(--Hz_1 "$hz1")
    [[ -n "${hz2// }" ]] && args+=(--Hz_2 "$hz2")
    "$PY" "$MAKE_CONTRAST_SCRIPT" "${args[@]}" ${MAKE_CONTRAST_EXTRA_ARGS:-}
done < "$CONTRAST_MANIFEST"

echo "Done contrasts. Manifest rows processed: $count"
