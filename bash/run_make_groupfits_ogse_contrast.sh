#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

PY="${PY:-python}"

# Configuracion
FITS_ROOT="$REPO_ROOT/analysis/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr"
MODEL="rest"
PIPE_SCRIPT="$REPO_ROOT/nogse_pipeline/scripts/run_tc_pipeline.py"
OUT_PREFIX="$FITS_ROOT/groupfits_${MODEL}"
# ROIS="ALL"
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC"

if [[ ! -d "$FITS_ROOT" ]]; then
    echo "ERROR: Fits root not found: $FITS_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PIPE_SCRIPT" ]]; then
    echo "ERROR: Pipeline script not found: $PIPE_SCRIPT" >&2
    exit 1
fi

OUT_XLSX="${OUT_PREFIX}.xlsx"
OUT_PARQUET="${OUT_PREFIX}.parquet"
mkdir -p "$(dirname "$OUT_XLSX")"

roi_args=()
if [[ "$ROIS" != "ALL" ]]; then
    IFS=',' read -r -a roi_list <<< "$ROIS"
    if (( ${#roi_list[@]} > 0 )); then
        roi_args+=(--rois "${roi_list[@]}")
    fi
fi

echo "============================================================"
echo "Building groupfits"
echo "  Fits root : $FITS_ROOT"
echo "  Model     : $MODEL"
echo "  ROIs      : $ROIS"
echo "  Out xlsx  : $OUT_XLSX"
echo "  Out parquet: $OUT_PARQUET"

"$PY" "$PIPE_SCRIPT" \
    "$FITS_ROOT" \
    --models "$MODEL" \
    "${roi_args[@]}" \
    --out-xlsx "$OUT_XLSX" \
    --out-parquet "$OUT_PARQUET"

echo
echo "Finished."
echo "  groupfits xlsx   : $OUT_XLSX"
echo "  groupfits parquet: $OUT_PARQUET"
