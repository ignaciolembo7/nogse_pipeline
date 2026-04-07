#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"

# Configuracion
ANALYSIS_ROOT="$PROJECT_ROOT/analysis/brains/ogse_experiments"
TABLES_ROOT="$ANALYSIS_ROOT/contrast-data-rotated/tables"
OUT_ROOT="$ANALYSIS_ROOT/fits/fit_rest_ogse_contrast_rotated_corr"
FIT_SCRIPT="$REPO_ROOT/scripts/fit_ogse-contrast_vs_g.py"
GRAD_CORR_XLSX="$ANALYSIS_ROOT/fits/grad_correction_rotated/Syringe.grad_correction_rotated.xlsx"
FILE_PATTERN="*.long.parquet"

MODEL="rest"
GBASE="g_thorsten_1"
YCOL="value_norm"
CORR_ROI="Syringe"
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Left-Lateral-Ventricle,Right-Lateral-Ventricle"

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "ERROR: Tables root not found: $TABLES_ROOT" >&2
    exit 1
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: Fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$GRAD_CORR_XLSX" ]]; then
    echo "ERROR: Gradient correction file not found: $GRAD_CORR_XLSX" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

roi_args=()
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        roi_args+=(--rois "${roi_list[@]}")
    fi
fi

total=0
ok=0
failed=0
declare -a failed_files=()

while read -r file; do
    [[ -z "$file" ]] && continue

    total=$((total + 1))
    base_name="$(basename "$file")"

    echo "============================================================"
    echo "Job $total"
    echo "  File: $base_name"
    echo "  ROIs  : $ROIS"

    if "$PY" "$FIT_SCRIPT" \
        "$file" \
        --model "$MODEL" \
        --gbase "$GBASE" \
        --ycol "$YCOL" \
        --directions long tra \
        --peak_D0_fix 3.2e-12 \
        --fix_M0 1.0 \
        --fix_D0 3.2e-12 \
        --apply_grad_corr \
        --corr_xlsx "$GRAD_CORR_XLSX" \
        --corr_roi "$CORR_ROI" \
        "${roi_args[@]}" \
        --out_root "$OUT_ROOT"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_files+=("$file")
        echo "  WARNING: failed file: $base_name (exit code: $status)" >&2
        echo "  Continuing with next file..." >&2
    fi

done < <(find "$TABLES_ROOT" -type f -name "$FILE_PATTERN" | sort)

echo
echo "Finished."
echo "  Total files   : $total"
echo "  Successful    : $ok"
echo "  Failed        : $failed"

if (( failed > 0 )); then
    echo
    echo "Failed files:"
    for f in "${failed_files[@]}"; do
        echo "  - $f"
    done
fi
