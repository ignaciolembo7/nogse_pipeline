#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

PY="${PY:-python}"

CONTRAST_ROOT="${1:-$REPO_ROOT/analysis/ogse_experiments/contrast-data-rotated/tables}"
MODEL="${2:-rest}"
FIT_SCRIPT="${3:-$REPO_ROOT/nogse_pipeline/scripts/fit_ogse-contrast_vs_g.py}"
GRAD_CORR_XLSX="${4:-$REPO_ROOT/analysis/ogse_experiments/fits/grad_correction_rotated/Syringe.grad_correction_rotated.xlsx}"

ALPHA_SUMMARY="${5:-$REPO_ROOT/analysis/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx}"
FIT_ROOT="${6:-$REPO_ROOT/analysis/ogse_experiments/fits/fit_${MODEL}_ogse_contrast_rotated_corr}"
FILE_PATTERN="${7:-*.long.parquet}"

if [[ ! -d "$CONTRAST_ROOT" ]]; then
    echo "ERROR: Contrast root not found: $CONTRAST_ROOT" >&2
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

mkdir -p "$FIT_ROOT"

total=0
ok=0
failed=0
declare -a failed_files=()

while read -r f; do
    [[ -z "$f" ]] && continue

    total=$((total + 1))
    base_name="$(basename "$f")"

    echo "============================================================"
    echo "Job $total"
    echo "  File: $base_name"

    if "$PY" "$FIT_SCRIPT" "$f" \
        --model "$MODEL" \
        --gbase g_thorsten_1 \
        --ycol value_norm \
        --directions long tra \
        --rois AntCC MidAntCC CentralCC MidPostCC PostCC Left-Lateral-Ventricle Right-Lateral-Ventricle \
        --peak_D0_fix 3.2e-12 \
        --fix_M0 1.0 \
        --fix_D0 3.2e-12 \
        --apply_grad_corr \
        --corr_xlsx "$GRAD_CORR_XLSX" \
        --corr_roi Syringe \
        --out_root "$FIT_ROOT"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_files+=("$f")
        echo "  WARNING: failed file: $base_name (exit code: $status)" >&2
        echo "  Continuing with next file..." >&2
    fi
done < <(find "$CONTRAST_ROOT" -type f -name "$FILE_PATTERN" | sort)

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
