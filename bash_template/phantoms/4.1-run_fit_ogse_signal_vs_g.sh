#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
FIT_SCRIPT="${1:-$REPO_ROOT/scripts/fit_ogse_signal_vs_g.py}"
DATA_ROOT="${2:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data}"
FITS_DIR="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits"
EXPERIMENT="ogse_signal_vs_g"
MODEL="monoexp"
APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-false}"
CORR_XLSX="${CORR_XLSX:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/grad_correction/water.grad_correction.xlsx}"
CORR_ROI="${CORR_ROI:-water}"
CORR_TD_MS="${CORR_TD_MS:-}"
CORR_SHEET="${CORR_SHEET:-}"
CORR_N1="${CORR_N1:-}"
CORR_N2="${CORR_N2:-}"
DEFAULT_OUT_ROOT="$FITS_DIR/${EXPERIMENT}_${MODEL}"
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    DEFAULT_OUT_ROOT="${DEFAULT_OUT_ROOT}_corr"
fi
OUT_ROOT="${3:-$DEFAULT_OUT_ROOT}"
DPROJ_ROOT="${4:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data}"
YCOL="value_norm"
G_TYPE="bvalue"
DIRECTIONS="ALL"
ROIS="ALL"
FIX_M0="1.0"
AUTO_FIT_MIN_POINTS="3"
AUTO_FIT_MAX_POINTS="11"
AUTO_FIT_ERR_FLOOR="0.05"
AUTO_FIT_TOL="0.15"

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
direction_args=()
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        direction_args+=(--directions "${dir_list[@]}")
    fi
fi

roi_args=()
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        roi_args+=(--rois "${roi_list[@]}")
    fi
fi

corr_args=(--no_grad_corr)
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    corr_args=(
        --apply_grad_corr
        --corr_xlsx "$CORR_XLSX"
        --corr_roi "$CORR_ROI"
    )
    if [[ -n "${CORR_TD_MS// }" ]]; then
        corr_args+=(--corr_td_ms "$CORR_TD_MS")
    fi
    if [[ -n "${CORR_SHEET// }" ]]; then
        corr_args+=(--corr_sheet "$CORR_SHEET")
    fi
fi

total=0
ok=0
failed=0
declare -a failed_jobs=()

while read -r file_path; do
    [[ -z "$file_path" ]] && continue
    total=$((total + 1))
    fname="$(basename "$file_path")"

    echo "============================================================"
    echo "Job $total"
    echo "  File: $fname"

    if "$PY" "$FIT_SCRIPT" \
        "$file_path" \
        --model "$MODEL" \
        --out_root "$OUT_ROOT" \
        --out_dproj_root "$DPROJ_ROOT" \
        "${direction_args[@]}" \
        --ycol "$YCOL" \
        --g_type "$G_TYPE" \
        --fix_M0 "$FIX_M0" \
        "${roi_args[@]}" \
        "${corr_args[@]}" \
        --auto_fit_points \
        --auto_fit_min_points "$AUTO_FIT_MIN_POINTS" \
        --auto_fit_max_points "$AUTO_FIT_MAX_POINTS" \
        --auto_fit_err_floor "$AUTO_FIT_ERR_FLOOR" \
        --auto_fit_tol "$AUTO_FIT_TOL"; then
        ok=$((ok + 1))
        echo "  OK"
    else
        status=$?
        failed=$((failed + 1))
        failed_jobs+=("exit $status :: $file_path")
        echo "  WARNING: command failed with exit code $status" >&2
        echo "  Continuing with next job..." >&2
    fi
done < <(
    find "$DATA_ROOT" -type f -name "*.long.parquet" | sort | while read -r candidate; do
        if "$PY" - "$candidate" "$G_TYPE" <<'PY'
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


path = sys.argv[1]
g_type = sys.argv[2]
col_map = {
    "bvalue": ["bvalue"],
    "g": ["bvalue_g"],
    "g_lin_max": ["bvalue_g_lin_max"],
    "g_thorsten": ["bvalue_thorsten"],
}

target_cols = col_map.get(g_type, [g_type])
df = pd.read_parquet(path)
usable = False
for col in target_cols:
    if col not in df.columns:
        continue
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).any() and np.nanmax(values) > 0:
        usable = True
        break

raise SystemExit(0 if usable else 1)
PY
        then
            printf '%s\n' "$candidate"
        fi
    done
)

echo
echo "Finished."
echo "  Total jobs  : $total"
echo "  Successful  : $ok"
echo "  Failed      : $failed"

if (( total == 0 )); then
    echo "  Notes       : no files expose a usable $G_TYPE axis, so monoexp fitting was skipped."
fi

if (( failed > 0 )); then
    echo
    echo "Failed jobs:"
    for item in "${failed_jobs[@]}"; do
        echo "  - $item"
    done
fi
