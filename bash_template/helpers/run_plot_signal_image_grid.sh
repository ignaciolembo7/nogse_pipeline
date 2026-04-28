#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi

PY="${PY:-$DEFAULT_PY}"
PLOT_SCRIPT="${PLOT_SCRIPT:-$REPO_ROOT/scripts/plot_signal_image_grid.py}"
SIGNAL_ROOT="${SIGNAL_ROOT:-$PROJECT_ROOT/Data-signals}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/qc/signal_image_grids}"
NAME="${NAME:-}"
IMAGE_NAME="${IMAGE_NAME:-mean.nii.gz}"
GRADIENT_TYPE="${GRADIENT_TYPE:-g}"
GRADIENT_VALUES="${GRADIENT_VALUES:-}"
ROWS_CSV="${ROWS_CSV:-}"
INCLUDE_TOKENS_CSV="${INCLUDE_TOKENS_CSV:-}"
EXCLUDE_TOKENS_CSV="${EXCLUDE_TOKENS_CSV:-}"
TITLE="${TITLE:-}"
OUT_STEM="${OUT_STEM:-}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
SLICE_AXIS="${SLICE_AXIS:-2}"
SLICE_INDEX="${SLICE_INDEX:-}"
VOLUME_INDEX="${VOLUME_INDEX:-}"
NO_CROP="${NO_CROP:-0}"
INTENSITY_PERCENTILE="${INTENSITY_PERCENTILE:-99}"
DIFF_PERCENTILE="${DIFF_PERCENTILE:-99}"
DPI="${DPI:-220}"

if [[ -z "${EXPERIMENT:-}" ]]; then
    echo "ERROR: EXPERIMENT is required." >&2
    exit 1
fi
if [[ -z "$NAME" ]]; then
    echo "ERROR: NAME is required." >&2
    exit 1
fi
if [[ -z "$GRADIENT_VALUES" ]]; then
    echo "ERROR: GRADIENT_VALUES is required." >&2
    exit 1
fi
if [[ -z "$ROWS_CSV" ]]; then
    echo "ERROR: ROWS_CSV is required." >&2
    exit 1
fi
if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

read -r -a gradient_values <<< "${GRADIENT_VALUES//,/ }"

IFS=',' read -r -a rows <<< "$ROWS_CSV"
for idx in "${!rows[@]}"; do
    rows[$idx]="$(echo "${rows[$idx]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
done

extra_args=()
if [[ -n "$INCLUDE_TOKENS_CSV" ]]; then
    IFS=',' read -r -a include_tokens <<< "$INCLUDE_TOKENS_CSV"
    for token in "${include_tokens[@]}"; do
        token="$(echo "$token" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -n "$token" ]] && extra_args+=(--include-token "$token")
    done
fi
if [[ -n "$EXCLUDE_TOKENS_CSV" ]]; then
    IFS=',' read -r -a exclude_tokens <<< "$EXCLUDE_TOKENS_CSV"
    for token in "${exclude_tokens[@]}"; do
        token="$(echo "$token" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -n "$token" ]] && extra_args+=(--exclude-token "$token")
    done
fi
if [[ -n "$TITLE" ]]; then
    extra_args+=(--title "$TITLE")
fi
if [[ -n "$OUT_STEM" ]]; then
    extra_args+=(--out-stem "$OUT_STEM")
fi
if [[ "$ALLOW_MISSING" == "1" ]]; then
    extra_args+=(--allow-missing)
fi
if [[ -n "$SLICE_INDEX" ]]; then
    extra_args+=(--slice-index "$SLICE_INDEX")
fi
if [[ -n "$VOLUME_INDEX" ]]; then
    extra_args+=(--volume-index "$VOLUME_INDEX")
fi
if [[ "$NO_CROP" == "1" ]]; then
    extra_args+=(--no-crop)
fi

"$PY" "$PLOT_SCRIPT" \
    --signal-root "$SIGNAL_ROOT" \
    --experiment "$EXPERIMENT" \
    --name "$NAME" \
    --out-root "$OUT_ROOT" \
    --gradient-type "$GRADIENT_TYPE" \
    --gradient-values "${gradient_values[@]}" \
    --rows "${rows[@]}" \
    --image-name "$IMAGE_NAME" \
    --slice-axis "$SLICE_AXIS" \
    --intensity-percentile "$INTENSITY_PERCENTILE" \
    --diff-percentile "$DIFF_PERCENTILE" \
    --dpi "$DPI" \
    "${extra_args[@]}"
