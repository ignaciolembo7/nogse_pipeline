#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
FITS_ROOT="${1:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_monoexp_ogse-signal}"
OUT_ROOT="${2:-$FITS_ROOT/summary_plots}"
PLOT_SCRIPT="${3:-$REPO_ROOT/scripts/plot_monoexp_D_vs_time.py}"
ROIS="ALL"
DIRECTIONS="ALL"
<<<<<<< HEAD
=======
NS="1,4,8"
>>>>>>> origin/main

if [[ ! -d "$FITS_ROOT" ]]; then
    echo "ERROR: Fits root not found: $FITS_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

if [[ -z "$(find "$FITS_ROOT" -type f -name 'fit_params.parquet' -print -quit)" ]]; then
    echo "No monoexp fit tables were found in $FITS_ROOT. Skipping summary plots."
    exit 0
fi

mkdir -p "$OUT_ROOT"

extra_args=()
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        extra_args+=(--dirs "${dir_list[@]}")
    fi
fi
<<<<<<< HEAD
=======
if [[ "$NS" != "ALL" ]]; then
    read -r -a n_list <<< "${NS//,/ }"
    if (( ${#n_list[@]} > 0 )); then
        extra_args+=(--Ns "${n_list[@]}")
    fi
fi
>>>>>>> origin/main

"$PY" "$PLOT_SCRIPT" \
    --fits-root "$FITS_ROOT" \
    --out-dir "$OUT_ROOT" \
    "${extra_args[@]}"
