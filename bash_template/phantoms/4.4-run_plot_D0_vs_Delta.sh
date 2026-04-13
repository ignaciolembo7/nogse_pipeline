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
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
PLOT_SCRIPT="$REPO_ROOT/scripts/plot_D0_vs_Delta.py"

DPROJ_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data"
SUBJS="ALL"
ROIS="ALL"
DIRS="ALL"
N_VALUE="1"
BVALMAX="7"
OUT_DIR="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DPROJ_ROOT" ]]; then
    echo "ERROR: Dproj root not found: $DPROJ_ROOT" >&2
    exit 1
fi

if [[ -z "$(find "$DPROJ_ROOT" -type f -name '*.Dproj.long.parquet' -print -quit)" ]]; then
    echo "No monoexp Dproj tables were found in $DPROJ_ROOT. Skipping D0 vs Delta."
    exit 0
fi

mkdir -p "$OUT_DIR"

extra_args=()
if [[ "$SUBJS" != "ALL" ]]; then
    read -r -a subj_list <<< "${SUBJS//,/ }"
    if (( ${#subj_list[@]} > 0 )); then
        extra_args+=(--brains "${subj_list[@]}")
    fi
fi
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi
if [[ "$DIRS" != "ALL" ]]; then
    read -r -a dir_list <<< "${DIRS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        extra_args+=(--dirs "${dir_list[@]}")
    fi
fi

echo "============================================================"
echo "Dataset    : phantoms"
echo "Dproj root : $DPROJ_ROOT"
echo "Subjs      : $SUBJS"
echo "ROIs       : $ROIS"
echo "Dirs       : $DIRS"
echo "N          : $N_VALUE"
echo "Bstep alpha: $BVALMAX"
echo "Output dir : $OUT_DIR"

# Legacy CLI alias accepted by the plotting script.
"$PY" "$PLOT_SCRIPT" \
    --dproj-root "$DPROJ_ROOT" \
    "${extra_args[@]}" \
    --N "$N_VALUE" \
    --bvalmax "$BVALMAX" \
    --out-dir "$OUT_DIR" \
    --reference-D0 0.0023

echo
echo "Finished."
