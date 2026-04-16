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
SUMMARY_SCRIPT="$REPO_ROOT/scripts/make_alpha_macro_summary.py"

DPROJ_ROOT="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/data"
SUBJS="ALL"
ROIS="ALL"
DIRS="1 2 3"
N_VALUE="1"
PLOT_ROIS="ALL"
PLOT_DIRECTIONS="ALL"
BVALMAX="7"
OUT_SUMMARY="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"

if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "ERROR: Script not found: $SUMMARY_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DPROJ_ROOT" ]]; then
    echo "ERROR: Dproj root not found: $DPROJ_ROOT" >&2
    exit 1
fi

if [[ -z "$(find "$DPROJ_ROOT" -type f -name '*.Dproj.long.parquet' -print -quit)" ]]; then
    echo "No monoexp Dproj tables were found in $DPROJ_ROOT. Skipping alpha macro summary."
    exit 0
fi

mkdir -p "$(dirname "$OUT_SUMMARY")"

extra_args=()
if [[ "$SUBJS" != "ALL" ]]; then
    read -r -a subj_list <<< "${SUBJS//,/ }"
    if (( ${#subj_list[@]} > 0 )); then
        extra_args+=(--subj "${subj_list[@]}")
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
if [[ "$PLOT_ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${PLOT_ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--plot-rois "${roi_list[@]}")
    fi
fi
if [[ "$PLOT_DIRECTIONS" != "ALL" ]]; then
    read -r -a dir_list <<< "${PLOT_DIRECTIONS//,/ }"
    if (( ${#dir_list[@]} > 0 )); then
        extra_args+=(--plot-directions "${dir_list[@]}")
    fi
fi

echo "============================================================"
echo "Dataset       : phantoms"
echo "Dproj root    : $DPROJ_ROOT"
echo "Subjs         : $SUBJS"
echo "ROIs          : $ROIS"
echo "Dirs          : $DIRS"
echo "N             : $N_VALUE"
echo "Plot ROIs     : $PLOT_ROIS"
echo "Plot dirs     : $PLOT_DIRECTIONS"
echo "Bstep alpha   : $BVALMAX"
echo "Out summary   : $OUT_SUMMARY"

"$PY" "$SUMMARY_SCRIPT" \
    --dproj-root "$DPROJ_ROOT" \
    "${extra_args[@]}" \
    --N "$N_VALUE" \
    --bvalmax "$BVALMAX" \
    --out-summary "$OUT_SUMMARY" \
    --reference-D0 0.0023 

echo
echo "Finished."
