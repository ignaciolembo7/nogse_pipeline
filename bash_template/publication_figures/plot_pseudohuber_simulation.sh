#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY="python"
    elif [[ -x "$HOME/.conda/envs/nogse_pipe_env/bin/python" ]]; then
        PY="$HOME/.conda/envs/nogse_pipe_env/bin/python"
    else
        PY="python3"
    fi
fi

PLOT_SCRIPT="$REPO_ROOT/scripts/plot_publication_pseudohuber_simulation.py"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/analysis/publication_figures/pseudohuber_simulation}"

# Edit these values to change the simulated model curves.
C_MS="${C_MS:-20.0}"
ALPHA_MACRO="${ALPHA_MACRO:-0.55}"
DELTAS_MS="${DELTAS_MS:-250}"
TD_MIN_MS="${TD_MIN_MS:-0}"
TD_MAX_MS="${TD_MAX_MS:-300}"
N_POINTS="${N_POINTS:-600}"
LINEAR_START_FACTOR="${LINEAR_START_FACTOR:-1.45}"
FIGSIZE="${FIGSIZE:-6,4}"
FORMATS="${FORMATS:-png pdf}"
DPI="${DPI:-400}"
TITLE="${TITLE:-}"
COMMON_SLOPE_GUIDE="${COMMON_SLOPE_GUIDE:-false}"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

read -r -a delta_args <<< "${DELTAS_MS//,/ }"
read -r -a format_args <<< "${FORMATS//,/ }"

title_args=()
if [[ -n "${TITLE// }" ]]; then
    title_args=(--title "$TITLE")
fi
guide_args=()
if [[ "${COMMON_SLOPE_GUIDE,,}" == "true" || "${COMMON_SLOPE_GUIDE}" == "1" ]]; then
    guide_args=(--common-slope-guide)
fi

mkdir -p "$OUT_DIR"

"$PY" "$PLOT_SCRIPT" \
    --project-root "$PROJECT_ROOT" \
    --out-dir "$OUT_DIR" \
    --c "$C_MS" \
    --alpha-macro "$ALPHA_MACRO" \
    --deltas-ms "${delta_args[@]}" \
    --td-min-ms "$TD_MIN_MS" \
    --td-max-ms "$TD_MAX_MS" \
    --n-points "$N_POINTS" \
    --linear-start-factor "$LINEAR_START_FACTOR" \
    --figsize "$FIGSIZE" \
    --formats "${format_args[@]}" \
    --dpi "$DPI" \
    "${guide_args[@]}" \
    "${title_args[@]}"

echo
echo "Finished publication figure in: $OUT_DIR"
