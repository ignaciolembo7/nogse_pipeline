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
PLOT_SCRIPT="$REPO_ROOT/scripts/plot_publication_delta_alpha.py"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/analysis/publication_figures/delta_alpha_ogse}"

# Edit these lists to change what appears in the final figure.
BRAINS_ROIS="${BRAINS_ROIS:-AntCC MidAntCC CentralCC MidPostCC PostCC}"
PHANTOMS_ROIS="${PHANTOMS_ROIS:-fiber1 fiber2}"
DIRECTIONS="${DIRECTIONS:-longitudinal transversal}"
FORMATS="${FORMATS:-png pdf}"
PLOT_MODES="${PLOT_MODES:-points bars}"
DPI="${DPI:-400}"
DELTA_YMIN="${DELTA_YMIN:-}"
PHANTOMS_DELTA_YMIN="${PHANTOMS_DELTA_YMIN:-0}"
ALPHA_YMIN="${ALPHA_YMIN:-0}"
BRAINS_ALPHA_YMAX="${BRAINS_ALPHA_YMAX:-0.4}"
PHANTOMS_DIRECTION_ALIAS="${PHANTOMS_DIRECTION_ALIAS:-1=transversal 2=transversal 3=longitudinal}"

# Override these paths when comparing a different fit family or an alternate alpha table.
BRAINS_ALPHA_TABLE="${BRAINS_ALPHA_TABLE:-$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1/summary_alpha_values.csv}"
BRAINS_DELTA_TABLE="${BRAINS_DELTA_TABLE:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx}"
PHANTOMS_ALPHA_TABLE="${PHANTOMS_ALPHA_TABLE:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/summary_alpha_values.csv}"
PHANTOMS_DELTA_TABLE="${PHANTOMS_DELTA_TABLE:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx}"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

for table in "$BRAINS_ALPHA_TABLE" "$BRAINS_DELTA_TABLE" "$PHANTOMS_ALPHA_TABLE" "$PHANTOMS_DELTA_TABLE"; do
    if [[ ! -f "$table" ]]; then
        echo "ERROR: Required table not found: $table" >&2
        exit 1
    fi
done

read -r -a brain_roi_args <<< "${BRAINS_ROIS//,/ }"
read -r -a phantom_roi_args <<< "${PHANTOMS_ROIS//,/ }"
read -r -a direction_args <<< "${DIRECTIONS//,/ }"
read -r -a format_args <<< "${FORMATS//,/ }"
read -r -a plot_mode_args <<< "${PLOT_MODES//,/ }"
read -r -a phantoms_alias_args <<< "${PHANTOMS_DIRECTION_ALIAS//,/ }"
phantoms_alias_cli=()
for alias in "${phantoms_alias_args[@]}"; do
    if [[ -n "${alias// }" ]]; then
        phantoms_alias_cli+=(--phantoms-direction-alias "$alias")
    fi
done

mkdir -p "$OUT_DIR"

limit_args=(--alpha-ymin "$ALPHA_YMIN")
if [[ -n "${DELTA_YMIN// }" ]]; then
    limit_args+=(--delta-ymin "$DELTA_YMIN")
fi
if [[ -n "${PHANTOMS_DELTA_YMIN// }" ]]; then
    limit_args+=(--phantoms-delta-ymin "$PHANTOMS_DELTA_YMIN")
fi
if [[ -n "${BRAINS_ALPHA_YMAX// }" ]]; then
    limit_args+=(--brains-alpha-ymax "$BRAINS_ALPHA_YMAX")
fi

"$PY" "$PLOT_SCRIPT" \
    --project-root "$PROJECT_ROOT" \
    --out-dir "$OUT_DIR" \
    --brains-alpha-table "$BRAINS_ALPHA_TABLE" \
    --brains-delta-table "$BRAINS_DELTA_TABLE" \
    --phantoms-alpha-table "$PHANTOMS_ALPHA_TABLE" \
    --phantoms-delta-table "$PHANTOMS_DELTA_TABLE" \
    --brains-rois "${brain_roi_args[@]}" \
    --phantoms-rois "${phantom_roi_args[@]}" \
    --directions "${direction_args[@]}" \
    "${phantoms_alias_cli[@]}" \
    --formats "${format_args[@]}" \
    --plot-modes "${plot_mode_args[@]}" \
    --dpi "$DPI" \
    "${limit_args[@]}"

echo
echo "Finished publication figures in: $OUT_DIR"
