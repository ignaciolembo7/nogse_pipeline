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

PLOT_SCRIPT="$REPO_ROOT/scripts/plot_publication_contrast_lcf_tcpeak.py"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/analysis/publication_figures/nogse_contrast_lcf_tcpeak}"

# Edit these values to change the exact publication panel.
FORMATS="${FORMATS:-png pdf}"
DPI="${DPI:-400}"
POINT_SIZE="${POINT_SIZE:-82}"
LINE_WIDTH="${LINE_WIDTH:-3.0}"

BRAINS_SUBJ="${BRAINS_SUBJ:-BRAIN}"
BRAINS_ROI="${BRAINS_ROI:-CentralCC}"
BRAINS_ROI_LABEL="${BRAINS_ROI_LABEL:-CentralCC}"
BRAINS_DIRECTIONS="${BRAINS_DIRECTIONS:-long tra}"
BRAINS_EXCLUDE_TD_MS="${BRAINS_EXCLUDE_TD_MS:-}"
BRAINS_D0="${BRAINS_D0:-3.2e-12}"

PHANTOMS_SUBJ="${PHANTOMS_SUBJ:-PHANTOM3}"
PHANTOMS_ROI="${PHANTOMS_ROI:-fiber1}"
PHANTOMS_ROI_LABEL="${PHANTOMS_ROI_LABEL:-less packed}"
PHANTOMS_DIRECTIONS="${PHANTOMS_DIRECTIONS:-1 3}"
PHANTOMS_EXCLUDE_TD_MS="${PHANTOMS_EXCLUDE_TD_MS:-209.1}"
PHANTOMS_D0="${PHANTOMS_D0:-2.3e-12}"

# Default phantoms aliases for this requested figure: 1=longitudinal, 3=transversal.
PHANTOMS_DIRECTION_ALIAS="${PHANTOMS_DIRECTION_ALIAS:-1=longitudinal 3=transversal}"

# Override these paths when the NOGSE-specific fits/tables are available.
BRAINS_FITS="${BRAINS_FITS:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_g_rest_corr/groupfits_rest.parquet}"
BRAINS_CONTRAST_ROOT="${BRAINS_CONTRAST_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments/contrast-data-rotated}"
BRAINS_TC_TABLE="${BRAINS_TC_TABLE:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_g_rest_corr/tcpeak_resampled_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_ms_k=None.xlsx}"

PHANTOMS_FITS="${PHANTOMS_FITS:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_g_rest_corr/groupfits_rest.parquet}"
PHANTOMS_CONTRAST_ROOT="${PHANTOMS_CONTRAST_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/contrast-data}"
PHANTOMS_TC_TABLE="${PHANTOMS_TC_TABLE:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_g_rest_corr/tcpeak_resampled_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_ms_k=None.xlsx}"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

for path in "$BRAINS_FITS" "$BRAINS_TC_TABLE" "$PHANTOMS_FITS" "$PHANTOMS_TC_TABLE"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required table not found: $path" >&2
        exit 1
    fi
done

for path in "$BRAINS_CONTRAST_ROOT" "$PHANTOMS_CONTRAST_ROOT"; do
    if [[ ! -d "$path" ]]; then
        echo "ERROR: Required contrast root not found: $path" >&2
        exit 1
    fi
done

read -r -a format_args <<< "${FORMATS//,/ }"
read -r -a brains_direction_args <<< "${BRAINS_DIRECTIONS//,/ }"
read -r -a phantoms_direction_args <<< "${PHANTOMS_DIRECTIONS//,/ }"
read -r -a brains_exclude_args <<< "${BRAINS_EXCLUDE_TD_MS//,/ }"
read -r -a phantoms_exclude_args <<< "${PHANTOMS_EXCLUDE_TD_MS//,/ }"
read -r -a phantoms_alias_args <<< "${PHANTOMS_DIRECTION_ALIAS//,/ }"
phantoms_alias_cli=()
for alias in "${phantoms_alias_args[@]}"; do
    if [[ -n "${alias// }" ]]; then
        phantoms_alias_cli+=(--phantoms-direction-alias "$alias")
    fi
done

mkdir -p "$OUT_DIR"

"$PY" "$PLOT_SCRIPT" \
    --project-root "$PROJECT_ROOT" \
    --out-dir "$OUT_DIR" \
    --formats "${format_args[@]}" \
    --dpi "$DPI" \
    --point-size "$POINT_SIZE" \
    --line-width "$LINE_WIDTH" \
    --brains-fits "$BRAINS_FITS" \
    --brains-contrast-root "$BRAINS_CONTRAST_ROOT" \
    --brains-tc-table "$BRAINS_TC_TABLE" \
    --brains-subj "$BRAINS_SUBJ" \
    --brains-roi "$BRAINS_ROI" \
    --brains-roi-label "$BRAINS_ROI_LABEL" \
    --brains-directions "${brains_direction_args[@]}" \
    --brains-peak-D0-fix "$BRAINS_D0" \
    --brains-exclude-td-ms "${brains_exclude_args[@]}" \
    --phantoms-fits "$PHANTOMS_FITS" \
    --phantoms-contrast-root "$PHANTOMS_CONTRAST_ROOT" \
    --phantoms-tc-table "$PHANTOMS_TC_TABLE" \
    --phantoms-subj "$PHANTOMS_SUBJ" \
    --phantoms-roi "$PHANTOMS_ROI" \
    --phantoms-roi-label "$PHANTOMS_ROI_LABEL" \
    --phantoms-directions "${phantoms_direction_args[@]}" \
    --phantoms-peak-D0-fix "$PHANTOMS_D0" \
    --phantoms-exclude-td-ms "${phantoms_exclude_args[@]}" \
    "${phantoms_alias_cli[@]}"

echo
echo "Finished publication figure in: $OUT_DIR"
