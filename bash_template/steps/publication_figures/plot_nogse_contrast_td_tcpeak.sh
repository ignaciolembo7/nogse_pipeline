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

PLOT_SCRIPT="$REPO_ROOT/scripts/plotting/plot_publication_contrast_td_tcpeak.py"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/analysis/publication_figures/nogse_contrast_td_tcpeak}"

# Edit these values to change the exact publication panels.
FORMATS="${FORMATS:-png pdf}"
DPI="${DPI:-400}"
POINT_SIZE="${POINT_SIZE:-150}"
LINE_WIDTH="${LINE_WIDTH:-2.0}"
LCF_MIN="${LCF_MIN:-3}"
LCF_MAX="${LCF_MAX:-12}"

BRAINS_SUBJ="${BRAINS_SUBJ:-LUDG}"
BRAINS_ROI="${BRAINS_ROI:-PostCC}" #CentralCC
BRAINS_ROI_LABEL="${BRAINS_ROI_LABEL:-PostCC}" #CentralCC
BRAINS_DIRECTION="${BRAINS_DIRECTION:-long}"
BRAINS_TD_MS="${BRAINS_TD_MS:-143.4}"
BRAINS_EXCLUDE_TD_MS="${BRAINS_EXCLUDE_TD_MS:-}"
BRAINS_D0="${BRAINS_D0:-3.2e-12}"

PHANTOMS_SUBJ="${PHANTOMS_SUBJ:-PHANTOM3}"
PHANTOMS_ROI="${PHANTOMS_ROI:-fiber1}"
PHANTOMS_ROI_LABEL="${PHANTOMS_ROI_LABEL:-less packed}"
PHANTOMS_DIRECTION="${PHANTOMS_DIRECTION:-1}"
PHANTOMS_TD_MS="${PHANTOMS_TD_MS:-142.5}"
PHANTOMS_EXCLUDE_TD_MS="${PHANTOMS_EXCLUDE_TD_MS:-75.1}"
PHANTOMS_D0="${PHANTOMS_D0:-2.3e-12}"

PHANTOMS_DIRECTION_ALIAS="${PHANTOMS_DIRECTION_ALIAS:-1=longitudinal 3=transversal}"

BRAINS_FITS="${BRAINS_FITS:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/groupfits_rest.parquet}"
BRAINS_CONTRAST_ROOT="${BRAINS_CONTRAST_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/contrast}"
BRAINS_TC_TABLE="${BRAINS_TC_TABLE:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx}"
BRAINS_MODEL="${BRAINS_MODEL:-rest_offset_globC}"

PHANTOMS_FITS="${PHANTOMS_FITS:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/groupfits_rest.parquet}"
PHANTOMS_CONTRAST_ROOT="${PHANTOMS_CONTRAST_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/contrast}"
PHANTOMS_TC_TABLE="${PHANTOMS_TC_TABLE:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx}"
PHANTOMS_MODEL="${PHANTOMS_MODEL:-rest_offset_globC}"

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
    --lcf-min "$LCF_MIN" \
    --lcf-max "$LCF_MAX" \
    --brains-fits "$BRAINS_FITS" \
    --brains-contrast-root "$BRAINS_CONTRAST_ROOT" \
    --brains-tc-table "$BRAINS_TC_TABLE" \
    --brains-model "$BRAINS_MODEL" \
    --brains-subj "$BRAINS_SUBJ" \
    --brains-roi "$BRAINS_ROI" \
    --brains-roi-label "$BRAINS_ROI_LABEL" \
    --brains-direction "$BRAINS_DIRECTION" \
    --brains-td-ms "$BRAINS_TD_MS" \
    --brains-peak-D0-fix "$BRAINS_D0" \
    --brains-exclude-td-ms "${brains_exclude_args[@]}" \
    --phantoms-fits "$PHANTOMS_FITS" \
    --phantoms-contrast-root "$PHANTOMS_CONTRAST_ROOT" \
    --phantoms-tc-table "$PHANTOMS_TC_TABLE" \
    --phantoms-model "$PHANTOMS_MODEL" \
    --phantoms-subj "$PHANTOMS_SUBJ" \
    --phantoms-roi "$PHANTOMS_ROI" \
    --phantoms-roi-label "$PHANTOMS_ROI_LABEL" \
    --phantoms-direction "$PHANTOMS_DIRECTION" \
    --phantoms-td-ms "$PHANTOMS_TD_MS" \
    --phantoms-peak-D0-fix "$PHANTOMS_D0" \
    --phantoms-exclude-td-ms "${phantoms_exclude_args[@]}" \
    "${phantoms_alias_cli[@]}"

echo
echo "Finished publication figure in: $OUT_DIR"
