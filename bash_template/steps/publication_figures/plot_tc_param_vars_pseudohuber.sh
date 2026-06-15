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

PLOT_SCRIPT="$REPO_ROOT/scripts/publication/plot_publication_tc_param_vars.py"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/analysis/publication_figures/tc_param_vars}"

# Edit these lists to change what appears in the final figure.
VARIABLES="${VARIABLES:-q_quad alpha_macro delta A c sqrt_q}"
PLOT_TYPES="${PLOT_TYPES:-vars delta-alpha}"
BRAINS_ROIS="${BRAINS_ROIS:-AntCC CentralCC PostCC}" #AntCC MidAntCC CentralCC MidPostCC PostCC
PHANTOMS_ROIS="${PHANTOMS_ROIS:-fiber1 fiber2}"
BRAINS_DIRECTIONS="${BRAINS_DIRECTIONS:-longitudinal transversal}"
PHANTOMS_DIRECTIONS="${PHANTOMS_DIRECTIONS:-longitudinal transversal}"
FORMATS="${FORMATS:-png pdf}"
DPI="${DPI:-400}"

# Defaults used for the publication figure labels.
BRAINS_SUBJECT_ALIAS="${BRAINS_SUBJECT_ALIAS:-BRAIN=subj1 LUDG=subj2 MBBL=subj3}"
PHANTOMS_SUBJECT_ALIAS="${PHANTOMS_SUBJECT_ALIAS:-PHANTOM3=subj1}"
BRAINS_DIRECTION_ALIAS="${BRAINS_DIRECTION_ALIAS:-long=longitudinal tra=transversal x=longitudinal y=transversal z=transversal}"
PHANTOMS_DIRECTION_ALIAS="${PHANTOMS_DIRECTION_ALIAS:-1=transversal 2=transversal 3=longitudinal}"

# Override these paths when comparing a different fit family.
BRAINS_TABLE="${BRAINS_TABLE:-$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx}"
PHANTOMS_TABLE="${PHANTOMS_TABLE:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/ogse_contrast_vs_gresampled_rest_offset_globC_corr/tcpeak_resampled_data_vs_td/pseudohuber_fixed_macro/params_pseudohuber_mode=fixed_macro_y=tc_peak_resampled_data_ms_k=None.xlsx}"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: Script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

for table in "$BRAINS_TABLE" "$PHANTOMS_TABLE"; do
    if [[ ! -f "$table" ]]; then
        echo "ERROR: Required table not found: $table" >&2
        exit 1
    fi
done

read -r -a variable_args <<< "${VARIABLES//,/ }"
read -r -a plot_type_args <<< "${PLOT_TYPES//,/ }"
read -r -a brain_roi_args <<< "${BRAINS_ROIS//,/ }"
read -r -a phantom_roi_args <<< "${PHANTOMS_ROIS//,/ }"
read -r -a brain_direction_args <<< "${BRAINS_DIRECTIONS//,/ }"
read -r -a phantom_direction_args <<< "${PHANTOMS_DIRECTIONS//,/ }"
read -r -a format_args <<< "${FORMATS//,/ }"
read -r -a brain_subject_alias_args <<< "${BRAINS_SUBJECT_ALIAS//,/ }"
read -r -a phantom_subject_alias_args <<< "${PHANTOMS_SUBJECT_ALIAS//,/ }"
read -r -a brain_direction_alias_args <<< "${BRAINS_DIRECTION_ALIAS//,/ }"
read -r -a phantom_direction_alias_args <<< "${PHANTOMS_DIRECTION_ALIAS//,/ }"

brain_subject_alias_cli=()
for alias in "${brain_subject_alias_args[@]}"; do
    if [[ -n "${alias// }" ]]; then
        brain_subject_alias_cli+=(--brains-subject-alias "$alias")
    fi
done

phantom_subject_alias_cli=()
for alias in "${phantom_subject_alias_args[@]}"; do
    if [[ -n "${alias// }" ]]; then
        phantom_subject_alias_cli+=(--phantoms-subject-alias "$alias")
    fi
done

brain_direction_alias_cli=()
for alias in "${brain_direction_alias_args[@]}"; do
    if [[ -n "${alias// }" ]]; then
        brain_direction_alias_cli+=(--brains-direction-alias "$alias")
    fi
done

phantom_direction_alias_cli=()
for alias in "${phantom_direction_alias_args[@]}"; do
    if [[ -n "${alias// }" ]]; then
        phantom_direction_alias_cli+=(--phantoms-direction-alias "$alias")
    fi
done

mkdir -p "$OUT_DIR"

"$PY" "$PLOT_SCRIPT" \
    --project-root "$PROJECT_ROOT" \
    --out-dir "$OUT_DIR" \
    --brains-table "$BRAINS_TABLE" \
    --phantoms-table "$PHANTOMS_TABLE" \
    --plot-types "${plot_type_args[@]}" \
    --variables "${variable_args[@]}" \
    --brains-rois "${brain_roi_args[@]}" \
    --phantoms-rois "${phantom_roi_args[@]}" \
    --brains-directions "${brain_direction_args[@]}" \
    --phantoms-directions "${phantom_direction_args[@]}" \
    "${brain_subject_alias_cli[@]}" \
    "${phantom_subject_alias_cli[@]}" \
    "${brain_direction_alias_cli[@]}" \
    "${phantom_direction_alias_cli[@]}" \
    --formats "${format_args[@]}" \
    --dpi "$DPI"

echo
echo "Finished publication figures in: $OUT_DIR"
