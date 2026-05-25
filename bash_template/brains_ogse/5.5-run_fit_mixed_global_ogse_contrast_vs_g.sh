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

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
export FIT_SCRIPT="${FIT_SCRIPT:-$REPO_ROOT/scripts/fit_ogse_contrast_vs_g.py}"
export ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments}"
export TABLES_ROOT="${TABLES_ROOT:-$ANALYSIS_ROOT/contrast-data-rotated/tables}"
export FITS_DIR="${FITS_DIR:-$ANALYSIS_ROOT/fits}"
export OUT_ROOT="${OUT_ROOT:-$FITS_DIR/ogse_contrast_vs_g_mixed_global}"
export FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"

export SUBJS="${SUBJS:-ALL}"

# ROIs to fit. Use ALL to keep every ROI in the input tables.
export ROIS="${ROIS:-AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Syringe,Right-Lateral-Ventricle,Left-Lateral-Ventricle}"
# export ROIS="Syringe"
# export ROIS="Right-Lateral-Ventricle,Left-Lateral-Ventricle"
# export ROIS="ALL"

export DIRECTIONS="${DIRECTIONS:-long tra}"
export GBASE="${GBASE:-g_thorsten}"
export YCOL="${YCOL:-value_norm}"
export STAT="${STAT:-avg}"
export ONEG="${ONEG:-false}"
export NO_PLOTS="${NO_PLOTS:-false}"

# The default summary has one alpha per subject/ROI/direction. To use one
# alpha per td, point ALPHA_TABLE to a table with subj, roi, direction, td_ms,
# and alpha columns, then set ALPHA_COL=alpha.
export ALPHA_TABLE="${ALPHA_TABLE:-$ANALYSIS_ROOT/alpha_macro/N1/summary_alpha_values.csv}"
export ALPHA_COL="${ALPHA_COL:-alpha_macro}"
export ALPHA_TD_COL="${ALPHA_TD_COL:-td_ms}"
export ALPHA_TD_TOL_MS="${ALPHA_TD_TOL_MS:-1e-3}"

# M0 mode. Keep one block active.
export FIX_M0="${FIX_M0:-1.0}"
export FREE_M0="${FREE_M0:-}"
# export FIX_M0=
# export FREE_M0=1.0

# D0 mode. D0 is in m^2/ms. Keep one block active.
export FIX_D0="${FIX_D0:-3.2e-12}"
export FREE_D0="${FREE_D0:-}"
# export FIX_D0=
# export FREE_D0=3.2e-12

# Mixed-global always fits tc globally; TC_INIT is the initial seed.
export TC_INIT="${TC_INIT:-5.0}"

# Fit bounds. Each variable is "MIN MAX".
export M0_BOUNDS="${M0_BOUNDS:-0.0 5.0}"
export D0_BOUNDS="${D0_BOUNDS:-1e-16 1e-10}"
export TC_BOUNDS="${TC_BOUNDS:-0.1 1000.0}"

# Gradient correction is applied per curve using the matching sheet, td_ms,
# direction, N_1, and N_2 correction factors.
export APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-true}"
export CORR_XLSX="${CORR_XLSX:-$FITS_DIR/grad_correction_rotated/Syringe.grad_correction_rotated.xlsx}"
export CORR_ROI="${CORR_ROI:-Syringe}"
export CORR_TD_MS="${CORR_TD_MS:-}"
export CORR_SHEET="${CORR_SHEET:-}"
export CORR_TOL_MS="${CORR_TOL_MS:-1e-3}"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$TABLES_ROOT" ]]; then
    echo "ERROR: contrast tables root not found: $TABLES_ROOT" >&2
    exit 1
fi

if [[ ! -f "$ALPHA_TABLE" ]]; then
    echo "ERROR: alpha table not found: $ALPHA_TABLE" >&2
    echo "Run 4.3-run_make_alpha_macro_summary.sh first or set ALPHA_TABLE." >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

declare -a files=()
if [[ "$SUBJS" == "ALL" ]]; then
    while IFS= read -r file; do
        files+=("$file")
    done < <(find "$TABLES_ROOT" -type f -name "$FILE_PATTERN" | sort)
else
    read -r -a subj_list <<< "${SUBJS//,/ }"
    for subj in "${subj_list[@]}"; do
        subj_root="$TABLES_ROOT/$subj"
        if [[ ! -d "$subj_root" ]]; then
            echo "WARNING: subject contrast root not found: $subj_root" >&2
            continue
        fi
        while IFS= read -r file; do
            files+=("$file")
        done < <(find "$subj_root" -type f -name "$FILE_PATTERN" | sort)
    done
fi

if (( ${#files[@]} == 0 )); then
    echo "ERROR: no contrast tables found under $TABLES_ROOT with pattern $FILE_PATTERN" >&2
    exit 1
fi

roi_args=()
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        roi_args+=(--rois "${roi_list[@]}")
    fi
fi

direction_args=()
if [[ "$DIRECTIONS" != "ALL" ]]; then
    read -r -a direction_list <<< "${DIRECTIONS//,/ }"
    if (( ${#direction_list[@]} > 0 )); then
        direction_args+=(--directions "${direction_list[@]}")
    fi
fi

oneg_args=()
if [[ "${ONEG,,}" == "true" ]]; then
    oneg_args+=(--oneg)
fi

m0_args=()
if [[ -n "${M0_BOUNDS// }" ]]; then
    read -r -a m0_bound_values <<< "${M0_BOUNDS//,/ }"
    m0_args+=(--M0_bounds "${m0_bound_values[@]}")
fi
if [[ -n "${FREE_M0// }" ]]; then
    m0_args+=(--free_M0 "$FREE_M0")
elif [[ -n "${FIX_M0// }" ]]; then
    m0_args+=(--fix_M0 "$FIX_M0")
fi

d0_args=()
if [[ -n "${D0_BOUNDS// }" ]]; then
    read -r -a d0_bound_values <<< "${D0_BOUNDS//,/ }"
    d0_args+=(--D0_bounds "${d0_bound_values[@]}")
fi
if [[ -n "${FREE_D0// }" ]]; then
    d0_args+=(--free_D0 "$FREE_D0")
elif [[ -n "${FIX_D0// }" ]]; then
    d0_args+=(--fix_D0 "$FIX_D0")
fi

tc_args=(--tc_init "$TC_INIT")
if [[ -n "${TC_BOUNDS// }" ]]; then
    read -r -a tc_bound_values <<< "${TC_BOUNDS//,/ }"
    tc_args+=(--tc_bounds "${tc_bound_values[@]}")
fi

corr_args=(--no_grad_corr)
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    corr_args=(
        --apply_grad_corr
        --corr_xlsx "$CORR_XLSX"
        --corr_roi "$CORR_ROI"
        --corr_tol_ms "$CORR_TOL_MS"
    )
    if [[ -n "${CORR_TD_MS// }" ]]; then
        corr_args+=(--corr_td_ms "$CORR_TD_MS")
    fi
    if [[ -n "${CORR_SHEET// }" ]]; then
        corr_args+=(--corr_sheet "$CORR_SHEET")
    fi
fi

plot_args=()
if [[ "${NO_PLOTS,,}" == "true" ]]; then
    plot_args+=(--no_plots)
fi

echo "============================================================"
echo "Brains OGSE mixed-global contrast fit"
echo "Fit script     : $FIT_SCRIPT"
echo "Tables root    : $TABLES_ROOT"
echo "Input files    : ${#files[@]}"
echo "Out root       : $OUT_ROOT"
echo "ROIs           : $ROIS"
echo "Directions     : $DIRECTIONS"
echo "gbase / ycol   : $GBASE / $YCOL"
echo "Alpha table    : $ALPHA_TABLE"
echo "Alpha column   : $ALPHA_COL"
echo "M0 fix/free    : ${FIX_M0:-<none>} / ${FREE_M0:-<none>}"
echo "D0 fix/free    : ${FIX_D0:-<none>} / ${FREE_D0:-<none>}"
echo "tc init/bounds : $TC_INIT / $TC_BOUNDS"
echo "Grad correction: $APPLY_GRAD_CORR"
echo "============================================================"

if "$PY" "$FIT_SCRIPT" \
    "${files[@]}" \
    --model mixed_global \
    --out_root "$OUT_ROOT" \
    --gbase "$GBASE" \
    --ycol "$YCOL" \
    --stat "$STAT" \
    --alpha_table "$ALPHA_TABLE" \
    --alpha_col "$ALPHA_COL" \
    --alpha_td_col "$ALPHA_TD_COL" \
    --alpha_td_tol_ms "$ALPHA_TD_TOL_MS" \
    "${tc_args[@]}" \
    "${m0_args[@]}" \
    "${d0_args[@]}" \
    "${roi_args[@]}" \
    "${direction_args[@]}" \
    "${oneg_args[@]}" \
    "${corr_args[@]}" \
    "${plot_args[@]}"; then
    echo
    echo "Finished: OK"
else
    status=$?
    echo
    echo "Finished: FAILED with exit code $status" >&2
    exit "$status"
fi
