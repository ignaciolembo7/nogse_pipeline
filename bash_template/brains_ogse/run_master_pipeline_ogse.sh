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
elif [[ -x "$HOME/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="$HOME/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

SIGNALS_ROOT="${SIGNALS_ROOT:-$PROJECT_ROOT/Data-signals}"
RESULTS_ROOT="${RESULTS_ROOT:-$SIGNALS_ROOT/Results}"
PARAMS_XLSX="${PARAMS_XLSX:-$SIGNALS_ROOT/sequence_parameters_brains.xlsx}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments}"
MASTER_PARQUET="${MASTER_PARQUET:-$ANALYSIS_ROOT/master.long.parquet}"
MASTER_FIT_PARAMS="${MASTER_FIT_PARAMS:-$ANALYSIS_ROOT/master_fit_params.parquet}"

PROCESS_SCRIPT="${PROCESS_SCRIPT:-$REPO_ROOT/scripts/process_one_results.py}"
ROTATE_SCRIPT="${ROTATE_SCRIPT:-$REPO_ROOT/scripts/rotate_ogse_tensor.py}"
MAKE_CONTRAST_SCRIPT="${MAKE_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/make_contrast.py}"
FIT_CONTRAST_SCRIPT="${FIT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/fit_ogse_contrast_vs_g.py}"
ALPHA_MACRO_SCRIPT="${ALPHA_MACRO_SCRIPT:-$REPO_ROOT/scripts/make_alpha_macro_summary.py}"
TC_VS_TD_SCRIPT="${TC_VS_TD_SCRIPT:-$REPO_ROOT/scripts/run_tc_vs_td.py}"
PLOT_SIGNAL_SCRIPT="${PLOT_SIGNAL_SCRIPT:-$REPO_ROOT/scripts/plot_ogse_signal_vs_g.py}"
PLOT_CONTRAST_SCRIPT="${PLOT_CONTRAST_SCRIPT:-$REPO_ROOT/scripts/plot_ogse_contrast_vs_g.py}"

DIRS_TXT="${DIRS_TXT:-$REPO_ROOT/assets/dirs/dirs_6.txt}"
PROCESS_OUT_ROOT="${PROCESS_OUT_ROOT:-$ANALYSIS_ROOT/data/tables}"
ROTATED_OUT_ROOT="${ROTATED_OUT_ROOT:-$ANALYSIS_ROOT/data-rotated/tables}"
CONTRAST_OUT_ROOT="${CONTRAST_OUT_ROOT:-$ANALYSIS_ROOT/contrast-data-master}"
FIT_OUT_ROOT="${FIT_OUT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_master}"
ALPHA_OUT_DIR="${ALPHA_OUT_DIR:-$ANALYSIS_ROOT/alpha_macro/master}"
TC_OUT_DIR="${TC_OUT_DIR:-$ANALYSIS_ROOT/fits/tc_vs_td_master}"
PLOT_OUT_ROOT="${PLOT_OUT_ROOT:-$ANALYSIS_ROOT/plots-master}"

# Declarative contrast specs. Format:
#   subj|sheet|roi|direction|td_ms|N_1|N_2|Hz_1|Hz_2
# Use ALL for subj/sheet/roi/direction to avoid that selector.
CONTRAST_SPECS="${CONTRAST_SPECS:-}"
RUN_ALPHA_MACRO="${RUN_ALPHA_MACRO:-0}"
RUN_TC_VS_TD="${RUN_TC_VS_TD:-0}"
RUN_MASTER_PLOTS="${RUN_MASTER_PLOTS:-0}"

mkdir -p "$ANALYSIS_ROOT" "$PROCESS_OUT_ROOT" "$ROTATED_OUT_ROOT" "$CONTRAST_OUT_ROOT" "$FIT_OUT_ROOT"

echo "Master table       : $MASTER_PARQUET"
echo "Master fit params  : $MASTER_FIT_PARAMS"

echo
echo "1) Ingest Results into master"
while read -r file; do
    [[ -z "$file" ]] && continue
    "$PY" "$PROCESS_SCRIPT" "$file" "$PARAMS_XLSX" \
        --out_dir "$PROCESS_OUT_ROOT" \
        --master-parquet "$MASTER_PARQUET"
done < <(find "$RESULTS_ROOT" -type f -name "*_results.xlsx" | sort)

echo
echo "2) Rotate signals from master back into master"
"$PY" - "$MASTER_PARQUET" <<'PY' | while IFS=$'\t' read -r subj sheet td_ms n hz; do
import pandas as pd
import sys

df = pd.read_parquet(sys.argv[1])
sig = df[df["row_kind"].astype(str).eq("signal")].copy()
cols = [c for c in ["subj", "sheet", "td_ms", "N", "Hz"] if c in sig.columns]
for row in sig[cols].drop_duplicates().sort_values(cols).itertuples(index=False):
    print("\t".join(str(x) for x in row))
PY
    "$PY" "$ROTATE_SCRIPT" \
        --master-parquet "$MASTER_PARQUET" \
        --subj "$subj" \
        --sheet "$sheet" \
        --td_ms "$td_ms" \
        --N "$n" \
        --Hz "$hz" \
        --dirs_txt "$DIRS_TXT" \
        --out_dir "$ROTATED_OUT_ROOT"
done

if [[ -n "$CONTRAST_SPECS" ]]; then
    echo
    echo "3) Build declared contrasts from master"
    IFS=';' read -r -a specs <<< "$CONTRAST_SPECS"
    for spec in "${specs[@]}"; do
        [[ -z "$spec" ]] && continue
        IFS='|' read -r subj sheet roi direction td_ms n1 n2 hz1 hz2 <<< "$spec"
        args=(--master-parquet "$MASTER_PARQUET" --append-master --out_root "$CONTRAST_OUT_ROOT")
        [[ "${subj:-ALL}" != "ALL" ]] && args+=(--subj "$subj")
        [[ "${sheet:-ALL}" != "ALL" ]] && args+=(--sheet "$sheet")
        [[ "${roi:-ALL}" != "ALL" ]] && args+=(--roi "$roi")
        [[ "${direction:-ALL}" != "ALL" ]] && args+=(--direction "$direction")
        [[ -n "${td_ms:-}" ]] && args+=(--td_ms "$td_ms")
        [[ -n "${n1:-}" ]] && args+=(--N_1 "$n1")
        [[ -n "${n2:-}" ]] && args+=(--N_2 "$n2")
        [[ -n "${hz1:-}" ]] && args+=(--Hz_1 "$hz1")
        [[ -n "${hz2:-}" ]] && args+=(--Hz_2 "$hz2")
        "$PY" "$MAKE_CONTRAST_SCRIPT" "${args[@]}"
    done
else
    echo
    echo "3) No CONTRAST_SPECS provided; skipping contrast build."
fi

echo
echo "4) Fit contrast rows from master"
"$PY" "$FIT_CONTRAST_SCRIPT" \
    --master-parquet "$MASTER_PARQUET" \
    --master-fit-params "$MASTER_FIT_PARAMS" \
    --model "${FIT_MODEL:-ogse_free}" \
    --out_root "$FIT_OUT_ROOT" \
    --gbase "${FIT_GBASE:-g_lin_max}" \
    --ycol "${FIT_YCOL:-value_norm}" \
    --stat "${FIT_STAT:-avg}" \
    ${FIT_EXTRA_ARGS:-}

if [[ "$RUN_ALPHA_MACRO" == "1" ]]; then
    echo
    echo "5) Build alpha_macro summary from master D_proj rows"
    "$PY" "$ALPHA_MACRO_SCRIPT" \
        --master-parquet "$MASTER_PARQUET" \
        --master-fit-params "$MASTER_FIT_PARAMS" \
        --N "${ALPHA_N:-1}" \
        --out-summary "$ALPHA_OUT_DIR/summary_alpha_values.xlsx" \
        --out-avg "$ALPHA_OUT_DIR/D_vs_delta_app.combined.xlsx" \
        ${ALPHA_EXTRA_ARGS:-}
fi

if [[ "$RUN_TC_VS_TD" == "1" ]]; then
    echo
    echo "6) Fit tc-vs-td from master_fit_params"
    "$PY" "$TC_VS_TD_SCRIPT" \
        --master-fit-params "$MASTER_FIT_PARAMS" \
        --method "${TC_METHOD:-pseudohuber_fixed_macro}" \
        --y-col "${TC_Y_COL:-tc_peak_ms}" \
        --out-dir "$TC_OUT_DIR/${TC_METHOD:-pseudohuber_fixed_macro}" \
        ${TC_EXTRA_ARGS:-}
fi

if [[ "$RUN_MASTER_PLOTS" == "1" ]]; then
    echo
    echo "7) Plot selected master rows"
    "$PY" "$PLOT_SIGNAL_SCRIPT" \
        --master-parquet "$MASTER_PARQUET" \
        --out_root "$PLOT_OUT_ROOT/signal" \
        --ycol "${PLOT_SIGNAL_YCOL:-value_norm}" \
        --xcol "${PLOT_SIGNAL_XCOL:-g_thorsten}" \
        --stat "${PLOT_STAT:-avg}" \
        ${PLOT_SIGNAL_EXTRA_ARGS:-}
    "$PY" "$PLOT_CONTRAST_SCRIPT" \
        --master-parquet "$MASTER_PARQUET" \
        --out_root "$PLOT_OUT_ROOT/contrast" \
        --ycol "${PLOT_CONTRAST_YCOL:-value_norm}" \
        --xcol "${PLOT_CONTRAST_XCOL:-g_thorsten_1}" \
        --stat "${PLOT_STAT:-avg}" \
        ${PLOT_CONTRAST_EXTRA_ARGS:-}
fi

echo
echo "Done."
