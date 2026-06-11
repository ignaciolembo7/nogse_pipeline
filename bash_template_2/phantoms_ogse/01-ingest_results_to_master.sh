#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash 01-ingest_results_to_master.sh [RESULTS_ROOT ...]

What it does:
  Reads one or more Results folders, converts each *_results.xlsx to long rows,
  writes legacy per-experiment tables, and appends row_kind='signal' rows to
  analysis/phantoms/ogse_experiments/master.long.parquet.

Arguments:
  RESULTS_ROOT ...   Optional Results folders. If omitted, uses $RESULTS_ROOT,
                     or Data-signals/Results by default.

Environment:
  PARAMS_XLSX        Sequence-parameter workbook. Default:
                     Data-signals/sequence_parameters_phantoms.xlsx
  MASTER_PARQUET     Master table output. Default:
                     analysis/phantoms/ogse_experiments/master.long.parquet
  RESULTS_GLOB       File pattern inside each Results root. Default:
                     *_results.xlsx

Examples:
  bash 01-ingest_results_to_master.sh Data-signals/Results/20260122-PHANTOM_FIBER

  bash 01-ingest_results_to_master.sh \
    Data-signals/Results/20260122-PHANTOM_FIBER \
    Data-signals/Results/20260210-PHANTOM_FIBER

  PARAMS_XLSX=Data-signals/sequence_parameters_phantoms.xlsx \
    bash 01-ingest_results_to_master.sh Data-signals/Results/20260122-PHANTOM_FIBER
EOF
    exit 0
fi
if (( $# > 0 )); then
    export RESULTS_ROOTS="$*"
fi
exec bash "$SCRIPT_DIR/../run_dataset.sh" phantoms ingest
