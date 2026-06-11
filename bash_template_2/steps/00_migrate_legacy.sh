#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help migrate "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

MIGRATE_SCRIPT="${MIGRATE_SCRIPT:-$REPO_ROOT/scripts/data/migrate_analysis_to_master.py}"
MIGRATION_REPORT_DIR="${MIGRATION_REPORT_DIR:-$ANALYSIS_ROOT/master_migration_report}"

bt2_require_file "$MIGRATE_SCRIPT" "migration script"
mkdir -p "$MIGRATION_REPORT_DIR" "$(dirname "$MASTER_PARQUET")"

"$PY" "$MIGRATE_SCRIPT" "$ANALYSIS_ROOT" \
    --out "$MASTER_PARQUET" \
    --report-dir "$MIGRATION_REPORT_DIR" \
    --drop-exact-duplicates \
    ${MIGRATE_EXTRA_ARGS:-}
