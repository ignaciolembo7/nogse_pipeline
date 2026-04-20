#!/usr/bin/env bash

set -u -o pipefail

# Shared helpers for DICOM-to-NIfTI batch runners.
# This file is meant to be sourced by small driver scripts that define their
# own paths and dataset-specific settings before calling `init_run`.
#
# Expected variables from the caller:
#   PROJECT_ROOT   Project root directory.
#   REPO_ROOT      Repository root directory.
#   INPUT_ROOT     Base directory that contains the DICOM experiment folders.
#   OUTPUT_ROOT    Base directory where converted NIfTI folders are written.
#
# Optional variables from the caller/environment:
#   CASE_FILTER    Comma-separated list of input/output case names to run.
#   SKIP_EXISTING  When set to 1, skip series whose target folder already
#                  contains `.nii` or `.nii.gz` files. Default: 1.
#   DCM2NIIX_FILENAME_PATTERN
#                  Optional dcm2niix -f pattern. Leave empty for dcm2niix default.
#   DCM2NIIX_WRITE_BEHAVIOR
#                  Optional dcm2niix -w behavior: 0=skip, 1=overwrite,
#                  2=add suffix. Leave empty for dcm2niix default.
#   LOG_ROOT       Directory where timestamped run logs are written.
#                  Default: "$REPO_ROOT/logs".

DICOM2NIFTI_LIB_PATH="${BASH_SOURCE[0]}"

declare -ag DICOM2NIFTI_FAILURES=()

TOTAL_CASES=0
TOTAL_CASES_OK=0
TOTAL_CASES_FAILED=0
TOTAL_CASES_SKIPPED=0
TOTAL_SERIES=0
TOTAL_SERIES_CONVERTED=0
TOTAL_SERIES_SKIPPED=0
TOTAL_SERIES_FAILED=0

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_defined_vars() {
  local var_name
  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      echo "ERROR: required variable '$var_name' is not defined before running dicom2nifti_batch_lib.sh"
      exit 1
    fi
  done
}

append_version() {
  local title="$1"
  shift
  {
    echo "------------------------------------------------------------"
    echo "$title"
    "$@"
    echo
  } >> "$VERSIONS_FILE" 2>&1
}

resolve_default_project_root() {
  local repo_root="$1"
  local repo_parent
  local shared_root
  local project_candidate

  repo_parent="$(cd "$repo_root/.." && pwd)"
  if [[ -d "$repo_parent/Data-DICOM" ]]; then
    echo "$repo_parent"
    return 0
  fi

  shared_root="$(cd "$repo_root/../../.." && pwd)"
  project_candidate="$shared_root/Project-Balseiro-Microstructure"
  if [[ -d "$project_candidate/Data-DICOM" ]]; then
    echo "$project_candidate"
    return 0
  fi

  echo "$repo_parent"
}

case_matches_filter() {
  local input_case="$1"
  local output_case="$2"
  local filter_raw="${CASE_FILTER:-}"
  local filter_item

  if [[ -z "$filter_raw" ]]; then
    return 0
  fi

  IFS=',' read -r -a filter_items <<< "$filter_raw"
  for filter_item in "${filter_items[@]}"; do
    filter_item="${filter_item#"${filter_item%%[![:space:]]*}"}"
    filter_item="${filter_item%"${filter_item##*[![:space:]]}"}"
    if [[ -z "$filter_item" ]]; then
      continue
    fi
    if [[ "$input_case" == "$filter_item" || "$output_case" == "$filter_item" ]]; then
      return 0
    fi
  done

  return 1
}

series_has_outputs() {
  local out_dir="$1"

  compgen -G "$out_dir/*.nii" >/dev/null || compgen -G "$out_dir/*.nii.gz" >/dev/null
}

init_run() {
  local run_name="$1"

  require_defined_vars PROJECT_ROOT REPO_ROOT INPUT_ROOT OUTPUT_ROOT

  if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: directory PROJECT_ROOT does not exist: $PROJECT_ROOT"
    exit 1
  fi

  if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: directory REPO_ROOT does not exist: $REPO_ROOT"
    exit 1
  fi

  if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "ERROR: input root does not exist: $INPUT_ROOT"
    exit 1
  fi

  if ! command_exists dcm2niix; then
    echo "ERROR: dcm2niix is not available in PATH"
    exit 1
  fi

  mkdir -p "$OUTPUT_ROOT"

  cd "$PROJECT_ROOT"

  RUN_ID="${run_name}_$(date +%Y%m%d_%H%M%S)"
  LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
  LOG_DIR="$LOG_ROOT/$RUN_ID"

  mkdir -p "$LOG_DIR"

  LOG_FILE="$LOG_DIR/run.log"
  COMMANDS_FILE="$LOG_DIR/commands_used.txt"
  VERSIONS_FILE="$LOG_DIR/software_versions.txt"
  ENV_FILE="$LOG_DIR/environment.txt"
  GIT_FILE="$LOG_DIR/git_info.txt"

  exec > >(tee -a "$LOG_FILE") 2>&1

  cp "$0" "$LOG_DIR/$(basename "$0")"
  cp "$DICOM2NIFTI_LIB_PATH" "$LOG_DIR/dicom2nifti_batch_lib.sh"

  {
    echo "RUN_ID=$RUN_ID"
    echo "DATE_START=$(date '+%F %T')"
    echo "PROJECT_ROOT=$PROJECT_ROOT"
    echo "REPO_ROOT=$REPO_ROOT"
    echo "INPUT_ROOT=$INPUT_ROOT"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    echo "LOG_ROOT=$LOG_ROOT"
    echo "CASE_FILTER=${CASE_FILTER:-}"
    echo "SKIP_EXISTING=${SKIP_EXISTING:-1}"
    echo "DCM2NIIX_FILENAME_PATTERN=${DCM2NIIX_FILENAME_PATTERN:-}"
    echo "DCM2NIIX_WRITE_BEHAVIOR=${DCM2NIIX_WRITE_BEHAVIOR:-}"
    echo "PWD_AT_START=$(pwd)"
    echo "USER=${USER:-}"
    echo "HOSTNAME=$(hostname)"
    echo "SHELL=${SHELL:-}"
    echo "PATH=$PATH"
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
  } > "$ENV_FILE"

  {
    echo "REPO_ROOT=$REPO_ROOT"
    echo
    if command_exists git; then
      echo "[git rev-parse --show-toplevel]"
      git -C "$REPO_ROOT" rev-parse --show-toplevel || true
      echo

      echo "[git branch --show-current]"
      git -C "$REPO_ROOT" branch --show-current || true
      echo

      echo "[git rev-parse HEAD]"
      git -C "$REPO_ROOT" rev-parse HEAD || true
      echo

      echo "[git describe --always --dirty --tags]"
      git -C "$REPO_ROOT" describe --always --dirty --tags || true
      echo

      echo "[git remote -v]"
      git -C "$REPO_ROOT" remote -v || true
      echo

      echo "[git status --short]"
      git -C "$REPO_ROOT" status --short || true
      echo

      echo "[git diff --stat]"
      git -C "$REPO_ROOT" diff --stat || true
      echo
    else
      echo "git not found"
    fi
  } > "$GIT_FILE"

  : > "$VERSIONS_FILE"

  append_version "bash --version" bash --version
  append_version "python --version" python --version
  append_version "dcm2niix --version" dcm2niix --version

  if command_exists git; then
    git -C "$REPO_ROOT" diff > "$LOG_DIR/repo_worktree.diff" || true
    git -C "$REPO_ROOT" diff --cached > "$LOG_DIR/repo_index.diff" || true
  fi

  if command_exists conda; then
    append_version "conda --version" conda --version
    conda env export > "$LOG_DIR/conda_env_export.yml" 2>/dev/null || true
  fi

  if command_exists pip; then
    append_version "pip --version" pip --version
    pip freeze > "$LOG_DIR/pip_freeze.txt" 2>/dev/null || true
  fi

  echo "============================================================"
  echo "Start       : $(date '+%F %T')"
  echo "RUN_ID      : $RUN_ID"
  echo "PROJECT_ROOT: $PROJECT_ROOT"
  echo "REPO_ROOT   : $REPO_ROOT"
  echo "INPUT_ROOT  : $INPUT_ROOT"
  echo "OUTPUT_ROOT : $OUTPUT_ROOT"
  echo "CASE_FILTER : ${CASE_FILTER:-<all>}"
  echo "SKIP_EXISTING: ${SKIP_EXISTING:-1}"
  echo "Log file    : $LOG_FILE"
  echo "Logs dir    : $LOG_DIR"
  echo "============================================================"
}

run_case() {
  local input_case="$1"
  local output_case="${2:-$1}"
  local dicom_case_dir="$INPUT_ROOT/$input_case"
  local nifti_case_dir="$OUTPUT_ROOT/$output_case"
  local case_failed=0
  local case_series=0
  local case_series_converted=0
  local case_series_skipped=0
  local case_series_failed=0
  local series_dir

  if ! case_matches_filter "$input_case" "$output_case"; then
    echo
    echo "------------------------------------------------------------"
    echo "[SKIP] Filter excluded case: $input_case -> $output_case"
    echo "------------------------------------------------------------"
    TOTAL_CASES_SKIPPED=$((TOTAL_CASES_SKIPPED + 1))
    return 0
  fi

  TOTAL_CASES=$((TOTAL_CASES + 1))

  echo
  echo "------------------------------------------------------------"
  echo "Input case  : $input_case"
  echo "Output case : $output_case"
  echo "DICOM dir   : $dicom_case_dir"
  echo "Output dir  : $nifti_case_dir"
  echo "------------------------------------------------------------"

  if [[ ! -d "$dicom_case_dir" ]]; then
    echo "ERROR: DICOM directory not found: $dicom_case_dir"
    TOTAL_CASES_FAILED=$((TOTAL_CASES_FAILED + 1))
    DICOM2NIFTI_FAILURES+=("$input_case -> $output_case :: missing input directory")
    return 1
  fi

  mkdir -p "$nifti_case_dir"
  printf '%s\t%s\t%s\n' "$input_case" "$output_case" "$dicom_case_dir" >> "$nifti_case_dir/${output_case}_orig-MR-code.txt"

  shopt -s nullglob
  for series_dir in "$dicom_case_dir"/*; do
    local series_name
    local out_series_dir
    local skip_existing

    [[ -d "$series_dir" ]] || continue

    series_name="$(basename "$series_dir")"
    out_series_dir="$nifti_case_dir/$series_name"
    skip_existing="${SKIP_EXISTING:-1}"

    case_series=$((case_series + 1))
    TOTAL_SERIES=$((TOTAL_SERIES + 1))

    mkdir -p "$out_series_dir"

    echo
    echo "  Series     : $series_name"
    echo "  Input      : $series_dir"
    echo "  Output     : $out_series_dir"

    if [[ "$skip_existing" == "1" ]] && series_has_outputs "$out_series_dir"; then
      echo "  Status     : skipped (target already contains NIfTI files)"
      case_series_skipped=$((case_series_skipped + 1))
      TOTAL_SERIES_SKIPPED=$((TOTAL_SERIES_SKIPPED + 1))
      continue
    fi

    local cmd=(dcm2niix -z y -o "$out_series_dir")
    if [[ -n "${DCM2NIIX_FILENAME_PATTERN:-}" ]]; then
      cmd+=(-f "$DCM2NIIX_FILENAME_PATTERN")
    fi
    if [[ -n "${DCM2NIIX_WRITE_BEHAVIOR:-}" ]]; then
      cmd+=(-w "$DCM2NIIX_WRITE_BEHAVIOR")
    fi
    cmd+=("$series_dir")

    printf '%q ' "${cmd[@]}" >> "$COMMANDS_FILE"
    printf '\n' >> "$COMMANDS_FILE"

    if "${cmd[@]}"; then
      echo "  Status     : converted"
      case_series_converted=$((case_series_converted + 1))
      TOTAL_SERIES_CONVERTED=$((TOTAL_SERIES_CONVERTED + 1))
    else
      local rc=$?
      echo "  Status     : failed (exit $rc)"
      case_failed=1
      case_series_failed=$((case_series_failed + 1))
      TOTAL_SERIES_FAILED=$((TOTAL_SERIES_FAILED + 1))
      DICOM2NIFTI_FAILURES+=("$input_case/$series_name :: exit $rc")
    fi
  done
  shopt -u nullglob

  if (( case_series == 0 )); then
    echo "ERROR: no DICOM series directories were found in: $dicom_case_dir"
    TOTAL_CASES_FAILED=$((TOTAL_CASES_FAILED + 1))
    DICOM2NIFTI_FAILURES+=("$input_case -> $output_case :: no series found")
    return 1
  fi

  echo
  echo "Case summary:"
  echo "  Series found    : $case_series"
  echo "  Converted       : $case_series_converted"
  echo "  Skipped         : $case_series_skipped"
  echo "  Failed          : $case_series_failed"

  if (( case_failed == 1 )); then
    TOTAL_CASES_FAILED=$((TOTAL_CASES_FAILED + 1))
    return 1
  fi

  TOTAL_CASES_OK=$((TOTAL_CASES_OK + 1))
  return 0
}

finish_run() {
  echo "DATE_END=$(date '+%F %T')" >> "$ENV_FILE"

  echo
  echo "============================================================"
  echo "End: $(date '+%F %T')"
  echo "Cases processed : $TOTAL_CASES"
  echo "Cases ok        : $TOTAL_CASES_OK"
  echo "Cases failed    : $TOTAL_CASES_FAILED"
  echo "Cases skipped   : $TOTAL_CASES_SKIPPED"
  echo "Series found    : $TOTAL_SERIES"
  echo "Series converted: $TOTAL_SERIES_CONVERTED"
  echo "Series skipped  : $TOTAL_SERIES_SKIPPED"
  echo "Series failed   : $TOTAL_SERIES_FAILED"

  if (( ${#DICOM2NIFTI_FAILURES[@]} > 0 )); then
    echo
    echo "Failures:"
    printf ' - %s\n' "${DICOM2NIFTI_FAILURES[@]}"
    echo "Check:"
    echo "  $LOG_FILE"
    echo "  $LOG_DIR"
    return 1
  fi

  echo "[OK] All selected cases completed successfully."
  echo "Check:"
  echo "  $LOG_FILE"
  echo "  $LOG_DIR"
  return 0
}
