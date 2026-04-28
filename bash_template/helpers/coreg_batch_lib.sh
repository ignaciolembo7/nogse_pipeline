#!/usr/bin/env bash

set -u -o pipefail

# Shared helpers for the signal-extraction batch runners.
# This file is meant to be sourced by small driver scripts that define their
# own paths and dataset-specific settings before calling `init_run`.
#
# Expected variables from the caller:
#   PROJECT_ROOT   Project root directory.
#   REPO_ROOT      Repository root, typically "$PROJECT_ROOT/nogse_pipeline".
#   COREG_SCRIPT   Python entry point to execute for each case.
#   OUT_ROOT       Base output directory for extracted signal tables.
#
# Optional variables from the caller:
#   SUBJECTS_DIR         FreeSurfer subjects directory, required only when
#                        REQUIRE_SUBJECTS_DIR=1.
#   REQUIRE_SUBJECTS_DIR Set to 1 for brain workflows that need SUBJECTS_DIR.

COREG_BATCH_LIB_PATH="${BASH_SOURCE[0]}"

declare -ag FAILURES=()

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_defined_vars() {
  local var_name
  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      echo "ERROR: required variable '$var_name' is not defined before sourcing/running coreg_batch_lib.sh"
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

init_run() {
  local run_name="$1"
  local require_subjects_dir="${REQUIRE_SUBJECTS_DIR:-0}"

  require_defined_vars PROJECT_ROOT REPO_ROOT COREG_SCRIPT OUT_ROOT

  if [[ ! -f "$COREG_SCRIPT" ]]; then
    echo "ERROR: file $COREG_SCRIPT does not exist"
    exit 1
  fi

  if [[ "$require_subjects_dir" == "1" ]]; then
    require_defined_vars SUBJECTS_DIR
    if [[ ! -d "$SUBJECTS_DIR" ]]; then
      echo "ERROR: directory $SUBJECTS_DIR does not exist"
      exit 1
    fi
  fi

  cd "$PROJECT_ROOT"

  RUN_ID="${run_name}_$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="$PROJECT_ROOT/nogse_pipeline/logs/$RUN_ID"

  mkdir -p "$LOG_DIR"

  LOG_FILE="$LOG_DIR/run.log"
  COMMANDS_FILE="$LOG_DIR/commands_used.txt"
  VERSIONS_FILE="$LOG_DIR/software_versions.txt"
  ENV_FILE="$LOG_DIR/environment.txt"
  GIT_FILE="$LOG_DIR/git_info.txt"

  exec > >(tee -a "$LOG_FILE") 2>&1

  cp "$0" "$LOG_DIR/$(basename "$0")"
  cp "$COREG_BATCH_LIB_PATH" "$LOG_DIR/coreg_batch_lib.sh"

  {
    echo "RUN_ID=$RUN_ID"
    echo "DATE_START=$(date '+%F %T')"
    echo "PROJECT_ROOT=$PROJECT_ROOT"
    echo "REPO_ROOT=$REPO_ROOT"
    echo "COREG_SCRIPT=$COREG_SCRIPT"
    if [[ -n "${SUBJECTS_DIR:-}" ]]; then
      echo "SUBJECTS_DIR=$SUBJECTS_DIR"
    fi
    echo "OUT_ROOT=$OUT_ROOT"
    echo "PWD_AT_START=$(pwd)"
    echo "USER=${USER:-}"
    echo "HOSTNAME=$(hostname)"
    echo "SHELL=${SHELL:-}"
    echo "PATH=$PATH"
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
    echo "FSLDIR=${FSLDIR:-}"
    echo "FREESURFER_HOME=${FREESURFER_HOME:-}"
    echo "ANTSPATH=${ANTSPATH:-}"
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

      echo "[git status]"
      git -C "$REPO_ROOT" status || true
      echo

      echo "[git diff --stat]"
      git -C "$REPO_ROOT" diff --stat || true
      echo
    else
      echo "git not found"
    fi
  } > "$GIT_FILE"

  : > "$VERSIONS_FILE"

  append_version "python --version" python --version

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

  if command_exists bet; then
    append_version "bet --version" bet --version
  fi

  if command_exists fslmaths; then
    append_version "fslmaths --version" fslmaths --version
  fi

  if [[ -n "${FSLDIR:-}" && -f "$FSLDIR/etc/fslversion" ]]; then
    {
      echo "------------------------------------------------------------"
      echo "FSL version file"
      cat "$FSLDIR/etc/fslversion"
      echo
    } >> "$VERSIONS_FILE"
  fi

  if command_exists dwiextract; then
    append_version "dwiextract -version" dwiextract -version
  fi

  if command_exists mrconvert; then
    append_version "mrconvert -version" mrconvert -version
  fi

  if command_exists antsRegistration; then
    append_version "antsRegistration --version" antsRegistration --version
  fi

  if command_exists antsApplyTransforms; then
    append_version "which antsApplyTransforms" bash -lc 'which antsApplyTransforms'
  fi

  if command_exists mri_convert; then
    append_version "mri_convert --version" mri_convert --version
  fi

  if command_exists recon-all; then
    append_version "recon-all -version" recon-all -version
  fi

  echo "============================================================"
  echo "Inicio      : $(date '+%F %T')"
  echo "RUN_ID      : $RUN_ID"
  echo "PROJECT_ROOT: $PROJECT_ROOT"
  echo "REPO_ROOT   : $REPO_ROOT"
  echo "SCRIPT      : $COREG_SCRIPT"
  if [[ -n "${SUBJECTS_DIR:-}" ]]; then
    echo "SUBJECTS_DIR: $SUBJECTS_DIR"
  fi
  echo "OUT_ROOT    : $OUT_ROOT"
  echo "Log file    : $LOG_FILE"
  echo "Logs dir    : $LOG_DIR"
  echo "============================================================"
}

run_case() {
  local label="$1"
  shift

  echo
  echo "------------------------------------------------------------"
  echo "[$(date '+%F %T')] Executing: $label"
  echo "Command: python $COREG_SCRIPT $*"
  echo "python $COREG_SCRIPT $*" >> "$COMMANDS_FILE"
  echo "------------------------------------------------------------"

  if python "$COREG_SCRIPT" "$@"; then
    echo "[$(date '+%F %T')] OK: $label"
  else
    local rc=$?
    echo "[$(date '+%F %T')] ERROR ($rc): $label"
    FAILURES+=("$label")
  fi
}

finish_run() {
  echo "DATE_END=$(date '+%F %T')" >> "$ENV_FILE"

  echo
  echo "============================================================"
  echo "End: $(date '+%F %T')"
  if (( ${#FAILURES[@]} > 0 )); then
    echo "Cases with errors: ${#FAILURES[@]}"
    printf ' - %s\n' "${FAILURES[@]}"
    echo "Check:"
    echo "  $LOG_FILE"
    echo "  $LOG_DIR"
    return 1
  else
    echo "[OK] All cases completed successfully."
    echo "Check:"
    echo "  $LOG_FILE"
    echo "  $LOG_DIR"
    return 0
  fi
}
