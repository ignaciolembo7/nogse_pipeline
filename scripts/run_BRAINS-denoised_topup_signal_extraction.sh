#!/usr/bin/env bash

set -u -o pipefail

# Este .sh vive en: Project-Balseiro-Microstructure/nogse_pipeline/scripts
SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project-Balseiro-Microstructure
PROJECT_ROOT="$(cd "$SCRIPT_HOME/../.." && pwd)"

# Repo Python: Project-Balseiro-Microstructure/nogse_pipeline
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

# Script principal
COREG_SCRIPT="$REPO_ROOT/src/coreg_extract.py"

if [[ ! -f "$COREG_SCRIPT" ]]; then
  echo "ERROR: no existe $COREG_SCRIPT"
  exit 1
fi

cd "$PROJECT_ROOT"

# SUBJECTS_DIR esperado por coreg_extract.py:
# debe contener sub-<subject>/mri/T1.mgz etc.
SUBJECTS_DIR="$PROJECT_ROOT/Data_signals/DATA_PROCESSED/subjects"

if [[ ! -d "$SUBJECTS_DIR" ]]; then
  echo "ERROR: SUBJECTS_DIR no existe: $SUBJECTS_DIR"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/run_BRAINS-denoised_topup_signal_extraction_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Inicio      : $(date '+%F %T')"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "REPO_ROOT   : $REPO_ROOT"
echo "SCRIPT      : $COREG_SCRIPT"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "Python      : $(command -v python || echo 'no encontrado')"
python --version || true
echo "Log file    : $LOG_FILE"
echo "============================================================"

declare -a FAILURES=()

run_case() {
  local label="$1"
  shift

  echo
  echo "------------------------------------------------------------"
  echo "[$(date '+%F %T')] Ejecutando: $label"
  echo "Comando: python $COREG_SCRIPT $*"
  echo "------------------------------------------------------------"

  if python "$COREG_SCRIPT" "$@"; then
    echo "[$(date '+%F %T')] OK: $label"
  else
    local rc=$?
    echo "[$(date '+%F %T')] ERROR ($rc): $label"
    FAILURES+=("$label")
  fi
}

run_case "20220622_BRAIN" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20220622_BRAIN" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle

run_case "20230619_BRAIN-3" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230619_BRAIN-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230623_BRAIN-4" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230623_BRAIN-4" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230623_LUDG-2" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230623_LUDG-2" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230629_MBBL-2" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230629_MBBL-2" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230710_LUDG-3" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230710_LUDG-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-dwi "syringe_mask.nii.gz"

run_case "20230630_MBBL-3" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230630_MBBL-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$PROJECT_ROOT/Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

echo
echo "============================================================"
echo "Fin: $(date '+%F %T')"
if (( ${#FAILURES[@]} > 0 )); then
  echo "Casos con error: ${#FAILURES[@]}"
  printf ' - %s\n' "${FAILURES[@]}"
  echo "Revisa el log: $LOG_FILE"
  exit 1
else
  echo "Todos los casos terminaron OK."
  echo "Log: $LOG_FILE"
  exit 0
fi