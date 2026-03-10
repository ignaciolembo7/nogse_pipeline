#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_HOME/src/coreg_extract.py" ]]; then
  REPO_ROOT="$SCRIPT_HOME"
elif [[ -f "$SCRIPT_HOME/coreg_extract.py" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_HOME/.." && pwd)"
else
  echo "ERROR: no pude localizar coreg_extract.py."
  echo "Guarda este .sh en la raiz de nogse_pipeline o dentro de nogse_pipeline/src/."
  exit 1
fi

cd "$REPO_ROOT"

if [[ -z "${SUBJECTS_DIR:-}" ]]; then
  echo "ERROR: SUBJECTS_DIR no esta definido."
  echo 'Ejemplo:'
  echo '  export SUBJECTS_DIR="$PWD/Data-signal-extracted/DATA_PROCESSED"'
  exit 1
fi

mkdir -p logs
LOG_FILE="logs/run_BRAINS-denoised_topup_signal_extraction_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Inicio: $(date '+%F %T')"
echo "Repo root   : $REPO_ROOT"
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
  echo "Comando: python src/coreg_extract.py $*"
  echo "------------------------------------------------------------"

  if python src/coreg_extract.py "$@"; then
    echo "[$(date '+%F %T')] OK: $label"
  else
    local rc=$?
    echo "[$(date '+%F %T')] ERROR ($rc): $label"
    FAILURES+=("$label")
  fi
}

run_case "20220622_BRAIN" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20220622_BRAIN" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle

run_case "20230619_BRAIN-3" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20230619_BRAIN-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230623_BRAIN-4" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20230623_BRAIN-4" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230623_LUDG-2" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20230623_LUDG-2" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230629_MBBL-2" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20230629_MBBL-2" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230710_LUDG-3" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20230710_LUDG-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230630_MBBL-3" \
  --exp-root "Data-NIFTI-BRAINS-denoised_topup/20230630_MBBL-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "Data_signals" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-dwi "syringe_mask.nii.gz"

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