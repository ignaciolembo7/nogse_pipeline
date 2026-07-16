#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Portable whole-corpus-callosum erosion runner for FastSurfer-CC outputs.
#
# The script:
#   1) Selects labels 251..255 together from callosum.CC.orig.mgz.
#   2) Erodes only the outer boundary of the combined corpus callosum mask.
#   3) Applies that eroded mask to the original multi-label segmentation.
#   4) Preserves the internal boundaries and original labels 251..255.
#
# Usage:
#   bash run_fastsurfer_erode.sh SUBJECT_ID [ERODE_STEPS]
#
# Example:
#   bash run_fastsurfer_erode.sh BRAIN-1 1
#
# Expected folder structure:
#   <script_directory>/
#   ├── run_fastsurfer_erode.sh
#   └── FASTSURFER_CC/
#       ├── license.txt
#       └── OUT/
#           └── <SUBJECT_ID>/
#               └── mri/
#                   └── callosum.CC.orig.mgz
#
# Main output:
#   FASTSURFER_CC/OUT/<SUBJECT_ID>/mri/
#   callosum.CC.orig.erode<ERODE_STEPS>.mgz
#
# Optional environment variables:
#   IMAGE        Docker image to use.
#   PROJECT_DIR  Root FastSurfer data directory.
#   OUTPUT_DIR   FastSurfer subjects directory. Defaults to PROJECT_DIR/OUT.
#   FORCE        Set to 1 to overwrite existing erosion outputs.
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '3,36p' "${BASH_SOURCE[0]}"
    exit 0
fi

SUBJECT_ID="${1:-${SUBJECT_ID:-BRAIN-1}}"
ERODE_STEPS="${2:-${ERODE_STEPS:-1}}"
IMAGE="${IMAGE:-deepmi/fastsurfer:cpu-v2.5.4}"
FORCE="${FORCE:-0}"

if [[ ! "${SUBJECT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: SUBJECT_ID may contain only letters, numbers, dots, underscores, and hyphens."
    exit 1
fi

if [[ ! "${ERODE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ERODE_STEPS must be a positive integer."
    exit 1
fi

if [[ "${FORCE}" != "0" && "${FORCE}" != "1" ]]; then
    echo "Error: FORCE must be 0 or 1."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/FASTSURFER_CC}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/OUT}"
LICENSE_FILE="${PROJECT_DIR}/license.txt"

SUBJECT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}"
MRI_DIR="${SUBJECT_DIR}/mri"
INPUT_LABELS="${MRI_DIR}/callosum.CC.orig.mgz"
EROSION_DIR="${MRI_DIR}/cc_erosion"
EROSION_TAG="erode${ERODE_STEPS}"

WHOLE_CC_MASK="${EROSION_DIR}/callosum.CC.whole.mask.mgz"
WHOLE_CC_ERODED="${EROSION_DIR}/callosum.CC.whole.mask.${EROSION_TAG}.mgz"
FINAL_LABELS="${MRI_DIR}/callosum.CC.orig.${EROSION_TAG}.mgz"
STATS_BEFORE="${EROSION_DIR}/callosum.CC.stats.before.txt"
STATS_AFTER="${EROSION_DIR}/callosum.CC.stats.after.${EROSION_TAG}.txt"

LOG_DIR="${PROJECT_DIR}/LOGS/CC_EROSION"
PROVENANCE_ROOT="${PROJECT_DIR}/PROVENANCE/CC_EROSION"

RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/${SUBJECT_ID}_${EROSION_TAG}_${RUN_TIMESTAMP}.log"
PROVENANCE_DIR="${PROVENANCE_ROOT}/${SUBJECT_ID}/${EROSION_TAG}/${RUN_TIMESTAMP}"

mkdir -p "${EROSION_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"

# Save all output to a timestamped log while keeping it visible in the terminal.
exec > >(tee -a "${LOG_FILE}") 2>&1

on_exit() {
    local status=$?
    echo
    if [[ ${status} -eq 0 ]]; then
        echo "Run completed successfully: $(date --iso-8601=seconds 2>/dev/null || date)"
    else
        echo "Run failed with exit code ${status}: $(date --iso-8601=seconds 2>/dev/null || date)"
    fi
    echo "Log file: ${LOG_FILE}"
}
trap on_exit EXIT

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"

case "${UNAME_S}" in
    MINGW*|MSYS*|CYGWIN*)
        PLATFORM="git-bash"
        ;;
    Darwin*)
        PLATFORM="macos"
        ;;
    Linux*)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            PLATFORM="wsl"
        else
            PLATFORM="linux"
        fi
        ;;
    *)
        PLATFORM="unknown"
        ;;
esac

# Git Bash rewrites arguments beginning with "/" before passing them to
# Windows executables. Disable that behavior for Linux container paths.
docker_safe() {
    if [[ "${PLATFORM}" == "git-bash" ]]; then
        MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker "$@"
    else
        docker "$@"
    fi
}

# Bind-mount source paths must use the host operating system path format.
to_docker_host_path() {
    local path="$1"

    if [[ "${PLATFORM}" == "git-bash" ]]; then
        if ! command -v cygpath >/dev/null 2>&1; then
            echo "Error: cygpath is required when running from Git Bash."
            return 1
        fi
        cygpath -m "${path}"
    else
        printf '%s\n' "${path}"
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not available in this terminal."
    exit 1
fi

if ! docker_safe info >/dev/null 2>&1; then
    echo "Error: Docker Desktop or the Docker daemon is not running."
    exit 1
fi

if [[ ! -f "${LICENSE_FILE}" ]]; then
    echo "Error: FreeSurfer license not found:"
    echo "  ${LICENSE_FILE}"
    exit 1
fi

if [[ ! -f "${INPUT_LABELS}" ]]; then
    echo "Error: FastSurfer-CC segmentation not found:"
    echo "  ${INPUT_LABELS}"
    exit 1
fi

if [[ -f "${FINAL_LABELS}" && "${FORCE}" != "1" ]]; then
    echo "Error: eroded corpus callosum segmentation already exists:"
    echo "  ${FINAL_LABELS}"
    echo
    echo "To overwrite it, run:"
    echo "  FORCE=1 bash $(basename "${BASH_SOURCE[0]}") ${SUBJECT_ID} ${ERODE_STEPS}"
    exit 1
fi

DOCKER_MRI_DIR="$(to_docker_host_path "${MRI_DIR}")"
DOCKER_LICENSE_FILE="$(to_docker_host_path "${LICENSE_FILE}")"

DOCKER_USER_ARGS=()

if [[ "${PLATFORM}" == "git-bash" ]]; then
    # Windows bind mounts do not benefit from Linux UID/GID mapping.
    DOCKER_USER_ARGS=(--user root)
elif [[ "$(id -u)" -eq 0 ]]; then
    DOCKER_USER_ARGS=(--user root)
else
    # Avoid root-owned output files on Linux, macOS, and WSL.
    DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")
fi

echo "Run started: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Platform: ${PLATFORM}"
echo "Subject ID: ${SUBJECT_ID}"
echo "Erosion steps: ${ERODE_STEPS}"
echo "Erosion tag: ${EROSION_TAG}"
echo "Docker image: ${IMAGE}"
echo "Input labels: ${INPUT_LABELS}"
echo "MRI mount: ${DOCKER_MRI_DIR} -> /data"
echo "FreeSurfer license mount: ${DOCKER_LICENSE_FILE} -> /fs_license/license.txt"
echo "Final output: ${FINAL_LABELS}"
echo "Log file: ${LOG_FILE}"
echo "Provenance directory: ${PROVENANCE_DIR}"
echo

echo "Pulling Docker image if necessary: ${IMAGE}"
docker_safe pull "${IMAGE}"

echo
echo "Recording run configuration and provenance."

{
    echo "Run timestamp: ${RUN_TIMESTAMP}"
    echo "Platform: ${PLATFORM}"
    echo "Subject ID: ${SUBJECT_ID}"
    echo "Erosion steps: ${ERODE_STEPS}"
    echo "Erosion tag: ${EROSION_TAG}"
    echo "Docker image tag: ${IMAGE}"
    echo "Input labels: ${INPUT_LABELS}"
    echo "Final labels: ${FINAL_LABELS}"
    echo "FreeSurfer license file: ${LICENSE_FILE}"
    echo "Container license path: /fs_license/license.txt"
    echo
    echo "Host uname:"
    uname -a 2>&1 || true
} > "${PROVENANCE_DIR}/run_configuration.txt"

cp "${BASH_SOURCE[0]}" "${PROVENANCE_DIR}/script_used.sh"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${INPUT_LABELS}" > "${PROVENANCE_DIR}/input_sha256.txt"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${INPUT_LABELS}" > "${PROVENANCE_DIR}/input_sha256.txt"
else
    echo "SHA-256 utility not available." > "${PROVENANCE_DIR}/input_sha256.txt"
fi

docker_safe version > "${PROVENANCE_DIR}/docker_version.txt" 2>&1 || true
docker_safe image inspect "${IMAGE}" \
    > "${PROVENANCE_DIR}/docker_image_inspect.json" 2>&1 || true
docker_safe image inspect \
    --format '{{json .RepoDigests}}' \
    "${IMAGE}" > "${PROVENANCE_DIR}/docker_image_digests.json" 2>&1 || true

echo
echo "Creating and eroding the whole corpus callosum mask."

docker_safe run \
    --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -e FS_LICENSE=/fs_license/license.txt \
    -e ERODE_STEPS="${ERODE_STEPS}" \
    -e EROSION_TAG="${EROSION_TAG}" \
    -v "${DOCKER_MRI_DIR}:/data" \
    -v "${DOCKER_LICENSE_FILE}:/fs_license/license.txt:ro" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '
        set -Eeuo pipefail

        INPUT_LABELS="/data/callosum.CC.orig.mgz"
        EROSION_DIR="/data/cc_erosion"

        WHOLE_CC_MASK="${EROSION_DIR}/callosum.CC.whole.mask.mgz"
        WHOLE_CC_ERODED="${EROSION_DIR}/callosum.CC.whole.mask.${EROSION_TAG}.mgz"
        FINAL_LABELS="/data/callosum.CC.orig.${EROSION_TAG}.mgz"
        STATS_BEFORE="${EROSION_DIR}/callosum.CC.stats.before.txt"
        STATS_AFTER="${EROSION_DIR}/callosum.CC.stats.after.${EROSION_TAG}.txt"

        WORK_DIR="$(mktemp -d /tmp/fastsurfer_cc_erode.XXXXXX)"
        trap '\''rm -rf "${WORK_DIR}"'\'' EXIT

        TMP_WHOLE="${WORK_DIR}/callosum.CC.whole.mask.mgz"
        TMP_ERODED="${WORK_DIR}/callosum.CC.whole.mask.${EROSION_TAG}.mgz"
        TMP_FINAL="${WORK_DIR}/callosum.CC.orig.${EROSION_TAG}.mgz"
        TMP_STATS_BEFORE="${WORK_DIR}/callosum.CC.stats.before.txt"
        TMP_STATS_AFTER="${WORK_DIR}/callosum.CC.stats.after.${EROSION_TAG}.txt"

        mkdir -p "${EROSION_DIR}"

        if [[ ! -f "${INPUT_LABELS}" ]]; then
            echo "Error: input segmentation not found: ${INPUT_LABELS}"
            exit 1
        fi

        echo "Step 1/4: merging labels 251..255 into one binary CC mask."

        mri_binarize \
            --i "${INPUT_LABELS}" \
            --match 251 252 253 254 255 \
            --o "${TMP_WHOLE}"

        echo "Step 2/4: eroding only the outer boundary of the whole CC."

        mri_binarize \
            --i "${TMP_WHOLE}" \
            --min 0.5 \
            --erode "${ERODE_STEPS}" \
            --o "${TMP_ERODED}"

        echo "Step 3/4: restoring the original internal labels inside the eroded mask."

        mri_mask \
            "${INPUT_LABELS}" \
            "${TMP_ERODED}" \
            "${TMP_FINAL}"

        echo "Step 4/4: recording voxel counts for labels 251..255."

        mri_segstats \
            --seg "${INPUT_LABELS}" \
            --id 251 252 253 254 255 \
            --sum "${TMP_STATS_BEFORE}"

        mri_segstats \
            --seg "${TMP_FINAL}" \
            --id 251 252 253 254 255 \
            --sum "${TMP_STATS_AFTER}"

        TOTAL_AFTER="$(
            awk '\''!/^#/ {sum += $3} END {print sum + 0}'\'' \
                "${TMP_STATS_AFTER}"
        )"

        if [[ "${TOTAL_AFTER}" -le 0 ]]; then
            echo "Error: the whole corpus callosum became empty after erosion."
            exit 1
        fi

        cp "${TMP_WHOLE}" "${WHOLE_CC_MASK}"
        cp "${TMP_ERODED}" "${WHOLE_CC_ERODED}"
        cp "${TMP_FINAL}" "${FINAL_LABELS}"
        cp "${TMP_STATS_BEFORE}" "${STATS_BEFORE}"
        cp "${TMP_STATS_AFTER}" "${STATS_AFTER}"

        echo
        echo "Voxel statistics before erosion:"
        cat "${STATS_BEFORE}"

        echo
        echo "Voxel statistics after erosion:"
        cat "${STATS_AFTER}"

        echo
        echo "Created:"
        echo "  ${WHOLE_CC_MASK}"
        echo "  ${WHOLE_CC_ERODED}"
        echo "  ${FINAL_LABELS}"
    '

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${FINAL_LABELS}" > "${PROVENANCE_DIR}/output_sha256.txt"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${FINAL_LABELS}" > "${PROVENANCE_DIR}/output_sha256.txt"
fi

cp "${STATS_BEFORE}" "${PROVENANCE_DIR}/callosum.CC.stats.before.txt"
cp "${STATS_AFTER}" "${PROVENANCE_DIR}/callosum.CC.stats.after.${EROSION_TAG}.txt"

echo
echo "Erosion results:"
echo "  Whole CC binary mask:"
echo "    ${WHOLE_CC_MASK}"
echo
echo "  Eroded whole CC binary mask:"
echo "    ${WHOLE_CC_ERODED}"
echo
echo "  Final multi-label segmentation:"
echo "    ${FINAL_LABELS}"
echo
echo "Use this file as the CC source in the DWI extraction pipeline:"
echo "  ${FINAL_LABELS}"
