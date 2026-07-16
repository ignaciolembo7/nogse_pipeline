#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Portable FastSurfer-CC runner for Linux, macOS, WSL, and Git Bash on Windows.
#
# Usage:
#   bash run_fastsurfer_cc_portable.sh SUBJECT_ID T1_FILENAME [THREADS]
#
# Example:
#   bash run_fastsurfer_cc_portable.sh BRAIN-1 BRAIN-1.nii.gz 8
#
# Expected folder structure:
#   <script_directory>/
#   ├── run_fastsurfer_cc_portable.sh
#   └── FASTSURFER_CC/
#       ├── license.txt
#       └── IN/
#           └── <T1_FILENAME>
#
# Optional environment variables:
#   IMAGE        Docker image to use.
#   PROJECT_DIR  Root data directory. Defaults to FASTSURFER_CC next to script.
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '3,25p' "${BASH_SOURCE[0]}"
    exit 0
fi

SUBJECT_ID="${1:-${SUBJECT_ID:-BRAIN-1}}"
T1_FILENAME="${2:-${T1_FILENAME:-BRAIN-1.nii.gz}}"
THREADS="${3:-${THREADS:-8}}"
IMAGE="${IMAGE:-deepmi/fastsurfer:cpu-v2.5.4}"

if [[ ! "${SUBJECT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: SUBJECT_ID may contain only letters, numbers, dots, underscores, and hyphens."
    exit 1
fi

if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: THREADS must be a positive integer."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/FASTSURFER_CC}"
INPUT_DIR="${PROJECT_DIR}/IN"
OUTPUT_DIR="${PROJECT_DIR}/OUT"
LOG_DIR="${PROJECT_DIR}/LOGS"
PROVENANCE_ROOT="${PROJECT_DIR}/PROVENANCE"
LICENSE_FILE="${PROJECT_DIR}/license.txt"

RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/${SUBJECT_ID}_${RUN_TIMESTAMP}.log"
PROVENANCE_DIR="${PROVENANCE_ROOT}/${SUBJECT_ID}/${RUN_TIMESTAMP}"
INPUT_FILE="${INPUT_DIR}/${T1_FILENAME}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"

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

# Git Bash automatically rewrites arguments beginning with "/" before passing
# them to Windows executables. That behavior is useful for host paths but wrong
# for Linux paths inside Docker containers, such as /data and /output.
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

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: T1 image not found:"
    echo "  ${INPUT_FILE}"
    exit 1
fi

if [[ ! -f "${LICENSE_FILE}" ]]; then
    echo "Error: FreeSurfer license not found:"
    echo "  ${LICENSE_FILE}"
    echo
    echo "Place the license file at:"
    echo "  ${PROJECT_DIR}/license.txt"
    exit 1
fi

DOCKER_INPUT_DIR="$(to_docker_host_path "${INPUT_DIR}")"
DOCKER_OUTPUT_DIR="$(to_docker_host_path "${OUTPUT_DIR}")"
DOCKER_LICENSE_FILE="$(to_docker_host_path "${LICENSE_FILE}")"

DOCKER_USER_ARGS=()
FASTSURFER_USER_ARGS=()

if [[ "${PLATFORM}" == "git-bash" ]]; then
    # Windows bind mounts do not benefit from Linux UID/GID mapping.
    DOCKER_USER_ARGS=(--user root)
    FASTSURFER_USER_ARGS=(--allow_root)
elif [[ "$(id -u)" -eq 0 ]]; then
    DOCKER_USER_ARGS=(--user root)
    FASTSURFER_USER_ARGS=(--allow_root)
else
    # Avoid root-owned output files on Linux, macOS, and WSL.
    DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")
fi

echo "Run started: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Platform: ${PLATFORM}"
echo "Subject ID: ${SUBJECT_ID}"
echo "Input image: ${INPUT_FILE}"
echo "Docker image: ${IMAGE}"
echo "Threads: ${THREADS}"
echo "Docker input mount: ${DOCKER_INPUT_DIR} -> /data"
echo "Docker output mount: ${DOCKER_OUTPUT_DIR} -> /output"
echo "FreeSurfer license mount: ${DOCKER_LICENSE_FILE} -> /fs_license/license.txt"
echo "Log file: ${LOG_FILE}"
echo "Provenance directory: ${PROVENANCE_DIR}"
echo

echo "Pulling Docker image if necessary: ${IMAGE}"
docker_safe pull "${IMAGE}"

echo
echo "Recording run configuration and host provenance."

{
    echo "Run timestamp: ${RUN_TIMESTAMP}"
    echo "Platform: ${PLATFORM}"
    echo "Subject ID: ${SUBJECT_ID}"
    echo "T1 filename: ${T1_FILENAME}"
    echo "Input file: ${INPUT_FILE}"
    echo "Threads: ${THREADS}"
    echo "Docker image tag: ${IMAGE}"
    echo "Docker input mount: ${DOCKER_INPUT_DIR}"
    echo "Docker output mount: ${DOCKER_OUTPUT_DIR}"
    echo "FreeSurfer license file: ${LICENSE_FILE}"
    echo "Container license path: /fs_license/license.txt"
    echo
    echo "Host uname:"
    uname -a 2>&1 || true
} > "${PROVENANCE_DIR}/run_configuration.txt"

cp "${BASH_SOURCE[0]}" "${PROVENANCE_DIR}/script_used.sh"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${INPUT_FILE}" > "${PROVENANCE_DIR}/input_sha256.txt"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${INPUT_FILE}" > "${PROVENANCE_DIR}/input_sha256.txt"
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
echo "Recording package versions from inside the container."

if ! docker_safe run \
    --rm \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '
        set +e

        echo "===== OPERATING SYSTEM ====="
        cat /etc/os-release 2>/dev/null

        echo
        echo "===== FASTSURFER VERSION FILES ====="
        for file in \
            /fastsurfer/VERSION \
            /fastsurfer/version.txt \
            /fastsurfer/BUILD.info \
            /fastsurfer/BUILD_INFO; do
            if [[ -f "${file}" ]]; then
                echo "--- ${file} ---"
                cat "${file}"
            fi
        done

        if [[ -d /fastsurfer/.git ]] && command -v git >/dev/null 2>&1; then
            echo
            echo "===== FASTSURFER GIT COMMIT ====="
            git -C /fastsurfer rev-parse HEAD
        fi

        echo
        echo "===== PYTHON ====="
        PYTHON_BIN="$(command -v python3 || command -v python || true)"
        if [[ -n "${PYTHON_BIN}" ]]; then
            "${PYTHON_BIN}" --version 2>&1
            "${PYTHON_BIN}" -m pip --version 2>&1

            echo
            echo "===== PYTHON PACKAGES ====="
            "${PYTHON_BIN}" -m pip freeze 2>&1
        else
            echo "Python executable not found."
        fi

        echo
        echo "===== CONDA PACKAGES ====="
        if command -v conda >/dev/null 2>&1; then
            conda list 2>&1
        else
            echo "Conda not found."
        fi

        echo
        echo "===== SYSTEM PACKAGES ====="
        if command -v dpkg-query >/dev/null 2>&1; then
            dpkg-query -W 2>&1
        elif command -v rpm >/dev/null 2>&1; then
            rpm -qa 2>&1
        else
            echo "No supported system package manager found."
        fi
    ' > "${PROVENANCE_DIR}/container_packages.txt" 2>&1
then
    echo "Warning: package inventory failed; segmentation will continue."
    echo "See: ${PROVENANCE_DIR}/container_packages.txt"
fi

echo
echo "Running FastSurfer-CC for subject: ${SUBJECT_ID}"

docker_safe run \
    --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${DOCKER_INPUT_DIR}:/data:ro" \
    -v "${DOCKER_OUTPUT_DIR}:/output" \
    -v "${DOCKER_LICENSE_FILE}:/fs_license/license.txt:ro" \
    "${IMAGE}" \
    --t1 "/data/${T1_FILENAME}" \
    --sid "${SUBJECT_ID}" \
    --sd /output \
    --fs_license /fs_license/license.txt \
    --seg_only \
    --no_cereb \
    --no_hypothal \
    --device cpu \
    --threads "${THREADS}" \
    --qc_snap \
    "${FASTSURFER_USER_ARGS[@]}"

echo
echo "Creating skull-stripped bias-corrected T1 image."

docker_safe run \
    --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -e SUBJECT_ID="${SUBJECT_ID}" \
    -e FS_LICENSE=/fs_license/license.txt \
    -v "${DOCKER_OUTPUT_DIR}:/output" \
    -v "${DOCKER_LICENSE_FILE}:/fs_license/license.txt:ro" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '
        set -euo pipefail

        MRI_DIR="/output/${SUBJECT_ID}/mri"
        INPUT_IMAGE="${MRI_DIR}/orig_nu.mgz"
        BRAIN_MASK="${MRI_DIR}/mask.mgz"
        OUTPUT_IMAGE="${MRI_DIR}/orig_nu_brain.mgz"

        if [[ ! -f "${INPUT_IMAGE}" ]]; then
            echo "Error: bias-corrected image not found: ${INPUT_IMAGE}"
            exit 1
        fi

        if [[ ! -f "${BRAIN_MASK}" ]]; then
            echo "Error: brain mask not found: ${BRAIN_MASK}"
            exit 1
        fi

        mri_mask "${INPUT_IMAGE}" "${BRAIN_MASK}" "${OUTPUT_IMAGE}"

        if [[ ! -f "${OUTPUT_IMAGE}" ]]; then
            echo "Error: skull-stripped image was not created: ${OUTPUT_IMAGE}"
            exit 1
        fi

        echo "Created: ${OUTPUT_IMAGE}"
    '

echo
echo "Subject results:"
echo "  ${OUTPUT_DIR}/${SUBJECT_ID}"
echo
echo "Bias-corrected skull-stripped T1:"
echo "  ${OUTPUT_DIR}/${SUBJECT_ID}/mri/orig_nu_brain.mgz"
echo
echo "Primary corpus callosum segmentation:"
echo "  ${OUTPUT_DIR}/${SUBJECT_ID}/mri/callosum.CC.orig.mgz"
echo
echo "QC snapshots:"
echo "  ${OUTPUT_DIR}/${SUBJECT_ID}/qc_snapshots"
echo
echo "Version and provenance files:"
echo "  ${PROVENANCE_DIR}"
