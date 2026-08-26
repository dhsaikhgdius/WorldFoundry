#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERIFY_SCRIPT="${ROOT}/scripts/setup/verify_flash_attn.py"
source "${ROOT}/scripts/setup/conda_utils.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup/install_flash_attn.sh flash_attn_fa25
  bash scripts/setup/install_flash_attn.sh flash_attn_fa28

Behavior:
  - validates the currently installed flash-attn in the WorldFoundry conda env first
  - reinstalls if the import or GPU kernel check fails
  - maps buckets to version pins: fa25 -> flash-attn>=2.5,<2.6 ; fa28 -> >=2.8,<2.9
  - prefers official prebuilt wheels; set WORLDFOUNDRY_FLASH_ATTN_FORCE_BUILD=1 to
    skip wheels and build from source
  - builds with CUDA_HOME pointing at the conda env prefix (nvcc from cuda-nvcc)
  - uses WORLDFOUNDRY_CONDA_ENV_PREFIX or WORLDFOUNDRY_CONDA_ENV_NAME to select the env
EOF
}

if (($# == 0)); then
  usage >&2
  exit 1
fi

BUCKET=""
while (($#)); do
  case "$1" in
    flash_attn_fa25|flash_attn_fa28)
      if [[ -n "$BUCKET" ]]; then
        echo "Only one flash-attn bucket may be specified." >&2
        usage >&2
        exit 2
      fi
      BUCKET="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "Missing flash-attn bucket." >&2
  usage >&2
  exit 2
fi

case "$BUCKET" in
  flash_attn_fa25) FLASH_ATTN_SPEC="${WORLDFOUNDRY_FLASH_ATTN_SPEC:-flash-attn>=2.5,<2.6}" ;;
  flash_attn_fa28) FLASH_ATTN_SPEC="${WORLDFOUNDRY_FLASH_ATTN_SPEC:-flash-attn>=2.8,<2.9}" ;;
  *)
    echo "Unsupported flash-attn bucket: ${BUCKET}" >&2
    exit 2
    ;;
esac

if ! command -v "${CONDA_EXE_PATH}" >/dev/null 2>&1; then
  echo "conda executable not found. Install Miniconda/Anaconda or set CONDA_EXE." >&2
  exit 1
fi

resolve_cuda_home() {
  if [[ -n "${WORLDFOUNDRY_CONDA_ENV_PREFIX:-}" && -d "${WORLDFOUNDRY_CONDA_ENV_PREFIX}" ]]; then
    printf '%s' "${WORLDFOUNDRY_CONDA_ENV_PREFIX}"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}" ]]; then
    printf '%s' "${CONDA_PREFIX}"
    return 0
  fi
  printf '%s' "${CUDA_HOME:-/usr/local/cuda}"
}

pip_trusted_host_args() {
  local host
  if [[ -z "${WORLDFOUNDRY_PIP_TRUSTED_HOST:-}" ]]; then
    return 0
  fi
  for host in ${WORLDFOUNDRY_PIP_TRUSTED_HOST}; do
    printf '%s\n' --trusted-host "${host}"
  done
}

verify_current() {
  worldfoundry_conda_run python "${VERIFY_SCRIPT}"
}

if [[ "${WORLDFOUNDRY_FORCE_FLASH_ATTN_REINSTALL:-0}" != "1" ]] && verify_current; then
  echo "flash-attn is already healthy; skipping reinstall."
  exit 0
fi

CUDA_HOME_VALUE="$(resolve_cuda_home)"
TRUSTED_HOST_ARGS=()
mapfile -t TRUSTED_HOST_ARGS < <(pip_trusted_host_args)

install_wheel() {
  echo "Trying prebuilt flash-attn wheel for ${BUCKET} (${FLASH_ATTN_SPEC})."
  worldfoundry_conda_pip uninstall -y flash-attn >/dev/null 2>&1 || true
  CUDA_HOME="${CUDA_HOME_VALUE}" \
  worldfoundry_conda_pip install \
    "${TRUSTED_HOST_ARGS[@]}" \
    --no-deps \
    --only-binary=:all: \
    --no-cache-dir \
    --force-reinstall \
    "${FLASH_ATTN_SPEC}"
}

install_from_source() {
  echo "Installing flash-attn from source for ${BUCKET} (${FLASH_ATTN_SPEC})."
  echo "CUDA_HOME=${CUDA_HOME_VALUE}"
  worldfoundry_conda_pip uninstall -y flash-attn >/dev/null 2>&1 || true
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  CUDA_HOME="${CUDA_HOME_VALUE}" \
  MAX_JOBS="${MAX_JOBS:-8}" \
  worldfoundry_conda_pip install \
    "${TRUSTED_HOST_ARGS[@]}" \
    --no-deps \
    --no-build-isolation \
    --no-binary flash-attn \
    --no-cache-dir \
    --force-reinstall \
    "${FLASH_ATTN_SPEC}"
}

FORCE_BUILD="${WORLDFOUNDRY_FLASH_ATTN_FORCE_BUILD:-0}"
if [[ "${FORCE_BUILD}" == "1" ]]; then
  install_from_source
elif install_wheel && verify_current; then
  echo "flash-attn wheel install verified for ${BUCKET}."
  exit 0
else
  echo "Prebuilt flash-attn wheel unavailable or failed verification; falling back to source build."
  install_from_source
fi

verify_current
