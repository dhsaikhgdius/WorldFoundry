#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash docker/build_with_docker.sh [options] IMAGE[:TAG] [IMAGE[:TAG] ...]

Build the WorldFoundry CUDA base image with Docker Buildx.

Options:
  --push                 Push the image and manifest to the registry.
  --load                 Load a single-platform image into the local Docker daemon.
  --platform PLATFORMS   Build platform list. Default: linux/amd64 for --load,
                         linux/amd64,linux/arm64 for --push.
  --cuda-image IMAGE     Base CUDA image. Default:
                         nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04.
  -h, --help             Show this help.

Environment overrides:
  WORLDFOUNDRY_DOCKER_PUSH=1
  WORLDFOUNDRY_DOCKER_PLATFORMS=linux/amd64,linux/arm64
  WORLDFOUNDRY_DOCKER_CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
  WORLDFOUNDRY_DOCKER_BUILD_HOST_NETWORK=1
                         Opt into BuildKit host networking (off by default).

Examples:
  bash docker/build_with_docker.sh worldfoundry:dev
  bash docker/build_with_docker.sh --push ghcr.io/openenvision/worldfoundry:base
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PUSH="${WORLDFOUNDRY_DOCKER_PUSH:-0}"
PLATFORMS="${WORLDFOUNDRY_DOCKER_PLATFORMS:-}"
CUDA_IMAGE="${WORLDFOUNDRY_DOCKER_CUDA_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"
HOST_NETWORK="${WORLDFOUNDRY_DOCKER_BUILD_HOST_NETWORK:-0}"
TAGS=()

while (($#)); do
  case "$1" in
    --push)
      PUSH=1
      shift
      ;;
    --load)
      PUSH=0
      shift
      ;;
    --platform)
      PLATFORMS="$2"
      shift 2
      ;;
    --cuda-image)
      CUDA_IMAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      TAGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#TAGS[@]}" -eq 0 ]]; then
  echo "Error: at least one image tag is required." >&2
  usage >&2
  exit 2
fi

if [[ -z "${PLATFORMS}" ]]; then
  if [[ "${PUSH}" == "1" ]]; then
    PLATFORMS="linux/amd64,linux/arm64"
  else
    PLATFORMS="linux/amd64"
  fi
fi

ACTION_ARGS=()
if [[ "${PUSH}" == "1" ]]; then
  ACTION_ARGS+=(--push)
else
  if [[ "${PLATFORMS}" == *,* ]]; then
    echo "Error: --load supports only one platform. Use --push for multi-platform builds." >&2
    exit 2
  fi
  ACTION_ARGS+=(--load)
fi

TAG_ARGS=()
for tag in "${TAGS[@]}"; do
  TAG_ARGS+=(-t "${tag}")
done

NETWORK_ARGS=()
if [[ "${HOST_NETWORK}" == "1" ]]; then
  NETWORK_ARGS+=(--allow network.host --network host)
fi

docker buildx build \
  --platform "${PLATFORMS}" \
  "${NETWORK_ARGS[@]}" \
  --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
  "${ACTION_ARGS[@]}" \
  "${TAG_ARGS[@]}" \
  -f docker/Dockerfile .
