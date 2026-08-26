# WorldFoundry container images (plan D-07 / D-05 / D-10).
#
# Targets:
#   base-runtime  cudnn-runtime + minimal apt + Python 3.11 (default run image)
#   base-devel    cudnn-devel + build toolchain (flash-attn / native kernels)
#   dev           base-devel + interactive toolbox (nano/tmux/ping) — D-10
#   cpu           python:3.11-slim for CPU-only CI / light tooling
#
# Default final stage is base-devel so existing `docker build -f docker/Dockerfile`
# and `build_with_docker.sh` keep producing a compile-capable image.
#
# When this context later COPYs the repo, set SETUPTOOLS_SCM_PRETEND_VERSION
# if .git is excluded by .dockerignore.

ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# Backward-compatible alias used by older build scripts.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.12.6

# ---------------------------------------------------------------------------
# base-runtime: lean GPU runtime (no nvcc / no interactive toolbox)
# ---------------------------------------------------------------------------
FROM ${CUDA_RUNTIME_IMAGE} AS base-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG UV_VERSION

LABEL org.opencontainers.image.source="https://github.com/OpenEnvision/WorldFoundry" \
      org.opencontainers.image.title="worldfoundry-base-runtime" \
      org.opencontainers.image.description="WorldFoundry CUDA runtime base (cudnn-runtime)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/worldfoundry-venv \
    UV_PYTHON=${PYTHON_VERSION}

WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        procps \
        python3 \
        python3-venv \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

RUN uv python install "${PYTHON_VERSION}" \
    && PY="$(uv python find "${PYTHON_VERSION}")" \
    && ln -sf "${PY}" /usr/local/bin/python \
    && ln -sf "${PY}" /usr/local/bin/python3 \
    && python3 --version

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# base-devel: full toolchain for compiling flash-attn / native extensions
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS base-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG UV_VERSION

LABEL org.opencontainers.image.source="https://github.com/OpenEnvision/WorldFoundry" \
      org.opencontainers.image.title="worldfoundry-base-devel" \
      org.opencontainers.image.description="WorldFoundry CUDA devel base (cudnn-devel + build tools)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/worldfoundry-venv \
    UV_PYTHON=${PYTHON_VERSION}

WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        bash \
        bash-completion \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        gcc \
        git \
        git-lfs \
        iproute2 \
        jq \
        less \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        openssh-client \
        pkg-config \
        procps \
        rsync \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

RUN uv python install "${PYTHON_VERSION}" \
    && PY="$(uv python find "${PYTHON_VERSION}")" \
    && ln -sf "${PY}" /usr/local/bin/python \
    && ln -sf "${PY}" /usr/local/bin/python3 \
    && python3 --version

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# dev: interactive toolbox on top of base-devel (plan D-10)
# ---------------------------------------------------------------------------
FROM base-devel AS dev

ARG DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source="https://github.com/OpenEnvision/WorldFoundry" \
      org.opencontainers.image.title="worldfoundry-dev" \
      org.opencontainers.image.description="WorldFoundry devel image + interactive toolbox"

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        iputils-ping \
        nano \
        screen \
        tmux \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# cpu: slim interpreter for CPU CI / docs tooling (no CUDA)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS cpu

ARG UV_VERSION=0.12.6

LABEL org.opencontainers.image.source="https://github.com/OpenEnvision/WorldFoundry" \
      org.opencontainers.image.title="worldfoundry-cpu" \
      org.opencontainers.image.description="WorldFoundry CPU-only slim base"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/worldfoundry-venv

WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

CMD ["/bin/bash"]

# Default export: devel (backward compatible with existing :base builds).
FROM base-devel
