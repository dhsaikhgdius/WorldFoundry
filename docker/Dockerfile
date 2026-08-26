ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# Keep in sync with environment.yml (python=3.11) and infra plan D-05.
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.12.6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/worldfoundry-venv \
    UV_PYTHON=${PYTHON_VERSION}

RUN mkdir -p /workspace
WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        # Distro python3 stays available for apt tooling; the default
        # interpreter for WorldFoundry is ${PYTHON_VERSION} via uv below.
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
        iputils-ping \
        jq \
        less \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        nano \
        ninja-build \
        openssh-client \
        pkg-config \
        procps \
        rsync \
        screen \
        tmux \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

# Install a pinned CPython matching environment.yml; prefer it on PATH.
RUN uv python install "${PYTHON_VERSION}" \
    && PY="$(uv python find "${PYTHON_VERSION}")" \
    && ln -sf "${PY}" /usr/local/bin/python \
    && ln -sf "${PY}" /usr/local/bin/python3 \
    && python3 --version

CMD ["/bin/bash"]
