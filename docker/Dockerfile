ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/worldfoundry-venv

RUN mkdir -p /workspace
WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        # Note: distro python3 may be 3.10; conda/env.yml targets 3.11 for the unified runtime.
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

ARG UV_VERSION=0.12.6
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/

CMD ["/bin/bash"]
