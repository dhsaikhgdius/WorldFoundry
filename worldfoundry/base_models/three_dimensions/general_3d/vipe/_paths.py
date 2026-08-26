# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> three_dimensions -> general_3d -> vipe -> _paths.py functionality."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.utils import worldfoundry_data_path


def get_config_path() -> Path:
    """Get config path.

    Returns:
        The return value.
    """
    data_configs = worldfoundry_data_path("models", "runtime", "configs", "vipe")
    if (data_configs / "default.yaml").is_file():
        return data_configs
    # Prefer the in-package ViPE config tree when packaged data is absent.
    repo_configs = Path(__file__).resolve().parent / "configs"
    if repo_configs.is_dir():
        return repo_configs
    return Path(__file__).resolve().parent / "_configs"
