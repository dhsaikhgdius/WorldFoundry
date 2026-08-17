# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Facebook, Inc. and its affiliates.
from omegaconf import DictConfig, OmegaConf

from worldfoundry.core.configuration.lazy_config.config import LazyConfig
from worldfoundry.core.configuration.lazy_config.instantiate import instantiate
from worldfoundry.core.configuration.lazy_config.lazy_call import LazyCall
from worldfoundry.core.configuration.lazy_config.omegaconf_patch import to_object

# WARNING (CF-3): process-wide monkey-patch. Importing this package replaces
# ``OmegaConf.to_object`` for *every* caller in the process: DictConfigs that
# contain ``_target_`` are returned unchanged instead of converted to plain
# containers (all other inputs keep stock semantics via ``to_container``).
# Vendored model code built against this behavior may silently depend on it,
# so the patch cannot be dropped without an audit; ``instantiate`` in this
# package already calls the local ``to_object`` directly and does not need
# the patch. New code should do the same rather than rely on the global.
OmegaConf.to_object = to_object

PLACEHOLDER = None


class LazyDict(DictConfig):  # NOTE: to differentiate between LazyDict & DictConfig
    """Marker subclass for editable, lazily instantiated object graphs.

    It behaves like OmegaConf ``DictConfig`` but lets WorldFoundry distinguish
    a deferred constructor tree from an ordinary runtime mapping. Construct it
    through ``LazyCall`` in normal code.
    """

    def __init__(self, *args, **kwargs):
        """Forward content and OmegaConf flags to ``DictConfig``."""
        super().__init__(*args, **kwargs)


__all__ = ["instantiate", "LazyCall", "LazyConfig", "PLACEHOLDER", "LazyDict"]
