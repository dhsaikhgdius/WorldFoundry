# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compatibility exports for WorldFoundry's shared attention implementation."""

from worldfoundry.core.attention.varlen import attention, flash_attention

__all__ = ["attention", "flash_attention"]
