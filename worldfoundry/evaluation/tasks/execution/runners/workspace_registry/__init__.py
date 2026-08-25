"""Workspace-facing dispatch for in-tree benchmark runners."""

from .dispatch import *
from .specs import CLI_RUNNERS, GENERIC_EVALUATION_METRICS, RESULT_SUFFIXES, WorkspaceRunnerSpec

__all__ = [name for name in globals() if not name.startswith('_')]
