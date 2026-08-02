"""Optional WorldFoundry native-kernel package.

Package import is side-effect free: call :func:`inspect` to validate package
files or :func:`load` at an explicit prewarm boundary to import PyTorch and
load the ABI-qualified dispatcher DSO.
"""

from ._loader import (
    MANIFEST_SCHEMA_VERSION,
    OPERATOR_ABI_VERSION,
    OPERATOR_SCHEMA_HASH,
    NativeKernelCompatibilityError,
    NativeKernelManifestError,
    NativeKernelPackageError,
    inspect,
    is_loaded,
    load,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "OPERATOR_ABI_VERSION",
    "OPERATOR_SCHEMA_HASH",
    "NativeKernelCompatibilityError",
    "NativeKernelManifestError",
    "NativeKernelPackageError",
    "inspect",
    "is_loaded",
    "load",
]
