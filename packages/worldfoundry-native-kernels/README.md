# WorldFoundry native kernels

This optional package is deliberately separate from the main `worldfoundry`
wheel and from `[all]`. The main-package sdist intentionally prunes
`packages/worldfoundry-native-kernels` so ABI-qualified kernels cannot be
folded into a normal `pip install worldfoundry`.

NAT-00 contains only a C++ dispatcher DSO and the private
`worldfoundry_native::_build_info()` smoke operator; it does not contain a CUDA
kernel or change model execution.

Build it only inside the exact PyTorch environment being qualified. Build
isolation is intentionally unsupported because it could resolve a different
PyTorch ABI:

```bash
python -m pip install "scikit-build-core>=0.10,<2" build
python -m build --wheel --no-isolation packages/worldfoundry-native-kernels
```

Each Python/Torch/CUDA matrix entry must use a fresh CMake build directory.
Release installs intentionally disable binary stripping because the sidecar
hashes the final dispatcher DSO.

Importing `worldfoundry_native_kernels` is side-effect free. Prefer the control
plane helper first:

```python
from worldfoundry.core.kernels.native_provider import native_provider_status

status = native_provider_status(load=False)  # absent on CPU / when not installed
```

Call package `inspect()` to validate the sidecar and DSO hash without importing
PyTorch, then call `load()` once at an explicit prewarm boundary. `load()`
validates the complete Torch build string, Torch CUDA version, CXX11 ABI,
operator ABI, schema hash, build ID, and DSO hash before
`torch.ops.load_library()`.

CPU-only hosts should skip installing this package.

Public schemas, fake/meta implementations, CUDA sources, and model adapters are
added only in later promotion-gated milestones described in
`plan/cuda_cpp_operator_spec.md`.
