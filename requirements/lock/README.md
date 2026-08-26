# Unified environment lockfiles (plan I-05)

Per-CUDA-tier lockfiles for `requirements/worldfoundry-unified.txt`.

| File | CUDA wheel index | I-03 constraint |
| --- | --- | --- |
| `worldfoundry-unified.cu121.lock.txt` | `https://download.pytorch.org/whl/cu121` | `requirements/cuda/cu121-torch.txt` |
| `worldfoundry-unified.cu124.lock.txt` | `https://download.pytorch.org/whl/cu124` | `requirements/cuda/cu124-torch.txt` |
| `worldfoundry-unified.cu128.lock.txt` | `https://download.pytorch.org/whl/cu128` | `requirements/cuda/cu128-torch.txt` |

## Generating a lock

Lock bodies are **never invented in-repo**. Generate them on a machine that
can resolve the CUDA torch index and PyPI:

```bash
make lock-unified TIER=cu128
# or
bash scripts/setup/compile_unified_lock.sh cu128
```

The compile passes the per-tier torch constraint stub (`requirements/cuda/`)
into `uv pip compile`, so a lock can never pin a torch stack outside the
`worldfoundry.runtime.cuda_tiers.TIER_TORCH_SPECS` matrix.

## Checks

- `make lock-check` (part of `make lint`) runs
  `scripts/setup/check_unified_lock.py`: offline, verifies each lock's header
  names the right index, populated locks keep the torch stack `==`-pinned
  inside the I-03 bounds, and the CLIP git dependency stays pinned to a commit
  SHA.
- CI runs `bash scripts/setup/compile_unified_lock.sh cu128 --check`: while a
  lock is still a placeholder this is a fast no-op; once a real lock body is
  committed it recompiles and fails on drift.

## Installer behaviour

`scripts/setup/conda_install.sh` prefers the lock for the resolved CUDA tier
when the file contains resolved requirements (non-comment lines). Placeholder
locks fall back to the unconstrained `requirements/worldfoundry-unified.txt`.
Pass `--unlocked` to `conda_install.sh` / `unified_install.sh` (or set
`WORLDFOUNDRY_INSTALL_UNLOCKED=1`) to force the unlocked path.

## CLIP git pin

`worldfoundry-unified.txt` pins OpenAI CLIP to commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (resolved via
`git ls-remote https://github.com/openai/CLIP.git HEAD` on 2026-08-26).
Refresh intentionally when bumping CLIP.
