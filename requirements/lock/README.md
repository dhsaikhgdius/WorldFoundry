# Unified environment lockfiles

Per-CUDA-tier lockfiles for `requirements/worldfoundry-unified.txt` (plan I-05).

## Files

| File | CUDA wheel index |
| --- | --- |
| `worldfoundry-unified.cu121.lock.txt` | `https://download.pytorch.org/whl/cu121` |
| `worldfoundry-unified.cu124.lock.txt` | `https://download.pytorch.org/whl/cu124` |
| `worldfoundry-unified.cu128.lock.txt` | `https://download.pytorch.org/whl/cu128` |

As of 2026-08-26, `worldfoundry-unified.cu128.lock.txt` is populated via
`uv pip compile` against the cu128 torch index. `cu121` / `cu124` remain
placeholders until a real compile is run on a machine that can resolve those
CUDA indexes. `optimized_core` / `all` extras now use `torch>=2.4,<2.12.0`
so they intersect cu121/cu124/cu128 floors (I-03); compiles still pass
`requirements/cuda/<tier>-torch.txt` as `--constraint` and post-compile
validation checks the CUDA index listing. Do **not** invent placeholder lock
bodies.

Lock bodies are **not** invented in-repo. Generate them on a machine that can
resolve the CUDA torch index:

```bash
make lock-unified TIER=cu128
# or
bash scripts/setup/compile_unified_lock.sh cu128

# Drift check (recompiles; needs network + uv):
make check-unified-lock TIER=cu128
```

Installers prefer the lock for the resolved CUDA tier when the file contains
resolved requirements (lines that are not comments). Pass `--unlocked` to
`conda_install.sh` / `unified_install.sh` to force the unconstrained
`requirements/worldfoundry-unified.txt`.

## CLIP git pin

`worldfoundry-unified.txt` pins OpenAI CLIP to commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (resolved via `git ls-remote` on
2026-08-26). Refresh intentionally when bumping CLIP.
