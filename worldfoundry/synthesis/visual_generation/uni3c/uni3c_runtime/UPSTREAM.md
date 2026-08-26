# Upstream provenance — Uni3C (`uni3c_runtime`)

Vendored from [`alibaba-damo-academy/Uni3C`](https://github.com/alibaba-damo-academy/Uni3C)
at the revision recorded in `worldfoundry/data/models/runtime/profiles/uni3c.yaml`
(`source_repos[0].revision`, currently `75ed6e2180316b7f07398e4e88ea8bdba3e6970c`).

Upstream license: Apache-2.0 (`LICENSE` in this directory).

## Checkpoints (not vendored)

Resolved via the Uni3C runtime profile / HFD layout, including:

- `ewrfcas/Uni3C` (PCD controller)
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` (camera base)
- `theFoxofSky/RealisDance-DiT` (unified camera + human motion)

## Local WorldFoundry adapters

High-level synthesis / Studio wiring lives outside this directory
(`uni3c_synthesis.py` and catalog/pipeline bindings).
