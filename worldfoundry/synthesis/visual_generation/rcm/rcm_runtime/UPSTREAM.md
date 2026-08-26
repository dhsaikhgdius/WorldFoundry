# Upstream provenance — Causal-rCM (`rcm_runtime`)

Vendored from [`NVlabs/rcm`](https://github.com/NVlabs/rcm) (Apache-2.0).
Checkpoint / code revision pins live in the `causal-rcm` runtime profile
(`worldfoundry/data/models/runtime/profiles/causal-rcm.yaml`).

File-level provenance and the upstream SPDX notice are recorded in
`THIRD_PARTY_NOTICES.md`. The Apache-2.0 text is duplicated in `LICENSE` so
packaged wheels always carry an explicit license file beside the runtime.

## Local WorldFoundry adapters

- `io_compat.py` is WorldFoundry-owned glue (not upstream).
- Launch / conda / checkpoint resolution stay in
  `worldfoundry/synthesis/visual_generation/rcm/worldfoundry_runtime.py`.
