# Physics-IQ Original in-tree runtime provenance

`official/` adapts the google-deepmind physics-IQ-benchmark scoring code (mask/IoU/MSE raw metrics and the Original + Verified score adapters) for in-tree execution; the surrounding modules are WorldFoundry glue. Serves both the `physics-iq` and `physics-iq-verified` catalog entries.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/google-deepmind/physics-IQ-benchmark |
| Pinned revision | `b02cf26dc15d559d0ca4f63a6917070312dde185` |
| Paper | https://arxiv.org/abs/2501.09038 |
| Project page | https://physics-iq.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/physics-iq.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/physics-iq.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- `--run-official` computes raw metrics and official aggregation locally from downloaded official media plus generated videos; no hosted judge is involved.
- The official media (reference videos and masks) must be downloaded separately and are never vendored.
- Official-validation from an existing `raw_metrics.csv` is pure-local and covered by tests.

## Not vendored

- Physics-IQ official media (Original) and the Physics-IQ-Verified Hugging Face dataset.

## License status

Upstream code is published by google-deepmind at pinned revision `b02cf26dc15d559d0ca4f63a6917070312dde185`; the repository is Apache-2.0 upstream — retain notices when redistributing.
