# WorldScore in-tree runtime provenance

`worldscore/` is a vendored copy of the official WorldScore benchmark code (haoyi-duan/WorldScore, MIT) so the metric suite can run in tree.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/haoyi-duan/WorldScore |
| Pinned revision | `344983c1be7d515c1e4ecb167d40d622adf502bc` |
| Paper | https://arxiv.org/abs/2504.00983 |
| Project page | https://haoyi-duan.github.io/WorldScore/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/worldscore.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/worldscore.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- `--run-official` style execution computes WorldScore metrics locally; the metric stack needs its model checkpoints and a GPU (a bounded single-video GPU validation is recorded in the catalog; the full suite is not claimed).
- Result normalization from existing official outputs is pure-local.

## Not vendored

- Metric model checkpoints downloaded per the WorldScore requirements.
- The WorldScore dataset (MIT-licensed Hugging Face release).

## License status

Upstream repository is MIT at pinned revision `344983c1be7d515c1e4ecb167d40d622adf502bc`.
