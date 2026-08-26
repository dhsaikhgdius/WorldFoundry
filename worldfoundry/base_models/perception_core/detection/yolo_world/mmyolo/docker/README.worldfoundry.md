# NOT a supported WorldFoundry build path

The Dockerfiles in this vendored tree are upstream `open-mmlab/mmyolo`
artifacts, kept verbatim as part of the YOLO-World snapshot (provenance:
`../../UPSTREAM.md`):

- `docker/Dockerfile`
- `docker/Dockerfile_deployment`
- `../.circleci/docker/Dockerfile`

WorldFoundry does **not** build, lint, publish, or otherwise support these
images. They are excluded from the repo's Docker gates (hadolint /
docker-smoke cover `docker/**` at the repository root only) and are known to
be stale (PyTorch 1.9 / CUDA 11.1 base, ubuntu1804 apt keyrings). The
supported container entry point is the repo-root `docker/Dockerfile`, built
via `docker/build_with_docker.sh`.

Do not modify these files to "fix" them: keeping the upstream diff clean is
intentional so the snapshot stays comparable with upstream (see
`plan/infra_optimization_plan.txt`, section 1 boundaries). If a working mmyolo
image is ever needed, build it from upstream directly instead of patching
this copy.
