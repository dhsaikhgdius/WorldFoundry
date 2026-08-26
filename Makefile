.PHONY: help install-core install-dev docs-check lint ruff-check ruff-format-check syntax-check shell-check data-check runtime-registry-check check-cuda-constraints lock-unified lock-check compile-eval cli-check precommit precommit-install preflight test-eval-core test-training open-source-infer-repro

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
RELEASE_HFD_ROOT ?= $(if $(WORLDFOUNDRY_HFD_ROOT),$(WORLDFOUNDRY_HFD_ROOT),$(HOME)/.cache/worldfoundry/models/checkpoints/hfd)
CANONICAL_DIFFUSION_SOURCES ?= \
	worldfoundry/base_models/diffusion_model/*.py \
	worldfoundry/base_models/diffusion_model/extensions \
	worldfoundry/base_models/diffusion_model/loaders \
	worldfoundry/base_models/diffusion_model/models \
	worldfoundry/base_models/diffusion_model/optimizations \
	worldfoundry/base_models/diffusion_model/recipes \
	worldfoundry/base_models/diffusion_model/runners \
	worldfoundry/base_models/diffusion_model/schedulers
RUFF_SOURCES ?= \
	worldfoundry/cli \
	worldfoundry/evaluation/api \
	worldfoundry/evaluation/models/runtime \
	worldfoundry/evaluation/tasks/catalog \
	worldfoundry/evaluation/tasks/execution/orchestration \
	worldfoundry/mcp \
	worldfoundry/runtime \
	scripts/benchmark_zoo \
	scripts/model_zoo

help:
	@printf '%s\n' \
		'WorldFoundry development targets:' \
		'  make install-core      Install the editable core package.' \
		'  make install-dev       Install lightweight development dependencies.' \
		'  make docs-check        Validate documented CLI entrypoints.' \
		'  make lint              Run lightweight source and catalog checks.' \
		'  make preflight         Run the public runtime preflight.' \
		'  make test-eval-core    Run the eval_core release-gate pytest suite (CPU).' \
		'  make test-training     Run the tests/training pytest suite (CPU subset).' \
		'  make check-cuda-constraints  Dry-run: verify per-tier torch constraint stubs.' \
		'  make lock-unified TIER=cu128  Compile the per-tier unified lockfile (needs uv + network).' \
		'  make lock-check        Offline unified-lock scaffolding consistency check.'

install-core:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

docs-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli --help >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

lint: ruff-check syntax-check shell-check data-check runtime-registry-check lock-check

ruff-check:
	$(PYTHON) -m ruff check $(RUFF_SOURCES)

# Optional until RUFF_SOURCES are format-clean (many pre-existing diffs on main).
ruff-format-check:
	$(PYTHON) -m ruff format --check $(RUFF_SOURCES)

# Byte-compile the canonical diffusion package sources only (plan C-05).
# Formerly misnamed format-check.
syntax-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q $(CANONICAL_DIFFUSION_SOURCES)

# Compatibility alias for older CI/docs that still call format-check.
format-check: syntax-check

shell-check:
	# D-08: cover docker / embodied / test / fumadocs / scripts/dev, not only setup.
	find scripts/setup docker scripts/embodied test scripts/dev docs/fumadocs/scripts \
		-type f -name '*.sh' -print0 2>/dev/null | xargs -0 -r bash -n

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

runtime-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evaluation.models.runtime.validate import validate_runtime_registry; errors = [issue for issue in validate_runtime_registry() if issue.severity == "error"]; assert not errors, "\\n".join(f"[{issue.code}] {issue.message}" for issue in errors)'

# I-03: verify per-CUDA-tier torch constraint stubs match TIER_TORCH_SPECS (no downloads).
check-cuda-constraints:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/check_cuda_torch_constraints.py

# I-05: compile a per-tier unified lockfile (requires uv + network access).
TIER ?= cu128
lock-unified:
	bash scripts/setup/compile_unified_lock.sh $(TIER)

# I-05: offline consistency check for the lock scaffolding (no downloads).
lock-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/check_unified_lock.py

compile-eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q worldfoundry/evaluation scripts

cli-check:
	rm -rf $(CLI_CHECK_OUTPUT)
	mkdir -p $(CLI_CHECK_OUTPUT)/input
	printf '%s\n' '{"sample_id":"ci-0001","status":"success","artifacts":{"video":{"uri":"$(CLI_CHECK_OUTPUT)/input/demo.mp4","kind":"video"}}}' > $(CLI_CHECK_OUTPUT)/input/results.jsonl
	: > $(CLI_CHECK_OUTPUT)/input/demo.mp4
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli evaluate \
		--mode existing-results \
		--results-path $(CLI_CHECK_OUTPUT)/input/results.jsonl \
		--output-dir $(CLI_CHECK_OUTPUT)/run \
		--benchmark-id ci-existing-results \
		--model-id ci-package-check \
		--metric artifact_count \
		--required-artifact video \
		--json

precommit:
	$(PRE_COMMIT) run -a

precommit-install:
	$(PRE_COMMIT) install

preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli preflight runtime \
		--profile $(PREFLIGHT_PROFILE) \
		--output-dir $(PREFLIGHT_OUTPUT) \
		--json

# Optional extra pytest flags, e.g. PYTEST_ARGS='-m "not gpu and not network"'
PYTEST_ARGS ?=

test-eval-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest test/eval_core -q -p no:cacheprovider $(PYTEST_ARGS)

test-training:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/training -q -p no:cacheprovider $(PYTEST_ARGS)

OPEN_SOURCE_INFER_MODEL ?= matrix-game-2
OPEN_SOURCE_INFER_HFD_ROOT ?= $(RELEASE_HFD_ROOT)
OPEN_SOURCE_INFER_STRICT_LOCAL ?= 0

open-source-infer-repro:
	PYTHONPATH=$(PYTHONPATH) $(WORLDFOUNDRY_EVAL) zoo model-download \
		--model-id $(OPEN_SOURCE_INFER_MODEL) \
		--cache-dir $(OPEN_SOURCE_INFER_HFD_ROOT) \
		--check-local
	PYTHONPATH=$(PYTHONPATH) $(WORLDFOUNDRY_EVAL) zoo model-validate \
		--model-id $(OPEN_SOURCE_INFER_MODEL) \
		--cache-dir $(OPEN_SOURCE_INFER_HFD_ROOT) \
		--check-local
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/model_zoo/open_source_infer_repro.py \
		--model-id $(OPEN_SOURCE_INFER_MODEL) \
		--cache-dir $(OPEN_SOURCE_INFER_HFD_ROOT) \
		$(if $(filter 1,$(OPEN_SOURCE_INFER_STRICT_LOCAL)),--strict-local,) \
		--output-dir tmp/open-source-infer-repro \
		--json
