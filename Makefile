.PHONY: help install-core install-dev docs-check lint ruff-check format-check shell-check data-check runtime-registry-check compile-eval cli-check precommit precommit-install preflight test-eval-core test-training lock-unified validate-unified-lock

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
RELEASE_HFD_ROOT ?= $(if $(WORLDFOUNDRY_HFD_ROOT),$(WORLDFOUNDRY_HFD_ROOT),$(HOME)/.cache/worldfoundry/checkpoints/hfd)
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
		'  make validate-unified-lock  Verify cu128 lock torch pin is on CUDA index.' \
		'  make preflight         Run the public runtime preflight.' \
		'  make test-eval-core    Run the eval_core release-gate pytest suite (CPU).' \
		'  make test-training     Run the tests/training pytest suite (CPU subset).'

install-core:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .
	$(PIP) install build pre-commit PyYAML ruff

docs-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli --help >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

lint: ruff-check format-check shell-check data-check runtime-registry-check validate-unified-lock

ruff-check:
	$(PYTHON) -m ruff check $(RUFF_SOURCES)

format-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q $(CANONICAL_DIFFUSION_SOURCES) worldfoundry/evaluation scripts

shell-check:
	find scripts/setup -type f -name '*.sh' -exec bash -n {} +

data-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo models --json >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

runtime-registry-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'from worldfoundry.evaluation.models.runtime.validate import validate_runtime_registry; errors = [issue for issue in validate_runtime_registry() if issue.severity == "error"]; assert not errors, "\\n".join(f"[{issue.code}] {issue.message}" for issue in errors)'

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
# CUDA tier for make lock-unified (cu121|cu124|cu128).
TIER ?= cu128

lock-unified:
	bash scripts/setup/compile_unified_lock.sh $(TIER)

# I-05: refuse committing a cu128 lock whose torch pin is absent from the CUDA index.
# Placeholder stubs (still containing "Do not invent") are skipped until populated.
validate-unified-lock:
	@lock="requirements/lock/worldfoundry-unified.cu128.lock.txt"; \
	if [ ! -f "$$lock" ]; then echo "missing $$lock" >&2; exit 1; fi; \
	if grep -q 'Do not invent' "$$lock"; then \
	  echo "validate-unified-lock: cu128 lock still placeholder; skip"; \
	else \
	  PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup/validate_unified_lock_tier.py "$$lock" --tier cu128; \
	fi

test-eval-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest test/eval_core -q -p no:cacheprovider $(PYTEST_ARGS)

test-training:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/training -q -p no:cacheprovider $(PYTEST_ARGS)
