from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers import wan as wan_denoiser
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.evaluation.api import WorldModelConfig
from worldfoundry.evaluation.models.pipelines.loading import (
    build_pipeline_runner_spec,
    call_pipeline_from_pretrained,
)
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline


def _checkpoint(*, revision: str = "a" * 40) -> CheckpointSpec:
    return CheckpointSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        revision=revision,
        files=("diffusion_pytorch_model.safetensors",),
        file_sha256={"diffusion_pytorch_model.safetensors": "b" * 64},
        file_size_bytes={"diffusion_pytorch_model.safetensors": 123},
    )


def _context(
    *,
    checkpoint: CheckpointSpec | None = None,
    component_options: dict[str, object] | None = None,
) -> ComponentBuildContext:
    return ComponentBuildContext(
        model_id="wan2.1-t2v-1.3b",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(device=torch.device("cpu"), dtype=torch.float32),
        checkpoints={"weights": checkpoint or _checkpoint()},
        component_options=component_options or {},
    )


def test_benchmark_runner_config_executes_wan_peft_adapter_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_from_pretrained(
        cls: type[NativeDiffusionPipeline],
        model_id: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        del cls
        calls.append((model_id, dict(kwargs)))
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    adapter_path = tmp_path / "policy"
    config = WorldModelConfig(
        model_id="wan2.1-t2v-1.3b",
        runner="worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner",
        parameters={
            "pipeline_target": ("worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v:Wan2p1T2VPipeline"),
            "peft_adapter_path": str(adapter_path),
        },
        runtime={"device": "cpu"},
    )
    spec = build_pipeline_runner_spec(config)

    pipeline = call_pipeline_from_pretrained(
        Wan2p1T2VPipeline,
        model_path=spec.model_path,
        fallback_model_path=spec.fallback_model_path,
        required_components=spec.required_components,
        device=spec.device,
        model_id=spec.model_id,
    )

    assert isinstance(pipeline, Wan2p1T2VPipeline)
    assert len(calls) == 1
    loaded_model_id, loading_options = calls[0]
    assert loaded_model_id == "wan2.1-t2v-1.3b"
    assert loading_options["checkpoint_overrides"] is None
    assert loading_options["component_options"] == {"denoiser:main": {"peft_adapter_path": str(adapter_path)}}
    assert loading_options["policy"].device == torch.device("cpu")


def test_wan_pipeline_default_loading_does_not_inject_component_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(
        cls: type[NativeDiffusionPipeline],
        model_id: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        del cls
        captured.update(kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        NativeDiffusionPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    Wan2p1T2VPipeline.from_pretrained(device="cpu", model_id="wan2.1-t2v-1.3b")

    assert captured["component_options"] is None


def test_wan_denoiser_rejects_unknown_component_options_before_loading() -> None:
    with pytest.raises(ValueError, match="unsupported Wan denoiser options"):
        wan_denoiser.build_wan21_t2v_1p3b_denoiser(_context(component_options={"peft_adapter_pat": "/tmp/adapter"}))


def test_wan_adapter_checkpoint_must_match_selected_recipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from worldfoundry.training import tuning

    selected = _checkpoint()
    exported = wan_denoiser._checkpoint_payload(selected)
    artifact = SimpleNamespace(
        path=tmp_path / "adapter",
        metadata={
            "role": "policy",
            "roles": {
                "policy": {
                    "checkpoint": {
                        "checkpoint": exported,
                    }
                }
            },
        },
    )
    monkeypatch.setattr(tuning, "inspect_peft_adapter", lambda path: artifact)

    assert (
        wan_denoiser._validated_wan_peft_adapter(
            _context(checkpoint=selected),
            artifact.path,
        )
        == artifact.path
    )

    with pytest.raises(ValueError, match="base checkpoint differs"):
        wan_denoiser._validated_wan_peft_adapter(
            _context(checkpoint=_checkpoint(revision="c" * 40)),
            artifact.path,
        )
