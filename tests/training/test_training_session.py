from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import ObjectiveBatch, PreparedBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.data import (  # noqa: E402
    DeterministicDistributedSampler,
    build_stateful_dataloader,
)
from worldfoundry.training.engine import (  # noqa: E402
    OverfitGateError,
    SingleDeviceTrainEngine,
    SingleDeviceTrainingSession,
    build_adamw,
)
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.recipes import TrainingRecipe  # noqa: E402


class _TinyAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes: tuple[type, ...] = ()

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0],
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


def _recipe(
    *,
    accumulation: int = 2,
    checkpoint_interval: int = 0,
    async_checkpoint: bool = False,
) -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "execution_owner": "worldfoundry-native",
            "run": {"id": "tiny-session", "output_dir": "unused"},
            "model": {"recipe": "sana-tiny", "checkpoint": "default"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "manifest.jsonl",
                "cache": "cache",
                "shuffle": False,
                "tail_policy": "uneven",
            },
            "objective": {
                "type": "flow_matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "uniform",
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.05,
                "gradient_accumulation_steps": accumulation,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {
                "save_every_steps": checkpoint_interval,
                "async": async_checkpoint,
            },
        }
    )


def _batches() -> list[TrainingBatch]:
    return [
        TrainingBatch(
            sample_ids=(f"sample-{index}",),
            prompts=("cached",),
            pixel_values=torch.tensor([[[[[0.0, 0.5], [1.0, -0.5]]]]]) + index * 0.1,
        )
        for index in range(2)
    ]


def _session(output_dir: Path, *, accumulation: int = 2) -> SingleDeviceTrainingSession:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = build_adamw(
        adapter.trainable_module.parameters(),
        learning_rate=0.05,
        fused=False,
    )
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    return SingleDeviceTrainingSession(
        recipe=_recipe(accumulation=accumulation),
        engine=engine,
        dataloader=_batches(),
        output_dir=output_dir,
        data_identity={"dataset_digest": "0" * 64},
    )


class _StatefulBatchDataset:
    def __init__(self) -> None:
        self.values = tuple(_batches() * 3)
        self.sample_ids = tuple(f"resume-sample-{index}" for index in range(len(self.values)))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> TrainingBatch:
        value = self.values[index]
        return TrainingBatch(
            sample_ids=(self.sample_ids[index],),
            prompts=value.prompts,
            pixel_values=value.pixel_values,
        )


def _single_batch(values: list[TrainingBatch]) -> TrainingBatch:
    if len(values) != 1:
        raise ValueError("session resume test expects one sample per batch")
    return values[0]


def _stateful_session(output_dir: Path) -> SingleDeviceTrainingSession:
    torch.manual_seed(101)
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = build_adamw(
        adapter.trainable_module.parameters(),
        learning_rate=0.05,
        fused=False,
    )
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    dataset = _StatefulBatchDataset()
    sampler = DeterministicDistributedSampler(
        dataset,
        dataset_digest="8" * 64,
        seed=17,
        shuffle=True,
        rank=0,
        world_size=1,
        tail_policy="uneven",
    )
    loader = build_stateful_dataloader(
        dataset,
        sampler,
        batch_size=1,
        collate_fn=_single_batch,
        worker_seed=19,
    )
    return SingleDeviceTrainingSession(
        recipe=_recipe(
            accumulation=1,
            checkpoint_interval=2,
            async_checkpoint=True,
        ),
        engine=engine,
        dataloader=loader,
        output_dir=output_dir,
        data_identity={"dataset_digest": "8" * 64},
    )


def test_single_device_session_writes_metrics_manifest_and_parameter_delta(tmp_path: Path) -> None:
    session = _session(tmp_path / "run")

    summary = session.run(
        max_steps=4,
        seed=13,
        fixed_batch=True,
        fixed_corruption=True,
    )

    metric_rows = [
        json.loads(line) for line in (session.output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((session.output_dir / "run.json").read_text(encoding="utf-8"))
    assert len(metric_rows) == 4
    assert [row["optimizer_step"] for row in metric_rows] == [1, 2, 3, 4]
    assert all(row["microbatch_count"] == 2 for row in metric_rows)
    assert manifest["status"] == "complete"
    assert manifest["recipe_digest"] == session.recipe.digest
    assert manifest["summary"]["optimizer_steps"] == 4
    assert summary.microbatches == 8
    assert summary.changed_parameter_tensors == 1
    assert summary.parameter_delta_l2 > 0
    assert summary.overfit_gate_passed is None
    with pytest.raises(RuntimeError, match="only run once"):
        session.run(max_steps=1)


def test_single_device_session_persists_an_explicit_overfit_gate_failure(tmp_path: Path) -> None:
    session = _session(tmp_path / "failed-gate", accumulation=1)

    with pytest.raises(OverfitGateError) as captured:
        session.run(
            max_steps=1,
            seed=7,
            fixed_batch=True,
            fixed_corruption=True,
            maximum_final_to_initial_loss_ratio=0.01,
        )

    manifest = json.loads((session.output_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "gate-failed"
    assert captured.value.summary.overfit_gate_passed is False


def test_session_dcp_resume_matches_the_uninterrupted_next_step(tmp_path: Path) -> None:
    uninterrupted = _stateful_session(tmp_path / "uninterrupted")
    uninterrupted_summary = uninterrupted.run(max_steps=3, seed=29)

    interrupted = _stateful_session(tmp_path / "interrupted")
    interrupted.run(max_steps=2, seed=29)
    checkpoint = interrupted.output_dir / "checkpoints" / "step-00000002"
    assert checkpoint.is_dir()

    resumed = _stateful_session(tmp_path / "resumed")
    resumed_summary = resumed.run(
        max_steps=1,
        seed=29,
        resume_checkpoint=checkpoint,
    )

    assert resumed.engine.global_step == 3
    assert resumed.progress.optimizer_steps == 3
    assert resumed_summary.final_loss == uninterrupted_summary.final_loss
    for expected, actual in zip(
        uninterrupted.engine.parameters,
        resumed.engine.parameters,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["initial_global_step"] == 2
    assert manifest["resumed_from"]["global_step"] == 2
    assert manifest["cumulative_progress"]["optimizer_steps"] == 3


def test_session_facade_resolves_canonical_lifecycle_types() -> None:
    from worldfoundry.training.engine import FSDP2TrainingSession
    from worldfoundry.training.engine.sessions.fsdp2 import (
        FSDP2TrainingSession as CanonicalFSDP2TrainingSession,
    )
    from worldfoundry.training.engine.sessions.single_device import (
        SingleDeviceTrainingSession as CanonicalSingleDeviceTrainingSession,
    )

    assert SingleDeviceTrainingSession is CanonicalSingleDeviceTrainingSession
    assert FSDP2TrainingSession is CanonicalFSDP2TrainingSession


def test_session_resume_identity_uses_runtime_state_not_source_files(tmp_path: Path) -> None:
    session = _stateful_session(tmp_path / "identity")

    identity = session._resume_identity(
        seed=29,
        fixed_batch=False,
        fixed_corruption=False,
    )

    assert "source_sha256" not in identity
    assert identity["recipe_digest"] == session.recipe.digest
    assert identity["data"] == dict(session.data_identity)
    assert identity["environment"]["device_type"] == session.engine.device.type
