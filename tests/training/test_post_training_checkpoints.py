from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.training.post_training import resolve_role_checkpoint


def _default() -> CheckpointSpec:
    return CheckpointSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        revision="37ec512624d61f7aa208f7ea8140a131f93afc9a",
        files=("diffusion_pytorch_model.safetensors",),
        file_sha256={"diffusion_pytorch_model.safetensors": "a" * 64},
        file_size_bytes={"diffusion_pytorch_model.safetensors": 123},
    )


def test_role_checkpoint_resolves_default_and_explicit_native_identity() -> None:
    default = _default()
    implicit = resolve_role_checkpoint(
        role="real_score",
        reference="default",
        native_default=default,
    )
    explicit = resolve_role_checkpoint(
        role="real-score",
        reference=("Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"),
        native_default=default,
    )

    assert implicit.checkpoint is default
    assert explicit.checkpoint is default
    assert implicit.source_kind == explicit.source_kind == "native-default"
    assert implicit.role == explicit.role == "real-score"


def test_role_checkpoint_requires_pinned_hub_or_fully_audited_local(
    tmp_path: Path,
) -> None:
    default = _default()
    with pytest.raises(ValueError, match="40_HEX_COMMIT"):
        resolve_role_checkpoint(
            role="teacher",
            reference="Wan-AI/Wan2.1-T2V-1.3B",
            native_default=default,
        )
    with pytest.raises(ValueError, match="40_HEX_COMMIT"):
        resolve_role_checkpoint(
            role="teacher",
            reference=str(tmp_path),
            native_default=default,
        )
    unaudited = CheckpointSpec(source=tmp_path, files=("weights.safetensors",))
    with pytest.raises(ValueError, match="SHA-256 and byte size"):
        resolve_role_checkpoint(
            role="teacher",
            reference="default",
            native_default=default,
            audited_local_override=unaudited,
        )


def test_role_checkpoint_preserves_layout_without_wrong_hashes_for_other_commit() -> None:
    resolved = resolve_role_checkpoint(
        role="fake-score",
        reference="example/derived-wan@1111111111111111111111111111111111111111",
        native_default=_default(),
    )

    assert resolved.source_kind == "pinned-hub"
    assert resolved.checkpoint.repo_id == "example/derived-wan"
    assert resolved.checkpoint.revision == "1" * 40
    assert resolved.checkpoint.files == ("diffusion_pytorch_model.safetensors",)
    assert not resolved.checkpoint.file_sha256
    assert not resolved.checkpoint.file_size_bytes


def test_explicit_native_reference_accepts_only_an_exact_audited_local_mirror(
    tmp_path: Path,
) -> None:
    default = _default()
    mirror = CheckpointSpec(
        source=tmp_path,
        files=default.files,
        file_sha256=default.file_sha256,
        file_size_bytes=default.file_size_bytes,
    )
    reference = "Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"

    resolved = resolve_role_checkpoint(
        role="real-score",
        reference=reference,
        native_default=default,
        audited_local_override=mirror,
    )

    assert resolved.source_kind == "audited-local"
    assert resolved.requested_reference == reference
    assert resolved.checkpoint is mirror

    wrong_bytes = CheckpointSpec(
        source=tmp_path,
        files=default.files,
        file_sha256={default.files[0]: "b" * 64},
        file_size_bytes=default.file_size_bytes,
    )
    with pytest.raises(ValueError, match="bytes differ"):
        resolve_role_checkpoint(
            role="real-score",
            reference=reference,
            native_default=default,
            audited_local_override=wrong_bytes,
        )
    with pytest.raises(ValueError, match="repository and revision"):
        resolve_role_checkpoint(
            role="real-score",
            reference="example/derived@1111111111111111111111111111111111111111",
            native_default=default,
            audited_local_override=mirror,
        )
