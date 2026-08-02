from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.core.io.integrity import text_sha256  # noqa: E402
from worldfoundry.training.data import SharedConditioningStore  # noqa: E402


def _write(store: SharedConditioningStore):
    return store.write(
        branch="unconditional",
        prompt_sha256=text_sha256(""),
        model_recipe_digest="1" * 64,
        conditioner_digest="2" * 64,
        tokenizer_digest="3" * 64,
        tensors={"context": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)},
        layouts={"context": "sequence-features"},
    )


def test_shared_conditioning_round_trip_is_content_addressed_and_idempotent(tmp_path) -> None:
    store = SharedConditioningStore(tmp_path)
    artifact = _write(store)
    same = _write(store)
    loaded = store.read("unconditional")

    assert same == artifact
    assert loaded.artifact == artifact
    assert artifact.identity.prompt_sha256 == text_sha256("")
    assert artifact.object_path.startswith("shared-objects/")
    torch.testing.assert_close(
        loaded.tensors["context"],
        torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
    )


def test_shared_conditioning_detects_object_and_manifest_tampering(tmp_path) -> None:
    object_store = SharedConditioningStore(tmp_path / "object")
    artifact = _write(object_store)
    object_path = object_store.root / artifact.object_path
    payload = bytearray(object_path.read_bytes())
    payload[-1] ^= 1
    object_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        object_store.read("unconditional")

    manifest_store = SharedConditioningStore(tmp_path / "manifest")
    _write(manifest_store)
    manifest = manifest_store.root / "unconditional-conditioning.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"branch":"unconditional"', '"branch":"changed"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest digest"):
        manifest_store.read("unconditional")
