from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.core.io.integrity import canonical_sha256  # noqa: E402
from worldfoundry.training.data import (  # noqa: E402
    SanaCachedDataset,
    SanaCacheEntry,
    SanaCacheProvenance,
    SanaCacheStore,
    collate_sana_cached_samples,
    text_sha256,
)


def _digest(label: str) -> str:
    return text_sha256(label)


def _provenance(*, height: int = 64, width: int = 64) -> SanaCacheProvenance:
    return SanaCacheProvenance(
        media_sha256=_digest("media"),
        prompt_sha256=_digest("a blue cup"),
        model_recipe_digest=_digest("sana recipe"),
        codec_digest=_digest("dcae"),
        conditioner_digest=_digest("gemma"),
        tokenizer_digest=_digest("tokenizer"),
        safety_audit_digest=_digest("safe"),
        pixel_transform_digest=_digest("rgb[-1,1]"),
        prompt_enhancement_digest=_digest("enhancement enabled and pinned"),
        image_height=height,
        image_width=width,
        spatial_compression=32,
        latent_scaling_factor=0.41407,
        max_text_length=3,
    )


def _write_sample(
    store: SanaCacheStore,
    *,
    sample_id: str = "sample",
    provenance: SanaCacheProvenance | None = None,
):
    resolved = provenance or _provenance()
    return store.write_sample(
        sample_id=sample_id,
        provenance=resolved,
        clean_latents=torch.arange(32 * 2 * 2, dtype=torch.float32).reshape(32, 2, 2),
        context=torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4),
        context_mask=torch.tensor([1, 1, 0]),
        latent_loss_mask=torch.ones(1, 2, 2),
        sample_weight=torch.tensor(0.75),
    )


def test_sana_cache_round_trip_is_content_addressed_and_prompt_free(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    first = _write_sample(store, sample_id="first")
    second = _write_sample(store, sample_id="second")
    dataset_digest = _digest("dataset")
    index = store.write_index(dataset_digest=dataset_digest, entries=[first, second])

    assert first.object_sha256 == second.object_sha256
    assert first.identity_sha256 == second.identity_sha256
    assert first.object_path.endswith(f"{first.object_sha256}.safetensors")
    assert len(tuple((tmp_path / "objects").rglob("*.safetensors"))) == 1
    raw_index = (tmp_path / "index.json").read_text(encoding="utf-8")
    assert "a blue cup" not in raw_index
    assert json.loads(raw_index)["index_sha256"] == index.index_sha256

    dataset = SanaCachedDataset(tmp_path, expected_dataset_digest=dataset_digest)
    batch = collate_sana_cached_samples([dataset[0], dataset[1]])

    assert dataset.sample_ids == ("first", "second")
    assert batch.pixel_values is None
    assert batch.conditions["clean_latents"].shape == (2, 32, 2, 2)
    assert batch.conditions["context"].shape == (2, 1, 3, 4)
    assert batch.conditions["context_mask"].shape == (2, 3)
    assert batch.conditions["latent_loss_mask"].shape == (2, 1, 2, 2)
    torch.testing.assert_close(batch.sample_weights, torch.tensor([0.75, 0.75]))
    assert all(prompt.startswith("sha256:") for prompt in batch.prompts)


def test_sana_cache_rejects_object_tampering(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    entry = _write_sample(store)
    store.write_index(dataset_digest=_digest("dataset"), entries=[entry])
    object_path = tmp_path / entry.object_path
    payload = bytearray(object_path.read_bytes())
    payload[-1] ^= 1
    object_path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        SanaCachedDataset(tmp_path, audit_on_open=False)[0]


def test_sana_cache_rejects_logical_identity_mismatch(tmp_path: Path) -> None:
    entry = _write_sample(SanaCacheStore(tmp_path))
    payload = entry.to_dict()
    payload["provenance"]["max_text_length"] = 4

    with pytest.raises(ValueError, match="logical identity"):
        SanaCacheEntry.from_mapping(payload)


def test_sana_cache_rejects_index_tampering_and_wrong_dataset(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    entry = _write_sample(store)
    store.write_index(dataset_digest=_digest("dataset"), entries=[entry])
    index_path = tmp_path / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["dataset_digest"] = _digest("different dataset")
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="index digest"):
        SanaCachedDataset(tmp_path)

    store.write_index(dataset_digest=_digest("dataset"), entries=[entry])
    with pytest.raises(ValueError, match="dataset digest mismatch"):
        SanaCachedDataset(tmp_path, expected_dataset_digest=_digest("wrong"))


def test_sana_cache_collator_rejects_incompatible_preprocessing_buckets(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)
    small = _write_sample(store, sample_id="small")
    large_provenance = _provenance(height=96, width=64)
    large = store.write_sample(
        sample_id="large",
        provenance=large_provenance,
        clean_latents=torch.zeros(32, 3, 2),
        context=torch.zeros(1, 3, 4),
        context_mask=torch.ones(3, dtype=torch.long),
        latent_loss_mask=torch.ones(1, 3, 2),
        sample_weight=torch.tensor(1.0),
    )
    store.write_index(dataset_digest=canonical_sha256({"samples": 2}), entries=[small, large])
    dataset = SanaCachedDataset(tmp_path)

    with pytest.raises(ValueError, match="shapes, dtypes, and layouts"):
        collate_sana_cached_samples([dataset[0], dataset[1]])


def test_sana_cache_validates_tensor_contract_before_write(tmp_path: Path) -> None:
    store = SanaCacheStore(tmp_path)

    with pytest.raises(ValueError, match="context sequence length"):
        store.write_sample(
            sample_id="bad",
            provenance=_provenance(),
            clean_latents=torch.zeros(32, 2, 2),
            context=torch.zeros(1, 2, 4),
            context_mask=torch.ones(2, dtype=torch.long),
        )
