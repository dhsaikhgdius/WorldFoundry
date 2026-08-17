"""Map-style dataset views over validated training manifests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload

from .manifest import LoadedTrainingManifest, TrainingSample, load_training_manifest


class TrainingManifestDataset(Sequence[TrainingSample]):
    """A lightweight descriptor dataset ready for a media-decoding transform.

    ``__getitem__`` returns the immutable manifest record.  Decode, frame
    sampling, augmentation, and collation remain separate so their RNG and
    cache provenance can be checkpointed independently.
    """

    def __init__(self, manifest: LoadedTrainingManifest) -> None:
        if not isinstance(manifest, LoadedTrainingManifest):
            raise TypeError("manifest must be a LoadedTrainingManifest")
        if not manifest.samples:
            raise ValueError("training dataset cannot be empty")
        self._manifest = manifest
        self._samples = manifest.samples
        self._sample_ids = tuple(sample.sample_id for sample in self._samples)
        self._index_by_id = {sample_id: index for index, sample_id in enumerate(self._sample_ids)}
        if len(self._index_by_id) != len(self._sample_ids):
            raise ValueError("training dataset sample_ids must be unique")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        split: str | None = None,
        verify_files: bool = True,
    ) -> "TrainingManifestDataset":
        return cls(
            load_training_manifest(
                path,
                split=split,
                verify_files=verify_files,
            )
        )

    @property
    def manifest(self) -> LoadedTrainingManifest:
        return self._manifest

    @property
    def manifest_path(self) -> Path:
        return self._manifest.path

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def index_for_sample_id(self, sample_id: str) -> int:
        try:
            return self._index_by_id[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"unknown training sample_id: {sample_id!r}") from error

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> TrainingSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrainingSample, ...]: ...

    def __getitem__(self, index: int | slice) -> TrainingSample | tuple[TrainingSample, ...]:
        return self._samples[index]

    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self._samples)


__all__ = ["TrainingManifestDataset"]
