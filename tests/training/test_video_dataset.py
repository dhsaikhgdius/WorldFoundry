from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

av = pytest.importorskip("av")
numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from worldfoundry.training.data import (  # noqa: E402
    LatentTokenBatchSampler,
    MediaReference,
    TrainingManifestDataset,
    TrainingSample,
    VideoDecodeConfig,
    VideoDecodingDataset,
    VideoLatentGeometry,
    VideoResolutionBucket,
    assign_video_buckets,
    collate_decoded_video_samples,
    decode_video_sample,
    file_sha256,
    video_frame_indices,
)


def _write_video(path: Path, *, frames: int = 8, height: int = 48, width: int = 64, fps: int = 8) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = numpy.zeros((height, width, 3), dtype=numpy.uint8)
            array[:, :, 0] = index * 20
            array[:, :, 1] = numpy.arange(width, dtype=numpy.uint8)[None, :]
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _sample(path: Path, *, num_frames: int = 8) -> TrainingSample:
    return TrainingSample(
        sample_id="video-sample",
        task="t2v",
        prompt="a red gradient changes over time",
        media=MediaReference(
            uri=path.name,
            sha256=file_sha256(path),
            mime_type="video/x-matroska",
            size_bytes=path.stat().st_size,
        ),
        width=64,
        height=48,
        num_frames=num_frames,
        fps=8.0,
        conditions={},
        split="train",
        safety={"accepted": True},
    )


def _manifest_dataset(tmp_path: Path, sample: TrainingSample) -> TrainingManifestDataset:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    return TrainingManifestDataset.from_file(manifest, verify_hashes=True)


def _assignment(sample: TrainingSample):
    return assign_video_buckets(
        (sample,),
        buckets=(VideoResolutionBucket(4, 32, 32, "text"),),
        geometry=VideoLatentGeometry(8, 8, 1, "uniform"),
        conditioning_layout="text",
    )[0]


def test_video_frame_indices_are_exact_and_reject_implicit_padding() -> None:
    assert video_frame_indices(8, 4, mode="head") == (0, 1, 2, 3)
    assert video_frame_indices(8, 4, mode="uniform_full") == (0, 2, 5, 7)
    assert video_frame_indices(8, 1, mode="uniform-full") == (3,)
    with pytest.raises(ValueError, match="without padding"):
        video_frame_indices(3, 4, mode="head")


def test_video_decoding_dataset_emits_bucket_aligned_pixel_batch_and_transform_digests(tmp_path) -> None:
    video = tmp_path / "sample.mkv"
    _write_video(video)
    sample = _sample(video)
    manifest_dataset = _manifest_dataset(tmp_path, sample)
    assignments = (_assignment(sample),)
    config = VideoDecodeConfig(
        frame_sampling="uniform-full",
        decoder_thread_type="auto",
        verify_media_sha256=True,
    )
    dataset = VideoDecodingDataset(manifest_dataset, assignments, config=config)
    decoded = dataset[0]
    batch = collate_decoded_video_samples([decoded])

    assert decoded.selected_frame_indices == (0, 2, 5, 7)
    assert decoded.decoded_frame_count == 8
    assert decoded.decoded_fps == 8.0
    assert tuple(decoded.pixel_values.shape) == (3, 4, 32, 32)
    assert float(decoded.pixel_values.min()) >= -1.0
    assert float(decoded.pixel_values.max()) <= 1.0
    assert tuple(batch.pixel_values.shape) == (1, 3, 4, 32, 32)
    assert tuple(batch.valid_mask.shape) == (1, 1, 4, 32, 32)
    assert batch.metadata["latent_tokens_per_microbatch"] == 4 * 4 * 4
    assert len(batch.metadata["frame_sampling_digest"][0]) == 64
    assert dataset.index_sha256 != dataset.dataset_digest

    sampler = LatentTokenBatchSampler(
        dataset,
        max_latent_tokens=64,
        shuffle=False,
        tail_policy="pad",
    )
    assert list(sampler) == [[0]]
    assert sampler.data_content_digest == dataset.index_sha256


def test_video_decode_rejects_manifest_count_and_media_digest_drift(tmp_path) -> None:
    video = tmp_path / "sample.mkv"
    _write_video(video)
    sample = _sample(video)
    assignment = _assignment(sample)

    wrong_count_sample = replace(sample, num_frames=9)
    wrong_count_assignment = replace(assignment, source_num_frames=9)
    with pytest.raises(ValueError, match="decoded frame count differs"):
        decode_video_sample(
            wrong_count_sample,
            wrong_count_assignment,
            media_path=video,
            config=VideoDecodeConfig(verify_manifest_frame_count=True),
        )

    wrong_media = replace(sample, media=replace(sample.media, sha256="f" * 64))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        decode_video_sample(
            wrong_media,
            assignment,
            media_path=video,
            config=VideoDecodeConfig(verify_media_sha256=True),
        )
