"""Focused tests for the native GUI's GEN3C adapter."""

from __future__ import annotations

import asyncio

import numpy as np
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
	DiffusionOutput,
	DiffusionRequest,
	SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.initializers.cosmos1.component import (
	Cosmos1Gen3CInitializer,
)
from worldfoundry.core.world_explorer import default_camera_path
from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline

from .api_types import InferenceRequest, InferenceResult, SeedingRequest
from .server_gen3c import GEN3C_FRAMES_PER_CHUNK, Gen3CModel


class _FakePipeline:
	def __init__(self) -> None:
		self.device = torch.device("cpu")
		self.dtype = torch.float32
		self.calls: list[dict] = []

	def __call__(self, **kwargs):
		self.calls.append(kwargs)
		image = torch.from_numpy(np.asarray(kwargs["images"], dtype=np.float32)).permute(2, 0, 1)
		video = image[None, :, None].repeat(1, 1, kwargs["num_frames"], 1, 1)
		return {"video": video.mul(2.0).sub(1.0)}


class _FakeNativeDiffusion:
	def __init__(self) -> None:
		self.request = None

	def __call__(self, request):
		self.request = request
		video = torch.zeros((1, 3, request.num_frames, request.height, request.width))
		return DiffusionOutput(sample=video, latents=torch.zeros((1, 1, 1, 1, 1)))


class _FakeDepthModel:
	def eval(self):
		return self

	def to(self, device):
		return self

	def infer(self, image):
		height, width = image.shape[-2:]
		return {
			"depth": torch.ones((height, width), device=image.device),
			"mask": torch.ones((height, width), device=image.device),
			"intrinsics": torch.tensor(
				[[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]],
				device=image.device,
			),
		}


class _FakeLatentEncoder:
	def encode(self, video):
		latent_frames = 1 if video.shape[2] == 1 else 16
		return torch.zeros(
			(video.shape[0], 16, latent_frames, video.shape[-2] // 8, video.shape[-1] // 8),
			device=video.device,
			dtype=video.dtype,
		)


def _depth_prediction(image: np.ndarray):
	height, width = image.shape[:2]
	return {
		"depth": np.full((height, width), 2.0, dtype=np.float32),
		"mask": np.ones((height, width), dtype=np.bool_),
		"intrinsics": np.asarray(
			[[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]],
			dtype=np.float32,
		),
	}


def _camera_batch(count: int) -> np.ndarray:
	result = np.repeat(np.eye(4, dtype=np.float32)[None, :3], count, axis=0)
	result[:, 0, 3] = np.linspace(0.0, 0.1, count, dtype=np.float32)
	return result


def test_gen3c_adapter_preserves_authored_cameras_and_supports_revert(monkeypatch) -> None:
	monkeypatch.setenv("WORLDFOUNDRY_EXPLORER_WIDTH", "16")
	monkeypatch.setenv("WORLDFOUNDRY_EXPLORER_HEIGHT", "16")
	pipeline = _FakePipeline()
	model = Gen3CModel(pipeline=pipeline, depth_predictor=_depth_prediction)
	seed = SeedingRequest(
		request_id="seed",
		images=np.full((1, 12, 20, 3), 0.5, dtype=np.float32),
		depths=None,
		masks=None,
		cameras_to_world=_camera_batch(1),
		focal_lengths=np.asarray([[12.0, 12.0]], dtype=np.float32),
		principal_points=np.asarray([[0.5, 0.5]], dtype=np.float32),
	)
	seed_result = asyncio.run(model.seed_model(seed))
	assert seed_result.depths.shape == (1, 16, 16)
	assert model.model_seeded is True

	frame_count = GEN3C_FRAMES_PER_CHUNK
	cameras = _camera_batch(frame_count)
	request = InferenceRequest(
		request_id="inference",
		timestamps=np.arange(frame_count, dtype=np.float64) / 24.0,
		cameras_to_world=cameras,
		focal_lengths=np.full((frame_count, 2), 12.0, dtype=np.float32),
		principal_points=np.full((frame_count, 2), 0.5, dtype=np.float32),
		resolutions=np.tile([[20, 12]], (frame_count, 1)),
		framerate=24.0,
		return_depths=False,
		show_cache_renderings=False,
	)
	result = asyncio.run(model.request_inference_sync(request))

	assert isinstance(result, InferenceResult)
	assert result.images.shape == (frame_count, 16, 16, 3)
	assert result.predicted_depth_indices.tolist() == [frame_count - 1]
	assert np.allclose(result.cameras_to_world, cameras)
	assert pipeline.calls[0]["rendered_warp_images"].shape == (1, frame_count, 1, 3, 16, 16)
	assert np.allclose(
		pipeline.calls[0]["camera_to_world"][0, :, :3],
		_camera_batch(frame_count),
	)
	assert len(model.cache) == 2

	revert_result = asyncio.run(model.revert_last_generation())
	assert revert_result["success"] is True
	assert len(model.cache) == 1


def test_gen3c_pipeline_accepts_the_shared_camera_path_contract() -> None:
	native = _FakeNativeDiffusion()
	pipeline = Gen3CPipeline(native_pipeline=native, device="cpu")
	pipeline(
		images=np.full((16, 16, 3), 0.5, dtype=np.float32),
		camera_path=default_camera_path(fps=24.0),
		region_hint="continue the garden behind the arch",
		height=16,
		width=16,
		num_frames=GEN3C_FRAMES_PER_CHUNK,
		num_inference_steps=1,
		return_dict=True,
	)

	assert native.request.prompt == "continue the garden behind the arch"
	assert native.request.inputs["camera_to_world"].shape == (
		1,
		GEN3C_FRAMES_PER_CHUNK,
		4,
		4,
	)
	assert native.request.inputs["camera_intrinsics"].shape == (
		1,
		GEN3C_FRAMES_PER_CHUNK,
		3,
		3,
	)


def test_gen3c_initializer_renders_explicit_authored_cameras() -> None:
	frame_count = GEN3C_FRAMES_PER_CHUNK
	camera_to_world = np.repeat(
		np.eye(4, dtype=np.float32)[None, None],
		frame_count,
		axis=1,
	)
	camera_to_world[0, :, 0, 3] = np.linspace(0.0, 0.1, frame_count)
	intrinsics = np.repeat(
		np.asarray(
			[[[[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]]]],
			dtype=np.float32,
		),
		frame_count,
		axis=1,
	)
	request = DiffusionRequest(
		prompt="",
		height=16,
		width=16,
		num_frames=frame_count,
		sampling=SamplingConfig(num_inference_steps=1),
		inputs={
			"image": np.full((16, 16, 3), 0.5, dtype=np.float32),
			"camera_to_world": camera_to_world,
			"camera_intrinsics": intrinsics,
		},
	)
	initializer = Cosmos1Gen3CInitializer(
		depth_model=_FakeDepthModel(),
		offload_depth_model=False,
	)
	result = initializer.initialize_with_encoder(
		request,
		latent_encoder=_FakeLatentEncoder(),
		generator=torch.Generator().manual_seed(1),
		device=torch.device("cpu"),
		dtype=torch.float32,
	)

	assert torch.allclose(
		result.artifacts["camera_to_world"],
		torch.from_numpy(camera_to_world),
	)
	assert torch.allclose(
		result.artifacts["camera_intrinsics"],
		torch.from_numpy(intrinsics),
	)
