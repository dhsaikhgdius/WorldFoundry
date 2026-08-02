# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEN3C adapter for the native World Explorer camera-path GUI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .api_types import (
	InferenceRequest,
	InferenceResult,
	SeedingRequest,
	SeedingResult,
)
from .server_base import InferenceModel

GEN3C_WIDTH = 1280
GEN3C_HEIGHT = 704
GEN3C_FRAMES_PER_CHUNK = 121
GEN3C_FPS = 24.0


@dataclass(frozen=True)
class _CacheFrame:
	image: np.ndarray
	depth: np.ndarray
	mask: np.ndarray
	camera_to_world: np.ndarray
	intrinsic: np.ndarray


def _complete_camera_to_world(value: np.ndarray) -> np.ndarray:
	value = np.asarray(value, dtype=np.float32)
	if value.ndim != 3 or value.shape[1:] not in {(3, 4), (4, 4)}:
		raise ValueError(f"camera-to-world matrices must be [N,3,4] or [N,4,4], got {value.shape}")
	if value.shape[1:] == (4, 4):
		return value.copy()
	result = np.zeros((len(value), 4, 4), dtype=np.float32)
	result[:, :3] = value
	result[:, 3, 3] = 1.0
	return result


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
	return cv2.resize(
		np.asarray(image)[..., :3].astype(np.float32),
		(width, height),
		interpolation=cv2.INTER_AREA,
	)


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
	return cv2.resize(
		np.asarray(depth, dtype=np.float32),
		(width, height),
		interpolation=cv2.INTER_LINEAR,
	)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
	return cv2.resize(
		np.asarray(mask, dtype=np.uint8),
		(width, height),
		interpolation=cv2.INTER_NEAREST,
	).astype(np.bool_)


def _pixel_intrinsic(value: np.ndarray, width: int, height: int) -> np.ndarray:
	intrinsic = np.asarray(value, dtype=np.float32).copy()
	if intrinsic.shape != (3, 3):
		raise ValueError(f"intrinsic matrix must be [3,3], got {intrinsic.shape}")
	if abs(float(intrinsic[0, 2])) <= 1.5 and abs(float(intrinsic[1, 2])) <= 1.5:
		intrinsic[0] *= float(width)
		intrinsic[1] *= float(height)
	return intrinsic


def _video_to_thwc(value: Any) -> np.ndarray:
	try:
		import torch

		if isinstance(value, torch.Tensor):
			value = value.detach().float().cpu().numpy()
	except ImportError:
		pass
	video = np.asarray(value)
	if video.ndim == 5:
		if video.shape[0] != 1:
			raise ValueError(f"GEN3C GUI backend requires batch size 1, got {video.shape}")
		video = video[0]
	if video.ndim != 4:
		raise ValueError(f"GEN3C output must be a four- or five-dimensional video, got {video.shape}")
	if video.shape[0] == 3:
		video = np.transpose(video, (1, 2, 3, 0))
	elif video.shape[1] == 3:
		video = np.transpose(video, (0, 2, 3, 1))
	elif video.shape[-1] != 3:
		raise ValueError(f"GEN3C output does not contain an RGB channel dimension: {video.shape}")
	video = video.astype(np.float32)
	if video.size and float(video.min()) < -0.01:
		video = video * 0.5 + 0.5
	elif video.size and float(video.max()) > 1.5:
		video = video / 255.0
	return np.clip(video, 0.0, 1.0)


class Gen3CModel(InferenceModel):
	"""Run WorldFoundry's resident native GEN3C recipe from authored cameras."""

	def __init__(
		self,
		*,
		checkpoint_path: str | None = None,
		pipeline: Any = None,
		depth_predictor: Callable[[np.ndarray], Any] | None = None,
		**kwargs: Any,
	) -> None:
		super().__init__(checkpoint_path=checkpoint_path, **kwargs)
		if pipeline is None:
			from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline

			source: str | None = checkpoint_path
			if source and not Path(source).expanduser().exists() and source == "gen3c":
				source = None
			pipeline = Gen3CPipeline.from_pretrained(
				source,
				device=os.environ.get("WORLDFOUNDRY_EXPLORER_DEVICE", "cuda"),
				torch_dtype=os.environ.get("WORLDFOUNDRY_EXPLORER_DTYPE", "bfloat16"),
				offload_mode=os.environ.get("WORLDFOUNDRY_EXPLORER_OFFLOAD_MODE", "block"),
			)
		self.pipeline = pipeline
		self.depth_predictor = depth_predictor
		self.width = int(os.environ.get("WORLDFOUNDRY_EXPLORER_WIDTH", GEN3C_WIDTH))
		self.height = int(os.environ.get("WORLDFOUNDRY_EXPLORER_HEIGHT", GEN3C_HEIGHT))
		self.frames_per_chunk = GEN3C_FRAMES_PER_CHUNK
		self.framerate = GEN3C_FPS
		self.guidance_scale = float(os.environ.get("WORLDFOUNDRY_EXPLORER_GUIDANCE", 1.0))
		self.num_inference_steps = int(os.environ.get("WORLDFOUNDRY_EXPLORER_STEPS", 35))
		self.seed = int(os.environ.get("WORLDFOUNDRY_EXPLORER_SEED", 1))
		self.negative_prompt = os.environ.get("WORLDFOUNDRY_EXPLORER_NEGATIVE_PROMPT", "")
		self.cache_limit = max(2, int(os.environ.get("WORLDFOUNDRY_EXPLORER_CACHE_FRAMES", 8)))
		self.cache: list[_CacheFrame] = []
		self._revert_cache: list[_CacheFrame] | None = None
		self.aabb_min = np.asarray([-16.0, -16.0, -16.0], dtype=np.float32)
		self.aabb_max = np.asarray([16.0, 16.0, 16.0], dtype=np.float32)

	async def make_test_image(self) -> InferenceResult:
		raise NotImplementedError

	def _native_depth_prediction(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		import torch
		import torch.nn.functional as functional

		native_pipeline = getattr(self.pipeline, "native_pipeline", self.pipeline)
		components = native_pipeline.components
		initializer = components.latent_initializer
		model = initializer.depth_model
		device = torch.device(native_pipeline.device)
		pixels = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
		pixels = functional.interpolate(pixels.float(), (720, 1280), mode="bilinear", align_corners=False)
		model.to(device)
		with torch.no_grad():
			prediction = model.infer(pixels[0].to(device))
		depth = prediction["depth"].float()[None, None]
		mask = prediction["mask"].float()[None, None]
		depth = functional.interpolate(
			depth, (self.height, self.width), mode="bilinear", align_corners=False
		)[0, 0]
		mask = functional.interpolate(mask, (self.height, self.width), mode="nearest")[0, 0] > 0.5
		intrinsic = prediction["intrinsics"].float().cpu().numpy()
		if bool(getattr(initializer, "offload_depth_model", False)) and device.type == "cuda":
			model.to("cpu")
			torch.cuda.empty_cache()
		return (
			depth.cpu().numpy().astype(np.float32),
			mask.cpu().numpy().astype(np.bool_),
			_pixel_intrinsic(intrinsic, self.width, self.height),
		)

	def _predict_depth(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		if self.depth_predictor is None:
			return self._native_depth_prediction(image)
		prediction = self.depth_predictor(image)
		if isinstance(prediction, dict):
			depth = prediction["depth"]
			mask = prediction.get("mask")
			intrinsic = prediction.get("intrinsics")
		else:
			values = tuple(prediction)
			if len(values) not in {2, 3}:
				raise ValueError("depth_predictor must return depth/mask or depth/mask/intrinsics")
			depth, mask = values[:2]
			intrinsic = values[2] if len(values) == 3 else None
		depth = _resize_depth(depth, self.width, self.height)
		mask = depth > 0 if mask is None else _resize_mask(mask, self.width, self.height)
		if intrinsic is None:
			intrinsic = np.asarray(
				[[self.width, 0, self.width / 2], [0, self.height, self.height / 2], [0, 0, 1]],
				dtype=np.float32,
			)
		return depth, mask, _pixel_intrinsic(intrinsic, self.width, self.height)

	async def seed_model(self, req: SeedingRequest) -> SeedingResult:
		target_resolutions = np.tile([[self.width, self.height]], (len(req), 1))
		intrinsics = req.intrinsics_matrix(for_resolutions=target_resolutions).astype(np.float32)
		camera_to_world = _complete_camera_to_world(req.cameras_to_world)
		cache: list[_CacheFrame] = []
		output_depths, output_masks = [], []
		for index in range(len(req)):
			image = _resize_image(req.images[index], self.width, self.height)
			if req.depths is None:
				depth, mask, estimated_intrinsic = self._predict_depth(image)
				intrinsics[index] = estimated_intrinsic
				if req.masks is not None:
					mask &= _resize_mask(req.masks[index], self.width, self.height)
			else:
				depth = _resize_depth(req.depths[index], self.width, self.height)
				mask = depth > 0 if req.masks is None else _resize_mask(
					req.masks[index], self.width, self.height
				)
			depth = np.where(mask, depth, 0.0).astype(np.float32)
			cache.append(
				_CacheFrame(
					image=image,
					depth=depth,
					mask=mask,
					camera_to_world=camera_to_world[index],
					intrinsic=intrinsics[index],
				)
			)
			output_depths.append(depth)
			output_masks.append(mask)
		self.cache = cache[-self.cache_limit :]
		self._revert_cache = None
		self.model_seeded = True
		return SeedingResult(
			request_id=req.request_id,
			cameras_to_world=np.stack([frame.camera_to_world[:3] for frame in cache]),
			focal_lengths=np.stack(
				[[frame.intrinsic[0, 0], frame.intrinsic[1, 1]] for frame in cache]
			).astype(np.float32),
			principal_points=np.stack(
				[
					[frame.intrinsic[0, 2] / self.width, frame.intrinsic[1, 2] / self.height]
					for frame in cache
				]
			).astype(np.float32),
			resolutions=np.tile([[self.width, self.height]], (len(cache), 1)),
			depths=np.stack(output_depths),
			masks=np.stack(output_masks),
		)

	def _target_camera_data(self, req: InferenceRequest) -> tuple[np.ndarray, np.ndarray]:
		target_resolutions = np.tile([[self.width, self.height]], (len(req), 1))
		return (
			_complete_camera_to_world(req.cameras_to_world),
			req.intrinsics_matrix(for_resolutions=target_resolutions).astype(np.float32),
		)

	def _render_cache(
		self,
		target_camera_to_world: np.ndarray,
		target_intrinsics: np.ndarray,
	) -> tuple[Any, Any]:
		import torch

		from worldfoundry.core.spatial_warp import forward_warp_indexed_frames

		native_pipeline = getattr(self.pipeline, "native_pipeline", self.pipeline)
		device = torch.device(getattr(native_pipeline, "device", "cpu"))
		dtype = getattr(native_pipeline, "dtype", torch.float32)
		if device.type == "cpu":
			dtype = torch.float32
		source_pixels = torch.from_numpy(
			np.stack([frame.image for frame in self.cache])
		).permute(0, 3, 1, 2)
		source_pixels = source_pixels.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)
		cache_cameras = np.stack([frame.camera_to_world for frame in self.cache])
		cache_intrinsics = np.stack([frame.intrinsic for frame in self.cache])
		all_cameras = torch.from_numpy(
			np.concatenate((cache_cameras, target_camera_to_world), axis=0)
		).unsqueeze(0).to(device=device)
		all_intrinsics = torch.from_numpy(
			np.concatenate((cache_intrinsics, target_intrinsics), axis=0)
		).unsqueeze(0).to(device=device)
		source_depths = {
			index: torch.from_numpy(frame.depth)[None, None].to(device=device)
			for index, frame in enumerate(self.cache)
		}
		rendered = forward_warp_indexed_frames(
			source_pixels=source_pixels,
			source_indices=list(range(len(self.cache))),
			source_camera_indices=list(range(len(self.cache))),
			target_camera_indices=list(
				range(len(self.cache), len(self.cache) + len(target_camera_to_world))
			),
			camera_to_world=all_cameras,
			intrinsic=all_intrinsics,
			source_depths=source_depths,
			height=self.height,
			width=self.width,
			depth_threshold=0.05,
			fill_value=0.0,
		)
		if rendered is None:
			raise RuntimeError("GEN3C could not render its seeded 3D cache")
		images, masks = rendered
		return (
			images.permute(0, 2, 1, 3, 4).unsqueeze(2),
			masks.permute(0, 2, 1, 3, 4).unsqueeze(2),
		)

	def _infer_chunk(
		self,
		req: InferenceRequest,
		start: int,
		stop: int,
		chunk_index: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		target_cameras, target_intrinsics = self._target_camera_data(
			InferenceRequest(
				request_id=req.request_id,
				timestamps=req.timestamps[start:stop],
				cameras_to_world=req.cameras_to_world[start:stop],
				focal_lengths=req.focal_lengths[start:stop],
				principal_points=req.principal_points[start:stop],
				resolutions=req.resolutions[start:stop],
				framerate=req.framerate,
				return_depths=req.return_depths,
				show_cache_renderings=req.show_cache_renderings,
				region_hint=req.region_hint,
			)
		)
		warp_images, warp_masks = self._render_cache(target_cameras, target_intrinsics)
		result = self.pipeline(
			images=self.cache[-1].image,
			prompt=req.region_hint,
			negative_prompt=self.negative_prompt or None,
			height=self.height,
			width=self.width,
			num_frames=self.frames_per_chunk,
			num_inference_steps=self.num_inference_steps,
			guidance_scale=self.guidance_scale,
			fps=int(round(req.framerate)),
			seed=self.seed + chunk_index,
			rendered_warp_images=warp_images,
			rendered_warp_masks=warp_masks,
			camera_to_world=target_cameras[None],
			camera_intrinsics=target_intrinsics[None],
			return_dict=True,
		)
		video = _video_to_thwc(result["video"])
		if len(video) != self.frames_per_chunk:
			raise ValueError(
				f"GEN3C returned {len(video)} frames; expected {self.frames_per_chunk}"
			)
		depth, mask, _ = self._predict_depth(video[-1])
		self.cache.append(
			_CacheFrame(
				image=video[-1],
				depth=np.where(mask, depth, 0.0).astype(np.float32),
				mask=mask,
				camera_to_world=target_cameras[-1],
				intrinsic=target_intrinsics[-1],
			)
		)
		self.cache = self.cache[-self.cache_limit :]
		return video, depth, mask

	async def run_inference(self, req: InferenceRequest) -> InferenceResult:
		async with self.inference_lock:
			start_time = time.monotonic()
			self._revert_cache = list(self.cache)
			videos, depths, masks, depth_indices = [], [], [], []
			try:
				for chunk_index, start in enumerate(range(0, len(req), self.frames_per_chunk)):
					stop = start + self.frames_per_chunk
					video, depth, mask = self._infer_chunk(
						req, start, stop, chunk_index
					)
					videos.append(video)
					depths.append(depth)
					masks.append(mask)
					depth_indices.append(stop - 1)
			except Exception:
				self.cache = self._revert_cache
				self._revert_cache = None
				raise

		images = np.concatenate(videos, axis=0)
		target_cameras, target_intrinsics = self._target_camera_data(req)
		return InferenceResult(
			request_id=req.request_id,
			result_ids=[f"{req.request_id}__frame_{index}" for index in range(len(images))],
			timestamps=np.asarray(req.timestamps, dtype=np.float64),
			cameras_to_world=target_cameras[:, :3],
			focal_lengths=target_intrinsics[:, (0, 1), (0, 1)],
			principal_points=np.stack(
				(
					target_intrinsics[:, 0, 2] / self.width,
					target_intrinsics[:, 1, 2] / self.height,
				),
				axis=1,
			),
			resolutions=np.tile([[self.width, self.height]], (len(images), 1)),
			frame_count_without_padding=req.frame_count_without_padding,
			images=images,
			depths=None,
			runtime_ms=(time.monotonic() - start_time) * 1000.0,
			predicted_depths=np.stack(depths),
			predicted_depth_indices=np.asarray(depth_indices, dtype=np.int32),
			predicted_masks=np.stack(masks),
		)

	async def revert_last_generation(self) -> dict[str, Any]:
		async with self.inference_lock:
			if self._revert_cache is None:
				return {"success": False, "message": "No generated segment is available to revert."}
			self.cache = self._revert_cache
			self._revert_cache = None
			return {"success": True, "message": "The latest generated segment was reverted."}

	def min_frames_per_request(self) -> int:
		return self.frames_per_chunk

	def max_frames_per_request(self) -> int:
		return self.frames_per_chunk * 100

	def inference_time_per_frame(self) -> float:
		return 4.0

	def inference_resolution(self) -> list[tuple[int, int]]:
		return [(self.width, self.height)]

	def default_framerate(self) -> float:
		return self.framerate

	def requires_seeding(self) -> bool:
		return True

	def metadata(self) -> dict[str, Any]:
		return {
			"model_name": "GEN3C",
			"model_version": (1, 0, 0),
			"aabb_min": self.aabb_min.tolist(),
			"aabb_max": self.aabb_max.tolist(),
			"min_frames_per_request": self.min_frames_per_request(),
			"max_frames_per_request": self.max_frames_per_request(),
			"inference_resolution": self.inference_resolution(),
			"inference_time_per_frame": self.inference_time_per_frame(),
			"default_framerate": self.default_framerate(),
			"requires_seeding": self.requires_seeding(),
			"camera_control": "arbitrary-authored-path",
		}

	def cleanup(self) -> None:
		cleanup = getattr(self.pipeline, "cleanup", None)
		if callable(cleanup):
			cleanup()


__all__ = ["Gen3CModel"]
