"""Reusable camera-pose and Plücker-ray preprocessing primitives."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from packaging import version as pver
from PIL import Image


class RandomHorizontalFlipWithPose(nn.Module):
    """Random horizontal flip with pose implementation."""
    def __init__(self, p: float = 0.5) -> None:
        """Init.

        Args:
            p: The p.

        Returns:
            The return value.
        """
        super().__init__()
        self.p = p

    def get_flip_flag(self, n_image: int) -> torch.Tensor:
        """Get flip flag.

        Args:
            n_image: The n image.

        Returns:
            The return value.
        """
        if torch.rand(1).item() < self.p:
            return torch.zeros(n_image, dtype=torch.bool)
        return torch.ones(n_image, dtype=torch.bool)

    def forward(self, image: torch.Tensor, flip_flag: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            image: The image.
            flip_flag: The flip flag.

        Returns:
            The return value.
        """
        from torchvision.transforms import functional as tvF

        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for should_flip, img in zip(flip_flag, image):
            if should_flip:
                ret_images.append(tvF.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera:
    """Camera implementation."""
    def __init__(self, entry: Sequence[float]) -> None:
        """Init.

        Args:
            entry: The entry.

        Returns:
            The return value.
        """
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def create_camera_params_from_batch(extrinsics_np: np.ndarray, intrinsics_np: np.ndarray) -> list[Camera]:
    """Create camera params from batch.

    Args:
        extrinsics_np: The extrinsics np.
        intrinsics_np: The intrinsics np.

    Returns:
        The return value.
    """
    cam_params = []
    for i, (ext_mat, int_mat) in enumerate(zip(extrinsics_np, intrinsics_np)):
        fx, fy = int_mat[0, 0], int_mat[1, 1]
        cx, cy = int_mat[0, 2], int_mat[1, 2]
        entry = [i, fx, fy, cx, cy, 0, 0] + ext_mat.flatten().tolist()
        cam_params.append(Camera(entry))
    return cam_params


def custom_meshgrid(*args):
    """Custom meshgrid."""
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    return torch.meshgrid(*args, indexing="ij")


def ray_condition(
    K: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    device: str | torch.device,
    flip_flag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ray condition.

    Args:
        K: The k.
        c2w: The c2w.
        H: The h.
        W: The w.
        device: The device.
        flip_flag: The flip flag.

    Returns:
        The return value.
    """
    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(B, c2w.shape[1], H, W, 6)


class RealEstate10KPoseProcessor:
    """Real estate k pose processor implementation."""
    def __init__(
        self,
        sample_stride: int = 4,
        minimum_sample_stride: int = 1,
        sample_n_frames: int = 16,
        relative_pose: bool = False,
        zero_t_first_frame: bool = False,
        sample_size: int | Sequence[int] = (256, 384),
        rescale_fxy: bool = False,
        shuffle_frames: bool = False,
        use_flip: bool = False,
        return_clip_name: bool = False,
        is_i2v: bool = False,
        pose_encoding_to_extri_intri: Callable | None = None,
    ) -> None:
        """Init.

        Args:
            sample_stride: The sample stride.
            minimum_sample_stride: The minimum sample stride.
            sample_n_frames: The sample n frames.
            relative_pose: The relative pose.
            zero_t_first_frame: The zero t first frame.
            sample_size: The sample size.
            rescale_fxy: The rescale fxy.
            shuffle_frames: The shuffle frames.
            use_flip: The use flip.
            return_clip_name: The return clip name.
            is_i2v: The is i2v.
            pose_encoding_to_extri_intri: The pose encoding to extri intri.

        Returns:
            The return value.
        """
        import torchvision.transforms as transforms

        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name
        self.is_i2v = is_i2v
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

        size = (sample_size, sample_size) if isinstance(sample_size, int) else tuple(sample_size)
        self.sample_size = size
        if use_flip:
            pixel_transforms = [
                transforms.Resize(size),
                RandomHorizontalFlipWithPose(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        else:
            pixel_transforms = [
                transforms.Resize(size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = size[1] / size[0]
        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params: Sequence[Camera]) -> np.ndarray:
        """Get relative pose.

        Args:
            cam_params: The cam params.

        Returns:
            The return value.
        """
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        cam_to_origin = 0 if self.zero_t_first_frame else np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -cam_to_origin],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        return np.array(ret_poses, dtype=np.float32)

    def load_cameras(self, pose_file: str) -> list[Camera]:
        """Load cameras.

        Args:
            pose_file: The pose file.

        Returns:
            The return value.
        """
        with open(pose_file, "r") as f:
            poses = f.readlines()
        if "youtube" in poses[0]:
            poses = [pose.strip().split(" ") for pose in poses[1:]]
        else:
            poses = [pose.strip().split(" ") for pose in poses]
        cam_params = [[float(x) for x in pose] for pose in poses]
        return [Camera(cam_param) for cam_param in cam_params]

    def _sample_camera_params(self, cam_params: Sequence[Camera]) -> list[Camera]:
        """Helper function to sample camera params.

        Args:
            cam_params: The cam params.

        Returns:
            The return value.
        """
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)
        current_sample_stride = self.sample_stride
        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = 0
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)
        if self.shuffle_frames:
            frame_indices = frame_indices[np.random.permutation(self.sample_n_frames)]
        return [cam_params[index] for index in frame_indices]

    def _rescale_focal_lengths(self, cam_params: Sequence[Camera], image_path: str | None) -> None:
        """Helper function to rescale focal lengths.

        Args:
            cam_params: The cam params.
            image_path: The image path.

        Returns:
            The return value.
        """
        if not self.rescale_fxy:
            return
        if image_path is None:
            raise ValueError("image_path is required when rescale_fxy=True")
        ori_w, ori_h = Image.open(image_path).size
        ori_wh_ratio = ori_w / ori_h
        if ori_wh_ratio > self.sample_wh_ratio:
            resized_ori_w = self.sample_size[0] * ori_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
        else:
            resized_ori_h = self.sample_size[1] / ori_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]

    def _plucker_from_camera_params(
        self,
        cam_params: Sequence[Camera],
        image_path: str | None = None,
    ) -> torch.Tensor:
        """Helper function to plucker from camera params.

        Args:
            cam_params: The cam params.
            image_path: The image path.

        Returns:
            The return value.
        """
        cam_params = self._sample_camera_params(cam_params)
        self._rescale_focal_lengths(cam_params, image_path)

        intrinsics = np.asarray(
            [
                [
                    cam_param.fx * self.sample_size[1],
                    cam_param.fy * self.sample_size[0],
                    cam_param.cx * self.sample_size[1],
                    cam_param.cy * self.sample_size[0],
                ]
                for cam_param in cam_params
            ],
            dtype=np.float32,
        )
        intrinsics_tensor = torch.as_tensor(intrinsics)[None]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)
        return ray_condition(
            intrinsics_tensor,
            c2w,
            self.sample_size[0],
            self.sample_size[1],
            device="cpu",
            flip_flag=flip_flag,
        )

    def get_plucker_embedding(self, pose_file: str, image_path: str | None = None) -> torch.Tensor:
        """Get plucker embedding.

        Args:
            pose_file: The pose file.
            image_path: The image path.

        Returns:
            The return value.
        """
        return self._plucker_from_camera_params(self.load_cameras(pose_file), image_path=image_path)

    def get_plucker_embedding_direct_from_cam_params(
        self,
        pose_enc: torch.Tensor,
        image_size,
        image_path: str | None = None,
        pose_encoding_to_extri_intri: Callable | None = None,
    ) -> torch.Tensor:
        """Get plucker embedding direct from cam params.

        Args:
            pose_enc: The pose enc.
            image_size: The image size.
            image_path: The image path.
            pose_encoding_to_extri_intri: The pose encoding to extri intri.

        Returns:
            The return value.
        """
        converter = pose_encoding_to_extri_intri or self.pose_encoding_to_extri_intri
        if converter is None:
            raise ValueError("pose_encoding_to_extri_intri must be provided for pose encoding input.")
        extrinsic, intrinsic = converter(pose_enc, image_size, pose_encoding_type="absT_quaR_FoV")
        extrinsics_np = extrinsic.cpu().numpy().squeeze(0)
        intrinsics_np = intrinsic.cpu().numpy().squeeze(0)
        cam_params = create_camera_params_from_batch(extrinsics_np, intrinsics_np)
        return self._plucker_from_camera_params(cam_params, image_path=image_path)
