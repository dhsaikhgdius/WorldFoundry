"""Lyra Utils visual generation pipeline module."""

import importlib
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from PIL import Image
from worldfoundry.core.io import (
    VIDEO_EXTENSIONS,
    coerce_video_frames,
    load_video_frames,
    materialize_video_input,
    save_video_frames,
    video_tensor_to_uint8_frames,
)
from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path, project_root
from worldfoundry.core.io.scratch import make_scratch_dir


DEFAULT_LYRA2_ALIAS = "lyra-2"
DEFAULT_LYRA1_ALIAS = "lyra-1"


def lyra_runtime_root(name: str) -> Path:
    """Lyra runtime root helper function."""
    root = project_root(__file__)
    direct_root = root / "worldfoundry" / "synthesis" / "visual_generation"
    visual_generation_root = (
        direct_root
        if direct_root.is_dir()
        else root / "src" / "worldfoundry" / "synthesis" / "visual_generation"
    )
    if name == "lyra2_runtime":
        return visual_generation_root / "lyra_2"
    if name == "lyra1_runtime":
        return visual_generation_root / "lyra_1" / "lyra1_runtime"
    raise ValueError(f"Unsupported Lyra runtime name: {name}")


def _candidate_subdir(
    path_value: Path,
    repo_dir_name: str,
    sentinel_parts: Iterable[str],
) -> Optional[Path]:
    """Candidate subdir helper function."""
    sentinel_parts = tuple(sentinel_parts)
    if path_value.joinpath(*sentinel_parts).is_file():
        return path_value
    nested = path_value / repo_dir_name
    if nested.joinpath(*sentinel_parts).is_file():
        return nested
    return None


def _candidate_lyra2_subdir(path_value: Path) -> Optional[Path]:
    """Candidate lyra2 subdir helper function."""
    return _candidate_subdir(
        path_value,
        repo_dir_name="Lyra-2",
        sentinel_parts=("lyra_2", "__init__.py"),
    )


def _candidate_lyra2_checkpoint_subdir(path_value: Path) -> Optional[Path]:
    """Candidate lyra2 checkpoint subdir helper function."""
    candidates = [
        path_value,
        path_value / "Lyra-2.0",
        path_value / "Lyra-2",
    ]
    for candidate in candidates:
        checkpoint_dir = candidate / "checkpoints" / "model"
        negative_prompt_path = candidate / "checkpoints" / "text_encoder" / "negative_prompt.pt"
        da3_model_path = candidate / "checkpoints" / "recon" / "model.pt"
        if checkpoint_dir.is_dir() and negative_prompt_path.is_file() and da3_model_path.exists():
            return candidate
    return None


def _candidate_lyra1_subdir(path_value: Path) -> Optional[Path]:
    """Candidate lyra1 subdir helper function."""
    return _candidate_subdir(
        path_value,
        repo_dir_name="Lyra-1",
        sentinel_parts=("sample.py",),
    )


def _repo_candidates(repo_dir_name: str) -> list[Path]:
    """Repo candidates helper function."""
    runtime_name = "lyra2_runtime" if repo_dir_name == "Lyra-2" else "lyra1_runtime"
    return [
        lyra_runtime_root(runtime_name),
    ]


def local_repo_candidates() -> list[Path]:
    """Local repo candidates helper function."""
    return _repo_candidates("Lyra-2")


def local_lyra2_checkpoint_candidates() -> list[Path]:
    """Local lyra2 checkpoint candidates helper function."""
    ckpt_root = checkpoint_root_path()
    hfd_root = hfd_root_path()
    return [
        hfd_root / "Lyra-2.0",
        hfd_root / "nvidia--Lyra-2.0",
        ckpt_root / "Lyra-2.0",
        ckpt_root / "Lyra-2",
    ]


def default_local_repo() -> Optional[Path]:
    """Default local repo helper function."""
    for candidate in local_repo_candidates():
        resolved = _candidate_lyra2_subdir(candidate)
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def local_lyra1_repo_candidates() -> list[Path]:
    """Local lyra1 repo candidates helper function."""
    return _repo_candidates("Lyra-1")


def default_local_lyra2_checkpoint_root() -> Optional[Path]:
    """Default local lyra2 checkpoint root helper function."""
    for candidate in local_lyra2_checkpoint_candidates():
        resolved = _candidate_lyra2_checkpoint_subdir(candidate)
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def default_local_lyra1_repo() -> Optional[Path]:
    """Default local lyra1 repo helper function."""
    for candidate in local_lyra1_repo_candidates():
        resolved = _candidate_lyra1_subdir(candidate)
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def local_lyra1_checkpoint_candidates() -> list[Path]:
    """Local lyra1 checkpoint candidates helper function."""
    ckpt_root = checkpoint_root_path()
    hfd_root = hfd_root_path()
    return [
        hfd_root / "Lyra",
        hfd_root / "nvidia--Lyra",
        ckpt_root / "Lyra",
        ckpt_root / "Lyra-1" / "checkpoints",
    ]


def default_local_lyra1_checkpoint_root() -> Optional[Path]:
    """Default local lyra1 checkpoint root helper function."""
    for candidate in local_lyra1_checkpoint_candidates():
        if _candidate_lyra1_weights_subdir(candidate) is not None:
            return candidate.resolve()
    return None


def resolve_repo_root(pretrained_model_path: Optional[str]) -> str:
    """Resolve repo root helper function."""
    default_runtime = default_local_repo()
    if pretrained_model_path is None or str(pretrained_model_path).strip() in {
        "",
        DEFAULT_LYRA2_ALIAS,
        "lyra",
        "lyra2",
        "Lyra-2",
        "Lyra",
        "nvidia/Lyra-2.0",
    }:
        if default_runtime is None:
            raise FileNotFoundError(
                "Unable to locate the vendored Lyra-2 runtime under WorldFoundry."
            )
        return str(default_runtime)

    candidate = Path(str(pretrained_model_path)).expanduser()
    checkpoint_root = _candidate_lyra2_checkpoint_subdir(candidate)
    if checkpoint_root is not None:
        if default_runtime is None:
            raise FileNotFoundError(
                "Lyra-2 checkpoints were found, but the vendored Lyra-2 runtime is missing."
            )
        return str(default_runtime)

    raise FileNotFoundError(
        f"Lyra-2 checkpoint directory not found at '{pretrained_model_path}'. "
        "External Lyra-2 source repositories are not accepted at runtime."
    )


def resolve_lyra1_repo_root(pretrained_model_path: Optional[str]) -> str:
    """Resolve lyra1 repo root helper function."""
    default_runtime = default_local_lyra1_repo()
    if pretrained_model_path is None or str(pretrained_model_path).strip() in {
        "",
        DEFAULT_LYRA1_ALIAS,
        "lyra1",
        "lyra-1",
        "Lyra-1",
        "Lyra",
        "nvidia/Lyra",
    }:
        if default_runtime is None:
            raise FileNotFoundError(
                "Unable to locate the vendored Lyra-1 runtime under WorldFoundry."
            )
        return str(default_runtime)

    candidate = Path(str(pretrained_model_path)).expanduser()
    if _candidate_lyra1_weights_subdir(candidate) is not None or _lyra1_checkpoint_layout_complete(candidate):
        if default_runtime is None:
            raise FileNotFoundError(
                "Lyra-1 checkpoints were found, but the vendored Lyra-1 runtime is missing."
            )
        return str(default_runtime)

    raise FileNotFoundError(
        f"Lyra-1 checkpoint directory not found at '{pretrained_model_path}'. "
        "External Lyra-1 source repositories are not accepted at runtime."
    )


def _candidate_lyra1_weights_subdir(path_value: Path) -> Optional[Path]:
    """Candidate lyra1 weights subdir helper function."""
    candidates = [
        path_value,
        path_value / "Lyra",
        path_value / "checkpoints" / "Lyra",
    ]
    for candidate in candidates:
        if (
            (candidate / "lyra_static.pt").is_file()
            and (candidate / "lyra_dynamic.pt").is_file()
        ):
            return candidate.resolve()
    return None


def _lyra1_checkpoint_layout_complete(root: Path) -> bool:
    """Lyra1 checkpoint layout complete helper function."""
    required_files = [
        root / "Gen3C-Cosmos-7B" / "model.pt",
        root / "Cosmos-Tokenize1-CV8x8x8-720p" / "mean_std.pt",
        root / "google-t5" / "t5-11b" / "config.json",
        root / "Lyra" / "lyra_static.pt",
        root / "Lyra" / "lyra_dynamic.pt",
    ]
    return all(path.exists() for path in required_files)


def _reset_dir(path: Path):
    """Reset dir helper function."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        # Safely clean up any pre-existing output directory structure
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path):
    """Link or copy helper function."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _reset_dir(dst)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            # Copy source file to target directory while preserving original metadata
            shutil.copy2(src, dst)


def prepare_lyra1_checkpoint_root(
    checkpoint_dir: Optional[str],
    repo_root: Optional[str] = None,
) -> str:
    """Prepare lyra1 checkpoint root helper function."""
    from ...synthesis.visual_generation.gen3c.runtime_env import (
        prepare_gen3c_checkpoint_root,
    )

    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())
    if repo_root is not None:
        candidates.append(Path(repo_root).expanduser() / "checkpoints")

    for candidate in candidates:
        if candidate.exists() and _lyra1_checkpoint_layout_complete(candidate):
            return str(candidate.resolve())

    lyra_weights_root = None
    for candidate in candidates:
        if candidate.exists():
            lyra_weights_root = _candidate_lyra1_weights_subdir(candidate)
            if lyra_weights_root is not None:
                break
    if lyra_weights_root is None:
        lyra_weights_root = default_local_lyra1_checkpoint_root()
    if lyra_weights_root is None:
        raise FileNotFoundError(
            "Unable to locate local Lyra-1 weights. "
            "Expected a directory containing 'Lyra/lyra_static.pt' and 'Lyra/lyra_dynamic.pt', "
            "or pass required_components['checkpoint_dir'] pointing to that layout."
        )

    gen3c_checkpoint_root = Path(
        prepare_gen3c_checkpoint_root(
            checkpoint_dir=checkpoint_dir,
        )
    ).expanduser().resolve()

    if _lyra1_checkpoint_layout_complete(gen3c_checkpoint_root):
        return str(gen3c_checkpoint_root)

    stage_root = project_root(__file__) / "cache" / "runtime" / "lyra1_checkpoints"
    if stage_root.exists():
        _reset_dir(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    for item in gen3c_checkpoint_root.iterdir():
        _link_or_copy(item, stage_root / item.name)

    target_lyra_dir = stage_root / "Lyra"
    if lyra_weights_root.name == "Lyra":
        _link_or_copy(lyra_weights_root, target_lyra_dir)
    else:
        target_lyra_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(lyra_weights_root / "lyra_static.pt", target_lyra_dir / "lyra_static.pt")
        _link_or_copy(lyra_weights_root / "lyra_dynamic.pt", target_lyra_dir / "lyra_dynamic.pt")

    return str(stage_root.resolve())


def ensure_repo_on_path(repo_root: str):
    """Ensure repo on path helper function."""
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    importlib.invalidate_caches()


def configure_lyra_runtime_env():
    # TransformerEngine fused attention currently crashes in this Lyra-2 setup on H100.
    # Keep FlashAttention/unfused paths enabled, but disable the fused backend by default.
    """Configure lyra runtime env helper function."""
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")


def resolve_path(path_value: Optional[str], repo_root: str) -> Optional[str]:
    """Resolve path helper function."""
    if path_value is None:
        return None
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(repo_root) / candidate).resolve())


def resolve_checkpoint_root(
    pretrained_model_path: Optional[str],
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve checkpoint root helper function."""
    candidates = []
    if pretrained_model_path is not None and str(pretrained_model_path).strip():
        candidates.append(Path(str(pretrained_model_path)).expanduser())
    if repo_root is not None and str(repo_root).strip():
        candidates.append(Path(str(repo_root)).expanduser())

    for candidate in candidates:
        resolved = _candidate_lyra2_checkpoint_subdir(candidate)
        if resolved is not None:
            return str(resolved.resolve())

    default_root = default_local_lyra2_checkpoint_root()
    if default_root is not None:
        return str(default_root)
    return None


def resolve_required_paths(
    repo_root: str,
    checkpoint_dir: Optional[str] = None,
    negative_prompt_path: Optional[str] = None,
    da3_model_path_custom: Optional[str] = None,
    weights_root: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve required paths helper function."""
    base_root = weights_root or repo_root
    paths = {
        "checkpoint_dir": resolve_path(checkpoint_dir or "checkpoints/model", base_root),
        "negative_prompt_path": resolve_path(
            negative_prompt_path or "checkpoints/text_encoder/negative_prompt.pt",
            base_root,
        ),
        "da3_model_path_custom": resolve_path(
            da3_model_path_custom or "checkpoints/recon/model.pt",
            base_root,
        ),
    }
    return paths


def ensure_path_exists(path_value: Optional[str], name: str):
    """Ensure path exists helper function."""
    if path_value is None:
        raise FileNotFoundError(f"{name} is not configured.")
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{name} not found: {path_value}")


def device_index(device: Optional[str]) -> int:
    """Device index helper function."""
    if not device or not str(device).startswith("cuda"):
        return 0
    if ":" in str(device):
        return int(str(device).split(":", maxsplit=1)[1])
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def maybe_set_cuda_device(device: Optional[str]):
    """Maybe set cuda device helper function."""
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device_index(device))


def prepare_lyra2_runtime_root(repo_root: str, weights_root: Optional[str] = None) -> str:
    """Prepare lyra2 runtime root helper function."""
    repo_root_path = Path(repo_root).expanduser().resolve()
    weights_root_path = Path(weights_root).expanduser().resolve() if weights_root else repo_root_path

    if weights_root_path == repo_root_path and (repo_root_path / "checkpoints").is_dir():
        return str(repo_root_path)

    runtime_root = make_scratch_dir(prefix="lyra2_runtime_").resolve()
    (runtime_root / "lyra_2").symlink_to(repo_root_path / "lyra_2", target_is_directory=True)
    checkpoints_src = weights_root_path / "checkpoints"
    if checkpoints_src.is_dir():
        (runtime_root / "checkpoints").symlink_to(checkpoints_src, target_is_directory=True)
    return str(runtime_root)


@contextmanager
def working_directory(path: str):
    """Working directory helper function."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_pil_image(image_input) -> Image.Image:
    """Load pil image helper function."""
    if isinstance(image_input, (list, tuple)):
        items = [item for item in image_input if item is not None]
        if len(items) != 1:
            raise TypeError(f"Expected exactly one image input, got {len(items)}")
        return load_pil_image(items[0])
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, np.ndarray):
        array = image_input
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if torch.is_tensor(image_input):
        tensor = image_input.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def build_subprocess_env(repo_root: str) -> dict:
    """Build subprocess env helper function."""
    env = os.environ.copy()
    root = project_root(__file__)
    pythonpath_entries = [repo_root, str(root / "src"), str(root)]
    for candidate in [
        root / "src" / "worldfoundry" / "base_models" / "three_dimensions" / "general_3d" / "eastern_journalist",
    ]:
        if candidate.exists():
            pythonpath_entries.append(str(candidate.resolve()))
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def materialize_image_input(
    image_input,
    output_dir: str,
    filename: str = "input.png",
) -> str:
    """Materialize image input helper function."""
    image = load_pil_image(image_input)
    output_path = Path(output_dir).expanduser().resolve() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)
