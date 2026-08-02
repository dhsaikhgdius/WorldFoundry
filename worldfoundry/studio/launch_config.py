"""CLI and process environment for launching WorldFoundry Studio."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .catalog import CatalogEntry, find_entry
from .runtime_paths import studio_path_summary
from .visualization.core.registry import (
    EMBODIED_VISUALIZATION,
    INTERACTIVE_WORLD_VISUALIZATION,
    MEDIA_VISUALIZATION,
    NATIVE_WORLD_EXPLORER_VISUALIZATION,
    RERUN_VISUALIZATION,
    SPARK_VISUALIZATION,
    UNIFIED_VISUALIZATION,
    VISER_VISUALIZATION,
)

LINGBOT_WORLD_MODEL_ID = "lingbot-world"
LINGBOT_WORLD_V2_MODEL_ID = "lingbot-world-v2"
MATRIX_GAME3_MODEL_ID = "matrix-game-3"
LONGVIE2_MODEL_ID = "longvie-2"
HELIOS_MODEL_ID = "helios"
DREAMX_WORLD_MODEL_ID = "dreamx-world-5b-cam"
LINGBOT_VARIANT_FAST = "fast"
LINGBOT_FAST_NUM_PROCS_ENV_KEYS = (
    "WORLDFOUNDRY_REALTIME_NPROC_PER_NODE",
    "WORLDFOUNDRY_REALTIME_NPROC",
    "WORLDFOUNDRY_STUDIO_LINGBOT_TORCHRUN_NPROC",
    "WM_LINGBOTWORLDFAST_NUM_PROCS",
)
LINGBOT_FAST_USE_SP_ENV = "WORLDFOUNDRY_LINGBOT_FAST_USE_SP"
CLI_BACKEND_CHOICES = frozenset({"auto", "from_pretrained", "api_init"})
CLI_FRONTEND_CHOICES = frozenset(
    {
        "auto",
        INTERACTIVE_WORLD_VISUALIZATION,
        "interactive-world",
        "world-model",
        NATIVE_WORLD_EXPLORER_VISUALIZATION,
        "world-explorer",
        "imgui",
        VISER_VISUALIZATION,
        "viser",
        "geometry",
        "pointcloud",
        "point-cloud",
        EMBODIED_VISUALIZATION,
        "sim",
        "simulator",
        MEDIA_VISUALIZATION,
        "preview",
        "video",
        "image",
        RERUN_VISUALIZATION,
        "rrd",
        SPARK_VISUALIZATION,
        "3dgs",
        "splat",
        "gaussian-splat",
        UNIFIED_VISUALIZATION,
    }
)


def _torchrun_world_size() -> int:
    try:
        return max(int(os.getenv("WORLD_SIZE", "1") or "1"), 1)
    except Exception:
        return 1


def _cuda_visible_device_count(value: str | None = None) -> int:
    text = os.getenv("CUDA_VISIBLE_DEVICES", "") if value is None else str(value or "")
    text = text.strip()
    if not text:
        return 0
    return len([item for item in text.split(",") if item.strip()])


def resolve_lingbot_fast_num_procs(*, visible_count: int | None = None) -> int:
    """Resolve LingBot-World-Fast torchrun process count using WMFactory semantics."""

    if visible_count is None:
        visible_count = _cuda_visible_device_count()
        if visible_count <= 0:
            try:
                import torch

                visible_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            except Exception:
                visible_count = 0
    visible_count = max(int(visible_count or 0), 0)
    # The public LingBot 14B recipe is eight-way FSDP + Ulysses.  Four ranks
    # are the compact supported topology because 40 attention heads divide
    # evenly; prefer all eight ranks when the host exposes them.
    default = 8 if visible_count >= 8 else (4 if visible_count >= 4 else (2 if visible_count >= 2 else 1))
    for key in LINGBOT_FAST_NUM_PROCS_ENV_KEYS:
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            nproc = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer, got {raw!r}.") from exc
        if nproc < 1:
            raise ValueError(f"{key} must be >= 1.")
        if visible_count and nproc > visible_count:
            raise ValueError(f"{key}={nproc} exceeds visible CUDA devices ({visible_count}).")
        return nproc
    return default


def lingbot_fast_sequence_parallel_enabled(*, world_size: int | None = None) -> bool:
    """Return whether LingBot-World-Fast should use sequence parallel inference."""

    raw = os.getenv(LINGBOT_FAST_USE_SP_ENV, "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    size = _torchrun_world_size() if world_size is None else max(int(world_size or 1), 1)
    return size > 1


@dataclass(frozen=True)
class WMFactoryInteractiveModelSpec:
    model_id: str
    wmfactory_model_id: str
    env_prefix: str
    preferred_visible_devices: int | None = None
    use_dual_device_hint: bool = False
    supported_process_counts: tuple[int, ...] = ()


WMFACTORY_INTERACTIVE_MODEL_SPECS: tuple[WMFactoryInteractiveModelSpec, ...] = (
    WMFactoryInteractiveModelSpec("matrix-game-2", "matrixgame", "MATRIXGAME"),
    WMFactoryInteractiveModelSpec(
        MATRIX_GAME3_MODEL_ID,
        "matrixgame3",
        "MATRIXGAME3",
        preferred_visible_devices=4,
        supported_process_counts=(1, 2, 4, 8),
    ),
    WMFactoryInteractiveModelSpec("yume", "yume", "YUME"),
    WMFactoryInteractiveModelSpec("yume-1p5", "yume", "YUME"),
    WMFactoryInteractiveModelSpec("diamond", "diamond", "DIAMOND"),
    WMFactoryInteractiveModelSpec("oasis-500m", "open-oasis", "OPENOASIS"),
    WMFactoryInteractiveModelSpec("vid2world", "vid2world", "VID2WORLD"),
    WMFactoryInteractiveModelSpec(
        "infinite-world",
        "infinite-world",
        "INFINITEWORLD",
        preferred_visible_devices=2,
        use_dual_device_hint=True,
    ),
    WMFactoryInteractiveModelSpec(
        "hunyuan-worldplay",
        "worldplay",
        "WORLDPLAY",
        preferred_visible_devices=8,
        supported_process_counts=(1, 2, 3, 4, 6, 8),
    ),
    WMFactoryInteractiveModelSpec("mineworld", "mineworld", "MINEWORLD"),
    WMFactoryInteractiveModelSpec(
        "lingbot-world",
        "lingbot-world-fast",
        "LINGBOTWORLDFAST",
        preferred_visible_devices=8,
        supported_process_counts=(1, 4, 8),
    ),
    WMFactoryInteractiveModelSpec(
        "lingbot-world-v2",
        "lingbot-world-v2",
        "LINGBOTWORLDV2",
        preferred_visible_devices=8,
        supported_process_counts=(1, 4, 8),
    ),
)

_WMFACTORY_SPEC_BY_MODEL_ID = {spec.model_id: spec for spec in WMFACTORY_INTERACTIVE_MODEL_SPECS}
_WMFACTORY_MODEL_ALIASES = {
    "matrixgame": "matrix-game-2",
    "matrixgame2": "matrix-game-2",
    "matrix-game2": "matrix-game-2",
    "matrixgame3": "matrix-game-3",
    "matrix-game3": "matrix-game-3",
    "yume-1.5": "yume-1p5",
    "yume1.5": "yume-1p5",
    "open-oasis": "oasis-500m",
    "openoasis": "oasis-500m",
    "open_oasis": "oasis-500m",
    "infiniteworld": "infinite-world",
    "infinite_world": "infinite-world",
    "worldplay": "hunyuan-worldplay",
    "hy-worldplay": "hunyuan-worldplay",
    "hyworldplay": "hunyuan-worldplay",
    "lingbot-world-fast": "lingbot-world",
    "lingbotworld-fast": "lingbot-world",
    "lingbotworldfast": "lingbot-world",
}


def wmfactory_interactive_model_spec(
    model_id: str,
    *,
    load_kwargs: Mapping[str, Any] | None = None,
) -> WMFactoryInteractiveModelSpec | None:
    """Return the WMFactory compatibility spec for a WorldFoundry interactive model."""

    key = (model_id or "").strip().lower()
    key = _WMFACTORY_MODEL_ALIASES.get(key, key)
    spec = _WMFACTORY_SPEC_BY_MODEL_ID.get(key)
    if spec is None:
        return None
    if spec.model_id == LINGBOT_WORLD_MODEL_ID:
        runtime_variant = str((load_kwargs or {}).get("runtime_variant") or "").strip().lower()
        if runtime_variant and runtime_variant != LINGBOT_VARIANT_FAST:
            return None
    return spec


@dataclass(frozen=True)
class StudioLaunchConfig:
    model_id: str
    variant_id: str | None = None
    model_ref: str = ""
    device: str = "cuda"
    backend: str = "auto"
    endpoint: str = ""
    show_aux_panels: bool = False
    frontend: str = "auto"
    asset_path: str = ""
    simulator_url: str = ""
    host: str = ""
    port: int | None = None


def env_first(*names: str) -> str:
    """Return the first non-empty environment value from ordered keys.

    Args:
        names: Environment variable names in priority order.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return ""


def build_launch_argument_parser(prog: str = "worldfoundry.studio.app") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Launch a WorldFoundry Studio frontend with one fixed model session.",
        epilog=(
            "Path roots: WORLDFOUNDRY_STUDIO_WORKSPACE_DIR controls Studio runs; "
            "WORLDFOUNDRY_MODEL_DIR and WORLDFOUNDRY_CACHE_DIR are searched for local checkpoints/repos."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Studio model id or alias to lock at launch, for example `lingbot-world`.",
    )
    parser.add_argument(
        "--model",
        dest="model_flag",
        help="Studio model id or alias to lock at launch.",
    )
    parser.add_argument(
        "--variant",
        help="Optional variant id or alias, for example `fast` or `base-camera`.",
    )
    parser.add_argument(
        "--model-ref",
        "--ckpt",
        "--checkpoint",
        dest="model_ref",
        help="Checkpoint path or repo id override for the fixed model.",
    )
    parser.add_argument(
        "--device",
        help="Default runtime device shown in the UI, for example `cuda:1`.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(CLI_BACKEND_CHOICES),
        help="Loader override for the fixed session.",
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint override for hosted or API-backed models.",
    )
    parser.add_argument(
        "--show-aux-panels",
        action="store_true",
        help="Expose the optional history, notes, and catalog side panels.",
    )
    parser.add_argument(
        "--frontend",
        choices=sorted(CLI_FRONTEND_CHOICES),
        help=(
            "Frontend surface to launch. `native-world` is the CUDA/C++/ImGui world explorer, "
            "`world` is the game-console shell, "
            "`points` is native Viser, `spark` is a standalone 3DGS viewer, "
            "`embodied` preserves an external simulator UI, `unified` is the Gradio shell, "
            "and `auto` routes by model type."
        ),
    )
    parser.add_argument(
        "--asset",
        dest="asset_path",
        help="Frontend asset path for native viewers, e.g. a .ply/.npz point cloud or .splat/.spz/.ply 3DGS file.",
    )
    parser.add_argument(
        "--simulator-url",
        help="Native embodied simulator URL to advertise instead of embedding in the game-console frontend.",
    )
    parser.add_argument(
        "--host",
        help="Frontend bind host. Defaults to loopback unless environment variables override it.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Frontend bind port. Defaults depend on the selected frontend.",
    )
    return parser


def launch_environment_summary() -> dict[str, str | list[str]]:
    """Return startup path discovery details for logs, tests, and readiness checks.

    Args:
        None.
    """

    return studio_path_summary()


def launch_uses_lingbot_torchrun_rollout(launch_config: StudioLaunchConfig) -> bool:
    world_size = _torchrun_world_size()
    if launch_config.model_id == LONGVIE2_MODEL_ID:
        if world_size not in {1, 4}:
            raise ValueError(
                "LongVie 2 supports either one GPU or the official four-rank USP topology; "
                f"got WORLD_SIZE={world_size}."
            )
        return world_size == 4
    if world_size <= 1:
        return False
    if launch_config.model_id == LINGBOT_WORLD_V2_MODEL_ID:
        if world_size not in {4, 8}:
            raise ValueError(
                "LingBot-World-V2 Workspace supports four or eight torchrun ranks; "
                f"got WORLD_SIZE={world_size}."
            )
        return True
    if (
        launch_config.model_id == LINGBOT_WORLD_MODEL_ID
        and launch_config.variant_id == LINGBOT_VARIANT_FAST
    ):
        if world_size not in {4, 8}:
            raise ValueError(
                "LingBot-World Fast Workspace supports four or eight torchrun ranks; "
                f"got WORLD_SIZE={world_size}."
            )
        return True
    if launch_config.model_id in {
        MATRIX_GAME3_MODEL_ID,
        HELIOS_MODEL_ID,
        DREAMX_WORLD_MODEL_ID,
    }:
        return True
    return False


def parse_launch_config(
    argv: Sequence[str] | None = None,
    *,
    entries: Sequence[CatalogEntry] | None = None,
    studio_catalog: Callable[[Sequence[CatalogEntry] | None], tuple[CatalogEntry, ...]] | None = None,
    resolve_cli_variant_id: Callable[[CatalogEntry, str | None], str | None] | None = None,
) -> StudioLaunchConfig:
    """Parse argv plus env overrides into a frozen launch configuration.

    Args:
        argv: Optional CLI tokens; forwarded to argparse (defaults match ``parse_args(None)``).
        entries: Optional catalog rows for fallback model ordering when env is unset.
        studio_catalog: Sorted catalog factory; lazily resolves to the lightweight Studio catalog helper when omitted.
        resolve_cli_variant_id: Variant resolver; lazily resolves to the lightweight variant helper when omitted.
    """
    if studio_catalog is None or resolve_cli_variant_id is None:
        if studio_catalog is None:
            from .studio_catalog import _studio_catalog

            studio_catalog = _studio_catalog
        if resolve_cli_variant_id is None:
            from .variants import resolve_cli_variant_id

    parser = build_launch_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw_model_flag = (args.model_flag or "").strip()
    raw_model_positional = (args.model or "").strip()
    if raw_model_flag and raw_model_positional and raw_model_flag.lower() != raw_model_positional.lower():
        parser.error("Use either the positional model argument or `--model`, not two different model ids.")

    raw_model = (
        raw_model_flag
        or raw_model_positional
        or env_first("WORLDFOUNDRY_STUDIO_MODEL").strip()
    )
    if not raw_model:
        catalog = tuple(entries) if entries is not None else studio_catalog()
        fallback_model_id = catalog[0].model_id if catalog else ""
        raw_model = fallback_model_id
    if not raw_model:
        parser.error("Studio catalog is empty; no launchable model is available.")

    try:
        entry = find_entry(raw_model)
    except Exception as exc:
        parser.error(str(exc))

    try:
        variant_id = resolve_cli_variant_id(
            entry,
            (
                args.variant
                or env_first("WORLDFOUNDRY_STUDIO_VARIANT")
            ).strip()
            or None,
        )
    except Exception as exc:
        parser.error(str(exc))

    backend = (
        (
            args.backend
            or env_first("WORLDFOUNDRY_STUDIO_BACKEND")
        ).strip()
        or entry.default_backend
        or "auto"
    )
    if backend not in CLI_BACKEND_CHOICES:
        parser.error(
            f"Unsupported backend `{backend}`. Use one of: {', '.join(sorted(CLI_BACKEND_CHOICES))}."
        )

    frontend = (
        args.frontend
        or env_first("WORLDFOUNDRY_STUDIO_FRONTEND")
        or "auto"
    ).strip().lower()
    if frontend not in CLI_FRONTEND_CHOICES:
        parser.error(
            f"Unsupported frontend `{frontend}`. Use one of: {', '.join(sorted(CLI_FRONTEND_CHOICES))}."
        )

    port_text = (
        str(args.port)
        if args.port is not None
        else env_first("WORLDFOUNDRY_STUDIO_PORT")
        or os.getenv("GRADIO_SERVER_PORT")
        or ""
    ).strip()
    try:
        port = int(port_text) if port_text else None
    except ValueError:
        parser.error(f"Invalid Studio port `{port_text}`.")

    return StudioLaunchConfig(
        model_id=entry.model_id,
        variant_id=variant_id,
        model_ref=(
            args.model_ref
            or env_first("WORLDFOUNDRY_STUDIO_FIXED_MODEL_REF")
        ).strip(),
        device=(
            args.device
            or env_first("WORLDFOUNDRY_STUDIO_DEVICE")
        ).strip()
        or "cuda",
        backend=backend,
        endpoint=(
            args.endpoint
            or env_first("WORLDFOUNDRY_STUDIO_ENDPOINT")
        ).strip()
        or entry.default_endpoint
        or "",
        show_aux_panels=bool(
            args.show_aux_panels
            or env_first("WORLDFOUNDRY_STUDIO_SHOW_AUX_PANELS").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        frontend=frontend,
        asset_path=(
            args.asset_path
            or env_first("WORLDFOUNDRY_STUDIO_ASSET")
        ).strip(),
        simulator_url=(
            args.simulator_url
            or env_first("WORLDFOUNDRY_STUDIO_SIMULATOR_URL")
        ).strip(),
        host=(
            args.host
            or env_first("WORLDFOUNDRY_STUDIO_HOST")
            or os.getenv("GRADIO_SERVER_NAME")
            or ""
        ).strip(),
        port=port,
    )
