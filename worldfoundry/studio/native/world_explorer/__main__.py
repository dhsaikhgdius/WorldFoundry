"""Build and launch the native World Explorer."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from pathlib import Path

from worldfoundry.core.io.paths import project_root

from . import NATIVE_EXPLORER_ROOT

REPOSITORY_ROOT = project_root(__file__)
RUNTIME_ROOT = (
	REPOSITORY_ROOT
	/ "worldfoundry"
	/ "synthesis"
	/ "visual_generation"
	/ "lyra_2"
)

def _setup(_: argparse.Namespace) -> None:
	from .setup_dependencies import main

	main()


def _build(args: argparse.Namespace) -> None:
	from .setup_dependencies import main

	main()
	build_dir = Path(args.build_dir).expanduser().resolve()
	configure = [
		"cmake",
		"-S",
		str(NATIVE_EXPLORER_ROOT),
		"-B",
		str(build_dir),
		f"-DCMAKE_BUILD_TYPE={args.build_type}",
		"-DNGP_BUILD_WITH_GUI=ON",
		"-DNGP_BUILD_WITH_PYTHON_BINDINGS=ON",
	]
	if args.cuda_architectures:
		configure.append(f"-DTCNN_CUDA_ARCHITECTURES={args.cuda_architectures}")
	subprocess.run(configure, check=True)
	subprocess.run(
		("cmake", "--build", str(build_dir), "--config", args.build_type, "-j", str(args.jobs)),
		check=True,
	)


def _configure_model_environment(args: argparse.Namespace) -> None:
	os.environ["WORLDFOUNDRY_EXPLORER_RUNTIME_ROOT"] = str(RUNTIME_ROOT)
	if args.model_id:
		os.environ["WORLDFOUNDRY_EXPLORER_MODEL_ID"] = args.model_id
	if args.backend:
		os.environ["WORLDFOUNDRY_EXPLORER_BACKEND"] = args.backend
	if args.checkpoint:
		os.environ["WORLDFOUNDRY_EXPLORER_CHECKPOINT_PATH"] = str(
			Path(args.checkpoint).expanduser().resolve()
		)
	if args.negative_prompt:
		os.environ["WORLDFOUNDRY_EXPLORER_NEGATIVE_PROMPT_PATH"] = str(
			Path(args.negative_prompt).expanduser().resolve()
		)
	if args.da3_checkpoint:
		os.environ["WORLDFOUNDRY_EXPLORER_DA3_CHECKPOINT_PATH"] = str(
			Path(args.da3_checkpoint).expanduser().resolve()
		)
	if args.qwen_model:
		os.environ["WORLDFOUNDRY_EXPLORER_QWEN_MODEL"] = str(
			Path(args.qwen_model).expanduser().resolve()
		)
	if args.dummy:
		os.environ["WORLDFOUNDRY_EXPLORER_DUMMY"] = "1"


def _client(args: argparse.Namespace) -> None:
	_configure_model_environment(args)
	from .api.backend_loader import DUMMY_BACKEND, backend_class
	from .api.client import WorldExplorerClient

	model_class = backend_class(
		DUMMY_BACKEND if args.dummy else args.backend,
		model_id=args.model_id,
	)
	model = model_class(checkpoint_path=os.environ.get("WORLDFOUNDRY_EXPLORER_CHECKPOINT_PATH"))
	try:
		client = WorldExplorerClient(
			files=args.files,
			model=model,
			width=args.width,
			height=args.height,
			inference_resolution=(args.width, args.height),
			seed_max_frames=args.seed_max_frames,
			seed_stride=args.seed_stride,
			output_dir=args.output_dir,
		)
		asyncio.run(client.run())
	except KeyboardInterrupt:
		return
	finally:
		model.cleanup()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m worldfoundry.studio.native.world_explorer"
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	setup = subparsers.add_parser("setup", help="fetch pinned native dependencies")
	setup.set_defaults(handler=_setup)

	build = subparsers.add_parser("build", help="build the CUDA/C++/ImGui viewer")
	build.add_argument("--build-dir", default=str(NATIVE_EXPLORER_ROOT / "build"))
	build.add_argument("--build-type", default="RelWithDebInfo")
	build.add_argument("--cuda-architectures", default="")
	build.add_argument("-j", "--jobs", type=int, default=max(os.cpu_count() or 1, 1))
	build.set_defaults(handler=_build)

	client = subparsers.add_parser("client", help="launch a local native ImGui world-model session")
	client.add_argument("files", nargs="*")
	client.add_argument("--model-id", help="WorldFoundry model id used to select a bundled adapter")
	client.add_argument("--checkpoint")
	client.add_argument("--negative-prompt")
	client.add_argument("--da3-checkpoint")
	client.add_argument("--qwen-model")
	client.add_argument("--backend", help="model adapter as python.module:Class")
	client.add_argument("--dummy", action="store_true")
	client.add_argument("--width", type=int, default=768)
	client.add_argument("--height", type=int, default=448)
	client.add_argument("--seed-max-frames", type=int)
	client.add_argument("--seed-stride", type=int, default=1)
	client.add_argument("--output-dir")
	client.set_defaults(handler=_client)
	return parser


def main(argv: list[str] | None = None) -> None:
	args = build_parser().parse_args(argv)
	args.handler(args)


if __name__ == "__main__":
	main()
