"""Studio launcher for the local CUDA/C++/ImGui World Explorer."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

from worldfoundry.core.process import run_logged_subprocess
from worldfoundry.core.world_explorer import WORLD_EXPLORER_TAG

from . import NATIVE_EXPLORER_ROOT


def _binding_available(build_dir: Path) -> bool:
	patterns = ("pyngp*.so", "pyngp*.pyd", "pyngp*.dylib")
	return any(
		glob.glob(str(root / "**" / pattern), recursive=True)
		for root in (NATIVE_EXPLORER_ROOT, build_dir)
		for pattern in patterns
	)


def launch_from_studio(entry, launch_config) -> None:
	"""Load the selected model directly into the native viewer process."""
	tags = {str(tag).strip().lower() for tag in entry.tags}
	if WORLD_EXPLORER_TAG not in tags:
		raise SystemExit(
			"The selected model does not declare the World Explorer camera/session contract."
		)
	if launch_config.endpoint:
		raise SystemExit(
			"Native World Explorer now runs the selected model in the viewer process; "
			"remote HTTP endpoints are not supported."
		)

	build_dir = Path(
		os.environ.get(
			"WORLDFOUNDRY_EXPLORER_BUILD_DIR",
			str(NATIVE_EXPLORER_ROOT / "build"),
		)
	).expanduser().resolve()
	if not _binding_available(build_dir):
		raise SystemExit(
			"Native World Explorer bindings are not built. Run "
			"`python -m worldfoundry.studio.native.world_explorer build` first."
		)

	client_command = [
		sys.executable,
		"-m",
		"worldfoundry.studio.native.world_explorer",
		"client",
		"--model-id",
		str(entry.model_id),
	]
	required = dict(entry.default_load_kwargs.get("required_components") or {})
	checkpoint = launch_config.model_ref or required.get("checkpoint_dir")
	if checkpoint and str(checkpoint).strip().lower() != str(entry.model_id).strip().lower():
		client_command.extend(("--checkpoint", str(checkpoint)))
	if required.get("negative_prompt_path"):
		client_command.extend(("--negative-prompt", str(required["negative_prompt_path"])))
	if required.get("da3_model_path_custom"):
		client_command.extend(("--da3-checkpoint", str(required["da3_model_path_custom"])))
	log_dir = build_dir / "worldfoundry_launch_logs"
	stdout_path = log_dir / "client.stdout.log"
	stderr_path = log_dir / "client.stderr.log"
	completed = run_logged_subprocess(
		client_command,
		stdout_path=stdout_path,
		stderr_path=stderr_path,
		cwd=NATIVE_EXPLORER_ROOT,
		env=os.environ.copy(),
	)
	if completed.returncode != 0:
		raise SystemExit(
			f"Native World Explorer client exited with code {completed.returncode}; "
			f"see {stdout_path} and {stderr_path}"
		)


__all__ = ["launch_from_studio"]
