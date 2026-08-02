"""Fetch pinned third-party native dependencies into ignored source directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = ROOT / "dependencies"


@dataclass(frozen=True)
class Dependency:
	name: str
	url: str
	commit: str


PINNED_DEPENDENCIES = (
	Dependency("fmt", "https://github.com/fmtlib/fmt", "fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa"),
	Dependency("glfw", "https://github.com/Tom94/glfw", "1f5e30f2bacf8faa4a0d13f51bb55193b098a808"),
	Dependency("imgui", "https://github.com/ocornut/imgui.git", "fa2b318dd6190852a6fe7ebc952b6551e93899e0"),
	Dependency("pybind11", "https://github.com/Tom94/pybind11", "7a5068336979377fbf4aa66bbaa483c4cb1c76a7"),
	Dependency("tinylogger", "https://github.com/Tom94/tinylogger", "2b9858edd349d688501327376b889fb8347054ea"),
)


def _run(*args: str, cwd: Path | None = None) -> None:
	subprocess.run(args, cwd=cwd, check=True)


def _head(path: Path) -> str:
	try:
		return subprocess.check_output(
			("git", "-C", str(path), "rev-parse", "HEAD"),
			text=True,
			stderr=subprocess.DEVNULL,
		).strip()
	except subprocess.CalledProcessError:
		return ""


def install_dependency(dependency: Dependency) -> None:
	"""Clone or validate one dependency without overwriting an existing checkout."""
	target = DEPENDENCIES / dependency.name
	git_dir = target / ".git"
	if git_dir.exists():
		if _head(target) != dependency.commit:
			_run("git", "-C", str(target), "fetch", "--depth", "1", "origin", dependency.commit)
			_run("git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD")
	elif target.exists() and any(target.iterdir()):
		raise RuntimeError(
			f"Refusing to overwrite non-git dependency directory: {target}"
		)
	else:
		target.mkdir(parents=True, exist_ok=True)
		_run("git", "-C", str(target), "init")
		_run("git", "-C", str(target), "remote", "add", "origin", dependency.url)
		_run("git", "-C", str(target), "fetch", "--depth", "1", "origin", dependency.commit)
		_run("git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD")

	if _head(target) != dependency.commit:
		raise RuntimeError(f"Dependency pin mismatch for {dependency.name}")
	_run("git", "-C", str(target), "submodule", "update", "--init", "--recursive")


def main() -> None:
	DEPENDENCIES.mkdir(parents=True, exist_ok=True)
	for dependency in PINNED_DEPENDENCIES:
		print(f"[world-explorer] {dependency.name} @ {dependency.commit[:12]}", flush=True)
		install_dependency(dependency)


if __name__ == "__main__":
	main()
