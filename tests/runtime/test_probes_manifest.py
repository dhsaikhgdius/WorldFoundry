from __future__ import annotations

from pathlib import Path

from worldfoundry.runtime import probes


def test_probe_env_manifest_resolves_gpu_and_environment_paths(tmp_path: Path):
    conda_root = tmp_path / "legacy"
    conda_envs = tmp_path / "envs"
    model_root = tmp_path / "models"
    candidates = probes.resolve_gpu_probe_candidates(conda_root, conda_envs)
    assert candidates["benchmark_cu113"] == conda_envs / "worldfoundry-zeroscope-cu113" / "bin" / "python"
    assert candidates["benchmark_worldplay"] == conda_root / "worldplay" / "bin" / "python"

    specs = probes.resolve_environment_probe_specs(conda_root, conda_envs, model_root)
    assert specs["base_current"]["python"] == Path(probes.sys.executable)
    assert specs["worldplay_vbench"]["pythonpath"][0] == model_root / "VBench"
    assert "torch" in specs["benchmark_cu113"]["modules"]


def test_probe_env_manifest_override_env(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "custom.yaml"
    manifest.write_text(
        "gpu_candidates:\n  only: \"{conda_envs_root}/custom/bin/python\"\n"
        "environment_specs: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLDFOUNDRY_PROBE_ENVS_MANIFEST", str(manifest))
    probes._load_probe_env_manifest.cache_clear()
    candidates = probes.resolve_gpu_probe_candidates(tmp_path / "legacy", tmp_path / "envs")
    assert list(candidates) == ["only"]
    assert candidates["only"] == tmp_path / "envs" / "custom" / "bin" / "python"
    probes._load_probe_env_manifest.cache_clear()
