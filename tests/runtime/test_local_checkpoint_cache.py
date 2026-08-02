from pathlib import Path

from worldfoundry.runtime.local_checkpoint_cache import stage_checkpoint_for_realtime


def test_stage_checkpoint_is_disabled_without_explicit_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source" / "model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", raising=False)

    assert stage_checkpoint_for_realtime(source, required_paths=("config.json",)) == source


def test_stage_checkpoint_reuses_immutable_local_copy(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source" / "model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"weights")
    cache = tmp_path / "local"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))

    first = stage_checkpoint_for_realtime(
        source,
        required_paths=("config.json", "weights.bin"),
    )
    second = stage_checkpoint_for_realtime(
        source,
        required_paths=("config.json", "weights.bin"),
    )

    assert first == second
    assert first != source
    assert (first / "weights.bin").read_bytes() == b"weights"


def test_stage_checkpoint_can_copy_only_runtime_components(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source" / "model"
    (source / "tokenizer").mkdir(parents=True)
    (source / "tokenizer" / "config.json").write_text("{}", encoding="utf-8")
    (source / "vae.bin").write_bytes(b"vae")
    (source / "unused-transformer.bin").write_bytes(b"unused")
    cache = tmp_path / "local"
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_STAGE_CHECKPOINT", "1")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_LOCAL_CHECKPOINT_CACHE", str(cache))

    staged = stage_checkpoint_for_realtime(
        source,
        required_paths=("vae.bin", "tokenizer/config.json"),
        include_paths=("vae.bin", "tokenizer"),
    )

    assert (staged / "vae.bin").read_bytes() == b"vae"
    assert (staged / "tokenizer" / "config.json").is_file()
    assert not (staged / "unused-transformer.bin").exists()
