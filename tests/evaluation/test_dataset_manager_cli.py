"""DS-09: DatasetManager locate/plan CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.cli.dataset import _handle_dataset_locate, _handle_dataset_plan


def test_dataset_locate_prefers_env_override(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    dataset_dir = tmp_path / "Howieeeee" / "WorldScore"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "marker.txt").write_text("ok", encoding="utf-8")
    from worldfoundry.evaluation.tasks.datasets.manager import dataset_location_env_var

    env_name = dataset_location_env_var("Howieeeee/WorldScore")
    monkeypatch.setenv(env_name, str(dataset_dir))

    args = type(
        "Args",
        (),
        {
            "dataset_id": "Howieeeee/WorldScore",
            "data_root": None,
            "manifest": None,
            "cache_dir": tmp_path / "hf-cache",
            "json": True,
        },
    )()
    captured: list[dict] = []
    monkeypatch.setattr("worldfoundry.cli.dataset.json_dump", lambda payload: captured.append(payload))
    code = _handle_dataset_locate(args)
    assert code == 0
    assert captured and captured[0]["ok"] is True
    assert Path(str(captured[0]["path"])) == dataset_dir


def test_dataset_plan_emits_commands(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    args = type(
        "Args",
        (),
        {
            "dataset_id": ["acme/demo-dataset"],
            "cache_dir": tmp_path / "hf-cache",
            "check_local": False,
            "json": True,
        },
    )()
    captured: list[dict] = []
    monkeypatch.setattr("worldfoundry.cli.dataset.json_dump", lambda payload: captured.append(payload))
    code = _handle_dataset_plan(args)
    assert code == 0
    assert captured
    assert captured[0]["dataset_ids"] == ["acme/demo-dataset"]
    assert captured[0]["commands"]
